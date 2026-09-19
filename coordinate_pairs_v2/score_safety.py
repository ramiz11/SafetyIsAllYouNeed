"""Recompute the publication-compatible safety values after model inference."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from .metrics import read_json, sha256_file
from .safety_metrics import aggregate_safety, normalized_safety


def load_top1_predictions(path: str | Path) -> list[int]:
    payload = read_json(path)
    if "predictions" in payload:
        return [int(value) for value in payload["predictions"]]
    if "records" in payload:
        result = []
        for row in payload["records"]:
            beam = row.get("beam1") or row.get("generation", {}).get("beam1", {}).get("new_ids")
            result.append(int(beam[0]) if beam else -1)
        return result
    raise ValueError("Unsupported prediction artifact schema")


def _route_key(start: tuple[float, float], end: tuple[float, float], precision=None) -> str:
    if precision is not None:
        start = tuple(round(float(value), precision) for value in start)
        end = tuple(round(float(value), precision) for value in end)
    return f"{(float(start[0]), float(start[1]))}_{(float(end[0]), float(end[1]))}"


def get_route(start, end, cache, *, allow_live_osrm: bool):
    for key in (_route_key(start, end), _route_key(start, end, 6)):
        if key in cache:
            return cache[key], "frozen_cache"
    if start == end:
        return [list(start), list(end)], "same_coordinate"
    if not allow_live_osrm:
        return [list(start), list(end)], "straight_fallback_no_network"
    import requests

    url = (
        "https://router.project-osrm.org/route/v1/walking/"
        f"{start[0]},{start[1]};{end[0]},{end[1]}"
        "?overview=full&geometries=geojson&radiuses=1000;1000"
    )
    for attempt in range(3):
        try:
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "SafetyIsAllYouNeed/1.0"}
            )
            payload = response.json() if response.status_code == 200 else {}
            if payload.get("code") == "Ok" and payload.get("routes"):
                coordinates = payload["routes"][0]["geometry"]["coordinates"]
                cache[_route_key(start, end)] = coordinates
                return coordinates, "live_osrm"
        except Exception:
            pass
        time.sleep(0.5 * 2**attempt)
    return [list(start), list(end)], "straight_fallback_after_osrm"


def build_crimes(path: Path, city: str, timezone: str, projected_crs: int, *, major_nyc_only: bool):
    import geopandas as gpd
    import pandas as pd

    frame = pd.read_csv(path)
    if city == "NYC":
        start = pd.to_datetime(frame["complaint_date_start"], errors="coerce")
        end = pd.to_datetime(frame["complaint_date_end"], errors="coerce")
        frame["crime_start_utc"] = start.dt.tz_localize(
            timezone, nonexistent="shift_forward", ambiguous="NaT"
        ).dt.tz_convert("UTC")
        frame["crime_end_utc"] = end.dt.tz_localize(
            timezone, nonexistent="shift_forward", ambiguous="NaT"
        ).dt.tz_convert("UTC")
        if major_nyc_only and "LAW_CAT_CD" in frame.columns:
            frame = frame[frame["LAW_CAT_CD"].isin(["MISDEMEANOR", "FELONY"])].copy()
    else:
        stamp = pd.to_datetime(
            frame["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        stamp = stamp.dt.tz_localize(
            timezone, nonexistent="shift_forward", ambiguous="NaT"
        ).dt.tz_convert("UTC")
        frame["crime_start_utc"] = stamp
        frame["crime_end_utc"] = stamp
    frame = frame.dropna(
        subset=["Latitude", "Longitude", "crime_start_utc", "crime_end_utc"]
    ).copy()
    geometry = gpd.points_from_xy(frame["Longitude"], frame["Latitude"])
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326").to_crs(
        epsg=projected_crs
    )


def build_windowed_train_poi_catalog(trajectories) -> dict[int, tuple[float, float]]:
    catalog = {}
    for trajectory in trajectories:
        for row in trajectory.itertuples(index=False):
            catalog.setdefault(int(row.poi_id), (float(row.longitude), float(row.latitude)))
    return catalog


def _timestamp(row, frame, timezone):
    import pandas as pd

    value = next(
        (row[column] for column in ("event_time_utc", "utc_time", "local_time") if column in frame),
        None,
    )
    if value is None:
        raise KeyError("No trajectory timestamp column")
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    return result.tz_convert("UTC")


def score_safety(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    city: str,
    predictions_path: str | Path,
    output_path: str | Path,
    allow_live_osrm: bool = False,
) -> dict:
    import geopandas as gpd
    from shapely.geometry import LineString

    root = Path(repo_root).resolve()
    config = read_json(config_path)
    profile = config["data_profiles"][city]
    runtime = profile["safety_runtime"]
    data_root = root / profile["path"]
    with (data_root / "train_trajectories.pickle").open("rb") as handle:
        train = pickle.load(handle)
    with (data_root / "test_trajectories.pickle").open("rb") as handle:
        test = pickle.load(handle)
    predictions = load_top1_predictions(predictions_path)
    if len(predictions) != len(test):
        raise ValueError(f"Expected {len(test)} predictions, found {len(predictions)}")
    poi_map = build_windowed_train_poi_catalog(train)

    def checked_path(details):
        path = root / details["path"]
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"Frozen safety input hash changed: {path}")
        return path

    route_cache_path = checked_path(runtime["route_cache"])
    stats_path = checked_path(runtime["normalization_stats"])
    crime_path = checked_path(runtime["crime_source"])
    with route_cache_path.open("rb") as handle:
        route_cache = pickle.load(handle)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    crimes = build_crimes(
        crime_path,
        city,
        profile["timezone"],
        int(runtime["projected_crs"]),
        major_nyc_only=bool(runtime["major_nyc_offenses_only"]),
    )
    rows = []
    route_sources: Counter[str] = Counter()
    scores: list[float | None] = []
    for index, (prediction, trajectory) in enumerate(zip(predictions, test)):
        frame = trajectory.reset_index(drop=True)
        if prediction not in poi_map:
            rows.append({"index": index, "prediction": prediction, "status": "invalid_poi"})
            scores.append(None)
            continue
        # This historical post-inference definition uses the held-out visit as
        # route origin. It is never used in a prompt, ranker, or generation.
        origin = frame.iloc[-1]
        start = (float(origin.longitude), float(origin.latitude))
        destination = poi_map[prediction]
        route, route_source = get_route(
            start, destination, route_cache, allow_live_osrm=allow_live_osrm
        )
        route_sources[route_source] += 1
        line = gpd.GeoSeries([LineString(route)], crs="EPSG:4326").to_crs(
            epsg=int(runtime["projected_crs"])
        ).iloc[0]
        buffer = line.buffer(float(profile["crime_radius_m"]))
        timestamp = _timestamp(origin, frame, profile["timezone"])
        start_time = timestamp - timedelta(weeks=int(profile["crime_time_weeks"]))
        temporal = crimes[
            (crimes["crime_start_utc"] >= start_time)
            & (crimes["crime_end_utc"] <= timestamp)
        ]
        crime_count = int(temporal.geometry.intersects(buffer).sum())
        score = normalized_safety(crime_count, stats)
        scores.append(score)
        rows.append(
            {
                "index": index,
                "prediction": prediction,
                "route_source": route_source,
                "crime_count": crime_count,
                "safety": score,
                "status": "scored",
            }
        )
    summary = aggregate_safety(scores, total_predictions=len(predictions), aggregation="median")
    summary.update(
        {
            "contract": config["contract"],
            "city": city,
            "projected_crs": runtime["projected_crs"],
            "nominal_buffer": profile["crime_radius_m"],
            "effective_buffer_m": profile["crime_radius_m"] * (1200.0 / 3937.0),
            "crime_window_weeks": profile["crime_time_weeks"],
            "route_origin": runtime["route_origin"],
            "major_nyc_offenses_only": runtime["major_nyc_offenses_only"],
            "allow_live_osrm": allow_live_osrm,
            "route_source_counts": dict(route_sources),
            "poi_catalog_source": runtime["poi_catalog"],
            "poi_catalog_size": len(poi_map),
            "input_sha256": {
                "route_cache": runtime["route_cache"]["sha256"],
                "normalization_stats": runtime["normalization_stats"]["sha256"],
                "crime_source": runtime["crime_source"]["sha256"],
            },
        }
    )
    result = {"summary": summary, "records": rows}
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="configs/coordinate_pairs_v2.json")
    parser.add_argument("--city", choices=("NYC", "CHICAGO"), required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-live-osrm", action="store_true")
    args = parser.parse_args()
    result = score_safety(
        repo_root=args.repo_root,
        config_path=Path(args.repo_root) / args.config,
        city=args.city,
        predictions_path=args.predictions,
        output_path=args.output,
        allow_live_osrm=args.allow_live_osrm,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
