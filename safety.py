from __future__ import annotations

import gzip
import json
import math
import pickle
import pickle as pkl
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import preprocessing_config as pc
from text_utils import read_json, sha256_file


def robust_scale(x: int, stats: dict) -> float:
    median = stats['50%']
    iqr = stats['75%'] - stats['25%']
    if iqr == 0:
        return 0.0
    scaled = (x - median) / iqr
    return min(1.0, max(0.0, scaled))


def get_route_coordinates(
    current_coords: tuple, # (lon, lat)
    next_coords: tuple, # (lon, lat)
    route_coordinates_hashmap: dict,
    max_retries: int = 3,
    backoff_s: float = 0.5,
    snap_radius_m: int = 1000,
    min_hop_m: int = 25, # tiny steps -> straight line, skip OSRM
    key_precision: int = 6, # quantize for better cache hit rate
):
    """
    Query OSRM (walking) to get route geometry (lon,lat)->(lon,lat).
    Returns (route_coordinates, updated_hashmap).
    """

    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))

    lon1, lat1 = current_coords
    lon2, lat2 = next_coords
    # Quantize to stabilize keys (avoid float noise creating new cache keys)
    q = lambda x: round(float(x), key_precision)
    lon1, lat1, lon2, lat2 = q(lon1), q(lat1), q(lon2), q(lat2)
    # Return straight segment
    if lon1 == lon2 and lat1 == lat2:
        coords = [[lon1, lat1], [lon2, lat2]]
        route_coordinates_hashmap[f"{(lon1, lat1)}_{(lon2, lat2)}"] = coords
        return coords, route_coordinates_hashmap

    if _haversine_m(lat1, lon1, lat2, lon2) < min_hop_m:
        coords = [[lon1, lat1], [lon2, lat2]]
        route_coordinates_hashmap[f"{(lon1, lat1)}_{(lon2, lat2)}"] = coords
        return coords, route_coordinates_hashmap

    hash_id = f"{(lon1, lat1)}_{(lon2, lat2)}"
    if hash_id in route_coordinates_hashmap:
        return route_coordinates_hashmap[hash_id], route_coordinates_hashmap

    # Add radiuses to allow snapping
    base_url = pc.OSRM_BASE_URL.rstrip("/")
    url = (f"{base_url}/{lon1},{lat1};{lon2},{lat2}"
           f"?overview=full&geometries=geojson&radiuses={snap_radius_m};{snap_radius_m}")

    last_err = None
    for attempt in range(max_retries):
        try:
            import requests
            r = requests.get(url, timeout=20, headers={"User-Agent":"SafetyIsAllYouNeed/1.0"})
            if r.status_code == 200:
                data = r.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    coords = data["routes"][0]["geometry"]["coordinates"]
                    route_coordinates_hashmap[hash_id] = coords
                    return coords, route_coordinates_hashmap
                else:
                    last_err = f"OSRM code={data.get('code')} msg={data.get('message')}"
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        # backoff + jitter
        time.sleep(backoff_s * (2 ** attempt) + random.uniform(0, 0.1))

    # Fallback: straight line so pipeline continues
    coords = [[lon1, lat1], [lon2, lat2]]
    route_coordinates_hashmap[hash_id] = coords
    return coords, route_coordinates_hashmap


def compute_route_crimes(route_coords,
                         poi_timestamp,
                         crime_gdf: gpd.GeoDataFrame,
                         buffer_meters: int,
                         time_window_weeks: int) -> int:
    """
    Filter crimes in [poi_timestamp - time_window_weeks, poi_timestamp] (in UTC),
    buffer route (meters), count intersecting crime points.
    """

    import geopandas as gpd
    from shapely.geometry import LineString
    # ensure UTC timestamp
    if getattr(poi_timestamp, "tzinfo", None) is None:
        poi_ts_utc = pd.Timestamp(poi_timestamp).tz_localize(pc.CITY_TZ).tz_convert('UTC')
    else:
        poi_ts_utc = pd.Timestamp(poi_timestamp).tz_convert('UTC')

    start_utc = poi_ts_utc - timedelta(weeks=time_window_weeks)

    subset = crime_gdf[
        (crime_gdf['crime_start_utc'] >= start_utc) &
        (crime_gdf['crime_end_utc'] <= poi_ts_utc)
    ]

    route_line = LineString(route_coords)  # [[lon, lat], ...]
    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    crimes_proj = subset.to_crs(epsg=pc.PROJ_CRS_EPSG)
    route_proj = route_gdf.to_crs(epsg=pc.PROJ_CRS_EPSG)
    route_buffer = route_proj.buffer(buffer_meters).geometry.iloc[0]
    buffer_gdf = gpd.GeoDataFrame(geometry=[route_buffer], crs=crimes_proj.crs)
    # spatial join to count
    nearby = gpd.sjoin(crimes_proj, buffer_gdf, predicate='intersects', how='inner')
    return len(nearby)


def calculate_route_safety(route_coords, crime_data, poi_timestamp_utc, normalize, dist_stats):
    cnt = compute_route_crimes(route_coords, poi_timestamp_utc, crime_data,
                               buffer_meters=pc.CRIME_RADIUS,
                               time_window_weeks=pc.CRIME_TIME_WINDOW)
    if not normalize:
        return cnt
    if cnt == -1:
        return -1
    return 1 - robust_scale(cnt, dist_stats)


def compute_crime_scores_for_trajectories(trajectories: list,
                                          crime_gdf: gpd.GeoDataFrame,
                                          segments_crime_hashmap: dict,
                                          segments_coordinates_hashmap: dict,
                                          crime_radius: int,
                                          crime_time_window: int) -> list:
    """
    For each traj, compute raw crime counts for each segment.
    Cache both route coords and segment crime counts.
    """

    import geopandas as gpd
    all_crime_scores = []
    for traj_df in tqdm(trajectories, desc="Calculating crime counts"):
        current_crime_counts = [-1] * len(traj_df)
        df = traj_df.reset_index(drop=True)
        for i in range(len(df) - 1):
            lat1, lon1, poi_id1, t1 = df.loc[i, 'latitude'], df.loc[i, 'longitude'], df.loc[i, 'poi_id'], df.loc[i, 'event_time_utc']
            lat2, lon2, poi_id2, _ = df.loc[i + 1, 'latitude'], df.loc[i + 1, 'longitude'], df.loc[i + 1, 'poi_id'], df.loc[i + 1, 'event_time_utc']
            day1, month1, year1 = t1.day, t1.month, t1.year
            seg_id = f"{poi_id1}_{day1}/{month1}/{year1}_{poi_id2}"
            if seg_id in segments_crime_hashmap:
                crimes_count = segments_crime_hashmap[seg_id]
            else:
                route_coords, segments_coordinates_hashmap = get_route_coordinates((lon1, lat1), (lon2, lat2), segments_coordinates_hashmap)
                crimes_count = compute_route_crimes(route_coords, t1, crime_gdf, crime_radius, crime_time_window)
                segments_crime_hashmap[seg_id] = crimes_count


            current_crime_counts[i] = crimes_count


        # persist caches
        with open(pc.SEGMENTS_COORDINATES_HASHMAP_PKL_PATH, 'wb') as f:
            pkl.dump(segments_coordinates_hashmap, f)
        with open(pc.SEGMENTS_CRIMES_HASHMAP_JSON_PATH, 'w') as f:
            json.dump(segments_crime_hashmap, f)

        all_crime_scores.append(current_crime_counts)
    return all_crime_scores


def derive_distribution_stats(crime_scores_list: list):
    vals = [v for sub in crime_scores_list for v in sub if v != -1]
    arr = np.array(vals) if len(vals) else np.array([0])
    return {
        'min': float(arr.min()),
        '25%': float(np.percentile(arr, 25)),
        '50%': float(np.percentile(arr, 50)),
        '75%': float(np.percentile(arr, 75)),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std':  float(arr.std())
    }


def apply_safety_scores_to_train_trajectories(trajectories: list,
                                              crime_scores_list: list,
                                              dist_stats: dict) -> list:
    updated = []
    for i, traj_df in enumerate(trajectories):
        traj_df = traj_df.copy()
        scores = crime_scores_list[i]
        norm_safety = []
        for c in scores:
            if c == -1:
                norm_safety.append(-1)
            else:
                norm_safety.append(1 - robust_scale(c, dist_stats))
        traj_df['crimes_count'] = scores
        traj_df['normalized_safety'] = norm_safety
        updated.append(traj_df)
    return updated


def apply_safety_scores_to_non_train_trajectories(trajectories: list,
                                                  crime_gdf: gpd.GeoDataFrame,
                                                  dist_stats: dict,
                                                  segments_crime_hashmap: dict,
                                                  segments_coordinates_hashmap: dict,
                                                  crime_radius: int,
                                                  crime_time_window: int) -> list:

    import geopandas as gpd
    updated = []
    for traj_df in tqdm(trajectories):
        traj_df = traj_df.copy()
        safety_values = [-1] * len(traj_df)
        df = traj_df.reset_index(drop=True)
        for i in range(len(df) - 1):
            lat1, lon1, poi_id1, t1 = df.loc[i, 'latitude'], df.loc[i, 'longitude'], df.loc[i, 'poi_id'], df.loc[i, 'event_time_utc']
            lat2, lon2, poi_id2 = df.loc[i + 1, 'latitude'], df.loc[i + 1, 'longitude'], df.loc[i + 1, 'poi_id']
            day1, month1, year1 = t1.day, t1.month, t1.year
            seg_id = f"{poi_id1}_{day1}/{month1}/{year1}_{poi_id2}"

            if seg_id in segments_crime_hashmap:
                crimes_count = segments_crime_hashmap[seg_id]
            else:
                route_coords, segments_coordinates_hashmap = get_route_coordinates((lon1, lat1), (lon2, lat2), segments_coordinates_hashmap)
                crimes_count = compute_route_crimes(route_coords, t1, crime_gdf, crime_radius, crime_time_window)
                segments_crime_hashmap[seg_id] = crimes_count

            route_safety = 1 - robust_scale(crimes_count, dist_stats)
            safety_values[i] = route_safety

        with open(pc.SEGMENTS_COORDINATES_HASHMAP_PKL_PATH, 'wb') as f:
            pkl.dump(segments_coordinates_hashmap, f)
        with open(pc.SEGMENTS_CRIMES_HASHMAP_JSON_PATH, 'w') as f:
            json.dump(segments_crime_hashmap, f)

        traj_df['normalized_safety'] = safety_values
        updated.append(traj_df)
    return updated


US_SURVEY_FOOT_METERS = 1200.0 / 3937.0


@dataclass(frozen=True)
class SafetySemantics:
    name: str
    projected_crs: int
    nominal_buffer: float
    crime_window_weeks: int
    aggregation: str
    route_origin: str
    omit_invalid_predictions: bool
    nyc_major_offenses_only: bool

    @property
    def effective_buffer_meters(self) -> float:
        if self.projected_crs == 2263:
            return self.nominal_buffer * US_SURVEY_FOOT_METERS
        return self.nominal_buffer


def _evaluation_robust_scale(value: float, stats: dict[str, float]) -> float:
    iqr = float(stats["75%"]) - float(stats["25%"])
    if iqr == 0:
        return 0.0
    return min(1.0, max(0.0, (float(value) - float(stats["50%"])) / iqr))


def normalized_safety(crime_count: float, stats: dict[str, float]) -> float:
    return 1.0 - _evaluation_robust_scale(crime_count, stats)


def aggregate_safety(
    scores: Iterable[float | None],
    *,
    total_predictions: int,
    aggregation: str,
) -> dict:
    valid = [float(value) for value in scores if value is not None]
    # Compute the mean directly from the normalized float values above.
    mean = sum(valid) / len(valid) if valid else None
    median = statistics.median(valid) if valid else None
    if aggregation == "median":
        value = median
    elif aggregation == "mean":
        value = mean
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    return {
        "value": value,
        "aggregation": aggregation,
        "mean": mean,
        "median": median,
        "valid_routes": len(valid),
        "total_predictions": total_predictions,
        "coverage": len(valid) / total_predictions if total_predictions else 0.0,
    }


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
    config = pc.load_model_config(config_path)
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
        origin = frame.iloc[-2]
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
            "effective_buffer_m": profile["crime_radius_m"],
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
