from __future__ import annotations

import json
import math
import pickle
import pickle as pkl
import random
import statistics
import time
import warnings
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
    allow_straight_fallback: bool = True,
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

    if not allow_straight_fallback:
        return None, route_coordinates_hashmap
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


def _route_key(start: tuple[float, float], end: tuple[float, float], precision=None) -> str:
    if precision is not None:
        start = tuple(round(float(value), precision) for value in start)
        end = tuple(round(float(value), precision) for value in end)
    return f"{(float(start[0]), float(start[1]))}_{(float(end[0]), float(end[1]))}"


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


def training_poi_catalog(train, raw_path: Path, timezone: str) -> dict[int, tuple[float, float]]:
    import preprocessing as pp

    catalog = build_windowed_train_poi_catalog(train)
    previous_timezone = pc.CITY_TZ
    try:
        pc.CITY_TZ = timezone
        raw = pp.load_checkins_dataset(str(raw_path))
        raw_train, raw_validation, raw_test = pp.remove_unseen_pois_users(
            *pp.time_based_split(raw, pc.TIME_SPLIT_RATIOS)
        )
        raw_train, _, _ = pp.reindex_poi_ids(raw_train, raw_validation, raw_test)
    finally:
        pc.CITY_TZ = previous_timezone
    if not set(catalog).issubset(set(raw_train.poi_id)):
        raise ValueError("Windowed TRAIN POIs do not match raw TRAIN identifiers")
    for row in raw_train.itertuples(index=False):
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


def score_safety(*, repo_root, config_path, city, predictions_path, output_path):
    from pyproj import Transformer
    from shapely.geometry import LineString
    from shapely.ops import transform

    root = Path(repo_root).resolve()
    config = pc.load_model_config(config_path, repo_root=root)
    profile = config["data_profiles"][city]
    runtime = profile["safety_runtime"]
    data_root = root / profile["path"]
    with (data_root / "train_trajectories.pickle").open("rb") as handle:
        train = pickle.load(handle)
    with (data_root / "test_trajectories.pickle").open("rb") as handle:
        test = pickle.load(handle)
    prediction_payload = read_json(predictions_path)
    records = prediction_payload["records"]
    if len(records) != len(test):
        raise ValueError("Prediction and TEST row counts differ")
    row_key = prediction_payload.get("summary", {}).get("row_key")
    row = next((item for item in config.get("rows", []) if item["row_key"] == row_key), None)
    if row_key is not None and row is None:
        raise ValueError(f"Unknown inference row: {row_key}")
    parser = row["inference"]["parser"] if row else "new_ids"

    cache_path = root / runtime["route_cache"]["path"]
    fallback_routes = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    def save_cache():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(fallback_routes, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)

    with (data_root / "safety/train_trajs_with_safety.pickle").open("rb") as handle:
        scored_train = pickle.load(handle)
    train_counts = [int(value) for frame in scored_train for value in frame["crimes_count"]
                    if value >= 0]
    train_stats = derive_distribution_stats([train_counts])
    crimes = build_crimes(
        root / runtime["crime_source"]["path"], city, profile["timezone"],
        int(runtime["projected_crs"]),
        major_nyc_only=bool(runtime["major_nyc_offenses_only"]),
    )
    crimes = crimes[crimes.crime_end_utc >= crimes.crime_start_utc]
    raw_path = root / runtime["raw_checkins"]["path"]
    catalog = training_poi_catalog(train, raw_path, profile["timezone"])
    projection = Transformer.from_crs(4326, int(runtime["projected_crs"]), always_xy=True)
    buffers = {}
    rows = []
    sources = Counter()
    for index, (record, trajectory) in enumerate(zip(records, test)):
        raw = record["generation"]["beam1"][parser]
        poi = int(raw[0]) if raw else -1
        origin = trajectory.iloc[-2]
        start = (float(origin.longitude), float(origin.latitude))
        timestamp = _timestamp(origin, trajectory, profile["timezone"])
        candidate = {"poi": poi, "safety": None, "status": "invalid_poi"}
        if poi in catalog:
            key = _route_key(start, catalog[poi], 6)
            route = fallback_routes.get(key)
            source = "route_cache" if route is not None else None
            if route is None:
                route, fallback_routes = get_route_coordinates(
                    start, catalog[poi], fallback_routes, allow_straight_fallback=False
                )
                source = "osrm" if route is not None else None
                if route is not None:
                    save_cache()
            if route is None:
                raise RuntimeError(f"Could not route {city} prediction {index}, POI {poi}: {key}")
            if route is None or len(route) < 2 or len({tuple(point) for point in route}) < 2:
                candidate["status"] = "missing_route"
            else:
                if key not in buffers:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        line = transform(projection.transform, LineString(route))
                    buffers[key] = line.buffer(float(profile["crime_radius_m"]))
                nearby = crimes.iloc[crimes.sindex.query(buffers[key], predicate="intersects")]
                count = int(((nearby.crime_start_utc >= timestamp - pd.Timedelta(weeks=int(profile["crime_time_weeks"])))
                             & (nearby.crime_end_utc <= timestamp)).sum())
                candidate.update(status="scored", safety=normalized_safety(count, train_stats),
                                 crime_count=count, route_source=source)
                sources[source] += 1
        score = candidate["safety"]
        rows.append({"index": index, "prediction": poi,
                     "status": "scored" if score is not None else "unscored",
                     "safety": score, "route": candidate})
    scores = [row["safety"] for row in rows]
    summary = aggregate_safety(scores, total_predictions=len(rows), aggregation="median")
    summary.update({"contract": config["contract"], "city": city,
                    "projected_crs": runtime["projected_crs"],
                    "nominal_buffer": profile["crime_radius_m"],
                    "crime_window_weeks": profile["crime_time_weeks"],
                    "route_origin": runtime["route_origin"],
                    "major_nyc_offenses_only": runtime["major_nyc_offenses_only"],
                    "poi_catalog_source": runtime["poi_catalog"],
                    "allow_live_osrm": True, "route_source_counts": dict(sources),
                    "input_sha256": {
                        key: sha256_file(root / runtime[key]["path"])
                        for key in ("crime_source", "raw_checkins")
                    } | {"safety_train": sha256_file(data_root / "safety/train_trajs_with_safety.pickle")}})
    result = {"summary": summary, "records": rows}
    Path(output_path).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
