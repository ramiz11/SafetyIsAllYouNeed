import json
import time
import random
import math
import requests
import pickle as pkl
from datetime import timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from tqdm import tqdm
from configs import preprocessing_config as pc


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