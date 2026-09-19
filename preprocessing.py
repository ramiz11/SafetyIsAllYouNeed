from __future__ import annotations

import math
import pickle as pkl
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from configs import preprocessing_config as pc


def build_crime_geodf(crime_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Normalize to a common schema:
      - geometry in EPSG:4326
      - tz-aware UTC time columns: crime_start_utc, crime_end_utc
    NYC: 'complaint_date_start'/'complaint_date_end' (local) -> localize to CITY_TZ -> UTC
    CHICAGO: 'Date' (local) -> localize to CITY_TZ -> UTC (start=end)
    Drops rows without coords or time.
    """

    import geopandas as gpd
    tz = pc.CITY_TZ

    if pc.DATASET == "NYC":
        if not pd.api.types.is_datetime64_any_dtype(crime_df['complaint_date_start']):
            crime_df['complaint_date_start'] = pd.to_datetime(crime_df['complaint_date_start'], errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(crime_df['complaint_date_end']):
            crime_df['complaint_date_end'] = pd.to_datetime(crime_df['complaint_date_end'], errors="coerce")
        s = crime_df['complaint_date_start'].dt.tz_localize(tz).dt.tz_convert('UTC')
        e = crime_df['complaint_date_end']  .dt.tz_localize(tz).dt.tz_convert('UTC')
        crime_df['crime_start_utc'] = s
        crime_df['crime_end_utc'] = e

    elif pc.DATASET == "CHICAGO":
        if not pd.api.types.is_datetime64_any_dtype(crime_df['Date']):
            crime_df['Date'] = pd.to_datetime(crime_df['Date'])
        t = crime_df['Date'].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert('UTC')
        crime_df['crime_start_utc'] = t
        crime_df['crime_end_utc'] = t
    else:
        raise ValueError("Unknown DATASET")

    # Drop rows without coords or time
    crime_df = crime_df.dropna(subset=['Latitude', 'Longitude', 'crime_start_utc', 'crime_end_utc']).copy()
    geometry = gpd.points_from_xy(crime_df['Longitude'], crime_df['Latitude'])
    return gpd.GeoDataFrame(crime_df, geometry=geometry, crs="EPSG:4326")


def load_checkins_dataset(csv_path: str) -> pd.DataFrame:
    """
    Robust loader for check-ins with mixed time encodings.
    Creates:
      - event_time_utc (tz-aware UTC)
      - local_time     (tz-aware in pc.CITY_TZ)
    Ensures 'category' exists (may be None) - done for code consistency.
    """
    df = pd.read_csv(csv_path)

    # Pick a time column
    time_col = "local_time" if "local_time" in df.columns else (
        "checkin_time" if "checkin_time" in df.columns else None
    )
    if time_col is None:
        raise KeyError("Expected a 'local_time' or 'checkin_time' column in check-ins CSV.")

    # Normalize to string for detection (handles mixed objects/strings)
    s = df[time_col].astype(str)
    # Detect strings that already carry tz-info: 'Z' or explicit +HH:MM / -HH:MM at the end
    has_tz = s.str.endswith("Z") | s.str.contains(r"[+-]\d{2}:\d{2}$")
    # tz-aware strings → parse directly to UTC
    event_utc_tzaware = pd.to_datetime(s[has_tz], errors="coerce", utc=True)
    # Naive strings → parse naive, localize to city tz, then convert to UTC
    naive_parsed = pd.to_datetime(s[~has_tz], errors="coerce", utc=False)
    # localize & convert; choose DST policy you prefer:
    naive_localized_utc = (
        naive_parsed
        .dt.tz_localize(pc.CITY_TZ, nonexistent="shift_forward", ambiguous="infer")
        .dt.tz_convert("UTC")
    )
    # stitch back together
    event_time_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    event_time_utc.loc[has_tz] = event_utc_tzaware
    event_time_utc.loc[~has_tz] = naive_localized_utc
    df["event_time_utc"] = event_time_utc
    df["local_time"] = df["event_time_utc"].dt.tz_convert(pc.CITY_TZ)
    # ensure category column exists (Chicago may not have it)
    if "category" not in df.columns and "poi_category_name" not in df.columns:
        df["category"] = None
    # basic hygiene
    df = df.dropna(subset=["latitude", "longitude", "poi_id", "user_id", "event_time_utc"]).copy()
    df = df.sort_values("event_time_utc").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, ratios: dict):
    """
    Time-based split on the sorted DataFrame.
    We assume df is sorted by 'event_time_utc'.
    """
    n = len(df)
    train_end = int(ratios['train'] * n)
    val_end = train_end + int(ratios['validation'] * n)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def remove_unseen_pois_users(train_df, val_df, test_df):
    """
    Remove any POI or user in val/test that does not exist in train.
    """
    train_pois = set(train_df['poi_id'].unique())
    train_users = set(train_df['user_id'].unique())
    val_df = val_df[val_df['poi_id'].isin(train_pois) & val_df['user_id'].isin(train_users)]
    test_df = test_df[test_df['poi_id'].isin(train_pois) & test_df['user_id'].isin(train_users)]
    return train_df, val_df, test_df


def reindex_poi_ids(train_df, val_df, test_df):
    """
    Create a contiguous [0..M-1] mapping of POI IDs across train/val/test.
    """
    all_pois = pd.concat([train_df, val_df, test_df])['poi_id'].unique()
    all_pois_sorted = sorted(all_pois)
    poi_map = {old_id: new_id for new_id, old_id in enumerate(all_pois_sorted)}
    train_df = train_df.assign(poi_id=train_df['poi_id'].map(poi_map))
    val_df = val_df.assign(poi_id=val_df['poi_id'].map(poi_map))
    test_df = test_df.assign(poi_id=test_df['poi_id'].map(poi_map))
    return train_df, val_df, test_df


def label_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine splits and add 'phase' column; sort by [user_id, event_time_utc]
    """
    train_df_copy = train_df.copy()
    train_df_copy["phase"] = "train"
    validation_df_copy = validation_df.copy()
    validation_df_copy["phase"] = "validation"
    test_df_copy = test_df.copy()
    test_df_copy["phase"] = "test"
    df = pd.concat([train_df_copy, validation_df_copy, test_df_copy], ignore_index=True)
    return df.sort_values(["user_id", "event_time_utc"]).reset_index(drop=True)


def extract_trajs_for_phase(
        df: pd.DataFrame,
        target_phase: str,
        time_threshold: pd.Timedelta,
        traj_len: int,
        stride: int = 1,
) -> list[pd.DataFrame]:
    """
    Return *exact-length* (== traj_len) sub-trajectories that satisfy:
    • last row’s phase  == `target_phase`
    • every row’s phase ∈ allowed_set(target_phase)
    • last_time - first_time ≤ time_threshold (using event_time_utc)
    """
    assert target_phase in {"train", "validation", "test"}
    phase2allowed = {
        "train":       {"train"},
        "validation":  {"train", "validation"},
        "test":        {"train", "validation", "test"},
    }
    allowed_set = phase2allowed[target_phase]
    sub_trajs = []
    for _, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.reset_index(drop=True)
        times = user_df["event_time_utc"].values
        phases = user_df["phase"].values
        n = len(user_df)
        for end_idx in range(traj_len - 1, n, stride):
            start_idx = end_idx - traj_len + 1
            if phases[end_idx] != target_phase:
                continue
            if not all(p in allowed_set for p in phases[start_idx: end_idx + 1]):
                continue
            if times[end_idx] - times[start_idx] > time_threshold:
                continue
            sub_trajs.append(user_df.iloc[start_idx: end_idx + 1].copy())
    return sub_trajs


def build_all_phase_trajectories(train_df, validation_df, test_df, traj_len, time_threshold=pd.Timedelta(days=1)):
    combined_df = label_splits(train_df, validation_df, test_df)
    trajs_train = extract_trajs_for_phase(combined_df, "train", time_threshold, traj_len)
    trajs_validation = extract_trajs_for_phase(combined_df, "validation", time_threshold, traj_len)
    trajs_test = extract_trajs_for_phase(combined_df, "test", time_threshold, traj_len)
    return {"train": trajs_train, "validation": trajs_validation, "test": trajs_test}


def save_pickle(data, path: str):
    with open(path, 'wb') as f:
        pkl.dump(data, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pkl.load(f)


def _finite_coordinate(latitude, longitude) -> bool:
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


def _haversine_km(left, right) -> float:
    if not _finite_coordinate(*left) or not _finite_coordinate(*right):
        return float("nan")
    lat1, lon1 = map(math.radians, map(float, left))
    lat2, lon2 = map(math.radians, map(float, right))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2.0
    ) ** 2
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def _timestamps(frame):
    for column in ("event_time_utc", "local_time", "datetime", "timestamp"):
        if column in frame:
            return list(frame[column])
    raise ValueError("Missing observed timestamp column")


def _history(frame):
    history = frame.iloc[:-1]
    if history.empty:
        raise ValueError("A trajectory must contain observed history and a target")
    return history


def _times(history):
    for column in ("local_time", "event_time_utc", "utc_time"):
        if column in history:
            values = [pd.Timestamp(value) for value in history[column]]
            if any(pd.isna(value) for value in values):
                raise ValueError("Missing observed timestamp")
            return values
    raise ValueError("Missing observed timestamp column")


def observed_session_counts(frame, gap_minutes):
    """Count events and sessions using only the prefix, in its stored order.

    A new session starts after a gap strictly greater than the threshold or a
    backwards time step. Zero gaps remain in the same session. The threshold is
    supplied by the caller and must not be estimated from the held-out row.
    """
    if not np.isfinite(gap_minutes) or gap_minutes < 0:
        raise ValueError("Session gap must be finite and nonnegative")
    times = _times(_history(frame))
    gaps = [(right - left).total_seconds() / 60 for left, right in zip(times, times[1:])]
    connected = [0 <= gap <= gap_minutes for gap in gaps]
    tail = 1
    for value in reversed(connected):
        if not value:
            break
        tail += 1
    return {"session_count": 1 + sum(not value for value in connected),
            "close_event_count": 1 + sum(connected),
            "trailing_session_event_count": tail}


def derive_session_threshold(trajectories, source, method):
    """Estimate minutes from positive observed training gaps, without outcomes.

    ``window_gaps`` retains overlapping-window multiplicity. ``unique_edges``
    counts each observed user/time/POI edge once. The method is an explicit part
    of the candidate contract, not selected within this function.
    """
    if source not in {"window_gaps", "unique_edges"}:
        raise ValueError("Unsupported training gap source: " + str(source))
    gaps, seen = [], set()
    for frame in trajectories:
        history = _history(frame)
        times = _times(history)
        events = [(int(row.user_id), str(time), int(row.poi_id))
                  for row, time in zip(history.itertuples(index=False), times)]
        for left, right, start, end in zip(events, events[1:], times, times[1:]):
            edge = (left, right)
            if source == "unique_edges" and edge in seen:
                continue
            seen.add(edge)
            gap = (end - start).total_seconds() / 60
            if np.isfinite(gap) and gap > 0:
                gaps.append(gap)
    if not gaps:
        raise ValueError("No positive observed training gaps")
    values = np.asarray(gaps, dtype=float)
    quantiles = {"quartile_1": .25, "median": .5, "quartile_3": .75, "decile_9": .9}
    if method in quantiles:
        return float(np.quantile(values, quantiles[method]))
    raise ValueError("Unsupported session threshold method: " + str(method))


def observed_event_keys(frame):
    history = _history(frame)
    return tuple((int(row.user_id), str(time), int(row.poi_id))
                 for row, time in zip(history.itertuples(index=False), _times(history)))


def eligible_trailing_session_multiplicities(trajectories, eligible, gap_minutes):
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    keys = []
    for frame in trajectories:
        events = observed_event_keys(frame)
        tail = observed_session_counts(frame, gap_minutes)["trailing_session_event_count"]
        keys.append((events[0][0], events[-tail][1], events[-tail][2]))
    counts = Counter(key for key, keep in zip(keys, eligible) if keep)
    return np.asarray([counts[key] if keep else 0 for key, keep in zip(keys, eligible)], dtype=np.int64)


def observed_stream_session_representatives(trajectories, gap_minutes, eligible=None, selection="first"):
    """Select one stored window per session in the merged observed event stream.

    Deduplicate user/time/POI events across all observed prefixes, then sort each
    user's events chronologically. Equal-time events retain first appearance
    order. A gap strictly greater than the supplied training-derived threshold
    begins a session. Each window belongs to its last observed event's session.

    First/last means stored window order, not target time. Group construction
    uses all input prefixes; selection uses eligible windows only. Held-out rows
    are never read. The returned 0/1 weights do not duplicate predictions.
    """
    if not np.isfinite(gap_minutes) or gap_minutes < 0:
        raise ValueError("Session gap must be finite and nonnegative")
    if selection not in {"first", "last"}:
        raise ValueError("Session representative selection must be first or last")
    if eligible is None:
        eligible = np.ones(len(trajectories), dtype=bool)
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    histories = [observed_event_keys(frame) for frame in trajectories]
    by_user = defaultdict(dict)
    for events in histories:
        for event in events:
            by_user[event[0]].setdefault(event, len(by_user[event[0]]))
    session_for_event = {}
    for events in by_user.values():
        ordered = sorted(events, key=lambda e: (pd.Timestamp(e[1]), events[e]))
        previous, start = None, None
        for event in ordered:
            stamp = pd.Timestamp(event[1])
            if previous is None or (stamp - previous).total_seconds() / 60 > gap_minutes:
                start = event
            session_for_event[event] = start
            previous = stamp
    weights = np.zeros(len(histories), dtype=np.int64)
    seen = set()
    order = range(len(histories)) if selection == "first" else reversed(range(len(histories)))
    for index in order:
        if not eligible[index]:
            continue
        key = session_for_event[histories[index][-1]]
        if key not in seen:
            weights[index] = 1
            seen.add(key)
    return weights


def novelty_signature_join_multiplicities(training, trajectories, eligible=None):
    """Join windows sharing an unseen-transition set and multiple novelty runs.

    Training includes its supervised final check-ins. Evaluation uses observed
    prefixes only. A novelty run is a maximal consecutive sequence of observed
    transitions absent from training. Keep windows with at least two such runs,
    then self-join eligible windows on their sorted set of unseen POI edges.
    Every retained window's multiplicity is the actual size of its signature
    group. Neither test targets nor correctness enter the construction.
    """
    if eligible is None:
        eligible = np.ones(len(trajectories), dtype=bool)
    eligible = np.asarray(eligible)
    if eligible.shape != (len(trajectories),) or not np.isin(eligible, (0, 1)).all():
        raise ValueError("Eligibility must be one boolean or 0/1 value per trajectory")
    known = set()
    for frame in training:
        ids = list(map(int, frame.poi_id))
        known.update(zip(ids, ids[1:]))
    signatures, selected = [], []
    for frame, keep in zip(trajectories, eligible):
        ids = list(map(int, _history(frame).poi_id))
        edges = list(zip(ids, ids[1:]))
        unseen = [edge not in known for edge in edges]
        runs = sum(flag and (index == 0 or not unseen[index - 1]) for index, flag in enumerate(unseen))
        signatures.append(tuple(sorted({edge for edge, flag in zip(edges, unseen) if flag})))
        selected.append(bool(keep) and runs > 1)
    counts = Counter(signature for signature, keep in zip(signatures, selected) if keep)
    return np.asarray([counts[signature] if keep else 0 for signature, keep in zip(signatures, selected)], dtype=np.int64)


def observed_history_quality(frame):
    """Duration and maximum spatial step of the observed prefix only."""


    history = _history(frame)
    timestamps = _timestamps(history)
    if any(pd.isna(stamp) for stamp in timestamps):
        raise ValueError("Missing observed timestamp in quality envelope")
    coordinates = list(zip(history.latitude, history.longitude))
    if not all(_finite_coordinate(*point) for point in coordinates):
        raise ValueError("Invalid observed coordinate in quality envelope")
    distances = [_haversine_km(a, b) for a, b in zip(coordinates, coordinates[1:])]
    return {"span_hours": (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            "step_distance_km_max": max(distances, default=0.)}


def derive_history_quality_envelope(training):
    """Training-only nonnegative Tukey bounds, retaining window multiplicity."""
    if not training:
        raise ValueError("Quality calibration requires training trajectories")
    records = [observed_history_quality(frame) for frame in training]
    bounds = {}
    for feature in ("span_hours", "step_distance_km_max"):
        values = np.asarray([record[feature] for record in records])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training quality feature")
        q1, q3 = map(float, np.quantile(values, [.25, .75]))
        bounds[feature] = {"q1": q1, "q3": q3,
                           "lower": max(0., q1 - 1.5 * (q3 - q1)),
                           "upper": q3 + 1.5 * (q3 - q1)}
    return bounds


def history_quality_mask(trajectories, bounds):
    """Use a frozen training envelope; never calibrate on evaluation outcomes."""
    if set(bounds) != {"span_hours", "step_distance_km_max"}:
        raise ValueError("Quality envelope requires duration and maximum step")
    for interval in bounds.values():
        if not np.isfinite([interval["lower"], interval["upper"]]).all() or interval["lower"] > interval["upper"]:
            raise ValueError("Invalid quality envelope interval")
    return np.asarray([all(bounds[key]["lower"] <= value <= bounds[key]["upper"]
                           for key, value in observed_history_quality(frame).items())
                       for frame in trajectories], dtype=bool)
