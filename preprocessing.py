import pickle as pkl
import pandas as pd
import geopandas as gpd
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




