
import argparse
import os
import json
import pickle as pkl
from pathlib import Path
import pandas as pd
from configs import preprocessing_config as pc


def _ensure_time_cols(df: pd.DataFrame):
    # Ensure local_time exists and is tz-aware
    if 'local_time' not in df.columns:
        if 'event_time_utc' in df.columns:
            df['local_time'] = df['event_time_utc'].dt.tz_convert(pc.CITY_TZ)
        else:
            raise KeyError('Expected local_time or event_time_utc column')
    # Ensure utc_time exists
    if 'utc_time' not in df.columns:
        if 'event_time_utc' in df.columns:
            df['utc_time'] = df['event_time_utc']
        else:
            df['utc_time'] = df['local_time'].dt.tz_convert('UTC')
    return df


def _ensure_category_cols(df: pd.DataFrame, cat_map: dict):
    # Choose existing category name column if present
    if 'poi_category_name' in df.columns:
        cat_name_col = 'poi_category_name'
    elif 'category' in df.columns:
        cat_name_col = 'category'
        df['poi_category_name'] = df['category']
    else:
        cat_name_col = None
        df['poi_category_name'] = 'UNKNOWN'

    # Fill missing
    df['poi_category_name'] = df['poi_category_name'].fillna('UNKNOWN').astype(str)

    # Map to integer id (deterministic)
    df['poi_category_id'] = df['poi_category_name'].map(cat_map)
    return df


def _build_category_map(trajs):
    cats = []
    for df in trajs:
        if 'poi_category_name' in df.columns:
            cats.extend(df['poi_category_name'].fillna('UNKNOWN').astype(str).tolist())
        elif 'category' in df.columns:
            cats.extend(df['category'].fillna('UNKNOWN').astype(str).tolist())
        else:
            cats.append('UNKNOWN')
    unique = sorted(set(cats))
    return {c: i for i, c in enumerate(unique)}


def _add_timezone_offset(df: pd.DataFrame):
    # hours offset from UTC
    try:
        offsets = df['local_time'].dt.utcoffset()
        df['timezone_offset'] = offsets.dt.total_seconds() / 3600.0
    except Exception:
        # fallback per-row
        df['timezone_offset'] = df['local_time'].apply(lambda x: x.utcoffset().total_seconds()/3600.0)
    return df


def _prepare_trajs(trajs, cat_map):
    out = []
    for df in trajs:
        df = df.copy()
        df = _ensure_time_cols(df)
        df = _add_timezone_offset(df)
        df = _ensure_category_cols(df, cat_map)
        out.append(df)
    return out


def dump_stan(trajs, out_path):
    rows = []
    for df in trajs:
        df = df.sort_values('utc_time')
        for _, r in df.iterrows():
            rows.append([
                int(r.user_id),
                int(r.poi_id),
                int(r.poi_category_id),
                f"{float(r.latitude):.6f}",
                f"{float(r.longitude):.6f}",
                int(pd.Timestamp(r.utc_time).timestamp())
            ])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for row in rows:
            f.write('\t'.join(map(str, row)) + '\n')


def make_getnext_csv(trajs, out_path):
    rows = []
    for traj_idx, traj in enumerate(trajs):
        traj = traj.sort_values('local_time')
        user = traj.user_id.iloc[0]
        window_id = f"{user}_{traj_idx}"
        first_day = traj.local_time.dt.date.iloc[0]
        for pos, row in enumerate(traj.itertuples(index=False), start=1):
            day_of_week = row.local_time.weekday()
            tod = row.local_time.hour * 3600 + row.local_time.minute*60 + row.local_time.second
            norm_in_day = tod / (24*3600)
            shift = (row.local_time.date() - first_day).days
            norm_rel = (pos - 1) / (len(traj) - 1)
            rows.append({
                'user_id': user,
                'POI_id': row.poi_id,
                'POI_catid': row.poi_category_id,
                'POI_catid_code': row.poi_category_id,
                'POI_catname': row.poi_category_name,
                'latitude': row.latitude,
                'longitude': row.longitude,
                'timezone': row.timezone_offset,
                'UTC_time': row.utc_time,
                'local_time': row.local_time,
                'day_of_week': day_of_week,
                'norm_in_day_time': norm_in_day,
                'trajectory_id': window_id,
                'norm_day_shift': shift,
                'norm_relative_time': norm_rel,
            })
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False, sep=',', header=[
        'user_id','POI_id','POI_catid','POI_catid_code','POI_catname',
        'latitude','longitude','timezone','UTC_time','local_time',
        'day_of_week','norm_in_day_time','trajectory_id',
        'norm_day_shift','norm_relative_time'
    ])


def dump_sthg(trajs, out_path):
    rows = []
    for traj in trajs:
        traj = traj.sort_values('local_time')
        uid = traj.user_id.iloc[0]
        wid = f"{uid}_{traj.index.min()}"
        first_day = traj.local_time.dt.normalize().iloc[0]
        for _, row in traj.iterrows():
            lt = row.local_time
            dow = lt.weekday()
            secs = lt.hour*3600 + lt.minute*60 + lt.second
            norm_in_day = secs / (24*3600)
            day_shift = (lt.normalize() - first_day).days
            rows.append({
                'user_id': uid,
                'POI_id': row.poi_id,
                'POI_catid': row.poi_category_id,
                'POI_catid_code': row.poi_category_id,
                'POI_catname': row.poi_category_name,
                'latitude': row.latitude,
                'longitude': row.longitude,
                'timezone': row.timezone_offset,
                'UTC_time': row.utc_time,
                'local_time': lt,
                'day_of_week': dow,
                'norm_in_day_time': norm_in_day,
                'trajectory_id': wid,
                'norm_day_shift': day_shift,
            })
    df = pd.DataFrame(rows)
    df['pos'] = df.groupby('trajectory_id').cumcount()
    traj_lens = df.groupby('trajectory_id')['pos'].transform('max')
    df['norm_relative_time'] = df['pos'] / traj_lens.astype(float)
    out = df[[
        'user_id','POI_id','POI_catid','POI_catid_code','POI_catname',
        'latitude','longitude','timezone','UTC_time','local_time',
        'day_of_week','norm_in_day_time','trajectory_id',
        'norm_day_shift','norm_relative_time'
    ]]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, sep='	', header=True, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CHICAGO', choices=['NYC','CHICAGO'])
    parser.add_argument('--traj-len', type=int, default=20)
    parser.add_argument('--crime-radius', type=int, default=1000)
    parser.add_argument('--crime-time-weeks', type=int, default=3)
    parser.add_argument('--base-dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--baseline', default='all', choices=['all','stan','getnext','sthgcn'])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / 'baselines' / 'exports'
    out_dir.mkdir(parents=True, exist_ok=True)

    pc.update_config(args.dataset, args.traj_len, args.crime_radius, args.crime_time_weeks, base_dir=str(base_dir))

    # load trajectories
    for p in (pc.TRAIN_TRAJECTORIES_PKL_PATH, pc.VALIDATION_TRAJECTORIES_PKL_PATH, pc.TEST_TRAJECTORIES_PKL_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing trajectory file: {p}. Run run_preprocessing.py first.")

    with open(pc.TRAIN_TRAJECTORIES_PKL_PATH, 'rb') as f:
        train_trajs = pkl.load(f)
    with open(pc.VALIDATION_TRAJECTORIES_PKL_PATH, 'rb') as f:
        val_trajs = pkl.load(f)
    with open(pc.TEST_TRAJECTORIES_PKL_PATH, 'rb') as f:
        test_trajs = pkl.load(f)

    # category map
    cat_map = _build_category_map(train_trajs + val_trajs + test_trajs)
    (out_dir / 'category_map.json').write_text(json.dumps(cat_map, indent=2))

    train_trajs = _prepare_trajs(train_trajs, cat_map)
    val_trajs = _prepare_trajs(val_trajs, cat_map)
    test_trajs = _prepare_trajs(test_trajs, cat_map)

    if args.baseline in ('all','stan'):
        stan_dir = out_dir / 'stan' / f"{args.dataset.lower()}_len{args.traj_len}"
        dump_stan(train_trajs, str(stan_dir / 'train.txt'))
        dump_stan(val_trajs, str(stan_dir / 'val.txt'))
        dump_stan(test_trajs, str(stan_dir / 'test.txt'))

    if args.baseline in ('all','getnext'):
        get_dir = out_dir / 'getnext'
        make_getnext_csv(train_trajs, str(get_dir / f"{args.dataset}_train.csv"))
        make_getnext_csv(val_trajs, str(get_dir / f"{args.dataset}_val.csv"))
        make_getnext_csv(test_trajs, str(get_dir / f"{args.dataset}_test.csv"))

    if args.baseline in ('all','sthgcn'):
        sth_dir = out_dir / 'sthgcn'
        dump_sthg(train_trajs, str(sth_dir / f"{args.dataset}_train.tsv"))
        dump_sthg(val_trajs, str(sth_dir / f"{args.dataset}_val.tsv"))
        dump_sthg(test_trajs, str(sth_dir / f"{args.dataset}_test.tsv"))

    print(f"Baseline exports written to: {out_dir}")


if __name__ == '__main__':
    main()
