import argparse
import os
import json
import pickle as pkl
from typing import List, Dict
import pandas as pd
from configs import preprocessing_config as pc
from text_utils import _safe_time_fmt
from train import run_train
from eval import run_eval

PROMPT_PREFIX = "llm4poi_textual"


def _get_cat_col(df: pd.DataFrame):
    if "poi_category_name" in df.columns:
        return "poi_category_name"
    if "category" in df.columns:
        return "category"
    return None


def _ensure_time_cols(df: pd.DataFrame):
    if "local_time" not in df.columns:
        if "event_time_utc" in df.columns:
            df["local_time"] = df["event_time_utc"].dt.tz_convert(pc.CITY_TZ)
        else:
            raise KeyError("Expected local_time or event_time_utc")
    if "event_time_utc" not in df.columns:
        df["event_time_utc"] = df["local_time"].dt.tz_convert("UTC")
    return df


def _build_category_map(trajs: List[pd.DataFrame]) -> Dict[str, int]:
    cats = []
    for df in trajs:
        col = _get_cat_col(df)
        if col:
            cats.extend(df[col].fillna("UNKNOWN").astype(str).tolist())
        else:
            cats.append("UNKNOWN")
    unique = sorted(set(cats))
    return {c: i for i, c in enumerate(unique)}


def _sentence(row, uid, cat_name, cat_id, time_col="local_time"):
    t = _safe_time_fmt(row[time_col])
    return (
        f"At {t}, user {uid} visited POI id {int(row.poi_id)} "
        f"which is a/an {cat_name} with category id {cat_id}."
    )


def _build_prompt(
    current_df: pd.DataFrame,
    hist_dfs: List[pd.DataFrame],
    poi_id_range: int,
    cat_map: Dict[str, int],
):
    current_df = current_df.copy()
    current_df = _ensure_time_cols(current_df)
    uid = int(current_df["user_id"].iloc[0])
    cat_col = _get_cat_col(current_df)

    # Current trajectory block: all but last entry
    current_lines = []
    for _, row in current_df.iloc[:-1].iterrows():
        cat_name = str(row[cat_col]) if cat_col else "UNKNOWN"
        cat_name = cat_name if cat_name and cat_name == cat_name else "UNKNOWN"
        cat_id = cat_map.get(cat_name, 0)
        current_lines.append(_sentence(row, uid, cat_name, cat_id))

    # Historical trajectory block (same user, prior trajectories)
    hist_lines = []
    for hdf in hist_dfs:
        hdf = _ensure_time_cols(hdf)
        hcol = _get_cat_col(hdf)
        for _, row in hdf.iterrows():
            cat_name = str(row[hcol]) if hcol else "UNKNOWN"
            cat_name = cat_name if cat_name and cat_name == cat_name else "UNKNOWN"
            cat_id = cat_map.get(cat_name, 0)
            hist_lines.append(_sentence(row, uid, cat_name, cat_id))

    current_block = "\n".join(current_lines) if current_lines else ""
    hist_block = "\n".join(hist_lines) if hist_lines else "None."

    final_row = current_df.iloc[-1]
    final_time = _safe_time_fmt(final_row["local_time"])
    max_id = max(0, poi_id_range - 1)

    question = (
        f"<question> The following is a trajectory of user {uid}:\n"
        f"{current_block}\n\n"
        f"There is also historical data:\n{hist_block}\n\n"
        f"Given the data, at {final_time}, which POI id will user {uid} visit?\n"
        f"Note that POI id is an integer in the range from 0 to {max_id}."
    )
    answer = f"<answer>: At {final_time}, user {uid} will visit POI id {int(final_row.poi_id)}."
    return question + "\n" + answer


def _build_user_history(all_trajs: List[pd.DataFrame]):
    user_map = {}
    for df in all_trajs:
        df = _ensure_time_cols(df)
        uid = int(df["user_id"].iloc[0])
        start_t = df["event_time_utc"].iloc[0]
        end_t = df["event_time_utc"].iloc[-1]
        user_map.setdefault(uid, []).append((start_t, end_t, df))
    # sort by end time
    for uid in user_map:
        user_map[uid].sort(key=lambda x: x[1])
    return user_map


def build_prompts_for_split(trajs, user_hist_map, poi_id_range, cat_map, max_hist=0):
    prompts = []
    for df in trajs:
        df = _ensure_time_cols(df)
        uid = int(df["user_id"].iloc[0])
        cur_start = df["event_time_utc"].iloc[0]
        # prior trajectories for same user
        hist = [t[2] for t in user_hist_map.get(uid, []) if t[1] < cur_start]
        if max_hist and max_hist > 0:
            hist = hist[-max_hist:]
        prompts.append(_build_prompt(df, hist, poi_id_range, cat_map))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--dataset", default="NYC")
    parser.add_argument("--traj-len", type=int, default=20)
    parser.add_argument("--crime-radius", type=int, default=500)
    parser.add_argument("--crime-time-weeks", type=int, default=4)
    parser.add_argument("--base-dir", default="/absolute/path/to/SafetyIsAllYouNeed")
    # LLM4POI baseline model (as in LLM4POI paper)
    parser.add_argument("--model-name", default="Llama-2-7b-longlora-32k")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-eval-steps", type=int, default=500)
    parser.add_argument(
        "--max-hist-trajs",
        type=int,
        default=0,
        help="Max historical trajectories to include (0 = all)",
    )
    args = parser.parse_args()

    pc.update_config(
        args.dataset,
        args.traj_len,
        args.crime_radius,
        args.crime_time_weeks,
        base_dir=args.base_dir,
    )

    # load trajectories
    with open(pc.TRAIN_TRAJECTORIES_PKL_PATH, "rb") as f:
        train_trajs = pkl.load(f)
    with open(pc.VALIDATION_TRAJECTORIES_PKL_PATH, "rb") as f:
        val_trajs = pkl.load(f)
    with open(pc.TEST_TRAJECTORIES_PKL_PATH, "rb") as f:
        test_trajs = pkl.load(f)

    all_trajs = train_trajs + val_trajs + test_trajs

    # POI id range
    max_poi = -1
    for df in all_trajs:
        max_poi = max(max_poi, int(df["poi_id"].max()))
    poi_id_range = max_poi + 1 if max_poi >= 0 else 0

    # category map
    cat_map = _build_category_map(all_trajs)

    # build history map
    user_hist_map = _build_user_history(all_trajs)

    # build prompts
    train_prompts = build_prompts_for_split(
        train_trajs, user_hist_map, poi_id_range, cat_map, args.max_hist_trajs
    )
    val_prompts = build_prompts_for_split(
        val_trajs, user_hist_map, poi_id_range, cat_map, args.max_hist_trajs
    )
    test_prompts = build_prompts_for_split(
        test_trajs, user_hist_map, poi_id_range, cat_map, args.max_hist_trajs
    )

    # write prompt files
    os.makedirs(pc.CURRENT_DATA_DIR, exist_ok=True)
    with open(os.path.join(pc.CURRENT_DATA_DIR, f"{PROMPT_PREFIX}_train_trajs.json"), "w") as f:
        json.dump(train_prompts, f)
    with open(os.path.join(pc.CURRENT_DATA_DIR, f"{PROMPT_PREFIX}_validation_trajs.json"), "w") as f:
        json.dump(val_prompts, f)
    with open(os.path.join(pc.CURRENT_DATA_DIR, f"{PROMPT_PREFIX}_test_trajs.json"), "w") as f:
        json.dump(test_prompts, f)

    if args.model_name == "Llama-2-7b-longlora-32k":
        print(
            "[WARN] Using placeholder model name. Please pass the exact HF checkpoint or local path via --model-name."
        )

    if args.mode in ("train", "both"):
        run_train(
            dataset=args.dataset,
            traj_len=args.traj_len,
            crime_radius=args.crime_radius,
            crime_time_weeks=args.crime_time_weeks,
            base_dir=args.base_dir,
            use_safety=False,
            prompt_prefix=PROMPT_PREFIX,
            model_name=args.model_name,
            max_length=args.max_length,
            num_train_epochs=args.num_train_epochs,
            lr=args.lr,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_eval_steps=args.save_eval_steps,
        )

    if args.mode in ("eval", "both"):
        metrics = run_eval(
            dataset=args.dataset,
            traj_len=args.traj_len,
            crime_radius=args.crime_radius,
            crime_time_weeks=args.crime_time_weeks,
            base_dir=args.base_dir,
            use_safety=False,
            prompt_prefix=PROMPT_PREFIX,
            model_name=args.model_name,
        )
        print(metrics)


if __name__ == "__main__":
    main()
