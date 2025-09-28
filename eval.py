"""
Evaluate a trained LLM on textual trajectory prompts.

Metrics:
- Acc@1 (greedy)
- Acc@k (beam search)
- MRR (beam search)
- Inference safety summary: safety of routes from last GT POI -> predicted POI,
  using the same route/crime logic as in preprocessing.

Notes:
- Requires: transformers, datasets, peft, bitsandbytes, geopandas, shapely, tqdm.
- Make sure `train.py` has already produced the model folder and prompt JSONs.
"""

import os
import re
import json
import gc
import pickle as pkl
from typing import Iterable, Tuple, List, Dict, Optional
import torch
import pandas as pd
import geopandas as gpd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from configs import preprocessing_config as pc
import preprocessing as pp
import safety as sf


def _scope_model_dir(base_dir: str, model_name: str, use_safety: bool) -> str:
    """
    Build the scoped model directory (same pattern used by train.py).
    """
    name_short = model_name.split("/")[-1]
    sub = f"{name_short}_{'w' if use_safety else 'wo'}_safety"
    return os.path.join(
        base_dir, "models", pc.DATASET,
        f"traj_len-{pc.TRAJ_LENGTH}",
        f"crime_radius-{pc.CRIME_RADIUS}m",
        f"crime_time-{pc.CRIME_TIME_WINDOW}w",
        sub,
        "best",  # train.py saves best checkpoint here
    )


def _resolve_prompt_paths(use_safety: bool) -> Tuple[str, str, str]:
    """
    Return (train_json, val_json, test_json) based on safety toggle.
    """
    if use_safety:
        data_dir = pc.SAFETY_DATA_DIR
        return (
            os.path.join(data_dir, "safety_textual_train_trajs.json"),
            os.path.join(data_dir, "safety_textual_validation_trajs.json"),
            os.path.join(data_dir, "safety_textual_test_trajs.json"),
        )
    else:
        data_dir = pc.CURRENT_DATA_DIR
        return (
            os.path.join(data_dir, "textual_train_trajs.json"),
            os.path.join(data_dir, "textual_validation_trajs.json"),
            os.path.join(data_dir, "textual_test_trajs.json"),
        )


def _load_text_prompts(train_json: str, val_json: str, test_json: str):
    with open(train_json, "r") as f:
        train_texts = json.load(f)
    with open(val_json, "r") as f:
        val_texts = json.load(f)
    with open(test_json, "r") as f:
        test_texts = json.load(f)
    return train_texts, val_texts, test_texts


def _load_numeric_trajectories():
    """
    Load numeric trajectories (with safety) produced by run_processing.py.
    Used to build POI-> (lat, lon) hashmap and to get GT timestamps.
    """
    with open(pc.TRAIN_TRAJS_WITH_SAFETY_PKL_PATH, "rb") as f:
        train_trajs = pkl.load(f)
    with open(pc.VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH, "rb") as f:
        val_trajs = pkl.load(f)
    with open(pc.TEST_TRAJS_WITH_SAFETY_PKL_PATH, "rb") as f:
        test_trajs = pkl.load(f)
    return train_trajs, val_trajs, test_trajs


def _create_poi_id_hashmap(train_trajs, val_trajs, test_trajs) -> Dict[str, Dict[str, float]]:
    """
    Build POI -> coordinates map from all splits.
    """
    poi_map = {}
    for trajs in (train_trajs, val_trajs, test_trajs):
        for df in trajs:
            for _, row in df.iterrows():
                pid = str(int(row.poi_id))
                if pid not in poi_map:
                    poi_map[pid] = {"latitude": float(row.latitude), "longitude": float(row.longitude)}
    return poi_map


## Parsing POI from text
_ANS = "<answer>:"


def extract_poi_num(text: str) -> int:
    """
    Extract POI id as an int from the *answer span only*.
    Returns -1 if not found.
    """
    part = text.split(_ANS, 1)[1] if _ANS in text else text
    m = re.search(r"POI id\s+(\d+)", part)
    if m:
        return int(m.group(1))
    # Fallback: last integer in the span
    nums = re.findall(r"\d+", part)
    if nums:
        return int(nums[-1])
    return -1


@torch.no_grad()
def calc_top1_acc(model, tokenizer, test_trajs, device="cuda", max_new_tokens=128, return_vectors=True, debug_n=0):
    total = 0
    hit = 0
    gt_pois, pred_pois = [], []

    for idx, traj in enumerate(test_trajs):
        if _ANS not in traj:
            continue
        Q, A = traj.split(_ANS, 1)
        gt = extract_poi_num(traj)  # Extracts from the GT answer span only

        inputs = tokenizer(Q, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        # Decode the *new* tokens, not the input prompt
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_poi_num(pred_text)

        # Debug a few examples if you want
        if debug_n and idx < debug_n:
            print("\n=== DEBUG SAMPLE ===")
            print("Q (tail):", Q[-200:])
            print("GEN TEXT:", pred_text[:200])
            print("GT:", gt, "PRED:", pred)

        # Count only real matches; -1 means 'couldn't parse'
        total += 1
        if (gt != -1) and (pred != -1) and (gt == pred):
            hit += 1

        gt_pois.append(gt)
        pred_pois.append(pred)

    acc = hit / total if total else 0.0
    return (acc, gt_pois, pred_pois) if return_vectors else acc


@torch.no_grad()
def calc_topk_acc(model, tokenizer, test_texts: List[str], k: int = 3, device: str = "cuda", max_new_tokens: int = 128) -> float:
    total = 0
    hit = 0
    for traj in test_texts:
        if _ANS not in traj:
            continue
        q, a_gt = traj.split(_ANS, 1)
        gt = extract_poi_num(a_gt)

        inputs = tokenizer(q, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=k,
            num_return_sequences=k,
            do_sample=False,
            early_stopping=True,
        )
        preds = [extract_poi_num(tokenizer.decode(seq, skip_special_tokens=True)) for seq in out]
        if gt in preds:
            hit += 1
        total += 1
    return hit / total if total else 0.0


@torch.no_grad()
def calc_mrr(model, tokenizer, test_texts: List[str], k_max: int = 10, device: str = "cuda", max_new_tokens: int = 128) -> float:
    rrs = []
    for traj in test_texts:
        if _ANS not in traj:
            continue
        q, a_gt = traj.split(_ANS, 1)
        gt = extract_poi_num(a_gt)

        inputs = tokenizer(q, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=k_max,
            num_return_sequences=k_max,
            do_sample=False,
            early_stopping=True,
        )
        preds = [extract_poi_num(tokenizer.decode(seq, skip_special_tokens=True)) for seq in out]
        rank = None
        for idx, p in enumerate(preds, start=1):
            if p == gt:
                rank = idx
                break
        rrs.append(1.0 / rank if rank else 0.0)
    return float(sum(rrs) / len(rrs)) if rrs else 0.0


def _inference_safety_summary(
    gt_vec: List[int],
    pred_vec: List[int],
    test_trajs: List[pd.DataFrame],
    poi_id_map: Dict[str, Dict[str, float]],
    crime_gdf: gpd.GeoDataFrame,
    dist_stats: Dict,
    buffer_meters: int,
    time_window_weeks: int,
) -> pd.Series:
    """
    For each test trajectory, build route from last GT POI to predicted POI, count crimes
    within buffer & window, convert to normalized safety using train dist_stats (1 - robust_scale).
    Returns a pandas Series describe() summary of per-route safety scores.
    """
    scores = []

    for i, gt_poi in enumerate(gt_vec):
        pred_poi = pred_vec[i]
        if pred_poi < 0:
            continue
        pred_key = str(int(pred_poi))
        if pred_key not in poi_id_map:
            # hallucinated/unseen POI id
            continue

        # get last timestamp & coords from the numeric trajectory
        df = test_trajs[i].reset_index(drop=True)
        last = df.iloc[-1]
        gt_lat, gt_lon = float(last.latitude), float(last.longitude)
        # Prefer UTC timestamp for safety computations
        tstamp = last["event_time_utc"] if "event_time_utc" in df.columns else last["local_time"]
        pred_lat = poi_id_map[pred_key]["latitude"]
        pred_lon = poi_id_map[pred_key]["longitude"]
        # Route geometry (lon, lat)
        route_coords, _ = sf.get_route_coordinates(
            (gt_lon, gt_lat),
            (pred_lon, pred_lat),
            route_coordinates_hashmap={}
        )
        # Raw crimes count for this route
        cnt = sf.compute_route_crimes(
            route_coords=route_coords,
            poi_timestamp=tstamp,
            crime_gdf=crime_gdf,
            buffer_meters=buffer_meters,
            time_window_weeks=time_window_weeks,
        )
        # Normalize using train dist stats
        safety_val = 1.0 - sf.robust_scale(cnt, dist_stats)
        scores.append(safety_val)

    if not scores:
        return pd.Series(dtype=float)
    return pd.Series(scores).describe()


def run_eval(
    *,
    # trajectory hyperparams (change to any permutation)
    dataset: str = "NYC",
    traj_len: int = 10,
    crime_radius: int = 1000,
    crime_time_weeks: int = 4,
    base_dir: str = "/Users/ramizaboura/MSC/SafetyIsAllYouNeed",
    use_safety: bool = True,
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    use_4bit: bool = True,
    torch_dtype = torch.float16,
    model_dir_override: Optional[str] = None,  # if you want to point directly to a folder
    # Generation
    max_new_tokens: int = 128,
    topk_list: Iterable[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Evaluate a trained adapter with Acc@k, MRR, and safety summary.
    Returns a dict of scalar metrics and prints a safety describe().
    """
    # Configure paths
    pc.update_config(dataset, traj_len, crime_radius, crime_time_weeks, base_dir=base_dir)
    # Load prompts
    train_json, val_json, test_json = _resolve_prompt_paths(use_safety)
    _, _, test_texts = _load_text_prompts(train_json, val_json, test_json)
    # Load numeric trajectories for POI map & timestamps
    train_trajs, val_trajs, test_trajs = _load_numeric_trajectories()
    poi_id_map = _create_poi_id_hashmap(train_trajs, val_trajs, test_trajs)
    # Crime GeoDF
    crime_df = pd.read_csv(pc.CRIME_CSV)
    crime_gdf = pp.build_crime_geodf(crime_df)
    # Dist stats for normalization
    with open(pc.TRAIN_CRIME_DIST_JSON_PATH, "r") as f:
        dist_stats = json.load(f)
    # Tokenizer
    tok_from = model_dir_override or _scope_model_dir(base_dir, model_name, use_safety)
    if os.path.exists(tok_from):
        tokenizer = AutoTokenizer.from_pretrained(tok_from, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Base model + adapter
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True) if use_4bit else None
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
    )
    adapter_dir = model_dir_override or _scope_model_dir(base_dir, model_name, use_safety)
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ## Metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: Dict[str, float] = {}
    # Acc@1 (+ vectors for safety)
    acc1, gt_vec, pred_vec = calc_top1_acc(
        model, tokenizer, test_texts,
        device=device, max_new_tokens=max_new_tokens, return_vectors=True
    )
    results["acc@1"] = float(acc1)
    # Acc@k
    for k in topk_list:
        if k == 1:
            continue
        acc_k = calc_topk_acc(
            model, tokenizer, test_texts, k=k,
            device=device, max_new_tokens=max_new_tokens
        )
        results[f"acc@{k}"] = float(acc_k)
    # MRR@k_max (use largest k in topk_list or default 10)
    k_max = max(list(topk_list) + [10])
    mrr = calc_mrr(
        model, tokenizer, test_texts, k_max=k_max,
        device=device, max_new_tokens=max_new_tokens
    )
    results["mrr"] = float(mrr)
    # Inference safety summary
    safety_summary = _inference_safety_summary(
        gt_vec=gt_vec,
        pred_vec=pred_vec,
        test_trajs=test_trajs,
        poi_id_map=poi_id_map,
        crime_gdf=crime_gdf,
        dist_stats=dist_stats,
        buffer_meters=pc.CRIME_RADIUS,
        time_window_weeks=pc.CRIME_TIME_WINDOW,
    )
    print("\n--- Inference Safety (predicted routes) summary ---")
    if safety_summary.empty:
        print("No valid predicted routes to score.")
    else:
        print(safety_summary.to_string())
    # Clean up CUDA memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


if __name__ == "__main__":
    # change trajectory hyperparams to any permutation
    metrics = run_eval(
        dataset="NYC",
        traj_len=10,
        crime_radius=1000,
        crime_time_weeks=4,
        base_dir="/Users/ramizaboura/MSC/SafetyIsAllYouNeed",
        use_safety=True,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=128,
        topk_list=(1, 3, 5),
    )
    print("\n--- Scalar metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")