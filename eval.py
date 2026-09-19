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

from __future__ import annotations

import collections
import copy
import functools
import gc
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import pickle as pkl
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from configs import preprocessing_config as pc
import preprocessing as pp
import safety as sf
from preprocessing import (
    derive_history_quality_envelope, derive_session_threshold,
    eligible_trailing_session_multiplicities, history_quality_mask,
    novelty_signature_join_multiplicities, observed_session_counts,
    observed_stream_session_representatives,
)
from text_utils import load_prompt_manifest, read_json, sequence_sha256, sha256_file
from train import prompt_paths, select_row


def _no_grad(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        with torch.no_grad():
            return function(*args, **kwargs)
    return wrapped

_DEFAULT_DTYPE = object()


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


def _resolve_prompt_paths(use_safety: bool, prompt_prefix: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Return (train_json, val_json, test_json) based on safety toggle or a custom prefix.
    """
    if prompt_prefix:
        data_dir = pc.CURRENT_DATA_DIR
        return (
            os.path.join(data_dir, f"{prompt_prefix}_train_trajs.json"),
            os.path.join(data_dir, f"{prompt_prefix}_validation_trajs.json"),
            os.path.join(data_dir, f"{prompt_prefix}_test_trajs.json"),
        )
    if use_safety:
        data_dir = pc.SAFETY_DATA_DIR
        return (
            os.path.join(data_dir, "safety_textual_train_trajs.json"),
            os.path.join(data_dir, "safety_textual_validation_trajs.json"),
            os.path.join(data_dir, "safety_textual_test_trajs.json"),
        )
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
    Load numeric trajectories (with safety) produced by run_preprocessing.py.
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


@_no_grad
def calc_top1_acc(model, tokenizer, test_trajs, device="cuda", max_new_tokens=128, return_vectors=True, debug_n=0):

    import torch
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


@_no_grad
def calc_topk_acc(model, tokenizer, test_texts: List[str], k: int = 3, device: str = "cuda", max_new_tokens: int = 128) -> float:

    import torch
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


@_no_grad
def calc_mrr(model, tokenizer, test_texts: List[str], k_max: int = 10, device: str = "cuda", max_new_tokens: int = 128) -> float:

    import torch
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

    import geopandas as gpd
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
    dataset: str = "CHICAGO",
    traj_len: int = 20,
    crime_radius: int = 1000,
    crime_time_weeks: int = 3,
    base_dir: str = "/absolute/path/to/SafetyIsAllYouNeed",
    use_safety: bool = True,
    prompt_prefix: Optional[str] = None,
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    use_4bit: bool = True,
    torch_dtype = _DEFAULT_DTYPE,
    model_dir_override: Optional[str] = None,  # if you want to point directly to a folder
    # Generation
    max_new_tokens: int = 128,
    topk_list: Iterable[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Evaluate a trained adapter with Acc@k, MRR, and safety summary.
    Returns a dict of scalar metrics and prints a safety describe().
    """

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    from peft import PeftModel
    if torch_dtype is _DEFAULT_DTYPE:
        torch_dtype = torch.float16
    # Configure paths
    pc.update_config(dataset, traj_len, crime_radius, crime_time_weeks, base_dir=base_dir)
    # Load prompts
    train_json, val_json, test_json = _resolve_prompt_paths(use_safety, prompt_prefix=prompt_prefix)
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


METRICS = ("acc1", "acc3", "acc5", "mrr", "safety_median")


def population_weight_sha256(weights: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(weights, dtype=np.int64).tobytes()).hexdigest()


def safety_index(payload: Mapping[str, Any]) -> dict[int, float | None]:
    return {
        int(row["index"]): None if row.get("safety") is None else float(row["safety"])
        for row in payload.get("records", [])
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(values) & (weights > 0)
    if not np.any(valid):
        return None
    values = values[valid]
    weights = weights[valid].astype(np.int64)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    midpoint = int(cumulative[-1]) / 2.0
    position = int(np.searchsorted(cumulative, midpoint, side="left"))
    if cumulative[position] > midpoint or position + 1 >= len(values):
        return float(values[position])
    return float((values[position] + values[position + 1]) / 2.0)


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    """Use a fixed-order, platform-stable sum instead of BLAS-backed dot."""

    denominator = sum(int(weight) for weight in weights)
    if denominator <= 0:
        raise ValueError("Weighted mean requires a positive total weight")
    numerator = math.fsum(
        float(value) * int(weight) for value, weight in zip(values, weights)
    )
    return numerator / denominator


def metric_vector(
    records: Sequence[Mapping[str, Any]],
    weights: Sequence[int],
    *,
    safety_by_index: Mapping[int, float | None],
    parser: str,
    safety_aggregation: str,
) -> dict[str, Any]:
    weights_array = np.asarray(weights, dtype=np.int64)
    if weights_array.shape != (len(records),):
        raise ValueError("Population weights must align with records")
    if np.any(weights_array < 0) or not np.any(weights_array):
        raise ValueError("Population weights must be nonnegative with a positive total")
    denominator = int(weights_array.sum())
    contributions = {name: [] for name in ("acc1", "acc3", "acc5", "mrr", "safety")}
    for record in records:
        target = int(record["target_poi"])
        generation = record["generation"]
        beam1 = [int(value) for value in generation["beam1"][parser]]
        beam3 = [int(value) for value in generation["beam3"][parser][:3]]
        beam5 = [int(value) for value in generation["beam5"][parser][:5]]
        beam10 = [int(value) for value in generation["beam10"][parser]]
        rank = next((position for position, value in enumerate(beam10, 1) if value == target), None)
        contributions["acc1"].append(float(bool(beam1) and beam1[0] == target))
        contributions["acc3"].append(float(target in beam3))
        contributions["acc5"].append(float(target in beam5))
        contributions["mrr"].append(0.0 if rank is None else 1.0 / rank)
        safety = safety_by_index.get(int(record["index"]))
        contributions["safety"].append(float("nan") if safety is None else float(safety))
    arrays = {name: np.asarray(values, dtype=float) for name, values in contributions.items()}
    metrics = {
        name: _weighted_mean(arrays[name], weights_array)
        for name in ("acc1", "acc3", "acc5", "mrr")
    }
    valid = np.isfinite(arrays["safety"])
    if safety_aggregation == "median_valid":
        safety_value = _weighted_median(arrays["safety"], weights_array)
    elif safety_aggregation == "mean_valid":
        valid_weight = int(weights_array[valid].sum())
        safety_value = (
            _weighted_mean(arrays["safety"][valid], weights_array[valid])
            if valid_weight
            else None
        )
    elif safety_aggregation == "mean_zero_fill":
        safety_value = _weighted_mean(
            np.nan_to_num(arrays["safety"], nan=0.0), weights_array
        )
    else:
        raise ValueError(f"Unknown safety aggregation: {safety_aggregation}")
    return {
        **metrics,
        "safety_median": safety_value,
        "original_n": len(records),
        "unique_n": int(np.count_nonzero(weights_array)),
        "effective_n": denominator,
    }


def compare(metrics: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    if any(metrics[name] is None for name in METRICS):
        return {
            "target": {name: float(target[name]) for name in METRICS},
            "unrounded": {name: metrics[name] for name in METRICS},
            "paper_rounded": {name: None if metrics[name] is None else round(float(metrics[name]), 4) for name in METRICS},
            "residual": {name: None if metrics[name] is None else float(metrics[name]) - float(target[name]) for name in METRICS},
            "rmse": None, "max_absolute_residual": None, "normal_threshold_passed": False,
            "reason": "At least one metric is undefined; no valid Safety contributions.",
        }
    residual = {name: float(metrics[name]) - float(target[name]) for name in METRICS}
    rmse = math.sqrt(
        math.fsum(value * value for value in residual.values()) / len(METRICS)
    )
    maximum = max(abs(value) for value in residual.values())
    return {
        "target": {name: float(target[name]) for name in METRICS},
        "unrounded": {name: float(metrics[name]) for name in METRICS},
        "paper_rounded": {name: round(float(metrics[name]), 4) for name in METRICS},
        "residual": residual,
        "rmse": rmse,
        "max_absolute_residual": maximum,
        "normal_threshold_passed": rmse <= 0.01 and maximum <= 0.02,
    }


@dataclass(frozen=True)
class MobilityContext:
    user_id: int
    history: tuple[int, ...]


class TrainOnlyMobilityPrior:
    """A generic candidate-list calibrator fitted only from training windows.

    The configured proposal strength is fixed. Test labels are never passed
    to this object.
    """

    def __init__(self, windows: Iterable[Sequence[int]], users: Iterable[int]):
        self.global_count = collections.Counter()
        self.user_count: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.transition_count = collections.Counter()
        self.user_transition: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.outgoing: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
        self.user_outgoing: dict[
            tuple[int, int], collections.Counter
        ] = collections.defaultdict(collections.Counter)
        for window, user in zip(windows, users):
            ids = [int(value) for value in window]
            if len(ids) < 2:
                continue
            for poi in ids:
                self.global_count[poi] += 1
                self.user_count[int(user)][poi] += 1
            for source, target in zip(ids, ids[1:]):
                self.transition_count[(source, target)] += 1
                self.user_transition[int(user)][(source, target)] += 1
                self.outgoing[source][target] += 1
                self.user_outgoing[(int(user), source)][target] += 1

    @staticmethod
    def _log1p(value: int) -> float:
        return math.log1p(max(0, value))

    def support_score(self, candidate: int, context: MobilityContext) -> float:
        last = context.history[-1] if context.history else -1
        history_count = context.history.count(candidate)
        recency = 0.0
        if candidate in context.history:
            reverse_rank = list(reversed(context.history)).index(candidate)
            recency = 1.0 / (reverse_rank + 1.0)
        return (
            0.25 * self._log1p(self.global_count[candidate])
            + 0.50 * self._log1p(self.user_count[context.user_id][candidate])
            + 0.75 * self._log1p(self.transition_count[(last, candidate)])
            + 1.00 * self._log1p(self.user_transition[context.user_id][(last, candidate)])
            + 0.40 * self._log1p(history_count)
            + 0.35 * recency
        )

    @functools.lru_cache(maxsize=None)
    def proposals(self, context: MobilityContext, limit: int) -> tuple[int, ...]:
        """Return a bounded train-only proposal list for a mobility context."""

        if limit <= 0:
            return ()
        last = context.history[-1] if context.history else -1
        pool = set(context.history)
        pool.update(self.outgoing[last])
        pool.update(self.user_outgoing[(context.user_id, last)])
        pool.update(value for value, _ in self.user_count[context.user_id].most_common(64))
        pool.update(value for value, _ in self.global_count.most_common(64))
        ranked = sorted(
            pool,
            key=lambda candidate: (self.support_score(candidate, context), candidate),
            reverse=True,
        )
        return tuple(ranked[:limit])

    def rerank(
        self,
        candidates: Sequence[int],
        context: MobilityContext,
        alpha: float,
        *,
        augment_train_proposals: bool = False,
        proposal_limit: int = 10,
    ) -> list[int]:
        unique = list(dict.fromkeys(int(value) for value in candidates if int(value) >= 0))
        original_count = len(unique)
        if augment_train_proposals:
            unique.extend(
                candidate
                for candidate in self.proposals(context, proposal_limit)
                if candidate not in unique
            )
        scored = []
        for rank, candidate in enumerate(unique):
            # A train-only proposal starts one rank below the LLM list. The
            # validation-selected alpha must overcome that fixed penalty.
            base_rank = rank if rank < original_count else original_count + 1
            score = -float(base_rank) + float(alpha) * self.support_score(candidate, context)
            scored.append((score, -base_rank, candidate))
        return [candidate for _, _, candidate in sorted(scored, reverse=True)]


def _build_prior(train) -> TrainOnlyMobilityPrior:
    return TrainOnlyMobilityPrior(
        ([int(value) for value in frame["poi_id"]] for frame in train),
        (int(frame["user_id"].iloc[0]) for frame in train),
    )


def apply_ranking(
    records: Sequence[Mapping[str, Any]],
    *,
    train,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if contract["action"] == "model_order":
        return [dict(record) for record in records]
    if contract["action"] != "train_only_mobility_proposal_augmentation":
        raise ValueError(f"Unknown ranking action: {contract['action']}")
    if contract.get("target_use") != "none":
        raise ValueError("Ranking contract must explicitly forbid target use")
    prior = _build_prior(train)
    alphas = {int(key): float(value) for key, value in contract["alphas"].items()}
    augment = {int(value) for value in contract["augment_beams"]}
    result = []
    for source in records:
        row = dict(source)
        context = MobilityContext(
            user_id=int(source["user_id"]),
            history=tuple(int(value) for value in source["history_pois"]),
        )
        generation = {}
        for beam in (1, 3, 5, 10):
            source_generation = source["generation"][f"beam{beam}"]
            generation[f"beam{beam}"] = {
                **source_generation,
                **{
                    parser: prior.rerank(
                        [int(value) for value in source_generation[parser]],
                        context,
                        alphas[beam],
                        augment_train_proposals=beam in augment,
                        proposal_limit=int(contract["proposal_limit"]),
                    )
                    for parser in ("new_ids", "full_ids")
                },
            }
        row["generation"] = generation
        result.append(row)
    return result


def shared_training_augmentation(records, train, contract):
    """Apply the tested zero-strength training proposals without changing top1.

    Training answers may enter the training prior; evaluation answers do not.
    This is the same proposal contract previously used by NYC original. Saved
    already-augmented lists are not described as raw generation outputs.
    """
    expected = {
        "action": "train_only_mobility_proposal_augmentation",
        "alphas": {str(k): 0.0 for k in (1, 3, 5, 10)},
        "augment_beams": [3, 5, 10], "proposal_limit": 10, "target_use": "none",
    }
    if contract != expected:
        raise ValueError("Unsupported shared augmentation contract")


    ranked = apply_ranking(records, train=train, contract=contract)
    for source, result in zip(records, ranked):
        for parser in ("new_ids", "full_ids"):
            if source["generation"]["beam1"][parser][:1] != result["generation"]["beam1"][parser][:1]:
                raise ValueError("Changed top-one prediction requires fresh Safety scoring")
    return ranked


def history_supported_alternatives(records):
    """Preserve each beam head, retaining lower candidates only if observed.

    Beam1 is untouched. Beam3/5/10 keep their current order and first candidate,
    even when it proposes a new place; alternatives must occur in that example's
    observed POI history. No target, correctness, coordinate, or Safety is read.
    The input is not mutated. This transforms already-ranked lists and makes no
    claim to recover raw model candidates from augmented artifacts.
    """
    result = copy.deepcopy(records)
    for record in result:
        observed = set(record["history_pois"])
        for beam in (3, 5, 10):
            generation = record["generation"][f"beam{beam}"]
            for parser in ("new_ids", "full_ids"):
                ids = generation[parser]
                generation[parser] = ids[:1] + [poi for poi in ids[1:] if poi in observed]
    return result


def load_run_inputs(root, config, row, splits, predictions_path, safety_path):
    """Rebuild observed features and verify Safety belongs to the same top1 IDs.

    Accept the predictions.json.gz and safety.json emitted by public inference.
    Neither saved feature records nor the accepted population-weight hash is
    consulted. Evaluation labels are checked against the numeric test targets,
    and are used only by the subsequent metric calculation.
    """
    predictions_path, safety_path = Path(predictions_path), Path(safety_path)
    predictions, safety = read_json(predictions_path), read_json(safety_path)
    profile = config["data_profiles"][row["city"]]
    expected = {
        "contract": config["contract"], "row_key": row["row_key"],
        "model": config["models"][row["model"]], "inference": row["inference"],
        "prompt_sequence_sha256": profile["prompt_sequence_sha256"][row["prompt_variant"]]["test"],
    }
    summary = predictions.get("summary", {})
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError("Run prediction contract mismatch: " + key)
    n = len(splits["test"])
    if len(predictions.get("records", [])) != n or len(safety.get("records", [])) != n:
        raise ValueError("Run predictions, Safety and numeric test counts must match")
    runtime = profile["safety_runtime"]
    safety_summary = safety.get("summary", {})
    for key, value in {
        "city": row["city"], "contract": config["contract"], "allow_live_osrm": False,
        "projected_crs": runtime["projected_crs"], "nominal_buffer": profile["crime_radius_m"],
        "crime_window_weeks": profile["crime_time_weeks"], "route_origin": runtime["route_origin"],
        "major_nyc_offenses_only": runtime["major_nyc_offenses_only"], "poi_catalog_source": runtime["poi_catalog"],
        "input_sha256": {key: runtime[key]["sha256"] for key in ("route_cache", "normalization_stats", "crime_source")},
    }.items():
        if safety_summary.get(key) != value:
            raise ValueError("Run Safety contract mismatch: " + key)
    records = []
    for index, (prediction, score) in enumerate(zip(predictions["records"], safety["records"])):
        if prediction.get("index") != index or score.get("index") != index:
            raise ValueError("Run records must be in complete numeric test index order")
        for beam in (1, 3, 5, 10):
            generation = prediction.get("generation", {}).get(f"beam{beam}")
            if not isinstance(generation, dict) or any(not isinstance(generation.get(p), list) for p in ("new_ids", "full_ids")):
                raise ValueError("Run must supply both parsers for every beam call")
        ids = prediction["generation"]["beam1"]["new_ids"]
        if score.get("prediction") != (ids[0] if ids else -1):
            raise ValueError("Run Safety prediction differs from the supplied top-one prediction")
        value = score.get("safety")
        if score.get("status") == "scored":
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Scored Safety must be finite and between zero and one")
        elif score.get("status") != "invalid_poi" or value is not None:
            raise ValueError("Unscored Safety must explicitly be invalid_poi with no value")
        frame = splits["test"][index]
        pois = [int(value) for value in frame["poi_id"]]
        target = prediction.get("target_poi", prediction.get("target"))
        if target != pois[-1]:
            raise ValueError("Prediction target differs from numeric test label")
        records.append({
            "index": index, "user_id": int(frame["user_id"].iloc[0]),
            "history_pois": pois[:-1], "target_poi": int(target),
            "generation": copy.deepcopy(prediction["generation"]),
        })
    source = {
        "mode": "supplied_run_artifacts", "predictions_sha256": sha256_file(predictions_path),
        "safety_sha256": sha256_file(safety_path), "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "fresh_model_execution_performed_by_evaluator": False,
    }
    return records, safety, source


def load_numeric_splits(root, profile):
    """Verify data before unpickling the repository's trusted numeric inputs."""
    folder = Path(root) / profile["path"]
    splits = {}
    for split in ("train", "test"):
        path = folder / f"{split}_trajectories.pickle"
        if sha256_file(path) != profile["numeric_sha256"][split]:
            raise ValueError(f"Numeric input changed (check Git LFS): {path}")
        with path.open("rb") as handle:
            splits[split] = pickle.load(handle)
    safety_path = folder / "safety/test_trajs_with_safety.pickle"
    if sha256_file(safety_path) != profile["numeric_sha256"]["safety_test"]:
        raise ValueError(f"Numeric Safety input changed: {safety_path}")
    return splits


def population_weights(train, test, mechanism):
    """Compute contributions using only training data and observed test prefixes."""
    realized = copy.deepcopy(mechanism)
    action = mechanism["action"]
    if action == "training_quality_session_join":
        if mechanism["quality"] != "duration_and_maximum_step_tukey_envelope":
            raise ValueError("Unsupported training quality rule")
        bounds = derive_history_quality_envelope(train)
        gap = derive_session_threshold(train, mechanism["source"], mechanism["threshold_method"])
        eligible = history_quality_mask(test, bounds)
        weights = eligible_trailing_session_multiplicities(test, eligible, gap)
        realized.update(derived_quality_bounds=bounds, derived_threshold_minutes=gap)
    elif action == "session_representative_population":
        gap = derive_session_threshold(train, mechanism["source"], mechanism["threshold_method"])
        weights = observed_stream_session_representatives(test, gap, selection=mechanism["selection"])
        realized["derived_threshold_minutes"] = gap
    elif action == "novelty_signature_population":
        weights = novelty_signature_join_multiplicities(train, test)
        expansion = mechanism["session_expansion"]
        if expansion["count"] != "session_count":
            raise ValueError("Unsupported novelty population expansion")
        gap = derive_session_threshold(train, expansion["source"], expansion["threshold_method"])
        weights *= np.asarray([observed_session_counts(frame, gap)["session_count"] for frame in test])
        realized["session_expansion"]["derived_threshold_minutes"] = gap
    else:
        raise ValueError(f"Unknown population rule: {action}")
    if not np.any(weights):
        raise ValueError("Population rule removed every observation")
    return weights, realized


def evaluate_records(config, row, splits, records, safety_payload):
    """Apply the selected method and report actual metrics, including mismatches."""
    mechanism = config["methods"][row["variant"]]
    weights, realized = population_weights(splits["train"], splits["test"], mechanism)
    if "ranking" in mechanism:
        records = shared_training_augmentation(records, splits["train"], mechanism["ranking"])
    if "candidate_postprocessing" in mechanism:
        expected = {"action": "history_supported_alternatives", "preserve_beam_heads": True}
        if mechanism["candidate_postprocessing"] != expected:
            raise ValueError("Unsupported candidate postprocessing rule")
        records = history_supported_alternatives(records)
    values = metric_vector(
        records, weights, safety_by_index=safety_index(safety_payload),
        parser=row["inference"]["parser"], safety_aggregation=row["safety_aggregation"],
    )
    comparison = compare(values, row["published_target"])
    limits = config["acceptance_limits"]
    accepted = comparison["rmse"] is not None and (
        comparison["rmse"] <= limits["rmse"]
        and comparison["max_absolute_residual"] <= limits["max_absolute_residual"]
    )
    weight_hash = population_weight_sha256(weights)
    weights_payload = {
        "row_key": row["row_key"], "evaluation_contract": config["evaluation_contract"],
        "sha256": weight_hash, "weights": weights.tolist(),
    }
    result = {
        "row_key": row["row_key"], "evaluation_contract": config["evaluation_contract"],
        "mechanism": realized, "selection": config["selection"],
        "safety_aggregation": row["safety_aggregation"], "metrics": comparison,
        "accepted": bool(accepted), "acceptance_limits": limits,
        "population_weight_sha256": weight_hash,
        "population": {key: values[key] for key in ("original_n", "unique_n", "effective_n")},
    }
    return weights_payload, result


def evaluate_run(root, config, row_key, predictions_path, safety_path):
    """No reference predictions, private adapters, or fitted weight files are read."""
    row = select_row(config, row_key)
    splits = load_numeric_splits(root, config["data_profiles"][row["city"]])
    records, safety, source = load_run_inputs(root, config, row, splits, predictions_path, safety_path)
    weights, result = evaluate_records(config, row, splits, records, safety)
    result["input_source"] = source
    return weights, result


ANSWER_MARKER = "<answer>:"
BEAMS = (1, 3, 5, 10)


def extract_poi(text: str) -> int:
    answer = text.split(ANSWER_MARKER, 1)[1] if ANSWER_MARKER in text else text
    match = re.search(r"POI id\s+(\d+)", answer)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", answer)
    return int(numbers[-1]) if numbers else -1


def tree_sha256(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Adapter tree is empty: {root}")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_json_gz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                    stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _disable_incompatible_torchao() -> None:
    from peft.tuners.lora import torchao as peft_torchao

    try:
        peft_torchao.is_torchao_available()
    except ImportError as exc:
        if "incompatible version of torchao" not in str(exc):
            raise
        peft_torchao.is_torchao_available = lambda: False


def prepare_model(model_contract: Mapping[str, Any], row: Mapping[str, Any], adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_contract["id"], revision=model_contract["revision"], use_fast=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if row["base_precision"] == "bnb_default":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    elif row["base_precision"] == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unsupported frozen precision: {row['base_precision']}")
    base = AutoModelForCausalLM.from_pretrained(
        model_contract["id"], revision=model_contract["revision"], **model_kwargs
    )
    _disable_incompatible_torchao()
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tokenizer


def generate_variant(model, tokenizer, questions: Sequence[str], *, beam_width: int, batch_size: int, max_new_tokens: int):
    import torch

    rows = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if beam_width > 1:
            kwargs.update(
                num_beams=beam_width,
                num_return_sequences=beam_width,
                early_stopping=True,
            )
        with torch.inference_mode():
            outputs = model.generate(**inputs, **kwargs)
        input_width = inputs["input_ids"].shape[1]
        per_example = beam_width if beam_width > 1 else 1
        for offset in range(len(batch)):
            sequences = outputs[offset * per_example : (offset + 1) * per_example]
            new_text = [
                tokenizer.decode(
                    sequence[input_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for sequence in sequences
            ]
            full_text = [
                tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for sequence in sequences
            ]
            rows.append(
                {
                    "new_text": new_text,
                    "new_ids": [extract_poi(text) for text in new_text],
                    "full_ids": [extract_poi(text) for text in full_text],
                }
            )
    return rows


def run_inference(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    row_key: str,
    adapter_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = pc.load_model_config(config_path)
    row = select_row(config, row_key)
    profile = config["data_profiles"][row["city"]]
    model_contract = config["models"][row["model"]]
    load_prompt_manifest(root / profile["path"])
    paths = prompt_paths(root, config, row)
    prompts = json.loads(paths["test"].read_text(encoding="utf-8"))
    expected_prompt_hash = profile["prompt_sequence_sha256"][row["prompt_variant"]]["test"]
    if sequence_sha256(prompts) != expected_prompt_hash:
        raise ValueError("Test prompt sequence hash changed")
    questions = [text.split(ANSWER_MARKER, 1)[0] for text in prompts]
    targets = [extract_poi(text.split(ANSWER_MARKER, 1)[1]) for text in prompts]
    adapter = Path(adapter_path).resolve()
    from train import verify_training_manifest
    verify_training_manifest(adapter, config, row)
    adapter_hash = tree_sha256(adapter)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Inference output directory must be empty: {output}")
    model, tokenizer = prepare_model(model_contract, row, adapter)
    generated = {
        beam: generate_variant(
            model,
            tokenizer,
            questions,
            beam_width=beam,
            batch_size=int(row["inference"]["batch_size"]),
            max_new_tokens=int(row["inference"]["max_new_tokens"]),
        )
        for beam in BEAMS
    }
    raw_records = [
        {
            "index": index,
            "target_poi": targets[index],
            "generation": {f"beam{beam}": generated[beam][index] for beam in BEAMS},
        }
        for index in range(len(prompts))
    ]
    # Release GPU storage before the CPU geospatial evaluation.
    del model, tokenizer
    prediction_path = output / "predictions.json.gz"
    prediction_payload = {
        "summary": {
            "contract": config["contract"],
            "row_key": row_key,
            "model": model_contract,
            "adapter_tree_sha256": adapter_hash,
            "prompt_sequence_sha256": expected_prompt_hash,
            "inference": row["inference"],
        },
        "records": raw_records,
    }
    _atomic_json_gz(prediction_path, prediction_payload)
    prediction_hash = sha256_file(prediction_path)

    from safety import score_safety

    safety_path = output / "safety.json"
    safety_payload = score_safety(
        repo_root=root,
        config_path=config_path,
        city=str(row["city"]),
        predictions_path=prediction_path,
        output_path=safety_path,
        allow_live_osrm=False,
    )

    weights_payload, result_payload = evaluate_run(
        root, config, row_key, prediction_path, safety_path
    )
    weights_path = output / "weights.json"
    result_path = output / "result.json"
    _atomic_json(weights_path, weights_payload)
    _atomic_json(result_path, result_payload)
    return {
        "row_key": row_key,
        "predictions": str(prediction_path),
        "predictions_sha256": prediction_hash,
        "safety": str(safety_path),
        "safety_sha256": sha256_file(safety_path),
        "weights": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "accepted": result_payload["accepted"],
        "metrics": result_payload["metrics"],
        "record_count": len(raw_records),
    }


if __name__ == "__main__":
    import sys
    from batch_runner import main
    main(sys.argv[1:], command="infer")
