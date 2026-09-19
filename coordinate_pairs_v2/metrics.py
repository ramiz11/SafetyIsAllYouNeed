"""Metric definitions and JSON/hash utilities for fresh model evaluations."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np



METRICS = ("acc1", "acc3", "acc5", "mrr", "safety_median")


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
