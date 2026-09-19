"""Evaluate newly generated predictions using the connected population rules."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np

from .candidate_postprocessing import history_supported_alternatives, shared_training_augmentation
from .metrics import compare, metric_vector, population_weight_sha256, safety_index, sha256_file
from .population_mechanisms import (
    derive_history_quality_envelope,
    derive_session_threshold,
    eligible_trailing_session_multiplicities,
    history_quality_mask,
    novelty_signature_join_multiplicities,
    observed_session_counts,
    observed_stream_session_representatives,
)
from .run_inputs import load_run_inputs
from .train import select_row


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
