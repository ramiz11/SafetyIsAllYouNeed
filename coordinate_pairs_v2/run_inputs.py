"""Validate a supplied run and rebuild observed features from canonical data."""
import math
import copy
from pathlib import Path

from .metrics import read_json, sha256_file


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
