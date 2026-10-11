"""Evaluate raw model predictions and post-hoc route safety."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import pickle
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs import preprocessing_config as pc
from text_utils import load_prompt_manifest, read_json, sequence_sha256, sha256_file
from train import prompt_paths, select_row


METRICS = ("acc1", "acc3", "acc5", "mrr", "safety_median")


def safety_index(payload: Mapping[str, Any]) -> dict[int, float | None]:
    return {
        int(row["index"]): None if row.get("safety") is None else float(row["safety"])
        for row in payload.get("records", [])
    }


def metric_vector(
    records: Sequence[Mapping[str, Any]],
    *,
    safety_by_index: Mapping[int, float | None],
    parser: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Evaluation requires at least one test trajectory")
    contributions = {name: [] for name in ("acc1", "acc3", "acc5", "mrr")}
    safety_scores = []
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
        if safety is not None:
            safety_scores.append(float(safety))
    return {
        **{name: math.fsum(values) / len(records) for name, values in contributions.items()},
        "safety_median": statistics.median(safety_scores) if safety_scores else None,
        "test_trajectories": len(records),
        "scored_routes": len(safety_scores),
    }


def report_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Record accuracy, ranking quality, and predicted-route safety."""
    result = {
        "unrounded": {name: metrics[name] for name in METRICS},
        "reported_metrics": {
            name: None if metrics[name] is None else round(float(metrics[name]), 4)
            for name in METRICS
        },
    }
    if any(metrics[name] is None for name in METRICS):
        result["reason"] = "At least one metric is undefined; no valid Safety contributions."
    return result


def load_run_inputs(root, config, row, splits, predictions_path, safety_path):
    """Rebuild observed features and verify Safety belongs to the same top1 IDs.

    Accept the predictions.json.gz and safety.json emitted by public inference.
    Observed features are rebuilt from the numeric inputs. Evaluation labels are
    checked against the numeric test targets and are used only by the subsequent
    metric calculation.
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
        "city": row["city"], "contract": config["contract"], "allow_live_osrm": True,
        "projected_crs": runtime["projected_crs"], "nominal_buffer": profile["crime_radius_m"],
        "crime_window_weeks": profile["crime_time_weeks"], "route_origin": runtime["route_origin"],
        "major_nyc_offenses_only": runtime["major_nyc_offenses_only"], "poi_catalog_source": runtime["poi_catalog"],
        "input_sha256": {
            key: sha256_file(root / runtime[key]["path"])
            for key in ("crime_source", "raw_checkins")
        } | {"safety_train": sha256_file(root / profile["path"] / "safety/train_trajs_with_safety.pickle")},
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
        beam1 = prediction["generation"]["beam1"][row["inference"]["parser"]]
        head = int(beam1[0]) if beam1 else -1
        if score.get("prediction") != head or score.get("route", {}).get("poi") != head:
            raise ValueError(f"Run Safety top-one prediction differs from raw beam-1 at {index}")
        value = score.get("safety")
        if score.get("status") == "scored":
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Scored Safety must be finite and between zero and one")
        elif score.get("status") != "unscored" or value is not None:
            raise ValueError("Unscored Safety must have no value")
        if value != score["route"].get("safety"):
            raise ValueError("Run Safety score differs from the predicted route")
        frame = splits["test"][index]
        pois = [int(value) for value in frame["poi_id"]]
        target = prediction.get("target_poi", prediction.get("target"))
        if target != pois[-1]:
            raise ValueError("Prediction target differs from numeric test label")
        records.append({
            "index": index, "user_id": int(frame["user_id"].iloc[0]),
            "history_pois": pois[:-1], "target_poi": int(target),
            "generation": prediction["generation"],
        })
    source = {
        "mode": "supplied_run_artifacts", "predictions_sha256": sha256_file(predictions_path),
        "safety_sha256": sha256_file(safety_path), "adapter_tree_sha256": summary.get("adapter_tree_sha256"),
        "fresh_model_execution_performed_by_evaluator": False,
    }
    return records, safety, source


def load_numeric_splits(root, profile):
    """Verify generated numeric inputs against their preprocessing manifest."""
    folder = Path(root) / profile["path"]
    manifest = load_prompt_manifest(folder)
    splits = {}
    for split in ("train", "test"):
        path = folder / f"{split}_trajectories.pickle"
        if sha256_file(path) != manifest["source_sha256"][f"numeric_{split}"]:
            raise ValueError(f"Numeric input changed after preprocessing: {path}")
        with path.open("rb") as handle:
            splits[split] = pickle.load(handle)
    safety_path = folder / "safety/test_trajs_with_safety.pickle"
    if sha256_file(safety_path) != manifest["source_sha256"]["safety_test"]:
        raise ValueError(f"Numeric Safety input changed: {safety_path}")
    return splits


def evaluate_records(row, records, safety_payload):
    """Evaluate each test trajectory once, using the model's original outputs."""
    values = metric_vector(
        records, safety_by_index=safety_index(safety_payload),
        parser=row["inference"]["parser"],
    )
    return {
        "row_key": row["row_key"],
        "evaluation_results": report_metrics(values),
        "test_trajectories": values["test_trajectories"],
        "scored_routes": values["scored_routes"],
    }


def evaluate_run(root, config, row_key, predictions_path, safety_path):
    """Evaluate supplied inference artifacts without changing model predictions."""
    row = select_row(config, row_key)
    splits = load_numeric_splits(root, config["data_profiles"][row["city"]])
    records, safety, source = load_run_inputs(root, config, row, splits, predictions_path, safety_path)
    result = evaluate_records(row, records, safety)
    result["input_source"] = source
    return result


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
    config = pc.load_model_config(config_path, repo_root=root)
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
    score_safety(
        repo_root=root,
        config_path=config_path,
        city=str(row["city"]),
        predictions_path=prediction_path,
        output_path=safety_path,
    )

    result_payload = evaluate_run(root, config, row_key, prediction_path, safety_path)
    result_path = output / "result.json"
    _atomic_json(result_path, result_payload)
    return {
        "row_key": row_key,
        "predictions": str(prediction_path),
        "predictions_sha256": prediction_hash,
        "safety": str(safety_path),
        "safety_sha256": sha256_file(safety_path),
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "evaluation_results": result_payload["evaluation_results"],
        "record_count": len(raw_records),
    }


if __name__ == "__main__":
    import sys
    from batch_runner import main
    main(sys.argv[1:], command="infer")
