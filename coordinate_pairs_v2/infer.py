"""Run exact independent-beam inference and prepare final evaluation records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


from .prompts import load_prompt_manifest, sequence_sha256
from .metrics import read_json, sha256_file
from .train import prompt_paths, select_row


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
    config = read_json(config_path)
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
    from .train import verify_training_manifest
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

    from .score_safety import score_safety

    safety_path = output / "safety.json"
    safety_payload = score_safety(
        repo_root=root,
        config_path=config_path,
        city=str(row["city"]),
        predictions_path=prediction_path,
        output_path=safety_path,
        allow_live_osrm=False,
    )
    from .evaluate import evaluate_run
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="configs/coordinate_pairs_v2.json")
    parser.add_argument("--row", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_inference(
        repo_root=args.repo_root,
        config_path=Path(args.repo_root) / args.config,
        row_key=args.row,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
