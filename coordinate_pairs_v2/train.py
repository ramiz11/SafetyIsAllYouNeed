"""Train a configured LoRA model from the public base model and training data."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping

from .prompts import sequence_sha256, verify_canonical_prompts
from .metrics import read_json, sha256_file


ANSWER_MARKER = "<answer>:"


def select_row(config: Mapping[str, Any], row_key: str) -> Mapping[str, Any]:
    matches = [row for row in config["rows"] if row["row_key"] == row_key]
    if len(matches) != 1:
        raise ValueError(f"Expected one configuration for {row_key!r}, found {len(matches)}")
    return matches[0]


def prompt_paths(repo_root: Path, config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Path]:
    data_root = repo_root / config["data_profiles"][row["city"]]["path"]
    if row["prompt_variant"] == "with_safety":
        return {
            split: data_root / "safety" / f"safety_textual_{split}_trajs.json"
            for split in ("train", "validation", "test")
        }
    return {
        split: data_root / f"textual_{split}_trajs.json"
        for split in ("train", "validation", "test")
    }


def load_and_verify_prompts(
    repo_root: Path, config: Mapping[str, Any], row: Mapping[str, Any],
    *, verify_manifest: bool = True,
) -> dict[str, list[str]]:
    profile = config["data_profiles"][row["city"]]
    if verify_manifest:
        verify_canonical_prompts(repo_root / profile["path"])
    prompts = {
        split: json.loads(path.read_text(encoding="utf-8"))
        for split, path in prompt_paths(repo_root, config, row).items()
    }
    expected = profile["prompt_sequence_sha256"][row["prompt_variant"]]
    for split, values in prompts.items():
        observed = sequence_sha256(values)
        if observed != expected[split]:
            raise ValueError(f"{row['row_key']} {split} prompt sequence hash changed")
    return prompts


def verify_training_recipe(manifest, config, row):
    """Check the recipe without comparing learned parameters to a private run."""
    expected = {
        "contract": config["contract"],
        "serializer_version": config["serializer_version"],
        "model": config["models"][row["model"]],
        "training": config["training_defaults"],
        "prompt_sequence_sha256": config["data_profiles"][row["city"]]["prompt_sequence_sha256"][row["prompt_variant"]],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Training manifest mismatch: {key}")
    for key in ("row_key", "city", "variant", "prompt_variant", "model", "seed", "base_precision", "checkpoint"):
        if manifest.get("row", {}).get(key) != row[key]:
            raise ValueError(f"Training row mismatch: {key}")
    return manifest


def verify_training_manifest(adapter, config, row):
    """Accept the user's own trained adapter, checking its recipe, not our weights."""
    adapter = Path(adapter).resolve()
    manifest = verify_training_recipe(
        read_json(adapter.parent.parent / "run_manifest.json"), config, row
    )
    if adapter.name != f"checkpoint-{row['checkpoint']['step']}":
        raise ValueError("Use the configured checkpoint produced by your training run")
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("Training checkpoint has no adapter_config.json")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Training checkpoint has no adapter weights")
    return manifest


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_tokenize_fn(tokenizer, max_length: int):
    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        for index, text in enumerate(batch["text"]):
            answer_index = text.find(ANSWER_MARKER)
            if answer_index < 0:
                labels[index, :] = -100
                continue
            question_length = tokenizer(
                text[:answer_index],
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"].shape[1]
            labels[index, : min(question_length, labels.size(1))] = -100
        encoded["labels"] = labels
        return encoded

    return tokenize


def audit_lengths(tokenizer, prompts: Mapping[str, list[str]], max_length: int) -> dict:
    result = {"max_length": max_length, "splits": {}}
    for split, values in prompts.items():
        lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in values]
        result["splits"][split] = {
            "count": len(lengths),
            "maximum": max(lengths, default=0),
            "exceeding_count": sum(length > max_length for length in lengths),
        }
    return result


def run_training(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    row_key: str,
    output_dir: str | Path,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    root = Path(repo_root).resolve()
    config = read_json(config_path)
    row = select_row(config, row_key)
    defaults = config["training_defaults"]
    model_contract = config["models"][row["model"]]
    run_dir = Path(output_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and resume_from_checkpoint is None:
        raise FileExistsError(f"Training output directory must be empty: {run_dir}")
    if resume_from_checkpoint is not None:
        resume_path = Path(resume_from_checkpoint).resolve()
        if resume_path.parent != run_dir / "checkpoints" or not resume_path.is_dir():
            raise ValueError("Resume from a checkpoint inside this run's checkpoints directory")
        verify_training_recipe(read_json(run_dir / "run_manifest.json"), config, row)
    if not torch.cuda.is_available():
        raise RuntimeError("The configured 4-bit training recipe requires a CUDA GPU")
    prompts = load_and_verify_prompts(root, config, row)
    set_seed(int(row["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_contract["id"], revision=model_contract["revision"], use_fast=True
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    length_audit = audit_lengths(tokenizer, prompts, int(defaults["max_length"]))
    if any(item["exceeding_count"] for item in length_audit["splits"].values()):
        raise ValueError("Prompt truncation is not allowed by the frozen contract")
    model_kwargs: dict[str, Any] = {
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
    model = AutoModelForCausalLM.from_pretrained(
        model_contract["id"], revision=model_contract["revision"], **model_kwargs
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = defaults["lora"]
    from .infer import _disable_incompatible_torchao
    _disable_incompatible_torchao()
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            target_modules=list(lora["target_modules"]),
            lora_dropout=float(lora["dropout"]),
            bias=str(lora["bias"]),
            task_type="CAUSAL_LM",
        ),
    )
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The LoRA contract has no trainable parameters")

    tokenize = make_tokenize_fn(tokenizer, int(defaults["max_length"]))
    datasets = {
        split: Dataset.from_dict({"text": values}).map(
            tokenize, batched=True, remove_columns=["text"]
        )
        for split, values in prompts.items() if split != "test"
    }
    checkpoint_dir = run_dir / "checkpoints"
    arguments = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(defaults["epochs"]),
        learning_rate=float(defaults["learning_rate"]),
        warmup_steps=int(defaults["warmup_steps"]),
        weight_decay=float(defaults["weight_decay"]),
        per_device_train_batch_size=int(defaults["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(defaults["gradient_accumulation_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=int(defaults["save_steps"]),
        save_steps=int(defaults["save_steps"]),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=int(defaults["save_steps"]),
        save_total_limit=50,
        save_only_model=False,
        fp16=True,
        report_to="none",
        seed=int(row["seed"]),
        data_seed=int(row["seed"]),
    )
    selected_step = int(row["checkpoint"]["step"])
    stop_step = int(row["checkpoint"]["train_stop_step"])

    class StopAtFrozenStep(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step >= stop_step:
                control.should_save = True
                control.should_training_stop = True
            return control

    manifest = {
        "contract": config["contract"],
        "serializer_version": config["serializer_version"],
        "row": row,
        "model": model_contract,
        "training": defaults,
        "prompt_sequence_sha256": config["data_profiles"][row["city"]]["prompt_sequence_sha256"][row["prompt_variant"]],
        "prompt_file_sha256": {
            split: sha256_file(path)
            for split, path in prompt_paths(root, config, row).items()
        },
        "prompt_length_audit": length_audit,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        callbacks=[StopAtFrozenStep()],
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    selected = checkpoint_dir / f"checkpoint-{selected_step}"
    if not selected.is_dir():
        raise RuntimeError(f"Frozen selected checkpoint was not produced: {selected}")
    result = {
        "row_key": row_key,
        "global_step": trainer.state.global_step,
        "planned_max_steps": trainer.state.max_steps,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "selected_checkpoint": str(selected),
        "metrics": train_result.metrics,
    }
    (run_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", default="configs/coordinate_pairs_v2.json")
    parser.add_argument("--row", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    result = run_training(
        repo_root=args.repo_root,
        config_path=Path(args.repo_root) / args.config,
        row_key=args.row,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
