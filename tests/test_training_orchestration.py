"""Exercise the training/orchestration code with an explicitly fake ML runtime."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from coordinate_pairs_v2.cli import run_rows, write_new_json
from coordinate_pairs_v2.metrics import read_json
from coordinate_pairs_v2.train import run_training, verify_training_manifest
from baselines.llm4poi.run_llm4poi_baseline import main as baseline_main

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/coordinate_pairs_v2.json"
CONFIG = read_json(CONFIG_PATH)


class TrainingOrchestrationTests(unittest.TestCase):
    def test_training_uses_only_train_and_validation_and_emits_own_checkpoint(self):
        row = CONFIG["rows"][0]
        observed = {}
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(use_cache=True),
            parameters=lambda: [types.SimpleNamespace(requires_grad=True)],
        )
        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="<eos>")

        class FakeDataset:
            @staticmethod
            def from_dict(value):
                return types.SimpleNamespace(map=lambda *a, **k: value["text"])

        class FakeTrainer:
            def __init__(self, **kwargs):
                observed.update(kwargs)
                self.state = types.SimpleNamespace(
                    global_step=row["checkpoint"]["train_stop_step"], max_steps=12894,
                    best_model_checkpoint=None, best_metric=.5,
                )

            def train(self, resume_from_checkpoint=None):
                observed["resume"] = resume_from_checkpoint
                control = types.SimpleNamespace(should_save=False, should_training_stop=False)
                observed["callbacks"][0].on_step_end(observed["args"], self.state, control)
                self_test.assertTrue(control.should_training_stop)
                self_test.assertTrue(control.should_save)
                selected = Path(observed["args"].output_dir) / f"checkpoint-{row['checkpoint']['step']}"
                selected.mkdir(parents=True)
                (selected / "adapter_config.json").write_text("{}")
                (selected / "adapter_model.safetensors").write_bytes(b"fake test adapter")
                return types.SimpleNamespace(metrics={"test_runtime_only": True})

        self_test = self
        fake_torch = types.SimpleNamespace(float16="fake_fp16", cuda=types.SimpleNamespace(is_available=lambda: True))
        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
            BitsAndBytesConfig=lambda **kw: kw, TrainingArguments=lambda **kw: types.SimpleNamespace(**kw),
            TrainerCallback=object, Trainer=FakeTrainer,
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kw: kw, prepare_model_for_kbit_training=lambda m: m,
            get_peft_model=lambda m, settings: m,
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("sys.modules", {"torch": fake_torch, "datasets": types.SimpleNamespace(Dataset=FakeDataset),
                                        "peft": fake_peft, "transformers": fake_transformers}), \
             patch("coordinate_pairs_v2.train.load_and_verify_prompts", return_value={
                 "train": ["TRAIN"], "validation": ["VALIDATION"], "test": ["DO_NOT_TRAIN_ON_TEST"]}), \
             patch("coordinate_pairs_v2.train.set_seed") as seed, \
             patch("coordinate_pairs_v2.train.audit_lengths", return_value={"splits": {}}), \
             patch("coordinate_pairs_v2.infer._disable_incompatible_torchao"):
            result = run_training(repo_root=ROOT, config_path=CONFIG_PATH, row_key=row["row_key"], output_dir=directory)
            self.assertEqual(observed["train_dataset"], ["TRAIN"])
            self.assertEqual(observed["eval_dataset"], ["VALIDATION"])
            self.assertEqual(observed["args"].per_device_train_batch_size, 1)
            seed.assert_called_once_with(row["seed"])
            verify_training_manifest(result["selected_checkpoint"], CONFIG, row)
            with self.assertRaises(FileExistsError):
                run_training(repo_root=ROOT, config_path=CONFIG_PATH, row_key=row["row_key"], output_dir=directory)

    def test_all_six_runner_uses_new_training_outputs(self):
        trained, inferred = [], []
        def train(**kwargs):
            trained.append(kwargs["row_key"])
            return {"selected_checkpoint": str(Path(kwargs["output_dir"]) / "my-own-checkpoint")}
        def infer(**kwargs):
            inferred.append(kwargs["row_key"])
            self.assertTrue(str(kwargs["adapter_path"]).endswith("my-own-checkpoint"))
            output = Path(kwargs["output_dir"]) / "result.json"
            write_new_json(output, {"row_key": kwargs["row_key"], "accepted": False, "test_fixture": True})
            return {"result": str(output)}
        with tempfile.TemporaryDirectory() as directory, \
             patch("coordinate_pairs_v2.train.run_training", side_effect=train), \
             patch("coordinate_pairs_v2.infer.run_inference", side_effect=infer), \
             patch.dict("sys.modules", {"torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}):
            report = run_rows(ROOT, CONFIG_PATH, CONFIG, CONFIG["rows"], directory)
            self.assertEqual(trained, [r["row_key"] for r in CONFIG["rows"]])
            self.assertEqual(inferred, trained)
            self.assertFalse(report["all_rows_within_limits"])
            self.assertEqual(len(read_json(Path(directory) / "comparison.json")["rows"]), 6)

    def test_baseline_wrapper_delegates_to_public_recipe(self):
        with patch("baselines.llm4poi.run_llm4poi_baseline.reproduction_main") as run:
            baseline_main(["--dataset", "NYC", "--variant", "llm4poi_31", "--mode", "both", "--run-dir", "unused"])
            forwarded = run.call_args.args[0]
            self.assertEqual(forwarded[0], "run")
            self.assertIn("NYC|llm4poi_31", forwarded)
            self.assertNotIn("--seed", forwarded)
