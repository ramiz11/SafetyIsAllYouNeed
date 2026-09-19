"""Synthetic-output tests: these never execute or download a language model."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coordinate_pairs_v2.cli import main, run_rows, write_new_json
from coordinate_pairs_v2.evaluate import evaluate_records, evaluate_run, load_numeric_splits
from coordinate_pairs_v2.infer import _atomic_json, _atomic_json_gz, run_inference
from coordinate_pairs_v2.metrics import metric_vector, read_json, compare
from coordinate_pairs_v2.train import verify_training_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "configs/coordinate_pairs_v2.json")


def synthetic_inputs(row, splits):
    profile = CONFIG["data_profiles"][row["city"]]
    runtime = profile["safety_runtime"]
    records, scores = [], []
    for index, frame in enumerate(splits["test"]):
        target = int(frame.poi_id.iloc[-1])
        records.append({"index": index, "target_poi": target,
                        "generation": {f"beam{k}": {"new_ids": [target], "full_ids": [target]}
                                       for k in (1, 3, 5, 10)}})
        scores.append({"index": index, "prediction": target, "status": "scored", "safety": .5})
    predictions = {"summary": {
        "contract": CONFIG["contract"], "row_key": row["row_key"],
        "model": CONFIG["models"][row["model"]], "inference": row["inference"],
        "prompt_sequence_sha256": profile["prompt_sequence_sha256"][row["prompt_variant"]]["test"],
        "adapter_tree_sha256": "synthetic_test_not_a_model",
    }, "records": records}
    safety = {"summary": {
        "city": row["city"], "contract": CONFIG["contract"], "allow_live_osrm": False,
        "projected_crs": runtime["projected_crs"], "nominal_buffer": profile["crime_radius_m"],
        "crime_window_weeks": profile["crime_time_weeks"], "route_origin": runtime["route_origin"],
        "major_nyc_offenses_only": runtime["major_nyc_offenses_only"], "poi_catalog_source": runtime["poi_catalog"],
        "input_sha256": {key: runtime[key]["sha256"] for key in ("route_cache", "normalization_stats", "crime_source")},
    }, "records": scores}
    return predictions, safety


def own_adapter(root, row):
    root = Path(root)
    profile = CONFIG["data_profiles"][row["city"]]
    manifest = {
        "contract": CONFIG["contract"], "serializer_version": CONFIG["serializer_version"],
        "row": row, "model": CONFIG["models"][row["model"]],
        "training": CONFIG["training_defaults"],
        "prompt_sequence_sha256": profile["prompt_sequence_sha256"][row["prompt_variant"]],
    }
    _atomic_json(root / "run_manifest.json", manifest)
    adapter = root / "checkpoints" / f"checkpoint-{row['checkpoint']['step']}"
    adapter.mkdir(parents=True)
    _atomic_json(adapter / "adapter_config.json", {"test_fixture": True})
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic fixture; not model weights")
    return adapter


class PublicWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {city: load_numeric_splits(ROOT, profile) for city, profile in CONFIG["data_profiles"].items()}

    def evaluate_fixture(self, row, predictions, safety):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _atomic_json_gz(folder / "predictions.json.gz", predictions)
            _atomic_json(folder / "safety.json", safety)
            return evaluate_run(ROOT, CONFIG, row["row_key"], folder / "predictions.json.gz", folder / "safety.json")

    def test_all_six_compute_actual_metrics_not_reference_values(self):
        expected = [(58, 304), (26, 26), (33, 409), (172, 1258), (65, 65), (184, 1094)]
        for row, population in zip(CONFIG["rows"], expected):
            with self.subTest(row=row["row_key"]):
                predictions, safety = synthetic_inputs(row, self.splits[row["city"]])
                weights, result = self.evaluate_fixture(row, predictions, safety)
                self.assertEqual((result["population"]["unique_n"], result["population"]["effective_n"]), population)
                self.assertEqual(sum(weights["weights"]), population[1])
                for name in ("acc1", "acc3", "acc5", "mrr"):
                    self.assertEqual(result["metrics"]["unrounded"][name], 1.)
                self.assertEqual(result["metrics"]["unrounded"]["safety_median"], .5)
                self.assertFalse(result["accepted"])

    def test_changed_predictions_change_metrics_not_weights(self):
        row = CONFIG["rows"][1]
        predictions, safety = synthetic_inputs(row, self.splits[row["city"]])
        weights, result = self.evaluate_fixture(row, predictions, safety)
        index = next(i for i, w in enumerate(weights["weights"]) if w)
        predictions["records"][index]["generation"]["beam3"]["new_ids"] = [-1]
        changed_weights, changed = self.evaluate_fixture(row, predictions, safety)
        self.assertEqual(changed_weights, weights)
        self.assertLess(changed["metrics"]["unrounded"]["acc3"], result["metrics"]["unrounded"]["acc3"])

    def test_wrong_target_and_mismatched_safety_are_rejected(self):
        row = CONFIG["rows"][1]
        predictions, safety = synthetic_inputs(row, self.splits[row["city"]])
        altered = copy.deepcopy(predictions)
        altered["records"][0]["target_poi"] = -999
        with self.assertRaisesRegex(ValueError, "target differs"):
            self.evaluate_fixture(row, altered, safety)
        safety["records"][0]["prediction"] = -999
        with self.assertRaisesRegex(ValueError, "top-one"):
            self.evaluate_fixture(row, predictions, safety)

    def test_recipe_has_no_private_weights_or_old_population_tables(self):
        for row in CONFIG["rows"]:
            for key in ("artifacts", "adapter_tree_sha256", "population_weight_sha256", "population_rule"):
                self.assertNotIn(key, row)

    def test_own_adapter_is_accepted_but_wrong_recipe_is_not(self):
        row = CONFIG["rows"][0]
        with tempfile.TemporaryDirectory() as directory:
            adapter = own_adapter(directory, row)
            verify_training_manifest(adapter, CONFIG, row)
            (adapter / "adapter_model.safetensors").write_bytes(b"a different independently trained adapter")
            verify_training_manifest(adapter, CONFIG, row)
            wrong = copy.deepcopy(row)
            wrong["seed"] += 1
            with self.assertRaisesRegex(ValueError, "seed"):
                verify_training_manifest(adapter, CONFIG, wrong)

    def test_infer_automatically_evaluates_without_reference_artifacts(self):
        row = CONFIG["rows"][1]
        predictions, safety = synthetic_inputs(row, self.splits[row["city"]])
        with tempfile.TemporaryDirectory() as directory:
            adapter = own_adapter(Path(directory) / "training", row)
            output = Path(directory) / "evaluation"
            def generate(model, tokenizer, questions, *, beam_width, **kwargs):
                self.assertTrue(all("<answer>:" not in text for text in questions))
                return [r["generation"][f"beam{beam_width}"] for r in predictions["records"]]
            def score(**kwargs):
                _atomic_json(kwargs["output_path"], safety)
                return safety
            with patch("coordinate_pairs_v2.infer.prepare_model", return_value=(None, None)), \
                 patch("coordinate_pairs_v2.infer.generate_variant", side_effect=generate), \
                 patch("coordinate_pairs_v2.score_safety.score_safety", side_effect=score):
                result = run_inference(repo_root=ROOT, config_path=ROOT / "configs/coordinate_pairs_v2.json",
                                       row_key=row["row_key"], adapter_path=adapter, output_dir=output)
            self.assertEqual(result["metrics"]["unrounded"]["acc1"], 1.)
            self.assertEqual(read_json(output / "result.json")["evaluation_contract"], "connected_population_v1")
            self.assertTrue((output / "weights.json").is_file())
            with patch("coordinate_pairs_v2.infer.prepare_model", side_effect=AssertionError("Model should not load")):
                with self.assertRaises(FileExistsError):
                    run_inference(repo_root=ROOT, config_path=ROOT / "configs/coordinate_pairs_v2.json",
                                  row_key=row["row_key"], adapter_path=adapter, output_dir=output)

    def test_cli_requires_new_run_inputs_and_fresh_outputs(self):
        with self.assertRaises(SystemExit):
            main(["evaluate", "--row", "NYC|our_method", "--output-dir", "unused"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new_json(path, {"keep": True})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"keep": False})
            self.assertEqual(read_json(path), {"keep": True})


class MetricTests(unittest.TestCase):
    def test_weighted_metrics_and_invalid_safety(self):
        records = [
            {"index": 0, "target_poi": 2, "generation": {f"beam{k}": {"new_ids": [1, 2]} for k in (1, 3, 5, 10)}},
            {"index": 1, "target_poi": 1, "generation": {f"beam{k}": {"new_ids": [1]} for k in (1, 3, 5, 10)}},
        ]
        values = metric_vector(records, [3, 1], safety_by_index={0: .8, 1: None},
                               parser="new_ids", safety_aggregation="mean_zero_fill")
        self.assertEqual(values["acc1"], .25)
        self.assertEqual(values["acc3"], 1.)
        self.assertEqual(values["mrr"], .625)
        self.assertAlmostEqual(values["safety_median"], .6)
        missing = metric_vector(records, [3, 1], safety_by_index={}, parser="new_ids", safety_aggregation="mean_valid")
        comparison = compare(missing, CONFIG["rows"][0]["published_target"])
        self.assertIsNone(comparison["rmse"])
        self.assertFalse(comparison["normal_threshold_passed"])
        with self.assertRaises(ValueError):
            metric_vector(records, [-1, 2], safety_by_index={}, parser="new_ids", safety_aggregation="mean_valid")
