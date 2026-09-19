from __future__ import annotations
from configs import preprocessing_config as pc
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import text_utils
from text_utils import audit_prompts, materialize_prompts, validate_manifest_header, verify_canonical_prompts, write_canonical_prompts
from text_utils import read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = pc.load_model_config()

class PromptContractTests(unittest.TestCase):
    def test_canonical_manifest_round_trip_and_tampering(self):
        frame = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "poi_id": [11, 11, 12],
                "latitude": [40.1, 40.2, 40.3],
                "longitude": [-73.1, -73.2, -73.3],
                "local_time": pd.to_datetime(
                    ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]
                ),
            }
        )
        scored = frame.assign(normalized_safety=[0.7, 0.9, -1.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safety").mkdir()
            for split in ("train", "validation", "test"):
                with (root / f"{split}_trajectories.pickle").open("wb") as handle:
                    pickle.dump([frame], handle)
                with (root / "safety" / f"{split}_trajs_with_safety.pickle").open("wb") as handle:
                    pickle.dump([scored], handle)
            written = write_canonical_prompts(root)
            verified = verify_canonical_prompts(root)
            self.assertEqual(verified, json.loads(json.dumps(written)))
            path = root / "coordinate_pairs_v2_manifest.json"
            altered = json.loads(path.read_text())
            altered["source_audit"]["train"]["trajectory_count"] += 1
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_canonical_prompts(root)

    def test_repeated_poi_retains_each_rows_coordinates_and_target_is_hidden(self):
        frame = pd.DataFrame(
            {
                "user_id": [7, 7, 7],
                "poi_id": [11, 11, 12],
                "latitude": [40.1, 40.2, 40.3],
                "longitude": [-73.1, -73.2, -73.3],
                "category": ["observed-a", "observed-b", "held-out-category"],
                "local_time": pd.to_datetime(
                    ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]
                ),
            }
        )
        prompt = text_utils.build_prompt(
            frame, userid=7, poi_id_range=20, prompt_contract="coordinate_pairs_v2"
        )
        question, answer = prompt.split("<answer>:", 1)
        self.assertIn("POI id 11 at <40.100000, -73.100000>", question)
        self.assertIn("POI id 11 at <40.200000, -73.200000>", question)
        self.assertNotIn("40.300000", question)
        self.assertNotIn("January 01, 2020, at 3:00 AM", question)
        self.assertNotIn("POI id 12", question)
        self.assertNotIn("held-out-category", question)
        self.assertEqual(answer.strip(), "POI id 12.")
        audit = audit_prompts([prompt], [frame], include_safety=False)
        self.assertEqual(audit.row_coordinate_mismatches, 0)

    def test_invalid_coordinate_policy_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "user_id": [7, 7],
                "poi_id": [11, 12],
                "latitude": [float("nan"), 40.2],
                "longitude": [-73.1, -73.2],
                "local_time": pd.to_datetime(
                    ["2020-01-01 01:00", "2020-01-01 03:00"]
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "invalid coordinates"):
            text_utils.build_prompt(
                frame,
                userid=7,
                poi_id_range=20,
                prompt_contract="coordinate_pairs_v2",
                coordinate_missing="error",
            )
        prompt = text_utils.build_prompt(
            frame,
            userid=7,
            poi_id_range=20,
            prompt_contract="coordinate_pairs_v2",
            coordinate_missing="omit",
        )
        question, _ = prompt.split("<answer>:", 1)
        self.assertIn("visited POI id 11.", question)
        self.assertNotIn("<nan", question)
        audit_prompts(
            [prompt], [frame], include_safety=False, coordinate_missing="omit"
        )

    def test_canonical_sequences_regenerate_from_numeric_sources(self):
        for profile in CONFIG["data_profiles"].values():
            materialized, manifest = materialize_prompts(ROOT / profile["path"])
            validate_manifest_header(manifest)
            for variant in ("no_safety", "with_safety"):
                for split in ("train", "validation", "test"):
                    metadata = manifest["variants"][variant]["splits"][split]
                    self.assertEqual(
                        metadata["sequence_sha256"],
                        profile["prompt_sequence_sha256"][variant][split],
                    )
                    self.assertNotEqual(
                        metadata["sequence_sha256"],
                        metadata["coordinate_free_v1_control_sequence_sha256"],
                    )
                    source_audit = manifest["source_audit"][split]
                    self.assertEqual(source_audit["coordinate_invalid_rows"], 0)
                    self.assertEqual(
                        source_audit["coordinate_valid_rows"],
                        source_audit["row_count"],
                    )
                    self.assertEqual(
                        source_audit["trajectory_length_counts"],
                        {"20": source_audit["trajectory_count"]},
                    )
                    self.assertTrue(source_audit["numeric_safety_alignment"])
                    if variant == "no_safety":
                        path = ROOT / profile["path"] / f"textual_{split}_trajs.json"
                    else:
                        path = ROOT / profile["path"] / "safety" / f"safety_textual_{split}_trajs.json"
                    self.assertEqual(json.loads(path.read_text()), materialized[variant][split])
