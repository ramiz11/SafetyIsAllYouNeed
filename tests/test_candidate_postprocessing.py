import copy
import unittest
import json
from pathlib import Path

import pandas as pd

from coordinate_pairs_v2.candidate_postprocessing import history_supported_alternatives, shared_training_augmentation


class CandidatePostprocessingTests(unittest.TestCase):
    def test_shared_augmentation_target_independence_and_idempotence(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/coordinate_pairs_v2.json").read_text())
        contract = config["methods"]["llm4poi_original"]["ranking"]
        train = [pd.DataFrame({"user_id": [1, 1, 1], "poi_id": [2, 3, 4]})]
        record = {"user_id": 1, "history_pois": [2, 3], "target_poi": 777,
                  "generation": {f"beam{k}": {"new_ids": [99, 3], "full_ids": [99, 3]} for k in (1, 3, 5, 10)}}
        before = copy.deepcopy(record)
        actual = shared_training_augmentation([record], train, contract)
        self.assertEqual(record, before)
        self.assertEqual(actual[0]["generation"]["beam1"], before["generation"]["beam1"])
        self.assertEqual(actual[0]["generation"]["beam3"]["new_ids"], [99, 3, 4, 2])
        self.assertEqual(shared_training_augmentation(actual, train, contract), actual)
        record["target_poi"] = 2
        self.assertEqual(shared_training_augmentation([record], train, contract)[0]["generation"], actual[0]["generation"])
        invalid = copy.deepcopy(contract)
        invalid["alphas"]["1"] = 1.0
        with self.assertRaises(ValueError):
            shared_training_augmentation([record], train, invalid)

    def test_heads_and_order_preserved_without_input_mutation(self):
        record = {"history_pois": [2, 3], "target_poi": 99,
                  "generation": {f"beam{k}": {"new_ids": [99, 4, 3, 2], "full_ids": [98, 3, 4, 2]}
                                 for k in (1, 3, 5, 10)}}
        before = copy.deepcopy(record)
        actual = history_supported_alternatives([record])[0]
        self.assertEqual(record, before)
        self.assertEqual(actual["generation"]["beam1"], before["generation"]["beam1"])
        for k in (3, 5, 10):
            self.assertEqual(actual["generation"][f"beam{k}"]["new_ids"], [99, 3, 2])
            self.assertEqual(actual["generation"][f"beam{k}"]["full_ids"], [98, 3, 2])
        record["target_poi"] = 4
        self.assertEqual(history_supported_alternatives([record])[0]["generation"], actual["generation"])

    def test_empty_lists_and_no_target_field(self):
        record = {"history_pois": [], "generation": {
            f"beam{k}": {"new_ids": [], "full_ids": [7, 8]} for k in (1, 3, 5, 10)}}
        actual = history_supported_alternatives([record])[0]
        self.assertEqual(actual["generation"]["beam3"], {"new_ids": [], "full_ids": [7]})


if __name__ == "__main__":
    unittest.main()
