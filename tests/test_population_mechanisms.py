from configs import preprocessing_config as pc
import unittest
import numpy as np
import pandas as pd
from preprocessing import (
    derive_session_threshold, observed_session_counts,
    eligible_trailing_session_multiplicities,
    observed_stream_session_representatives,
    novelty_signature_join_multiplicities,
    derive_history_quality_envelope, history_quality_mask, observed_history_quality,
)

def trajectory(user=1, day="2020-01-01", minutes=(0, 5, 40, 41)):
    return pd.DataFrame({"user_id": [user] * len(minutes),
                         "poi_id": list(range(len(minutes))),
                         "local_time": [pd.Timestamp(day) + pd.Timedelta(minutes=m) for m in minutes]})


class PopulationMechanismTests(unittest.TestCase):
    def test_target_does_not_change_group_or_session(self):
        original = trajectory()
        altered = original.copy()
        altered.loc[altered.index[-1], "local_time"] = pd.Timestamp("2099-01-01")
        altered.loc[altered.index[-1], "user_id"] = 123456
        altered.loc[altered.index[-1], "poi_id"] = 987654
        self.assertEqual(observed_session_counts(original, 5), observed_session_counts(altered, 5))

    def test_session_threshold_inclusive_and_backwards_break(self):
        self.assertEqual(observed_session_counts(trajectory(), 5),
                         {"session_count": 2, "close_event_count": 2, "trailing_session_event_count": 1})
        self.assertEqual(observed_session_counts(trajectory(minutes=(0, 0, -1, 2)), 5),
                         {"session_count": 2, "close_event_count": 2, "trailing_session_event_count": 1})

    def test_invalid_mask_and_threshold_rejected(self):
        with self.assertRaises(ValueError):
            eligible_trailing_session_multiplicities([trajectory()], [2], 5)
        with self.assertRaises(ValueError):
            observed_session_counts(trajectory(), float("nan"))

    def test_training_cadence_and_unique_edge_threshold_derivation(self):
        frames = [trajectory(), trajectory()]
        self.assertEqual(derive_session_threshold(frames, "window_gaps", "quartile_3"), 35.0)
        self.assertEqual(derive_session_threshold(frames, "unique_edges", "quartile_3"), 27.5)
        self.assertEqual(derive_session_threshold(frames, "window_gaps", "decile_9"), 35.0)
        frames[0].loc[frames[0].index[-1], "local_time"] = pd.Timestamp("2099-01-01")
        self.assertEqual(derive_session_threshold(frames, "window_gaps", "quartile_3"), 35.0)

    def test_session_join_and_event_exposure(self):
        first = trajectory()
        second = trajectory(minutes=(5, 40, 42, 43))
        second["poi_id"] = [1, 2, 4, 5]
        np.testing.assert_array_equal(eligible_trailing_session_multiplicities([first, second], [1, 1], 10), [2, 2])
        np.testing.assert_array_equal(eligible_trailing_session_multiplicities([first, second], [1, 0], 10), [1, 0])

    def test_merged_observed_session_representatives(self):
        first = trajectory(minutes=(0, 5, 10, 15))
        second = trajectory(minutes=(5, 10, 15, 20))
        second["poi_id"] = [1, 2, 3, 4]
        later = trajectory(minutes=(60, 65, 70, 75))
        frames = [first, second, later, trajectory(user=2, minutes=(0, 5, 10, 15))]
        np.testing.assert_array_equal(observed_stream_session_representatives(frames, 5), [1, 0, 1, 1])
        np.testing.assert_array_equal(observed_stream_session_representatives(frames, 5, selection="last"), [0, 1, 1, 1])
        np.testing.assert_array_equal(observed_stream_session_representatives(frames, 5, [0, 1, 1, 1]), [0, 1, 1, 1])
        # Moving windows have different prefix starts but share the same real
        # session. Target attributes cannot create or bridge observed sessions.
        for frame in frames:
            frame.loc[frame.index[-1], "local_time"] = pd.Timestamp("2099-01-01")
            frame.loc[frame.index[-1], "poi_id"] = 999999
        np.testing.assert_array_equal(observed_stream_session_representatives(frames, 5), [1, 0, 1, 1])
        with self.assertRaises(ValueError):
            observed_stream_session_representatives(frames, 5, selection="random")
        with self.assertRaises(ValueError):
            observed_stream_session_representatives(frames, 5, [1, 1, 1, 2])

    def test_session_gap_ties_and_empty_collection(self):
        repeated_time = trajectory(minutes=(0, 0, 5, 999))
        np.testing.assert_array_equal(observed_stream_session_representatives([repeated_time, repeated_time.copy()], 5), [1, 0])
        self.assertEqual(observed_stream_session_representatives([], 5).tolist(), [])
        with self.assertRaises(ValueError):
            observed_stream_session_representatives([repeated_time], float("nan"))

    def test_novelty_signature_join_and_target_exclusion(self):
        train = pd.DataFrame({"poi_id": [0, 1, 2]})
        multiple = pd.DataFrame({"poi_id": [0, 9, 1, 2, 10, 999]})
        single = pd.DataFrame({"poi_id": [0, 9, 1, 2, 999]})
        other = multiple.copy()
        other.loc[other.index[-1], "poi_id"] = 123456
        frames = [multiple, other, single]
        np.testing.assert_array_equal(novelty_signature_join_multiplicities([train], frames), [2, 2, 0])
        np.testing.assert_array_equal(novelty_signature_join_multiplicities([train], frames, [1, 0, 1]), [1, 0, 0])
        with self.assertRaises(ValueError):
            novelty_signature_join_multiplicities([train], frames, [2, 1, 1])

    def test_training_quality_envelope_and_target_exclusion(self):
        def spatial(minutes):
            frame = trajectory(minutes=minutes)
            frame["latitude"] = 40.
            frame["longitude"] = -73.
            return frame
        training = [spatial((0,5,10,999)), spatial((0,10,20,999)),
                    spatial((0,15,30,999)), spatial((0,20,40,999))]
        bounds = derive_history_quality_envelope(training)
        self.assertAlmostEqual(bounds["span_hours"]["q1"], 17.5/60)
        self.assertAlmostEqual(bounds["span_hours"]["q3"], 32.5/60)
        self.assertAlmostEqual(bounds["span_hours"]["upper"], 55/60)
        self.assertEqual(bounds["step_distance_km_max"]["upper"], 0.)
        normal, outlier = spatial((0,10,20,999)), spatial((0,60,120,999))
        np.testing.assert_array_equal(history_quality_mask([normal,outlier],bounds),[True,False])
        old = observed_history_quality(normal)
        normal.loc[normal.index[-1], "latitude"] = float("nan")
        normal.loc[normal.index[-1], "local_time"] = pd.Timestamp("2099-01-01")
        self.assertEqual(observed_history_quality(normal),old)
        training[0].loc[training[0].index[-1], "latitude"] = float("nan")
        self.assertEqual(derive_history_quality_envelope(training),bounds)
        with self.assertRaises(ValueError):
            derive_history_quality_envelope([])
        normal.loc[normal.index[0], "latitude"] = float("nan")
        with self.assertRaises(ValueError):
            observed_history_quality(normal)

    def test_both_cities_share_one_method_definition(self):
        config = pc.load_model_config()
        self.assertEqual(set(config["methods"]), {"our_method", "llm4poi_31", "llm4poi_original"})
        for row in config["rows"]:
            self.assertIn(row["variant"], config["methods"])
            self.assertNotIn("population_rule", row)


if __name__ == "__main__":
    unittest.main()
