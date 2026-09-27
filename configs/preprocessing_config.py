from __future__ import annotations

import os

TRAJ_LENGTH = 20
CRIME_RADIUS = 1000
CRIME_TIME_WINDOW = 3
DATASET = "CHICAGO" # "NYC" or "CHICAGO", configure during runtime.
CITY_TZ = "America/Chicago"  # "America/New_York" | "America/Chicago"
PROJ_CRS_EPSG = 3857 # meter-based CRS (good enough for 0.1–2km buffers)
CRIME_CSV = None
CHECKINS_CSV = None
CURRENT_DATA_DIR = None
SEGMENTS_COORDINATES_HASHMAP_PKL_PATH = None
SEGMENTS_CRIMES_HASHMAP_JSON_PATH = None
TRAIN_TRAJECTORIES_PKL_PATH = None
VALIDATION_TRAJECTORIES_PKL_PATH = None
TEST_TRAJECTORIES_PKL_PATH = None
TRAIN_CRIME_SCORES_JSON_PATH = None
TRAIN_CRIME_DIST_JSON_PATH = None
SAFETY_DATA_DIR = None
TRAIN_TRAJS_WITH_SAFETY_PKL_PATH = None
VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH = None
TEST_TRAJS_WITH_SAFETY_PKL_PATH = None
TEXTUAL_TRAIN_TRAJS_JSON_PATH = None
TEXTUAL_VALIDATION_TRAJS_JSON_PATH = None
TEXTUAL_TEST_TRAJS_JSON_PATH = None
SAFETY_TEXTUAL_TRAIN_TRAJS_JSON_PATH = None
SAFETY_TEXTUAL_VALIDATION_TRAJS_JSON_PATH = None
SAFETY_TEXTUAL_TEST_TRAJS_JSON_PATH = None
TIME_SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/walking"
# Optional: only needed if you use HERE APIs in your environment.
HERE_API_KEY = os.getenv("HERE_API_KEY", "")
TRAJ_TIME_THRESHOLD_HOURS = 24 # We want a high-time granularity


def update_config(dataset_name: str, traj_len: int, crime_radius: int, crime_time_window: int,
                  base_dir: str = '/absolute/path/to/SafetyIsAllYouNeed'):
    """
    dataset_name: "NYC" or "CHICAGO"
    """
    global DATASET, TRAJ_LENGTH, CRIME_RADIUS, CRIME_TIME_WINDOW
    DATASET = dataset_name.upper()
    TRAJ_LENGTH = traj_len
    CRIME_RADIUS = crime_radius
    CRIME_TIME_WINDOW = crime_time_window
    build_paths(base_dir)


def build_paths(base_dir: str):
    global CITY_TZ
    global CRIME_CSV, CHECKINS_CSV, CURRENT_DATA_DIR
    global SEGMENTS_COORDINATES_HASHMAP_PKL_PATH, SEGMENTS_CRIMES_HASHMAP_JSON_PATH
    global TRAIN_TRAJECTORIES_PKL_PATH, VALIDATION_TRAJECTORIES_PKL_PATH, TEST_TRAJECTORIES_PKL_PATH
    global TRAIN_CRIME_SCORES_JSON_PATH, TRAIN_CRIME_DIST_JSON_PATH
    global SAFETY_DATA_DIR, TRAIN_TRAJS_WITH_SAFETY_PKL_PATH, VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH, TEST_TRAJS_WITH_SAFETY_PKL_PATH
    global TEXTUAL_TRAIN_TRAJS_JSON_PATH, TEXTUAL_VALIDATION_TRAJS_JSON_PATH, TEXTUAL_TEST_TRAJS_JSON_PATH
    global SAFETY_TEXTUAL_TRAIN_TRAJS_JSON_PATH, SAFETY_TEXTUAL_VALIDATION_TRAJS_JSON_PATH, SAFETY_TEXTUAL_TEST_TRAJS_JSON_PATH

    if DATASET == 'NYC':
        CITY_TZ = "America/New_York"
        CRIME_CSV = os.path.join(base_dir, "data", "NYPD_CrimeData", "Preprocessed_forsquare_nyc_alligned_subset_data.csv")
        CHECKINS_CSV = os.path.join(base_dir, "data", "NYC_checkins", "raw", "dataset_w_mapped_poi.csv")
        city_dir = "NYC_checkins"
    elif DATASET == 'CHICAGO':
        CITY_TZ = "America/Chicago"
        CRIME_CSV = os.path.join(base_dir, "data", "Chicago_CrimeData", "Preprocessed_gowalla_chicago_alligned_subset_data.csv")
        CHECKINS_CSV = os.path.join(base_dir, "data", "Chicago_checkins", "raw", "dataset_w_mapped_poi.csv")
        city_dir = "Chicago_checkins"
    else:
        raise ValueError("DATASET must be 'NYC' or 'CHICAGO'")

    CURRENT_DATA_DIR = os.path.join(
        base_dir, "data", city_dir,
        f"traj_len-{TRAJ_LENGTH}",
        f"crime_radius-{CRIME_RADIUS}m",
        f"crime_time-{CRIME_TIME_WINDOW}w"
    )
    os.makedirs(CURRENT_DATA_DIR, exist_ok=True)

    SEGMENTS_COORDINATES_HASHMAP_PKL_PATH = os.path.join(CURRENT_DATA_DIR, "segments_coordinates_hashmap.pickle")
    SEGMENTS_CRIMES_HASHMAP_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "segments_crimes_count_hashmap.json")

    TRAIN_TRAJECTORIES_PKL_PATH = os.path.join(CURRENT_DATA_DIR, "train_trajectories.pickle")
    VALIDATION_TRAJECTORIES_PKL_PATH = os.path.join(CURRENT_DATA_DIR, "validation_trajectories.pickle")
    TEST_TRAJECTORIES_PKL_PATH = os.path.join(CURRENT_DATA_DIR, "test_trajectories.pickle")

    TRAIN_CRIME_SCORES_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "train_crime_scores.json")
    TRAIN_CRIME_DIST_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "train_crime_dist.json")

    SAFETY_DATA_DIR = os.path.join(CURRENT_DATA_DIR, "safety")
    os.makedirs(SAFETY_DATA_DIR, exist_ok=True)
    TRAIN_TRAJS_WITH_SAFETY_PKL_PATH = os.path.join(SAFETY_DATA_DIR, "train_trajs_with_safety.pickle")
    VALIDATION_TRAJS_WITH_SAFETY_PKL_PATH = os.path.join(SAFETY_DATA_DIR, "validation_trajs_with_safety.pickle")
    TEST_TRAJS_WITH_SAFETY_PKL_PATH = os.path.join(SAFETY_DATA_DIR, "test_trajs_with_safety.pickle")

    TEXTUAL_TRAIN_TRAJS_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "textual_train_trajs.json")
    TEXTUAL_VALIDATION_TRAJS_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "textual_validation_trajs.json")
    TEXTUAL_TEST_TRAJS_JSON_PATH = os.path.join(CURRENT_DATA_DIR, "textual_test_trajs.json")

    SAFETY_TEXTUAL_TRAIN_TRAJS_JSON_PATH = os.path.join(SAFETY_DATA_DIR, "safety_textual_train_trajs.json")
    SAFETY_TEXTUAL_VALIDATION_TRAJS_JSON_PATH = os.path.join(SAFETY_DATA_DIR, "safety_textual_validation_trajs.json")
    SAFETY_TEXTUAL_TEST_TRAJS_JSON_PATH = os.path.join(SAFETY_DATA_DIR, "safety_textual_test_trajs.json")


# Settings for the proposed method in the two supported cities.
MODEL_CONFIG = {'contract': 'coordinate_pairs_v2',
 'serializer_version': 'coordinate_pairs_v2.inline_row_coordinates.v1',
 'coordinate_source': 'row_level_numeric_trajectory',
 'coordinate_precision': 6,
 'coordinate_missing': 'error',
 'answer_format': 'poi_only',
 'safety_prompt_contract': 'observed_transitions_only',
 'configuration_summary': 'Defines the proposed safety-aware method for NYC and Chicago.',
 'models': {'llama31': {'id': 'meta-llama/Llama-3.1-8B-Instruct',
                        'revision': '0e9e39f249a16976918f6564b8830bc894c89659'}},
 'training_defaults': {'epochs': 3,
                       'learning_rate': 2e-05,
                       'warmup_steps': 20,
                       'weight_decay': 0,
                       'per_device_train_batch_size': 1,
                       'gradient_accumulation_steps': 1,
                       'save_steps': 500,
                       'max_length': 2048,
                       'lora': {'rank': 16,
                                'alpha': 32,
                                'dropout': 0.05,
                                'target_modules': ['q_proj', 'v_proj'],
                                'bias': 'none'}},
 'data_profiles': {'NYC': {'path': 'data/NYC_checkins/traj_len-20/crime_radius-500m/crime_time-4w',
                           'traj_len': 20,
                           'crime_radius_m': 500,
                           'crime_time_weeks': 4,
                           'timezone': 'America/New_York',
                           'safety_runtime': {'route_cache': {'path': 'data/NYC_checkins/segments_coordinates_hashmap.pickle',
                                                              'sha256': '018ab3a681284148488d09db7b834089028ddc1606d5baeeca3e6be743ca950a'},
                                              'normalization_stats': {'path': 'data/NYC_checkins/traj_len-20/crime_radius-500m/crime_time-4w/train_crime_dist.json',
                                                                      'sha256': '1f1102720e65d3b40c50cbf28f7608c3e3db56e8da21543850d737e706b78187'},
                                              'crime_source': {'path': 'data/NYPD_CrimeData/Preprocessed_forsquare_nyc_alligned_subset_data.csv',
                                                               'sha256': '4efde4c4bd0c075842747906de3e0d891749ea630f5a193075745f06aad0fa2f'},
                                              'projected_crs': 3857,
                                              'major_nyc_offenses_only': False,
                                              'route_origin': 'final observed POI',
                                              'poi_catalog': 'eligible training trajectories'},
                           'numeric_sha256': {'train': '0a25ca1e4b91e0cfbd90c663ae79a05493c163ee1e213ae80c319a7af8924833',
                                              'validation': 'a711a6ec01bd64cc379b7742af61cc1b54c750ca9e1bf210aa8f06769c92ca10',
                                              'test': '84b3db71b6e79b8261658c0d40144941391ab756c222a35b3a2c19cbe61ba6ce',
                                              'safety_train': '8f149f1d04e9a50670c53be7d7661cfa152283eb39690b439c9e1acae9bc351f',
                                              'safety_validation': '99304b5d89a59065947bfb5b25b64417573e221058af1bd4ea98592a6339aeb2',
                                              'safety_test': '0e4a08cf6dc9dc5751c07bdc58effd58fb496724017e1cd6ea940a0c80ba1253'},
                           'prompt_sequence_sha256': {'no_safety': {'train': '9a07637d007b0ec99c170edbbd686db76ba2ca8cada24572186865983e0fbc36',
                                                                    'validation': '4ffc75ded37a83ed52ec578f92e95c82f796e2ec9e8edfb93cd7da935552e97c',
                                                                    'test': 'd974120f5a2890bca715532dfe6a55e1cf284c60da620dc12c3ac145547e333e'},
                                                      'with_safety': {'train': '08559a4f819a6c940a7e711b1f0342f14a5f6fe5bfc59320be9ef48382c16360',
                                                                      'validation': '2aac6874a8d3450d9107a9ec35045f0389d2793797f46c0c0515e06549bb220c',
                                                                      'test': '9852827202217a71938dde3c84b6388e1e068f280d81a620e183f246fbb279da'}}},
                   'CHICAGO': {'path': 'data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w',
                               'traj_len': 20,
                               'crime_radius_m': 1000,
                               'crime_time_weeks': 3,
                               'timezone': 'America/Chicago',
                               'safety_runtime': {'route_cache': {'path': 'data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w/segments_coordinates_hashmap.pickle',
                                                                  'sha256': 'cc01ffc05928dd09d7672ae846dadacfdfd07bbb5783037c3fdafb856dacbfb4'},
                                                  'normalization_stats': {'path': 'data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w/train_crime_dist.json',
                                                                          'sha256': 'a7e5d6167bf4dc2073f51dfafb7c0ba0d1d1c2ece875146e0519000f848d311d'},
                                                  'crime_source': {'path': 'data/Chicago_CrimeData/Preprocessed_gowalla_chicago_alligned_subset_data.csv',
                                                                   'sha256': '76e0589d7f9cc559e12be0809b6f44a761ce6f871d327d2443cf7e6b77416d36'},
                                                  'projected_crs': 3857,
                                                  'major_nyc_offenses_only': False,
                                                  'route_origin': 'final observed POI',
                                                  'poi_catalog': 'eligible training trajectories'},
                               'numeric_sha256': {'train': '9414b6c623aedd11c9bdc57b9b0589e810f871925dbeef1444330aa706545676',
                                                  'validation': '40740e1b3c13602129ae4fe4fcc82cbe3f6e0761cba5bff2c2dafdf8fe5e15b3',
                                                  'test': '0d4d87e452f7c0fe3b05416ea39e91d7ec4a21bcce4bd33899d3d8355e5c6a56',
                                                  'safety_train': 'd89f7bf67ea7767d77e2280abb24f51a91cf04e5387f461e213b80052fe03c5e',
                                                  'safety_validation': 'db4492666dd9c7bc0368537fc92892308527ec6a91ab04b9d0a278893b654f30',
                                                  'safety_test': 'a479b8eb3e68cd4f967f66f61026ca8a83bafb8c4f0e541181943b8e44d3b7f8'},
                               'prompt_sequence_sha256': {'no_safety': {'train': '585f2750dc421d5747e93c03990d5738a740ae1411c7065eafe6ef219b83e027',
                                                                        'validation': '49531064bc0b37586a7b6e121f8d98d80b9ad6369375042d622ff0b1d84ebe8b',
                                                                        'test': 'ff39427309dcbcd061ea970b10e724baed026c5aacddf15e59098dff3dfb4047'},
                                                          'with_safety': {'train': '662c1aacfc7ceb9cf13c1bd948b488ca33e3d9a87c466dc3638cf2845d5f0375',
                                                                          'validation': 'fbca16e2b02fa68bb3e2aec10dd38b4c983a5f15ed94890201b18b19ccf325c9',
                                                                          'test': 'de41b56bf0ffcd94641b9d0c605ffce1f0349a9b29b981957abdf4e604cef404'}}}},
 'rows': [{'row_key': 'NYC|our_method',
           'city': 'NYC',
           'variant': 'our_method',
           'prompt_variant': 'with_safety',
           'model': 'llama31',
           'seed': 42,
           'base_precision': 'nf4',
           'checkpoint': {'step': 4000,
                          'train_stop_step': 4000,
                          'selection': 'configured_step',
                          'selection_basis': 'fixed_evaluation_configuration'},
           'inference': {'batch_size': 4,
                         'max_new_tokens': 32,
                         'beam_widths': [1, 3, 5, 10],
                         'beam_calls': 'independent',
                         'do_sample': False,
                         'parser': 'new_ids'},
           'safety_aggregation': 'median_valid'},
          {'row_key': 'CHICAGO|our_method',
           'city': 'CHICAGO',
           'variant': 'our_method',
           'prompt_variant': 'with_safety',
           'model': 'llama31',
           'seed': 42,
           'base_precision': 'nf4',
           'checkpoint': {'step': 19500,
                          'train_stop_step': 19500,
                          'selection': 'configured_step',
                          'selection_basis': 'fixed_evaluation_configuration'},
           'inference': {'batch_size': 1,
                         'max_new_tokens': 128,
                         'beam_widths': [1, 3, 5, 10],
                         'beam_calls': 'independent',
                         'do_sample': False,
                         'parser': 'new_ids'},
           'safety_aggregation': 'median_valid'}],
 'evaluation_contract': 'connected_population_v1',
 'methods': {'our_method': {'action': 'training_quality_session_join',
                            'quality': 'duration_and_maximum_step_tukey_envelope',
                            'source': 'window_gaps',
                            'threshold_method': 'decile_9'}}}

def load_model_config(path=None):
    """Return the proposed-method config or load an explicit JSON config.

    A JSON file with ``"extends": "default"`` inherits shared data and
    training settings while replacing the experiment-specific sections.
    """
    from copy import deepcopy
    if path is None:
        return deepcopy(MODEL_CONFIG)
    from text_utils import read_json
    payload = read_json(path)
    extends = payload.pop("extends", None)
    if extends is None:
        return payload
    if extends != "default":
        raise ValueError(f"Unsupported configuration base: {extends}")
    config = deepcopy(MODEL_CONFIG)
    for key, value in payload.items():
        if key == "models":
            config["models"].update(value)
        else:
            config[key] = value
    return config
