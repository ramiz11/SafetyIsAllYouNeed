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
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org/route/v1/walking")
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
 'data_profiles': {
     'NYC': {
         'path': 'data/NYC_checkins/traj_len-20/crime_radius-500m/crime_time-4w',
         'traj_len': 20, 'crime_radius_m': 500, 'crime_time_weeks': 4,
         'timezone': 'America/New_York',
         'safety_runtime': {
             'raw_checkins': {'path': 'data/NYC_checkins/raw/dataset_w_mapped_poi.csv'},
             'crime_source': {'path': 'data/NYPD_CrimeData/Preprocessed_forsquare_nyc_alligned_subset_data.csv'},
             'route_cache': {'path': 'data/NYC_checkins/traj_len-20/crime_radius-500m/crime_time-4w/candidate_routes.json'},
             'projected_crs': 32618, 'major_nyc_offenses_only': False,
             'route_origin': 'final observed POI',
             'poi_catalog': 'raw and windowed training trajectories',
         },
     },
     'CHICAGO': {
         'path': 'data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w',
         'traj_len': 20, 'crime_radius_m': 1000, 'crime_time_weeks': 3,
         'timezone': 'America/Chicago',
         'safety_runtime': {
             'raw_checkins': {'path': 'data/Chicago_checkins/raw/dataset_w_mapped_poi.csv'},
             'crime_source': {'path': 'data/Chicago_CrimeData/Preprocessed_gowalla_chicago_alligned_subset_data.csv'},
             'route_cache': {'path': 'data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w/candidate_routes.json'},
             'projected_crs': 32616, 'major_nyc_offenses_only': False,
             'route_origin': 'final observed POI',
             'poi_catalog': 'raw and windowed training trajectories',
         },
     },
 },
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
                         'parser': 'new_ids'}},
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
                         'parser': 'new_ids'}}]}

def load_model_config(path=None, *, repo_root=None):
    """Return the proposed-method config or load an explicit JSON config.

    A JSON file with ``"extends": "default"`` inherits shared data and
    training settings while replacing the experiment-specific sections.
    """
    from copy import deepcopy
    if path is None:
        config = deepcopy(MODEL_CONFIG)
        return _with_prompt_hashes(config, repo_root)
    from text_utils import read_json
    payload = read_json(path)
    extends = payload.pop("extends", None)
    if extends is None:
        return _with_prompt_hashes(payload, repo_root)
    if extends != "default":
        raise ValueError(f"Unsupported configuration base: {extends}")
    config = deepcopy(MODEL_CONFIG)
    for key, value in payload.items():
        if key == "models":
            config["models"].update(value)
        else:
            config[key] = value
    return _with_prompt_hashes(config, repo_root)


def _with_prompt_hashes(config, repo_root):
    from pathlib import Path
    from text_utils import load_prompt_manifest

    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[1]
    for profile in config['data_profiles'].values():
        manifest_path = root / profile['path'] / 'coordinate_pairs_v2_manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = load_prompt_manifest(manifest_path.parent)
        profile['prompt_sequence_sha256'] = {
            variant: {split: details['sequence_sha256']
                      for split, details in variant_data['splits'].items()}
            for variant, variant_data in manifest['variants'].items()
        }
    return config
