import os

TRAJ_LENGTH = None
CRIME_RADIUS = None
CRIME_TIME_WINDOW = None
DATASET = None # "NYC" or "CHICAGO", configure during runtime.
CITY_TZ = None  # "America/New_York" | "America/Chicago"
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
HERE_API_KEY = "q1R28qK2r82A3R4pQ9qZ1RcnJ_BnH-siOjOIrzYvYZU"
TRAJ_TIME_THRESHOLD_HOURS = 24 # We want a high-time granularity


def update_config(dataset_name: str, traj_len: int, crime_radius: int, crime_time_window: int,
                  base_dir: str = '/Users/ramizaboura/MSC/SafetyIsAllYouNeed'):
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
