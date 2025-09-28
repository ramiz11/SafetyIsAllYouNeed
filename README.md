Leveraging Large Language Models for Safety-Aware Point-of-Interest Recommendations
POI (Point-of-Interest) next-visit prediction using LLMs fine-tuned on textualized trajectories, augmented with route safety derived from historical crime data.
The pipeline is city-agnostic (NYC / Chicago supported) and fully reproducible: preprocessing → prompt generation → LoRA fine-tuning → evaluation (Acc@k, MRR, safety at inference).

Features
Generic config: switch between NYC and CHICAGO with a single parameter; city-agnostic paths & time handling.

Robust time parsing: converts check-ins to UTC (event_time_utc) while keeping human-friendly local_time in the city timezone.

Trajectory extraction: time-based data split + fixed-length windows within a configurable time span.

Route safety: counts crimes along OSRM walking routes within a spatial buffer & temporal window; robust-scaled to [0, 1].

LLM training: LoRA fine-tuning (4-bit) on safety-aware or plain prompts; masks the question portion for loss.

Evaluation: Acc@1/3/5, MRR, and an inference-time safety summary for predicted routes.


Data Expectations
Check-ins (NYC / Chicago)
Required columns: user_id, poi_id, latitude, longitude, local_time or checkin_time (strings; can be naive or with Z/offset), category optional (NYC may use poi_category_name; Chicago lacks categories).

text_utils handles missing categories.

Crime data
NYC: complaint_date_start, complaint_date_end, LAW_CAT_CD, Latitude, Longitude.
Chicago: Date, Latitude, Longitude (single timestamp; treated as both start/end).

In preprocessing, times are localized to the city TZ then converted to UTC columns: crime_start_utc, crime_end_etc

Environment Setup
Python 3.10+ recommended.

Geo stack (recommended via conda)
conda create -n poi_env python=3.10 -y
conda activate poi_env
conda install -c conda-forge geopandas shapely pyproj rtree -y

ML & utilities (pip)
pip install pandas numpy tqdm requests
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or CPU/cu118 as needed
pip install transformers accelerate peft datasets bitsandbytes huggingface_hub
If you use gated models (e.g., Llama 3), make sure you have HF access and are logged in (huggingface-cli login).

Configuration
All configuration is centralized in configs/preprocessing_config.py.

Key parameters:

DATASET: "NYC" or "CHICAGO"

TRAJ_LENGTH, CRIME_RADIUS (meters), CRIME_TIME_WINDOW (weeks)

CITY_TZ handled automatically per dataset

Output directories are scoped by dataset & hyperparameters, e.g.:
data/NYC_checkins/traj_len-15/crime_radius-250m/crime_time-4w/

Set configuration by calling:
from configs import preprocessing_config as pc
pc.update_config(dataset_name="NYC", traj_len=15, crime_radius=250, crime_time_window=4,
                 base_dir="/absolute/path/to/SafetyIsAllYouNeed")

1. Preprocessing Pipeline
Generates trajectories, caches OSRM routes & crime counts, and builds textual prompts (textual_*.json and safety_textual_*.json).

Outputs (under the scoped CURRENT_DATA_DIR):
train_trajectories.pickle, validation_trajectories.pickle, test_trajectories.pickle
segments_coordinates_hashmap.pickle, segments_crimes_count_hashmap.json
train_crime_scores.json, train_crime_dist.json (robust-scaling stats)
textual_{train,val,test}_trajs.json
In safety/: *_trajs_with_safety.pickle, safety_textual_{train,val,test}_trajs.json

Uses public OSRM; caching minimizes repeated calls. If OSRM rate limits or latency are an issue, point OSRM_BASE_URL in the config to your own server.

## EXAMPLE RUN:
from run_processing import main

main(
    dataset="NYC",
    traj_len=20,
    crime_radius=1000, # meters
    crime_time_weeks=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
)


2. Training (LoRA, 4-bit)
Fine-tunes a base model (default: Llama-3.1-8B-Instruct) on either plain or safety-augmented prompts.
Saves the best checkpoint and tokenizer to:
models/{DATASET}/traj_len-XX/crime_radius-YYm/crime_time-ZZw/{model}_{w|wo}_safety/best/

## EXAMPLE RUN:
from train import run_train
run_train(
    dataset="NYC",
    traj_len=10, crime_radius=1000, crime_time_weeks=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
    use_safety=True, # True = safety_textual_*.json
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_train_epochs=3, lr=2e-5,
    per_device_train_batch_size=1, gradient_accumulation_steps=1,
    max_length=2048,                  # context length for prompts
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=("q_proj","v_proj"),
)

3) Evaluation
Computes Acc@k (greedy/beam), MRR, and an inference-time route safety summary for predicted POIs.
print(metrics)  # {'acc@1': ..., 'acc@3': ..., 'acc@5': ..., 'mrr': ...}
It also prints a pandas.Series.describe() of the per-route safety scores (normalized to [0,1] using train distribution stats).


**Batch Runs** 
Because everything is callable, you can easily grid-search:

## RUN EXAMPLE:
from run_processing import main
from train import run_train
from eval import run_eval
DATASET = "NYC" 
BASE = "/absolute/path/to/SafetyIsAllYouNeed"
for traj_len in [10, 15, 20, 25]:
    for radius in [250, 500, 750, 1000]:
        for tw in [1, 2, 3, 4]:
            main(DATASET, traj_len, radius, tw, base_dir=BASE)
            run_train(dataset=DATASET, traj_len=traj_len, crime_radius=radius, crime_time_weeks=tw,
                      base_dir=BASE, use_safety=True)
            metrics = run_eval(dataset=DATASET, traj_len=traj_len, crime_radius=radius, crime_time_weeks=tw,base_dir=BASE, use_safety=True)
            print(traj_len, radius, tw, metrics)


Notes & Design Choices

For convenience, we provide ready-to-use training, validation, and test trajectories with safety
scores pre-injected, limited to the optimal hyperparameter settings for each dataset. This resource
lowers barriers for the research community, making it easier to replicate our results and further
explore the role of safety in urban mobility modeling

Time zones: mixed timestamp formats are parsed; naive times are localized to CITY_TZ then converted to UTC for comparisons (event_time_utc).

CRS & buffering: route buffering occurs in a meter-based CRS; default EPSG selected in config. You can switch to a city-specific projected CRS if desired.

Safety normalization: robust scaling ((x - median) / IQR), clamped to [0,1]; safety = 1 - scaled.

Prompt masking: training ignores tokens before <answer>: so loss is computed only on the answer span.

Category text: if a dataset lacks POI categories, prompts omit the category phrase automatically.

Troubleshooting
tz-naive vs tz-aware errors: ensure you use the provided loaders; all internal comparisons use UTC columns.

Geo stack install: install geopandas + shapely via conda-forge to avoid binary issues.

OSRM errors: transient HTTP errors may occur; re-run, or host your own OSRM and change OSRM_BASE_URL.

CUDA OOM: reduce max_length, enable 4-bit (default), or increase gradient_accumulation_steps.

Gated models: make sure your HF account has access and you’re logged in.

Acknowledgments
Routing by OSRM (Open Source Routing Machine).
Crime datasets from NYPD and City of Chicago (licenses/terms apply).
Models and tooling: Hugging Face Transformers, PEFT, bitsandbytes, PyTorch.


Citation
If this work is used in academic publications, please cite appropriately.