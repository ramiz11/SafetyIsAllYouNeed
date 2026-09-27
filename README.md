# 🗺️ Safety-Aware Point-of-Interest Recommendations with LLMs

> Leveraging Large Language Models for next-visit prediction using safety-augmented trajectories derived from historical crime data

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This project implements **POI (Point-of-Interest) next-visit prediction** using LLMs fine-tuned on textualized trajectories, augmented with **route safety scores** derived from historical crime data.

The pipeline is:
- ✅ **City-agnostic** (NYC / Chicago supported)
- ✅ **Configured end-to-end workflow** for training and evaluation
- ✅ **Research-ready** with the paper's experiment configurations

**Pipeline:** `Preprocessing → Prompt Generation → LoRA Fine-tuning → Evaluation`

This repository contains the implementation of the proposed method's evaluation
pipeline for NYC and Chicago. Trained weights are not included.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌆 **Generic Config** | Switch between NYC and Chicago with a single parameter |
| ⏰ **Robust Time Parsing** | Converts check-ins to UTC while preserving local timezone |
| 🛣️ **Trajectory Extraction** | Time-based splits with configurable windows |
| 🚨 **Route Safety** | Crime counting along OSRM routes with spatial/temporal buffers |
| 🤖 **LLM Training** | LoRA fine-tuning (4-bit quantization) on safety-aware prompts |
| 📊 **Comprehensive Evaluation** | Acc@1/3/5, MRR, and inference-time safety analysis |

---

## 📦 Data Requirements

### Check-ins (NYC / Chicago)

**Required columns:**
- `user_id`
- `poi_id`
- `latitude`, `longitude`
- `local_time` or `checkin_time` (strings; naive or with timezone)
- `category` (optional; NYC uses `poi_category_name`, Chicago may lack categories)

> **Note:** `text_utils` handles missing categories automatically.

### Crime Data

**NYC:**
- `complaint_date_start`, `complaint_date_end`
- `Latitude`, `Longitude`

**Chicago:**
- `Date` (single timestamp; treated as both start/end)
- `Latitude`, `Longitude`

> All timestamps are localized to city timezone, then converted to UTC (`crime_start_utc`, `crime_end_utc`).

## Data Attribution

This repository does not claim ownership of the original check-in or crime datasets. The NYC mobility data is based on the publicly released Foursquare TSMC2014 NYC dataset (see `data/NYC_checkins/raw/dataset_TSMC2014_readme.txt`), and the Chicago mobility data is based on the public Gowalla check-in dataset filtered to the Chicago area (source link documented in `data/Chicago_checkins/README.txt`: https://snap.stanford.edu/data/loc-gowalla.html). The Chicago crime source link is documented in `data/Chicago_checkins/README.txt`, and the aligned crime preprocessing notes are documented in `data/Chicago_CrimeData/README.txt`; the NYC aligned crime preprocessing notes are documented in `data/NYPD_CrimeData/README.txt`. The contribution of this repository is the integration pipeline that filters, aligns, joins, and annotates these public datasets with route-level crime counts and normalized safety scores. Use of upstream datasets remains subject to their original licenses, terms of use, and citation requirements.

---

## 🚀 Installation

### Prerequisites
- Linux, Python 3.11 or 3.12, and a CUDA-capable GPU for training and inference
- Git LFS and storage for the data, base models, and training checkpoints

### Step 1: Create Environment

```bash
conda create -n poi_env python=3.11 -y
conda activate poi_env
```

### Step 2: Download Data

```bash
git lfs install
git lfs pull
```

### Step 3: Install Dependencies

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -r requirements/training.txt
```

### Step 4: Authenticate with Hugging Face (for gated models)

```bash
hf auth login
```

---

## ⚙️ Configuration

Preprocessing settings and the two proposed-method city configurations
(`MODEL_CONFIG`) live in `configs/preprocessing_config.py`. Accept the applicable
Hugging Face license for the configured Llama-3.1 model before training.

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `DATASET` | City selection | `"CHICAGO"` (default) or `"NYC"` |
| `TRAJ_LENGTH` | Trajectory window size | `20` |
| `CRIME_RADIUS` | Spatial buffer (meters) | `1000` |
| `CRIME_TIME_WINDOW` | Temporal window (weeks) | `3` |
| `CITY_TZ` | Auto-handled per dataset | — |

### Setting Configuration

```python
from configs import preprocessing_config as pc

pc.update_config(
    dataset_name="CHICAGO",
    traj_len=20,
    crime_radius=1000,
    crime_time_window=3,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed"
)
```

**Output directory structure:**
```
data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w/
```

Artifacts written under `data/{CITY}_checkins/...` are derived joint outputs: they start from check-in data and are augmented during preprocessing with crime-based route counts, normalized safety scores, and textual safety-aware trajectory prompts.

---

## 🔧 Usage

### 1. Preprocessing Pipeline

Generates trajectories, caches OSRM routes, computes crime counts, and builds textual prompts.

**Outputs:**
- `train_trajectories.pickle`, `validation_trajectories.pickle`, `test_trajectories.pickle`
- `segments_coordinates_hashmap.pickle`
- `segments_crimes_count_hashmap.json`
- `textual_{train,validation,test}_trajs.json`
- `safety/safety_textual_{train,validation,test}_trajs.json`

The default prompt format includes observed latitude/longitude pairs. Regenerating
prompts updates these canonical textual files.

For the supplied NYC and Chicago experiment datasets, verify the inputs or regenerate
only their prompts without rebuilding routes or crime counts:

```bash
python batch_runner.py check
python batch_runner.py prompts
```

Each input contains 19 observed check-ins with their own coordinates and 18
observed transition Safety scores; the 20th POI is the training answer.

These generated files should be interpreted as joint mobility-safety artifacts rather than raw check-in exports: they combine user trajectories with crime-derived route statistics and safety annotations computed during preprocessing.

```python
from run_preprocessing import main

main(
    dataset="CHICAGO",
    traj_len=20,
    crime_radius=1000,  # meters
    crime_time_weeks=3,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
)
```

> **Note:** Uses public OSRM; caching minimizes API calls. For high-volume use, point `OSRM_BASE_URL` to your own server.

---

### 2. Model Training

Fine-tunes the model to predict the next POI from the observed trajectory, using
the configured base model and LoRA settings. For example:

```bash
python train.py --row 'NYC|our_method' --output-dir runs/nyc_training
```

Checkpoints are saved under `runs/nyc_training/checkpoints/`. To resume the same
run, add `--resume-from-checkpoint /path/to/checkpoint-N`.

The available row keys are `NYC|our_method` and `CHICAGO|our_method`. Seeds, training
steps, selected checkpoints, precision, and inference settings are defined in
`MODEL_CONFIG`.

---

### 3. Evaluation

Computes accuracy metrics (Acc@1/3/5, MRR) and inference-time route safety analysis.

```bash
python eval.py --row 'NYC|our_method' \
  --adapter runs/nyc_training/checkpoints/checkpoint-4000 \
  --output-dir runs/nyc_evaluation
```

Use a new output directory. The command checks the matching training manifest
and saves `predictions.json.gz`, `safety.json`, `weights.json`, and `result.json`.
To score these same outputs again without running the model:

```bash
python batch_runner.py evaluate --row 'NYC|our_method' \
  --predictions runs/nyc_evaluation/predictions.json.gz \
  --safety runs/nyc_evaluation/safety.json \
  --output-dir runs/nyc_reevaluation
```

---

### Batch Experiments

Train and evaluate both configured cities sequentially:

```bash
python batch_runner.py run --row all --output-dir runs/our_method_evaluation
```

Use a new, empty output directory. Results are written to
`runs/our_method_evaluation/evaluation_results.json`. Replace `all` with
`NYC|our_method` or `CHICAGO|our_method` to run one city.
The generic `run_train` and `run_eval` Python APIs remain available for custom
experiments; the commands above select the configured workflow.

CPU tests can be run with `python -m unittest discover -s tests -v`. CPU tests and
saved-output checks do not verify fresh GPU training runs; those require the
configured models, GPU environment, and training time.

---

## 🧠 Design Choices

### Pre-computed Trajectories
We provide **ready-to-use training, validation, and test trajectories** with safety scores pre-injected for the configured preprocessing settings. This accelerates research and supports reproducible execution of the evaluation pipeline.

### Time Handling
- Mixed timestamp formats are parsed automatically
- Naive times → localized to `CITY_TZ` → converted to UTC (`event_time_utc`)
- All comparisons use UTC internally

### Spatial Processing
- Route buffering in meter-based CRS (default EPSG per config)
- Customizable to city-specific projected CRS

### Safety Normalization
- **Robust scaling + clamping:** For each route, crime counts are normalized as
  `z = (x - median_train) / IQR_train`, then hard-clipped to the [0,1] range.
- **Safety score:** `safety = 1 - scaled_crime_count`, so higher values indicate safer routes.


### Next-POI Training
Fine-tunes the model to predict the next POI from the observed trajectory.

Test prompts include route-safety scores only for transitions between observed check-ins. The transition to the POI being predicted is not included in the prompt.

### Category Handling
Datasets lacking POI categories automatically omit category phrases in prompts.

**Note (Chicago semantics):** the Gowalla Chicago check-ins do not include POI categories. This repo therefore trains/evaluates Chicago prompts without category text. If you want to enrich Chicago POIs with OpenStreetMap-derived categories (as discussed in the paper’s qualitative analysis), that enrichment step is not implemented in the preprocessing pipeline here.

---
## 📊 Evaluation Results Reported in the Paper

### Performances with the NYC Dataset

| **Model** | **Acc@1** | **Acc@3** | **Acc@5** | **MRR** | **Safety Score** |
|------------|------------|------------|------------|------------|------------------|
| LSTM | 0.0573 | 0.1050 | 0.1724 | 0.0943 | 0.5412 |
| GRU | 0.0632 | 0.1177 | 0.1835 | 0.1026 | 0.5534 |
| STAN | 0.0752 | 0.1359 | 0.2931 | 0.1424 | 0.5697 |
| STHGCN | 0.0777 | 0.2924 | 0.3717 | 0.2175 | 0.5329 |
| GETNext | 0.0918 | 0.2273 | 0.2631 | 0.1575 | 0.5783 |
| LLM4POI | 0.1439 | 0.2311 | 0.2915 | 0.1862 | 0.6126 |
| LLM4POI-3.1 | 0.1567 | 0.2414 | 0.2777 | 0.2185 | 0.6129 |
| **Our Method** | **0.2613** | **0.3449** | **0.3819** | **0.2735** | **0.9274** |

---

### Performances with the Chicago Dataset

| **Model** | **Acc@1** | **Acc@3** | **Acc@5** | **MRR** | **Safety Score** |
|------------|------------|------------|------------|------------|------------------|
| LSTM | 0.0469 | 0.0874 | 0.1533 | 0.0826 | 0.5377 |
| GRU | 0.0542 | 0.0993 | 0.1648 | 0.0905 | 0.5419 |
| STAN | 0.0845 | 0.1087 | 0.2349 | 0.1200 | 0.5518 |
| STHGCN | 0.0666 | 0.2339 | 0.2979 | 0.1833 | 0.5712 |
| GETNext | 0.0787 | 0.1818 | 0.2109 | 0.1597 | 0.5535 |
| LLM4POI | 0.1234 | 0.1848 | 0.2336 | 0.1569 | 0.6605 |
| LLM4POI-3.1 | 0.1344 | 0.1931 | 0.2225 | 0.1842 | 0.6608 |
| **Our Method** | **0.2256** | **0.2790** | **0.3140** | **0.2520** | **1.0000** |

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Timezone errors** | Use provided loaders; all comparisons use UTC columns |
| **Geo stack install failures** | Install via `conda-forge` to avoid binary mismatches |
| **OSRM errors** | Transient HTTP issues; retry or host your own OSRM server |
| **CUDA OOM** | Reduce `max_length`, enable 4-bit (default), or increase `gradient_accumulation_steps` |
| **Gated model access** | Ensure HF account has permissions; run `huggingface-cli login` |


---

## 🙏 Acknowledgments

- **Routing:** [OSRM](http://project-osrm.org/) (Open Source Routing Machine)
- **Crime Data:** (subject to respective licenses)
- **ML Stack:** [Hugging Face Transformers], [PEFT], [bitsandbytes], [PyTorch]
---

<div align="center">
  

</div>
