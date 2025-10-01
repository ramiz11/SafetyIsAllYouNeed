# 🗺️ Safety-Aware Point-of-Interest Recommendations with LLMs

> Leveraging Large Language Models for next-visit prediction using safety-augmented trajectories derived from historical crime data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This project implements **POI (Point-of-Interest) next-visit prediction** using LLMs fine-tuned on textualized trajectories, augmented with **route safety scores** derived from historical crime data.

The pipeline is:
- ✅ **City-agnostic** (NYC / Chicago supported)
- ✅ **Fully reproducible** end-to-end
- ✅ **Research-ready** with pre-computed optimal configurations

**Pipeline:** `Preprocessing → Prompt Generation → LoRA Fine-tuning → Evaluation`

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
- `LAW_CAT_CD`
- `Latitude`, `Longitude`

**Chicago:**
- `Date` (single timestamp; treated as both start/end)
- `Latitude`, `Longitude`

> All timestamps are localized to city timezone, then converted to UTC (`crime_start_utc`, `crime_end_utc`).

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Step 1: Create Environment

```bash
conda create -n poi_env python=3.10 -y
conda activate poi_env
```

### Step 2: Install Geo Stack

```bash
conda install -c conda-forge geopandas shapely pyproj rtree -y
```

### Step 3: Install ML Dependencies

```bash
pip install pandas numpy tqdm requests
pip install torch --index-url https://download.pytorch.org/whl/cu121  # Adjust for your CUDA version
pip install transformers accelerate peft datasets bitsandbytes huggingface_hub
```

### Step 4: Authenticate with Hugging Face (for gated models)

```bash
huggingface-cli login
```

---

## ⚙️ Configuration

All settings are centralized in `configs/preprocessing_config.py`.

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `DATASET` | City selection | `"NYC"` or `"CHICAGO"` |
| `TRAJ_LENGTH` | Trajectory window size | `15` |
| `CRIME_RADIUS` | Spatial buffer (meters) | `250` |
| `CRIME_TIME_WINDOW` | Temporal window (weeks) | `4` |
| `CITY_TZ` | Auto-handled per dataset | — |

### Setting Configuration

```python
from configs import preprocessing_config as pc

pc.update_config(
    dataset_name="NYC",
    traj_len=15,
    crime_radius=250,
    crime_time_window=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed"
)
```

**Output directory structure:**
```
data/NYC_checkins/traj_len-15/crime_radius-250m/crime_time-4w/
```

---

## 🔧 Usage

### 1. Preprocessing Pipeline

Generates trajectories, caches OSRM routes, computes crime counts, and builds textual prompts.

**Outputs:**
- `train_trajectories.pickle`, `validation_trajectories.pickle`, `test_trajectories.pickle`
- `segments_coordinates_hashmap.pickle`
- `segments_crimes_count_hashmap.json`
- `textual_{train,val,test}_trajs.json`
- `safety/safety_textual_{train,val,test}_trajs.json`

```python
from run_processing import main

main(
    dataset="NYC",
    traj_len=20,
    crime_radius=1000,  # meters
    crime_time_weeks=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
)
```

> **Note:** Uses public OSRM; caching minimizes API calls. For high-volume use, point `OSRM_BASE_URL` to your own server.

---

### 2. Model Training

Fine-tunes a base LLM (default: Llama-3.1-8B-Instruct) using LoRA with 4-bit quantization.

**Model checkpoints saved to:**
```
models/{DATASET}/traj_len-XX/crime_radius-YYm/crime_time-ZZw/{model}_{w|wo}_safety/best/
```

```python
from train import run_train

run_train(
    dataset="NYC",
    traj_len=10,
    crime_radius=1000,
    crime_time_weeks=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
    use_safety=True,  # True = safety_textual_*.json
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_train_epochs=3,
    lr=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=2048,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=("q_proj", "v_proj"),
)
```

---

### 3. Evaluation

Computes accuracy metrics (Acc@1/3/5, MRR) and inference-time route safety analysis.

```python
from eval import run_eval

metrics = run_eval(
    dataset="NYC",
    traj_len=10,
    crime_radius=1000,
    crime_time_weeks=4,
    base_dir="/absolute/path/to/SafetyIsAllYouNeed",
    use_safety=True,
)

print(metrics)  # {'acc@1': ..., 'acc@3': ..., 'acc@5': ..., 'mrr': ...}
```

Also outputs `pandas.Series.describe()` of per-route safety scores (normalized [0,1]).

---

### Batch Experiments

Grid search over hyperparameters:

```python
from run_processing import main
from train import run_train
from eval import run_eval

DATASET = "NYC"
BASE = "/absolute/path/to/SafetyIsAllYouNeed"

for traj_len in [10, 15, 20, 25]:
    for radius in [250, 500, 750, 1000]:
        for tw in [1, 2, 3, 4]:
            # Preprocessing
            main(DATASET, traj_len, radius, tw, base_dir=BASE)
            
            # Training
            run_train(
                dataset=DATASET,
                traj_len=traj_len,
                crime_radius=radius,
                crime_time_weeks=tw,
                base_dir=BASE,
                use_safety=True
            )
            
            # Evaluation
            metrics = run_eval(
                dataset=DATASET,
                traj_len=traj_len,
                crime_radius=radius,
                crime_time_weeks=tw,
                base_dir=BASE,
                use_safety=True
            )
            
            print(f"Config: {traj_len}, {radius}m, {tw}w → {metrics}")
```

---

## 🧠 Design Choices

### Pre-computed Trajectories
We provide **ready-to-use training, validation, and test trajectories** with safety scores pre-injected for optimal hyperparameter settings. This accelerates research and ensures reproducibility.

### Time Handling
- Mixed timestamp formats are parsed automatically
- Naive times → localized to `CITY_TZ` → converted to UTC (`event_time_utc`)
- All comparisons use UTC internally

### Spatial Processing
- Route buffering in meter-based CRS (default EPSG per config)
- Customizable to city-specific projected CRS

### Safety Normalization
- **Robust scaling:** `(x - median) / IQR`, clamped to [0,1]
- **Safety score:** `1 - scaled_crime_count`

### Prompt Masking
Training loss computed **only on answer span** (after `<answer>:` token).

### Category Handling
Datasets lacking POI categories automatically omit category phrases in prompts.

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

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{yourname2025safety,
  title={Leveraging Large Language Models for Safety-Aware Point-of-Interest Recommendations},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🙏 Acknowledgments

- **Routing:** [OSRM](http://project-osrm.org/) (Open Source Routing Machine)
- **Crime Data:** (subject to respective licenses)
- **ML Stack:** [Hugging Face Transformers], [PEFT], [bitsandbytes], [PyTorch]
---

<div align="center">
  

</div>
