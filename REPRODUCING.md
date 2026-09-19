# Reproduce the six coordinate-bearing LLM configurations

This is the supported path for NYC and Chicago, each with Our Method,
LLM4POI-3.1 and Original LLM4POI. It uses the public data and settings in
`configs/coordinate_pairs_v2.json`.

**Train your own models.** No trained adapters, checkpoints, private downloads,
Colab session, or archived experiment files are required or supplied. Training
downloads the two configured public base models and learns your own LoRA
adapters. An adapter is a training output, not a repository asset.

The reference values approximately match the published table within the agreed
per-row five-metric RMSE of .03 and maximum absolute difference of .06. They are
measurements from previously generated predictions, not a guarantee for a fresh
training run. The population/ranking procedures were selected with the published
numbers as a guide; they are not proven to be the deleted historical code.

## 1. Install and obtain data

Use Linux, Python 3.11 or 3.12, a CUDA-capable GPU, Git LFS, and sufficient storage
for base models and training checkpoints. The full six-model training workflow
is computationally expensive; CPU-only tests do not train a model.

```bash
git clone https://github.com/ramiz11/SafetyIsAllYouNeed.git
cd SafetyIsAllYouNeed
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -r requirements/reproduction.txt
hf auth login
```

Accept the applicable base-model licenses on Hugging Face. Base model IDs and
immutable revisions are recorded in the configuration:

- `meta-llama/Llama-3.1-8B-Instruct`
- `Yukang/Llama-2-7b-longlora-32k-ft`

These are base-model downloads, not our trained adapters. Keep credentials out
of the repository. Do not commit your environment or generated model files.

## 2. Verify or regenerate prompts

```bash
python scripts/reproduce_coordinate_pairs_v2.py check
python -m unittest discover -s tests -v
```

`check` verifies the numeric inputs, all twelve prompt files, their manifests,
and the fixed Safety inputs. It needs no GPU. To regenerate the textual files
from the supplied numeric trajectories:

```bash
python scripts/reproduce_coordinate_pairs_v2.py prompts
```

The canonical data directories are:

- `data/NYC_checkins/traj_len-20/crime_radius-500m/crime_time-4w/`
- `data/Chicago_checkins/traj_len-20/crime_radius-1000m/crime_time-3w/`

Each input contains the first 19 check-ins of a 20-check-in trajectory, with
each observed row's timestamp, POI, optional category and own latitude/longitude
pair. Repeated POI identifiers retain their row-specific coordinates. The
20th POI is the supervised answer, not part of the prediction input; its
timestamp, coordinates and category are omitted. Our Method additionally
includes the 18 observed-to-observed transition Safety scores. Both LLM4POI
variants omit Safety from their prompts.

Numeric trajectories, split membership and ordering are unchanged. This
workflow uses those supplied derived inputs, rather than querying live routing
services and rebuilding a potentially different crime/route dataset.

## 3. Train, infer and evaluate

Run all six configurations sequentially:

```bash
python scripts/reproduce_coordinate_pairs_v2.py run \
  --row all --output-dir runs/paper
```

Use a new, empty output directory. This command trains each configuration,
loads its configured checkpoint, runs inference and evaluation, and writes
the measured results to `runs/paper/comparison.json`.

For one configuration:

```bash
python scripts/reproduce_coordinate_pairs_v2.py run \
  --row 'NYC|our_method' --output-dir runs/nyc_ours
```

| Row key | Seed | Selected step | Training stop step |
| --- | ---: | ---: | ---: |
| NYC\|our_method | 42 | 4000 | 4000 |
| NYC\|llm4poi_31 | 7 | 5500 | 6000 |
| NYC\|llm4poi_original | 42 | 8000 | 8000 |
| CHICAGO\|our_method | 42 | 19500 | 19500 |
| CHICAGO\|llm4poi_31 | 42 | 3000 | 3000 |
| CHICAGO\|llm4poi_original | 42 | 12500 | 19944 |

The JSON configuration specifies the training and inference settings.
Training uses the training split, with loss evaluation on the validation split.

To run stages separately, or resume interrupted training:

```bash
python scripts/reproduce_coordinate_pairs_v2.py train \
  --row 'NYC|our_method' --output-dir runs/nyc_training
# Resume the same recipe, if necessary:
python scripts/reproduce_coordinate_pairs_v2.py train \
  --row 'NYC|our_method' --output-dir runs/nyc_training \
  --resume-from-checkpoint runs/nyc_training/checkpoints/checkpoint-3500
python scripts/reproduce_coordinate_pairs_v2.py infer \
  --row 'NYC|our_method' \
  --adapter runs/nyc_training/checkpoints/checkpoint-4000 \
  --output-dir runs/nyc_evaluation
```

Inference checks the local training manifest and saves `predictions.json.gz`,
`safety.json`, `weights.json`, and the measured `result.json`.

To recompute evaluation from your own existing inference outputs:

```bash
python scripts/reproduce_coordinate_pairs_v2.py evaluate \
  --row 'NYC|our_method' \
  --predictions runs/nyc_evaluation/predictions.json.gz \
  --safety runs/nyc_evaluation/safety.json \
  --output-dir runs/nyc_reevaluation
```

An out-of-tolerance run is still written as measured, with `accepted: false`.
This flag is only a numerical comparison with the declared reference tolerances.

## Reference results and verification limits

See [reference_metrics.json](results/coordinate_pairs_v2/reference_metrics.json)
for all six reference results, published targets and differences. This file is
not used by training or inference.

CPU tests and saved-output regression checks have passed. A complete fresh
six-model GPU training run through this public pipeline has not yet been
verified. Fresh training results may vary with hardware, libraries and randomness.
