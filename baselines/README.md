
# Baselines

This folder contains baseline implementations and helpers used in the paper. The goal is to make it easy to reproduce baseline results using the **same trajectories and safety-score computation** as the main pipeline.

## Coverage (what the paper mentions)

- **LSTM / GRU** (local, implemented here): `baselines/lstm_gru/run_lstm_gru.py`
- **LLM4POI** (local wrapper, implemented here): `baselines/llm4poi/run_llm4poi_baseline.py`
  - Use `--model-name Llama-2-7b-longlora-32k` for the **pre-3.1** baseline
  - Use `--model-name meta-llama/Llama-3.1-8B-Instruct` for the **LLM4POI-3.1** baseline
- **STAN / STHGCN / GETNext** (external repos): this repo exports inputs and provides copy helpers; the actual training code lives upstream.

## Why `lstm_gru/` is one baseline

LSTM and GRU share the same dataset construction, training loop, and evaluation. The code is packaged together to avoid duplication and to ensure both baselines differ only in the recurrent cell (`--model lstm` vs `--model gru`).

## Quick start (data export)

All baselines consume different input formats. Use the export script to convert the preprocessed trajectories into baseline‑specific files:

```bash
python scripts/export_baseline_inputs.py \
  --dataset NYC \
  --traj-len 20 \
  --crime-radius 500 \
  --crime-time-weeks 4 \
  --base-dir /absolute/path/to/SafetyIsAllYouNeed \
  --baseline all
```

Outputs are written under:
```
baselines/exports/
  stan/nyc_len20/{train,val,test}.txt
  getnext/NYC_{train,val,test}.csv
  sthgcn/NYC_{train,val,test}.tsv
  category_map.json
```

After exporting, you can use the helper scripts to copy files into each baseline repo:
```bash
scripts/run_stan.sh /absolute/path/to/SafetyIsAllYouNeed /absolute/path/to/Spatial-Temporal-Attention-Network-for-POI-Recommendation NYC 20
scripts/run_getnext.sh /absolute/path/to/SafetyIsAllYouNeed /absolute/path/to/GETNext NYC
scripts/run_sthgcn.sh /absolute/path/to/SafetyIsAllYouNeed /absolute/path/to/Spatio-Temporal-Hypergraph-Model NYC
```

## Notebooks (legacy / reference)

The notebooks in `baselines/notebooks/` were originally run in Colab and include drive mounts. They are kept as **reference**, but the recommended flow is:
1) run preprocessing in the main repo
2) run `scripts/export_baseline_inputs.py`
3) use `scripts/run_*.sh` to copy the exported files into the external baseline repos
4) follow the upstream baseline repos’ training instructions

## STAN
1. Clone the STAN repo: `https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation`
2. Copy the exported `train/val/test` TXT files to the repo’s data folder.
3. Use the notebook in `baselines/notebooks/STAN.ipynb` as a reference for config and training.

## STHGCN
1. Clone the STHGCN repo: `https://github.com/alipay/Spatio-Temporal-Hypergraph-Model`
2. Copy the exported TSV files into the expected `data/NYC/` (or `data/Chicago/`) directory.
3. Use the notebook in `baselines/notebooks/STHGCN.ipynb` as a reference for the expected format and training command.

## GETNext
1. Clone the GETNext repo: `https://github.com/songyangme/GETNext`
2. Copy the exported CSVs into `dataset/NYC/` or `dataset/Chicago/`.
3. Use `build_graph.py` then `train.py` (see `baselines/notebooks/GetNext.ipynb`).

## LLM4POI (pre‑3.1 and 3.1 baselines)

See `baselines/llm4poi/README.md` for training and evaluation.
This wrapper implements the **limited-history** variant (same-user history only; no key-query similarity).

## LSTM / GRU

```bash
python baselines/lstm_gru/run_lstm_gru.py \
  --model lstm \
  --mode both \
  --dataset NYC \
  --traj-len 20 \
  --crime-radius 500 \
  --crime-time-weeks 4 \
  --base-dir /absolute/path/to/SafetyIsAllYouNeed
```

Switch `--model gru` for the GRU baseline.
