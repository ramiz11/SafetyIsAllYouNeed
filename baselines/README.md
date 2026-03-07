
# Baselines

This folder contains baseline implementations and helpers used in the paper. The goal is to make it easy to reproduce baseline results using the **same trajectories and safety‑score computation** as the main pipeline.

**Included baselines**
- LSTM / GRU (classical sequential baselines)
- STAN (Spatio‑Temporal Attention Network)
- STHGCN (Spatio‑Temporal Hypergraph GCN)
- GETNext (Transformer‑based next‑POI)
- LLM4POI (pre‑Llama‑3.1 baseline; see `baselines/llm4poi`)

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

## Notebooks

The notebooks in `baselines/notebooks/` were originally run in Colab and include drive mounts. You can either adapt the paths to your local machine **or** use the exported files from `scripts/export_baseline_inputs.py` and follow the steps below.

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

## LLM4POI (pre‑3.1 baseline)
See `baselines/llm4poi/README.md` for training and evaluation.
This uses **Llama‑2‑7B‑LongLoRA‑32k** and the **limited‑history** variant (same‑user history only, no key‑query similarity).

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
