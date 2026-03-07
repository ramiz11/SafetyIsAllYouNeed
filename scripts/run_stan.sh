#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/absolute/path/to/SafetyIsAllYouNeed}"
STAN_REPO_DIR="${2:-/absolute/path/to/Spatial-Temporal-Attention-Network-for-POI-Recommendation}"
DATASET="${3:-NYC}"
TRAJ_LEN="${4:-20}"

dataset_lc=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
EXPORT_DIR="$BASE_DIR/baselines/exports/stan/${dataset_lc}_len${TRAJ_LEN}"

if [ ! -d "$EXPORT_DIR" ]; then
  echo "Missing exports at $EXPORT_DIR. Run scripts/export_baseline_inputs.py first."
  exit 1
fi

if [ ! -d "$STAN_REPO_DIR" ]; then
  echo "STAN repo not found at $STAN_REPO_DIR"
  echo "Clone: git clone https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation \"$STAN_REPO_DIR\""
  exit 1
fi

TARGET_DIR="$STAN_REPO_DIR/data/$DATASET"
mkdir -p "$TARGET_DIR"
cp "$EXPORT_DIR/train.txt" "$TARGET_DIR/train.txt"
cp "$EXPORT_DIR/val.txt" "$TARGET_DIR/val.txt"
cp "$EXPORT_DIR/test.txt" "$TARGET_DIR/test.txt"

echo "Copied STAN data to $TARGET_DIR"
echo "Next: follow the STAN repo instructions to train/evaluate."
