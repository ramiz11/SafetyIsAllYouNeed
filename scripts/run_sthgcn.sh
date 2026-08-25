#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/absolute/path/to/SafetyIsAllYouNeed}"
STHGCN_REPO_DIR="${2:-/absolute/path/to/Spatio-Temporal-Hypergraph-Model}"
DATASET="${3:-CHICAGO}"

EXPORT_DIR="$BASE_DIR/baselines/exports/sthgcn"
TRAIN_SRC="$EXPORT_DIR/${DATASET}_train.tsv"
VAL_SRC="$EXPORT_DIR/${DATASET}_val.tsv"
TEST_SRC="$EXPORT_DIR/${DATASET}_test.tsv"

if [ ! -f "$TRAIN_SRC" ]; then
  echo "Missing exports at $EXPORT_DIR. Run scripts/export_baseline_inputs.py first."
  exit 1
fi

if [ ! -d "$STHGCN_REPO_DIR" ]; then
  echo "STHGCN repo not found at $STHGCN_REPO_DIR"
  echo "Clone: git clone https://github.com/alipay/Spatio-Temporal-Hypergraph-Model \"$STHGCN_REPO_DIR\""
  exit 1
fi

TARGET_DIR="$STHGCN_REPO_DIR/data/$DATASET"
mkdir -p "$TARGET_DIR"
cp "$TRAIN_SRC" "$TARGET_DIR/${DATASET}_train.tsv"
cp "$VAL_SRC" "$TARGET_DIR/${DATASET}_val.tsv"
cp "$TEST_SRC" "$TARGET_DIR/${DATASET}_test.tsv"

echo "Copied STHGCN data to $TARGET_DIR"
echo "Next: follow the STHGCN repo instructions to train/evaluate."
