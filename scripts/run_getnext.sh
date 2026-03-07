#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/absolute/path/to/SafetyIsAllYouNeed}"
GETNEXT_REPO_DIR="${2:-/absolute/path/to/GETNext}"
DATASET="${3:-NYC}"

EXPORT_DIR="$BASE_DIR/baselines/exports/getnext"
TRAIN_SRC="$EXPORT_DIR/${DATASET}_train.csv"
VAL_SRC="$EXPORT_DIR/${DATASET}_val.csv"
TEST_SRC="$EXPORT_DIR/${DATASET}_test.csv"

if [ ! -f "$TRAIN_SRC" ]; then
  echo "Missing exports at $EXPORT_DIR. Run scripts/export_baseline_inputs.py first."
  exit 1
fi

if [ ! -d "$GETNEXT_REPO_DIR" ]; then
  echo "GETNext repo not found at $GETNEXT_REPO_DIR"
  echo "Clone: git clone https://github.com/songyangme/GETNext \"$GETNEXT_REPO_DIR\""
  exit 1
fi

TARGET_DIR="$GETNEXT_REPO_DIR/dataset/$DATASET"
mkdir -p "$TARGET_DIR"
cp "$TRAIN_SRC" "$TARGET_DIR/${DATASET}_train.csv"
cp "$VAL_SRC" "$TARGET_DIR/${DATASET}_val.csv"
cp "$TEST_SRC" "$TARGET_DIR/${DATASET}_test.csv"

echo "Copied GETNext data to $TARGET_DIR"
echo "Next: run build_graph.py then train.py in the GETNext repo."
