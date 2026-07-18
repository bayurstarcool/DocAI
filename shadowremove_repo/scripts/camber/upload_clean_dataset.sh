#!/usr/bin/env bash
set -euo pipefail

DATA_SRC="${DATA_SRC:-datasetku/cleans_rgb}"
DEST="${DEST:-stash://bayurstarcool/projects/shadowremove/data/clean_docs/}"

[ -d "$DATA_SRC" ] || { echo "Dataset source not found: $DATA_SRC"; exit 1; }
camber stash mkdir "$DEST" >/dev/null 2>&1 || true
camber stash cp -r --exclude '.DS_Store' --exclude '**/.DS_Store' "$DATA_SRC/" "$DEST"
echo "Uploaded clean dataset $DATA_SRC to $DEST"
