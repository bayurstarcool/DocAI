#!/usr/bin/env bash
set -euo pipefail

SHADOW_SRC="${SHADOW_SRC:-datasetku/shadow}"
CLEAN_SRC="${CLEAN_SRC:-datasetku/cleans_rgb}"
MASK_SRC="${MASK_SRC:-}"
BASE_DEST="${BASE_DEST:-stash://bayurstarcool/projects/shadowremove/data/user_pairs/}"

[ -d "$SHADOW_SRC" ] || { echo "Shadow source not found: $SHADOW_SRC"; exit 1; }
[ -d "$CLEAN_SRC" ] || { echo "Clean source not found: $CLEAN_SRC"; exit 1; }
camber stash mkdir "${BASE_DEST}shadow/" >/dev/null 2>&1 || true
camber stash mkdir "${BASE_DEST}clean/" >/dev/null 2>&1 || true
camber stash cp -r --exclude '.DS_Store' --exclude '**/.DS_Store' "$SHADOW_SRC/" "${BASE_DEST}shadow/"
camber stash cp -r --exclude '.DS_Store' --exclude '**/.DS_Store' "$CLEAN_SRC/" "${BASE_DEST}clean/"
if [ -n "$MASK_SRC" ]; then
  [ -d "$MASK_SRC" ] || { echo "Mask source not found: $MASK_SRC"; exit 1; }
  camber stash mkdir "${BASE_DEST}mask/" >/dev/null 2>&1 || true
  camber stash cp -r --exclude '.DS_Store' --exclude '**/.DS_Store' "$MASK_SRC/" "${BASE_DEST}mask/"
fi
echo "Uploaded paired dataset to $BASE_DEST"
