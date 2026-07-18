#!/usr/bin/env bash
set -euo pipefail

STASH_PATH="${STASH_PATH:-stash://bayurstarcool/projects/shadowremove/}"

camber stash mkdir "$STASH_PATH" >/dev/null 2>&1 || true
camber stash cp -r --use-gitignore \
  --exclude 'datasetku/**' \
  --exclude 'data/**' \
  --exclude 'models/**' \
  --exclude 'runs/**' \
  --exclude 'exports/**' \
  --exclude '.git/**' \
  . "$STASH_PATH"

echo "Uploaded project to $STASH_PATH"
