#!/usr/bin/env bash
set -euo pipefail

STASH_PATH="${STASH_PATH:-stash://bayurstarcool/projects/shadowremove/}"
CLEAN_SRC="${CLEAN_SRC:-datasetku/cleans_rgb}"
SHADOW_SRC="${SHADOW_SRC:-datasetku/shadow}"

command -v camber >/dev/null || { echo 'camber CLI not found. Run: source ~/.zshrc'; exit 1; }
camber me >/dev/null

echo "Camber CLI OK"
echo "Project stash: $STASH_PATH"

echo "Local project files to upload:"
git ls-files | wc -l | tr -d ' '

echo "Ignored heavy paths:"
git status --ignored --short datasetku data models runs exports __pycache__ .DS_Store | sed -n '1,80p'

if [ -d "$CLEAN_SRC" ]; then
  echo "Clean dataset $CLEAN_SRC images:"
  find "$CLEAN_SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l | tr -d ' '
else
  echo "Missing clean dataset: $CLEAN_SRC"
fi

if [ -d "$SHADOW_SRC" ]; then
  echo "Shadow dataset $SHADOW_SRC images:"
  find "$SHADOW_SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l | tr -d ' '
fi

camber stash test -e "$STASH_PATH" >/dev/null 2>&1 && echo "Stash path exists" || echo "Stash path will be created"
