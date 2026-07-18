#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-camber_l4_paired}"
SRC="${SRC:-stash://bayurstarcool/projects/shadowremove/outputs/$RUN_NAME/}"
DEST="${DEST:-runs/$RUN_NAME}"

mkdir -p "$DEST"
camber stash cp -r "$SRC" "$DEST/"
echo "Pulled $SRC to $DEST"
