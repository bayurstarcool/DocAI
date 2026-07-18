#!/usr/bin/env bash
set -euo pipefail

STASH_PATH="${STASH_PATH:-stash://bayurstarcool/projects/shadowremove/}"
SIZE="${SIZE:-xsmall}"
ENGINE="${ENGINE:-base}"
PAIR_LIMIT="${PAIR_LIMIT:-200}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-sd7k_subset}"
WORK_DIR="${WORK_DIR:-sd7k_work}"
MANIFEST_DIR="${MANIFEST_DIR:-sd7k_manifest}"

CMD="PAIR_LIMIT=$PAIR_LIMIT SEED=$SEED OUT_DIR=$OUT_DIR WORK_DIR=$WORK_DIR MANIFEST_DIR=$MANIFEST_DIR bash scripts/camber/prepare_sd7k_subset_openxlab.sh"

camber job create \
  --path "$STASH_PATH" \
  --engine "$ENGINE" \
  --size "$SIZE" \
  --num-nodes 1 \
  --cmd "$CMD"
