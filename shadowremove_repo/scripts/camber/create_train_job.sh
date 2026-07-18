#!/usr/bin/env bash
set -euo pipefail

STASH_PATH="${STASH_PATH:-stash://bayurstarcool/projects/shadowremove/}"
SIZE="${SIZE:-small}"
ENGINE="${ENGINE:-base}"
MODE="${MODE:-clean}"
DATA_DIR="${DATA_DIR:-data/clean_docs}"
OUT_DIR="${OUT_DIR:-runs/camber_l4}"
RUN_NAME="${RUN_NAME:-$(basename "$OUT_DIR")}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-2}"
IMG_SIZE="${IMG_SIZE:-1024}"
WORKERS="${WORKERS:-0}"
BASE="${BASE:-32}"
PAIRED_SHADOW="${PAIRED_SHADOW:-data/static_pairs/shadow}"
PAIRED_CLEAN="${PAIRED_CLEAN:-data/static_pairs/clean}"
PAIRED_MASK="${PAIRED_MASK:-data/static_pairs/mask}"

if [ "$MODE" = "paired" ]; then
  TRAIN_CMD="PAIRED_SHADOW=$PAIRED_SHADOW PAIRED_CLEAN=$PAIRED_CLEAN PAIRED_MASK=$PAIRED_MASK OUT_DIR=$OUT_DIR EPOCHS=$EPOCHS BATCH=$BATCH SIZE=$IMG_SIZE WORKERS=$WORKERS BASE=$BASE bash scripts/camber/train_camber_paired_l4.sh"
else
  TRAIN_CMD="DATA_DIR=$DATA_DIR OUT_DIR=$OUT_DIR EPOCHS=$EPOCHS BATCH=$BATCH SIZE=$IMG_SIZE WORKERS=$WORKERS BASE=$BASE bash scripts/camber/train_camber_l4.sh"
fi

CMD="bash scripts/camber/setup_camber.sh && $TRAIN_CMD"

echo "Output will be saved under stash project path: $OUT_DIR"

camber job create \
  --path "$STASH_PATH" \
  --engine "$ENGINE" \
  --gpu \
  --size "$SIZE" \
  --num-nodes 1 \
  --cmd "$CMD"
