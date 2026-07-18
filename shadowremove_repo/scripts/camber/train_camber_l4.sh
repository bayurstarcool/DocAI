#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/clean_docs}"
OUT_DIR="${OUT_DIR:-runs/camber_l4}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-4}"
SIZE="${SIZE:-1024}"
WORKERS="${WORKERS:-4}"
BASE="${BASE:-48}"

source .venv-camber/bin/activate
python train.py \
  --data "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --size "$SIZE" \
  --workers "$WORKERS" \
  --base "$BASE" \
  --amp \
  --out "$OUT_DIR"
