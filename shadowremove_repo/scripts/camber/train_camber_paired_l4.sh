#!/usr/bin/env bash
set -euo pipefail

PAIRED_SHADOW="${PAIRED_SHADOW:-data/user_pairs/shadow}"
PAIRED_CLEAN="${PAIRED_CLEAN:-data/user_pairs/clean}"
PAIRED_MASK="${PAIRED_MASK:-}"
OUT_DIR="${OUT_DIR:-runs/camber_l4_paired}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-4}"
SIZE="${SIZE:-1024}"
WORKERS="${WORKERS:-4}"
BASE="${BASE:-48}"

source .venv-camber/bin/activate
cmd=(python train.py --paired-shadow "$PAIRED_SHADOW" --paired-clean "$PAIRED_CLEAN" --epochs "$EPOCHS" --batch "$BATCH" --size "$SIZE" --workers "$WORKERS" --base "$BASE" --amp --out "$OUT_DIR")
if [ -n "$PAIRED_MASK" ]; then
  cmd+=(--paired-mask "$PAIRED_MASK")
fi
"${cmd[@]}"
