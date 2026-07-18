#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/datasets/paired/ShadowDocument7K}"
DOWNLOAD_DIR="${ROOT}/download"
mkdir -p "$DOWNLOAD_DIR"

if ! command -v openxlab >/dev/null 2>&1; then
  python -m pip install openxlab
fi

cat <<'EOF'
OpenXLab login required. Run once, then re-run this script:
  openxlab login
Use account with access to `lkljty/ShadowDocument7K`.
EOF

openxlab dataset download --repo lkljty/ShadowDocument7K --local "$DOWNLOAD_DIR"
echo "Download done: $DOWNLOAD_DIR"
echo "Inspect archive layout, then place official paired folders at: $ROOT/input and $ROOT/target"
