#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-camber
source .venv-camber/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
