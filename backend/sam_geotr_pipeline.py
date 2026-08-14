"""SAM mask/crop + GeoTr Doc3D + optional DocRes enhancement for DocAI."""
from __future__ import annotations

import subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

PYTHON = Path('/home/wahyu/miniconda3/bin/python3')
SAM_SCRIPT = Path('/home/wahyu/docai/backend/sam_document_crop.py')


def _pil_to_bgr(image):
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    return image


def sam_document_crop(image):
    img_bgr = _pil_to_bgr(image)
    with tempfile.TemporaryDirectory(prefix='docai_sam_') as td:
        inp = Path(td) / 'input.png'
        out = Path(td) / 'crop.png'
        cv2.imwrite(str(inp), img_bgr)
        proc = subprocess.run(
            [str(PYTHON), str(SAM_SCRIPT), str(inp), str(out)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout[-4000:])
        res = cv2.imread(str(out))
        if res is None:
            raise RuntimeError('SAM crop output missing: ' + proc.stdout[-2000:])
        return Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
