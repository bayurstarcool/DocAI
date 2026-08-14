"""GeoTr Doc3D inference wrapper for DocAI.
Runs FelixHertlein/inv3d-model geotr@doc3d in a subprocess.
Forces CPU to avoid GPU OOM with DocAI server.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

INV3D_REPO = Path("/home/wahyu/inv3d-model")
PYTHON = Path("/home/wahyu/miniconda3/bin/python3")
MODEL = "geotr@doc3d"


def geotr_doc3d_dewarp(image_bgr):
    if image_bgr is None:
        raise ValueError("image is None")
    if isinstance(image_bgr, Image.Image):
        image_bgr = cv2.cvtColor(np.array(image_bgr.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    stamp = time.strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory(prefix="docai_geotr_") as td:
        ds_name = f"docai_tmp_{stamp}_{abs(hash(td)) % 100000}"
        input_dir = INV3D_REPO / "input" / ds_name
        input_dir.mkdir(parents=True, exist_ok=True)
        inp = input_dir / "image_input.png"
        cv2.imwrite(str(inp), image_bgr)

        # Force CPU to avoid GPU OOM
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [
            str(PYTHON), "inference.py",
            "--model", MODEL,
            "--dataset", ds_name,
            "--gpu", "0",
            "--output_width", str(w),
            "--output_height", str(h),
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(INV3D_REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=180,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout[-4000:])
            out = INV3D_REPO / "output" / f"{ds_name} - {MODEL}" / "unwarped_input.png"
            result = cv2.imread(str(out))
            if result is None:
                raise RuntimeError(f"GeoTr output missing: {out}\n{proc.stdout[-2000:]}")
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        finally:
            try:
                for p in [input_dir, INV3D_REPO / "output" / f"{ds_name} - {MODEL}"]:
                    if p.exists():
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass
