"""DewarpNet inference wrapper for DocAI backend.
Two-stage: UNet shape network -> DNet backward mapping network.
Model: ICCV 2019 pretrained on Doc3D.
FIX: aspect ratio preserved, precise crop.
"""
import sys, os, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DEWARPNET_MODELS_DIR = BASE_DIR / "backend" / "models"
CKPT_DIR = BASE_DIR / "checkpoints" / "dewarpnet"

_wc_model = None
_bm_model = None
_htan = nn.Hardtanh(0, 1.0)

def _convert_state_dict(sd):
    keys = list(sd.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return OrderedDict((k[7:], v) for k, v in sd.items())
    return sd

def _load_models():
    global _wc_model, _bm_model
    if _wc_model is not None:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wc_path = CKPT_DIR / "unetnc_doc3d_final.pkl"
    bm_path = CKPT_DIR / "dnetccnl_doc3d_final.pkl"
    sys.path.insert(0, str(DEWARPNET_MODELS_DIR))
    from dewarpnet import get_model
    _wc_model = get_model("unetnc", 3, in_channels=3)
    _bm_model = get_model("dnetccnl", 2, in_channels=3)
    wc_sd = torch.load(str(wc_path), map_location="cpu")["model_state"]
    bm_sd = torch.load(str(bm_path), map_location="cpu")["model_state"]
    _wc_model.load_state_dict(_convert_state_dict(wc_sd))
    _bm_model.load_state_dict(_convert_state_dict(bm_sd))
    _wc_model.eval().to(device)
    _bm_model.eval().to(device)

def _unwarp(img_rgb, bm, out_h, out_w):
    """Unwarp using BM, output at out_h x out_w."""
    bm_np = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0, :, :, :]
    bm0 = cv2.blur(bm_np[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm_np[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (out_w, out_h))
    bm1 = cv2.resize(bm1, (out_w, out_h))
    bm_t = torch.from_numpy(np.stack([bm0, bm1], axis=-1)).double().unsqueeze(0)
    img_t = torch.from_numpy(img_rgb.astype(float) / 255.0).double().permute(2, 0, 1).unsqueeze(0)
    res = F.grid_sample(input=img_t, grid=bm_t, mode="bilinear", padding_mode="border", align_corners=True)
    return res[0].numpy().transpose(1, 2, 0)

def dewarpnet_infer(pil_image):
    """Run DewarpNet on a PIL RGB image, return PIL RGB result.
    Preserves aspect ratio for portrait/landscape images.
    Strategy: run on padded square, unwarp padded, crop result.
    """
    _load_models()
    device = next(_wc_model.parameters()).device
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    # Pad to square keeping aspect ratio
    sq = 256
    scale = sq / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h))
    canvas = np.full((sq, sq, 3), 128, dtype=np.uint8)
    y_off = (sq - new_h) // 2
    x_off = (sq - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # Stage 1: WC shape network (on padded square)
    img_t = torch.from_numpy(canvas[:, :, ::-1].copy().transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        wc_out = _wc_model(img_t)
        pred_wc = _htan(wc_out)

    # Stage 2: BM network (on padded square)
    bm_input = F.interpolate(pred_wc, (128, 128))
    with torch.no_grad():
        bm_out = _bm_model(bm_input)

    # Unwarp the FULL padded image (not cropped)
    # BM predicts displacement for the full 128x128 padded input
    padded_unwarped = _unwarp(canvas, bm_out, sq, sq)

    # Crop back to original aspect from the unwarped padded result
    result = padded_unwarped[y_off:y_off+new_h, x_off:x_off+new_w]

    # Resize back to original resolution
    result = cv2.resize(result, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    # Unsharp mask to recover detail lost during low-res BM
    blur = cv2.GaussianBlur(result, (0, 0), 3)
    result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)
