from __future__ import annotations
import os, random
from pathlib import Path
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_metric

IMG_EXTS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}

def list_images(root: str|Path):
    root = Path(root)
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])

def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img,0,255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def pad_to_multiple(img, m=32):
    h,w = img.shape[:2]
    ph = (m-h%m)%m; pw=(m-w%m)%m
    return cv2.copyMakeBorder(img,0,ph,0,pw,cv2.BORDER_REFLECT_101), (h,w)

def unpad(img, hw):
    h,w = hw
    return img[:h,:w]

def psnr(pred, target):
    mse = np.mean((pred.astype(np.float32)-target.astype(np.float32))**2)
    if mse <= 1e-10: return 99.0
    return 20*np.log10(255.0/np.sqrt(mse))

def ssim_rgb(pred, target):
    return float(ssim_metric(target, pred, channel_axis=2, data_range=255))
