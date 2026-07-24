#!/usr/bin/env python3
"""Test all 3 shadow removal methods from repo."""
import cv2
import numpy as np
from scipy import ndimage
from skimage import color, filters

def effective_bg_estimation(img_path, out_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    L = np.stack([ndimage.maximum_filter(img_rgb[:,:,i], size=13) for i in range(3)], axis=2)
    L[L == 0] = 1e-6
    
    G = np.zeros_like(L)
    for i in range(3):
        thresh = filters.threshold_otsu((L[:,:,i] * 255).astype(np.uint8))
        unshadowed = L[:,:,i] > thresh
        mean_val = np.mean(L[:,:,i][unshadowed]) if unshadowed.any() else np.mean(L[:,:,i])
        G[:,:,i] = mean_val
    
    result = np.clip((G / L * img_rgb) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"OK: {out_path}")

def water_filling(img_path, out_path):
    img = cv2.imread(img_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    h0, w0 = y.shape
    y_small = cv2.resize(y, (0,0), fx=0.2, fy=0.2).astype(np.float32)
    hs, ws = y_small.shape
    
    neta = 0.2
    w_ = np.zeros((hs, ws), dtype=np.float32)
    
    for t in range(2500):
        G_ = w_ + y_small
        G_peak = np.amax(G_)
        pouring = np.exp(-t) * (G_peak - G_)
        
        left = np.minimum(np.roll(G_, -1, axis=1) - G_, 0)
        right = np.minimum(np.roll(G_, 1, axis=1) - G_, 0)
        top = np.minimum(np.roll(G_, -1, axis=0) - G_, 0)
        btm = np.minimum(np.roll(G_, 1, axis=0) - G_, 0)
        
        del_w = neta * (left + right + top + btm)
        w_ = np.maximum(del_w + pouring + w_, 0)
    
    G_ = w_ + y_small
    G_ = cv2.resize(G_, (w0, h0))
    G_[G_ == 0] = 1e-6
    
    result_y = np.clip(0.85 * y.astype(np.float32) / G_ * 255, 0, 255).astype(np.uint8)
    merged = cv2.merge([result_y, cr, cb])
    cv2.imwrite(out_path, cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR))
    print(f"OK: {out_path}")

def iterative_removal(img_path, out_path):
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    block_size = max(51, min(201, max(gray.shape) // 3) | 1)
    thresh_mask = cv2.adaptiveThreshold((gray*255).astype(np.uint8), 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)
    binary = thresh_mask > 0
    
    result = img.copy()
    for _ in range(3):
        ks = max(101, min(301, max(gray.shape) // 3) | 1)
        for c in range(3):
            shading = cv2.GaussianBlur(result[:,:,c], (ks, ks), 0)
            shading[shading == 0] = 1e-6
            mean_bg = np.mean(result[:,:,c][binary]) if binary.any() else np.mean(result[:,:,c])
            ratio = np.clip(mean_bg / shading, 0.5, 2.0)
            result[:,:,c] = result[:,:,c] * ratio
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, result)
    print(f"OK: {out_path}")

# Test on both images
for img_path, name in [('/tmp/shadow_flower.jpg', 'flower'), ('/tmp/shadow_test_new.jpg', 'arabic')]:
    effective_bg_estimation(img_path, f'/tmp/m1_{name}.png')
    water_filling(img_path, f'/tmp/m2_{name}.png')
    iterative_removal(img_path, f'/tmp/m3_{name}.png')

print("ALL DONE")
