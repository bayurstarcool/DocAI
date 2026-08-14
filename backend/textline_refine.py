"""Experimental textline residual refinement for DocAI dewarp.

Pipeline: GeoTr Doc3D full-frame dewarp -> estimate residual local text-line shifts -> remap -> optional DocRes appearance by caller.
Designed for text-heavy pages (newspaper/book) without template.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(image):
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return image


def _bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _estimate_row_offsets(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Text/ink emphasis. Invert after adaptive threshold: dark text -> high values.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    # Keep horizontal text strokes, suppress vertical page borders/noise.
    kx = max(25, w // 32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    # Row projection = text-line energy.
    proj = horiz.sum(axis=1).astype(np.float32)
    if proj.max() <= 0:
        return None
    # Smooth projection; find text rows.
    win = max(9, h // 80)
    if win % 2 == 0:
        win += 1
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, win), 0).ravel()
    active = proj_s > max(proj_s.mean() + 0.35 * proj_s.std(), proj_s.max() * 0.18)
    # Estimate local residual skew per horizontal band from Hough angles.
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=max(40, w // 8), maxLineGap=18)
    if lines is None:
        return None
    bands = max(8, min(24, h // 70))
    band_h = h / bands
    offsets = np.zeros(h, dtype=np.float32)
    conf = np.zeros(h, dtype=np.float32)
    for b in range(bands):
        y0 = int(b * band_h)
        y1 = int((b + 1) * band_h)
        angles = []
        for l in lines:
            x1, y1l, x2, y2l = l.reshape(-1)
            cy = (y1l + y2l) / 2
            if y0 <= cy < y1:
                a = np.degrees(np.arctan2(y2l - y1l, x2 - x1))
                if abs(a) < 15:
                    angles.append(a)
        if len(angles) < 3:
            continue
        med = float(np.median(angles))
        # Convert residual angle to x-shift across vertical distance from band center.
        center = (y0 + y1) / 2
        ys = np.arange(y0, y1)
        shift = np.tan(np.deg2rad(med)) * (ys - center)
        # Limit. This is residual refinement only, not main dewarp.
        shift = np.clip(shift, -w * 0.025, w * 0.025)
        offsets[y0:y1] = shift
        conf[y0:y1] = 1.0
    if conf.max() <= 0:
        return None
    # Only apply on rows with text signal; interpolate through gaps.
    mask = (conf > 0) & active
    if mask.sum() < h * 0.08:
        mask = conf > 0
    idx = np.arange(h)
    offsets_interp = np.interp(idx, idx[mask], offsets[mask]).astype(np.float32) if mask.any() else offsets
    offsets_interp = cv2.GaussianBlur(offsets_interp.reshape(-1, 1), (1, max(9, win | 1)), 0).ravel()
    return offsets_interp


def textline_residual_refine(image, strength: float = 0.65):
    img = _pil_to_bgr(image)
    h, w = img.shape[:2]
    offsets = _estimate_row_offsets(img)
    if offsets is None:
        return image if isinstance(image, Image.Image) else _bgr_to_pil(img)
    offsets = offsets * float(strength)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + offsets[:, None].astype(np.float32)
    map_y = grid_y.astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return _bgr_to_pil(out)


def textline_refine_dewarp(image, geotr_doc3d_dewarp, docres_task_infer=None):
    base = geotr_doc3d_dewarp(image)
    refined = textline_residual_refine(base)
    if docres_task_infer is not None:
        refined = docres_task_infer(refined, 'appearance')
    return refined
