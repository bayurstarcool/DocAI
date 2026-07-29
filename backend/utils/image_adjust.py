"""
Image adjustment post-processing — user-controllable via UI.
Applied AFTER AI output. Defaults = no-op (1.0 / 0), preserving AI-only output.
"""
from PIL import Image, ImageEnhance
import numpy as np
import cv2


def apply_adjustments(img: Image.Image,
                      brightness: float = 1.0,
                      contrast: float = 1.0,
                      saturation: float = 1.0,
                      sharpness: float = 1.0,
                      gamma: float = 1.0,
                      white_balance: bool = False,
                      clahe_clip: float = 0.0) -> Image.Image:
    """
    Apply user-controllable adjustments.
    All multipliers default to 1.0 (no change). white_balance/clahe default off.

    brightness/contrast/saturation/sharpness: 0.0-3.0, 1.0 = no change
    gamma: 0.1-3.0, 1.0 = no change (>1 brighter mid, <1 darker mid)
    white_balance: bool, gray-world auto white balance
    clahe_clip: 0.0 = off, else clip limit (typical 1.0-4.0)
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # PIL enhancers (only run when != 1.0 to avoid unnecessary work)
    if abs(brightness - 1.0) > 1e-3:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if abs(contrast - 1.0) > 1e-3:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if abs(saturation - 1.0) > 1e-3:
        img = ImageEnhance.Color(img).enhance(saturation)
    if abs(sharpness - 1.0) > 1e-3:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # Gamma correction (numpy)
    if abs(gamma - 1.0) > 1e-3:
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.power(arr, 1.0 / max(gamma, 1e-3))
        img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    # Gray-world white balance
    if white_balance:
        arr = np.asarray(img).astype(np.float32)
        means = arr.reshape(-1, 3).mean(axis=0)
        gray = means.mean()
        scale = gray / np.clip(means, 1e-6, None)
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # CLAHE on L channel (LAB)
    if clahe_clip > 1e-3:
        arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(arr)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        l = clahe.apply(l)
        arr = cv2.merge([l, a, b])
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))

    return img
