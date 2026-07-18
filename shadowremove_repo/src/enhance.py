import cv2
import numpy as np


def white_balance_grayworld(img):
    f = img.astype(np.float32) + 1e-6
    avg = f.reshape(-1, 3).mean(axis=0)
    gray = avg.mean()
    return np.clip(f * (gray / avg), 0, 255).astype(np.uint8)


def estimate_illumination(img, k=45):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return bg


def classical_shadow_remove(img):
    """Fast fallback: illumination normalization + adaptive contrast."""
    img = white_balance_grayworld(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    bg = estimate_illumination(img)
    norm = cv2.divide(L, bg, scale=245)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    norm = clahe.apply(norm)
    out = cv2.cvtColor(cv2.merge([norm, A, B]), cv2.COLOR_LAB2BGR)
    out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    sharp = cv2.GaussianBlur(out, (0, 0), 1.1)
    out = cv2.addWeighted(out, 1.35, sharp, -0.35, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def binarize_document(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def enhance_document(img, mode="color"):
    clean = classical_shadow_remove(img)
    if mode == "bw":
        return binarize_document(clean)
    if mode == "magic":
        lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        L = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(L)
        clean = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
    return clean


def magic_document_enhance(img_rgb):
    """RGB-in/RGB-out wrapper around the classical magic enhancer."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = enhance_document(bgr, mode="magic")
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def adaptive_binarize(img_rgb):
    """RGB-in/RGB-out adaptive binarization for OCR-ready output."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    out = binarize_document(bgr)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def _soft_shadow_mask(mask):
    mask = mask.astype(np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    mask = np.clip((mask - 0.08) / 0.62, 0, 1)
    return mask[:, :, None]

def _restore_detail(original, enhanced):
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.GaussianBlur(edges, (0, 0), 0.8)[:, :, None]
    original_f = original.astype(np.float32)
    enhanced_f = enhanced.astype(np.float32)
    blended = enhanced_f * (1 - edges * 0.45) + original_f * (edges * 0.45)
    blur = cv2.GaussianBlur(blended, (0, 0), 0.75)
    sharp = cv2.addWeighted(blended, 1.18, blur, -0.18, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def ai_shadow_postprocess(original_rgb, ai_rgb, mask):
    """Keep AI correction mostly on shadow regions while preserving text/detail."""
    if mask is None:
        return _restore_detail(original_rgb, ai_rgb)
    alpha = _soft_shadow_mask(mask)
    original_lab = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB)
    ai_lab = cv2.cvtColor(ai_rgb, cv2.COLOR_RGB2LAB)
    original_l, original_a, original_b = cv2.split(original_lab)
    ai_l, ai_a, ai_b = cv2.split(ai_lab)
    alpha_2d = alpha[:, :, 0]
    l = original_l.astype(np.float32) * (1 - alpha_2d) + ai_l.astype(np.float32) * alpha_2d
    chroma_alpha = alpha_2d * 0.12
    a = original_a.astype(np.float32) * (1 - chroma_alpha) + ai_a.astype(np.float32) * chroma_alpha
    b = original_b.astype(np.float32) * (1 - chroma_alpha) + ai_b.astype(np.float32) * chroma_alpha
    balanced = cv2.cvtColor(cv2.merge([
        np.clip(l, 0, 255).astype(np.uint8),
        np.clip(a, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
    ]), cv2.COLOR_LAB2RGB)
    return _restore_detail(original_rgb, balanced)
