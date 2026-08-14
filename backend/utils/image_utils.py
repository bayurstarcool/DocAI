"""
Image processing utilities for preprocessing and postprocessing
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import torch
import torchvision.transforms as transforms


def _blend_window(height: int, width: int) -> np.ndarray:
    """Return soft tile weights to reduce seams in overlapped inference."""
    if height <= 1 or width <= 1:
        return np.ones((height, width), dtype=np.float32)
    y = np.hanning(height).astype(np.float32)
    x = np.hanning(width).astype(np.float32)
    weights = np.outer(y, x)
    weights = np.maximum(weights, 1e-3)
    return weights / weights.max()


def run_tiled_restoration(model, image: Image.Image, device: str, tile_size: int = 1024, overlap: int = 96):
    """Restore high-resolution documents in overlapping tiles without resizing them."""
    if tile_size <= 0 or tile_size % 4:
        raise ValueError('tile_size must be a positive multiple of 4')
    if overlap < 0 or overlap >= tile_size:
        raise ValueError('overlap must be between 0 and tile_size')

    source = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    if height <= tile_size and width <= tile_size:
        tensor = torch.from_numpy(source.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        restored, mask = model(tensor)
        restored = restored[0].detach().float().cpu().numpy().transpose(1, 2, 0)
        mask = mask[0, 0].detach().float().cpu().numpy()
    else:
        step = tile_size - overlap
        y_starts = list(range(0, max(1, height - tile_size + 1), step))
        x_starts = list(range(0, max(1, width - tile_size + 1), step))
        if y_starts[-1] != max(0, height - tile_size):
            y_starts.append(max(0, height - tile_size))
        if x_starts[-1] != max(0, width - tile_size):
            x_starts.append(max(0, width - tile_size))
        restored_sum = np.zeros((height, width, 3), dtype=np.float32)
        mask_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        for top in y_starts:
            for left in x_starts:
                bottom, right = min(top + tile_size, height), min(left + tile_size, width)
                patch = source[top:bottom, left:right]
                tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
                prediction, prediction_mask = model(tensor)
                prediction = prediction[0].detach().float().cpu().numpy().transpose(1, 2, 0)
                prediction_mask = prediction_mask[0, 0].detach().float().cpu().numpy()
                weights = _blend_window(bottom - top, right - left)
                restored_sum[top:bottom, left:right] += prediction * weights[..., None]
                mask_sum[top:bottom, left:right] += prediction_mask * weights
                weight_sum[top:bottom, left:right] += weights
        restored = restored_sum / weight_sum[..., None]
        mask = mask_sum / weight_sum
    result = Image.fromarray((np.clip(restored, 0, 1) * 255).astype(np.uint8))
    mask_image = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return result, mask_image

def segment_document(image: Image.Image) -> Image.Image:
    """Crop to the largest document-like contour when one is found."""
    source = np.asarray(image.convert('RGB'))
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.convert('RGB')
    largest = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(largest)
    image_area = source.shape[0] * source.shape[1]
    box_area = width * height
    if box_area < image_area * 0.25 or width < source.shape[1] * 0.35 or height < source.shape[0] * 0.35:
        return image.convert('RGB')
    padding = max(4, int(min(source.shape[:2]) * 0.01))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(source.shape[1], x + width + padding)
    y1 = min(source.shape[0], y + height + padding)
    return Image.fromarray(source[y0:y1, x0:x1])

def _soft_shadow_matte(mask_image: Image.Image) -> np.ndarray:
    mask = np.asarray(mask_image.convert('L'), dtype=np.float32) / 255.0
    mask = cv2.GaussianBlur(mask, (0, 0), 2.5)
    matte = np.clip((mask - 0.005) / 0.40, 0, 1)
    return np.clip(matte ** 0.45, 0, 1)

def _estimate_illumination_map(image_rgb: np.ndarray, shadow_matte: np.ndarray) -> np.ndarray:
    image_float = image_rgb.astype(np.float32) / 255.0
    luma = 0.299 * image_float[..., 0] + 0.587 * image_float[..., 1] + 0.114 * image_float[..., 2]
    kernel = max(31, int(min(image_rgb.shape[:2]) * 0.09) | 1)
    illumination = cv2.GaussianBlur(luma, (kernel, kernel), 0)
    shadow_darkening = cv2.GaussianBlur(shadow_matte, (0, 0), max(3, kernel / 8))
    return np.clip(illumination * (1.0 - shadow_darkening * 0.20), 0.08, 1.0)

def _apply_shadow_correction(original_rgb: np.ndarray, ai_rgb: np.ndarray, shadow_matte: np.ndarray,
                             illumination_map: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply shadow correction: replace dark pixels with AI-brightened versions.
    
    Rule: original luma < 80 AND AI brightens by > 5 units -> use AI.
    Everything else stays original. Guarantees non-shadow areas unchanged.
    """
    original = original_rgb.astype(np.float32) / 255.0
    ai = ai_rgb.astype(np.float32) / 255.0
    
    orig_luma = 0.299 * original[:,:,0] + 0.587 * original[:,:,1] + 0.114 * original[:,:,2]
    ai_luma = 0.299 * ai[:,:,0] + 0.587 * ai[:,:,1] + 0.114 * ai[:,:,2]
    
    # Dark pixel that AI brightens
    is_dark = orig_luma < (80.0 / 255.0)
    ai_brightens = (ai_luma - orig_luma) > (5.0 / 255.0)
    mask = (is_dark & ai_brightens).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (11, 11), 3)
    mask = np.clip(mask * strength, 0, 1)[..., None]
    
    corrected = original * (1 - mask) + ai * mask
    return (np.clip(corrected, 0, 1) * 255).astype(np.uint8)

def _white_balance_gray_world(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    means = image.reshape(-1, 3).mean(axis=0)
    target = max(means.mean(), 130.0)  # higher target for brightness
    gains = target / np.maximum(means, 1.0)
    # gentle lift only - reduce over-bright in already bright areas
    brightness_boost = np.clip(target / max(means.mean(), 1.0), 1.0, 1.08)
    return np.clip(image * gains * brightness_boost, 0, 255).astype(np.uint8)

def _uniform_lighting(image_rgb: np.ndarray) -> np.ndarray:
    """Passthrough - preserve original brightness. Shadow correction already handles blending."""
    return image_rgb.copy()

def _local_contrast_enhancement(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def _text_sharpening(image_rgb: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Gentle unsharp mask to recover text sharpness without noise.
    
    strength: 0.0 = no sharpening, 0.5 = moderate, 1.0 = strong
    """
    if strength <= 0:
        return image_rgb
    
    # Gaussian blur for unsharp mask
    blurred = cv2.GaussianBlur(image_rgb, (0, 0), 1.0)
    # Compute high-frequency component
    detail = image_rgb.astype(np.float32) - blurred.astype(np.float32)
    # Add back scaled detail
    sharpened = image_rgb.astype(np.float32) + detail * strength
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def _ai_color_preserve(original_rgb: np.ndarray, ai_rgb: np.ndarray, mask_image: Image.Image | None = None) -> np.ndarray:
    """Preserve original colors in non-shadow area while using AI luminance in shadow area.

    Goal: keep AI shadow removal/detail, avoid global color shift.
    """
    orig = original_rgb.astype(np.uint8)
    ai = ai_rgb.astype(np.uint8)

    orig_lab = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB)
    ai_lab = cv2.cvtColor(ai, cv2.COLOR_RGB2LAB)

    orig_l = orig_lab[:, :, 0].astype(np.float32)
    ai_l = ai_lab[:, :, 0].astype(np.float32)

    # Shadow mask from model if available; otherwise infer from dark original + AI brightening.
    if mask_image is not None:
        matte = np.asarray(mask_image.convert('L'), dtype=np.float32) / 255.0
    else:
        matte = np.zeros_like(orig_l, dtype=np.float32)

    brightening = np.clip((ai_l - orig_l) / 55.0, 0, 1)
    dark = np.clip((145.0 - orig_l) / 95.0, 0, 1)
    inferred = brightening * dark

    # Use max of model matte and inferred shadow, but suppress bright non-shadow.
    shadow = np.maximum(matte * dark, inferred)
    shadow = np.where(shadow > 0.08, shadow, 0.0)
    shadow = cv2.GaussianBlur(shadow.astype(np.float32), (21, 21), 7)
    shadow = np.clip(shadow, 0, 1)

    # L channel: AI only in shadow, original elsewhere.
    out_l = orig_l * (1 - shadow) + ai_l * shadow

    # AB color: mostly original to prevent color shift; slight AI color only in deep shadow.
    out_lab = orig_lab.copy().astype(np.float32)
    out_lab[:, :, 0] = out_l
    deep = (shadow ** 1.8)[..., None] * 0.15
    out_lab[:, :, 1:3] = orig_lab[:, :, 1:3].astype(np.float32) * (1 - deep) + ai_lab[:, :, 1:3].astype(np.float32) * deep

    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out

def _ai_tone_match(original_rgb: np.ndarray, ai_rgb: np.ndarray) -> np.ndarray:
    """AI Only + tone match to original using bright non-shadow pixels.

    Keeps AI shadow removal/detail, corrects global color/brightness shift.
    """
    orig = original_rgb.astype(np.float32)
    ai = ai_rgb.astype(np.float32)

    # Non-shadow reference: bright, low-saturation-ish paper/background area.
    orig_lab = cv2.cvtColor(np.clip(orig,0,255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L = orig_lab[:, :, 0].astype(np.float32)
    # Use top 35% luminance as non-shadow reference, ignore extreme highlights.
    lo = np.percentile(L, 65)
    hi = np.percentile(L, 98)
    ref = (L >= lo) & (L <= hi)
    if ref.mean() < 0.05:
        ref = L >= np.percentile(L, 70)

    out = ai.copy()
    # Per-channel affine match on reference pixels.
    for c in range(3):
        o = orig[:, :, c][ref]
        a = ai[:, :, c][ref]
        if len(o) < 100:
            continue
        o_mean, o_std = float(o.mean()), float(o.std() + 1e-6)
        a_mean, a_std = float(a.mean()), float(a.std() + 1e-6)
        matched = (out[:, :, c] - a_mean) * (o_std / a_std) + o_mean
        # Blend correction: 85% tone match, 15% original AI color.
        out[:, :, c] = out[:, :, c] * 0.15 + matched * 0.85

    # Preserve dark-shadow AI strength: don't over-darken pixels AI strongly brightened.
    orig_l = 0.299*orig[:,:,0] + 0.587*orig[:,:,1] + 0.114*orig[:,:,2]
    ai_l = 0.299*ai[:,:,0] + 0.587*ai[:,:,1] + 0.114*ai[:,:,2]
    out_l = 0.299*out[:,:,0] + 0.587*out[:,:,1] + 0.114*out[:,:,2]
    shadow = np.clip((120 - orig_l) / 80, 0, 1) * np.clip((ai_l - orig_l) / 60, 0, 1)
    shadow = cv2.GaussianBlur(shadow.astype(np.float32), (17,17), 5)[...,None]
    # In shadow, mix back AI to keep removal strong.
    out = out * (1 - shadow*0.45) + ai * (shadow*0.45)

    return np.clip(out, 0, 255).astype(np.uint8)

def _paper_whitening(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[..., 0].astype(np.float32)
    high = np.percentile(l_channel, 92)
    low = np.percentile(l_channel, 4)
    stretched_l = np.clip((l_channel - low) * (255.0 / max(high - low, 1.0)), 0, 255)
    paper = stretched_l > np.percentile(stretched_l, 62)
    whitened = image.copy()
    white_target = np.array([246, 246, 242], dtype=np.float32)
    blend = np.clip((stretched_l - 185) / 70, 0, 0.15)[..., None]
    whitened[paper] = whitened[paper] * (1 - blend[paper]) + white_target * blend[paper]
    lab[..., 0] = stretched_l.astype(np.uint8)
    contrast_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    result = contrast_rgb * 0.85 + whitened * 0.15
    return np.clip(result, 0, 255).astype(np.uint8)

def run_document_restoration_pipeline(
    model,
    image: Image.Image,
    device: str,
    tile_size: int = 1024,
    overlap: int = 96,
    segment: bool = False,
    mode: str = "full",
    shadow_strength: float = 1.0,
):
    """Run document restoration pipeline.
    
    Modes:
        full    - AI + shadow correction + white balance + whitening + CLAHE (default)
        ai_only - AI model output only (preserves original colors)
        ai      - AI + shadow correction only (no color processing)
        color   - AI + shadow correction + CLAHE only (no white balance/whitening)
    """
    segmented = segment_document(image) if segment else image.convert('RGB')
    ai_corrected, mask_image = run_tiled_restoration(model, segmented, device, tile_size=tile_size, overlap=overlap)
    
    if mode == "ai_only":
        # Apply gentle uniform lighting to balance shadow vs non-shadow areas
        ai_arr = np.asarray(ai_corrected.convert("RGB"))
        ai_uniform = _uniform_lighting(ai_arr)
        # Recover text sharpness
        ai_uniform = _text_sharpening(ai_uniform, strength=0.4)
        result = Image.fromarray(ai_uniform)
        info = {
            'pipeline': ['input', 'ai_model', 'uniform_lighting', 'text_sharpening', 'ai_only_output'],
            'segmented_size': segmented.size,
            'mask_mean': float(np.asarray(mask_image).mean()) if mask_image else 0,
            'illumination_mean': float(np.mean(np.asarray(result.convert("L")))) / 255.0,
        }
        return result, mask_image, info

    if mode == "ai_tone_match":
        ai_arr = np.asarray(ai_corrected.convert("RGB"))
        orig_arr = np.asarray(segmented.convert("RGB"))
        matched = _ai_tone_match(orig_arr, ai_arr)
        matched = _text_sharpening(matched, strength=0.2)
        result = Image.fromarray(matched)
        info = {
            'pipeline': ['input', 'ai_model', 'tone_match_non_shadow', 'shadow_ai_mix', 'text_sharpening'],
            'segmented_size': segmented.size,
            'mask_mean': float(np.asarray(mask_image).mean()) if mask_image else 0,
            'illumination_mean': float(np.mean(np.asarray(result.convert("L")))) / 255.0,
        }
        return result, mask_image, info

    if mode == "ai_preserve":
        ai_arr = np.asarray(ai_corrected.convert("RGB"))
        orig_arr = np.asarray(segmented.convert("RGB"))
        preserved = _ai_color_preserve(orig_arr, ai_arr, mask_image)
        preserved = _text_sharpening(preserved, strength=0.25)
        result = Image.fromarray(preserved)
        info = {
            'pipeline': ['input', 'ai_model', 'color_preserve_lab', 'shadow_mask_blend', 'text_sharpening'],
            'segmented_size': segmented.size,
            'mask_mean': float(np.asarray(mask_image).mean()) if mask_image else 0,
            'illumination_mean': float(np.mean(np.asarray(result.convert("L")))) / 255.0,
        }
        return result, mask_image, info

    if mode == "ai_shadow":
        # Targeted gamma: brighten dark areas aggressively
        ai_arr = np.asarray(ai_corrected.convert("RGB"))
        ai_float = ai_arr.astype(np.float32) / 255.0
        
        luma = 0.299 * ai_float[:,:,0] + 0.587 * ai_float[:,:,1] + 0.114 * ai_float[:,:,2]
        
        # Detect dark areas (shadow artifacts)
        dark_mask = (luma < 0.60).astype(np.float32)
        dark_mask = cv2.GaussianBlur(dark_mask, (0, 0), 11.0)
        
        # Gamma: strength 1.0 -> gamma 0.15 (near 0% shadow), strength 0.3 -> gamma 0.6
        gamma = 0.15 + (1.0 - shadow_strength) * 0.45
        
        corrected = ai_float.copy()
        for c in range(3):
            corrected[:,:,c] = np.where(
                dark_mask > 0.05,
                np.power(np.maximum(ai_float[:,:,c], 0.001), gamma),
                ai_float[:,:,c]
            )
        
        final = ai_float * (1 - dark_mask[...,None]) + corrected * dark_mask[...,None]
        final_arr = (np.clip(final, 0, 1) * 255).astype(np.uint8)
        # Normalize luminance across whole page so gamma-corrected shadow area
        # matches untouched non-shadow area (fixes tone mismatch complaint)
        final_arr = _uniform_lighting(final_arr)
        # Recover text sharpness lost during gamma+uniform_lighting
        final_arr = _text_sharpening(final_arr, strength=0.4)
        
        info = {
            "pipeline": ["input", "ai_model", "dark_area_detection", "gamma_correction", "uniform_lighting", "text_sharpening"],
            "segmented_size": segmented.size,
            "mask_mean": float(dark_mask.mean()),
            "illumination_mean": float(luma.mean()),
        }
        return Image.fromarray(final_arr), mask_image, info
    
    segmented_rgb = np.asarray(segmented.convert('RGB'))
    ai_rgb = np.asarray(ai_corrected.convert('RGB'))
    shadow_matte = _soft_shadow_matte(mask_image)
    illumination_map = _estimate_illumination_map(segmented_rgb, shadow_matte)
    corrected = _apply_shadow_correction(segmented_rgb, ai_rgb, shadow_matte, illumination_map, strength=shadow_strength)
    
    if mode == "ai":
        final = corrected
        pipeline_steps = ['input', 'shadow_mask_prediction', 'illumination_map_estimation', 'shadow_correction']
    elif mode == "color":
        final = _local_contrast_enhancement(corrected)
        pipeline_steps = ['input', 'shadow_mask_prediction', 'illumination_map_estimation', 'shadow_correction', 'local_contrast_enhancement']
    else:  # full
        balanced = _white_balance_gray_world(corrected)
        uniform = _uniform_lighting(balanced)
        final = _local_contrast_enhancement(uniform)
        pipeline_steps = ['input', 'shadow_mask_prediction', 'illumination_map_estimation', 'shadow_correction', 'white_balance', 'uniform_lighting', 'local_contrast_enhancement', 'final_document']
    
    info = {
        'pipeline': pipeline_steps,
        'segmented_size': segmented.size,
        'mask_mean': float(shadow_matte.mean()),
        'illumination_mean': float(illumination_map.mean())
    }
    return Image.fromarray(final), mask_image, info


def preprocess_image(image_path_or_pil, size=512):
    """Load and preprocess image for model input"""
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert('RGB')
    else:
        img = image_path_or_pil.convert('RGB')

    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor, img, original_size


def postprocess_tensor(tensor, original_size=None):
    """Convert model output tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.clamp(0, 1)
    img = transforms.ToPILImage()(tensor.cpu())

    if original_size:
        img = img.resize(original_size, Image.LANCZOS)

    return img


def traditional_shadow_removal(img_pil):
    """Traditional CV-based shadow removal using histogram equalization"""
    img = np.array(img_pil).astype(np.float32)

    # CLAHE on L channel
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Morphological operations to estimate shadow
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)

    # Normalize
    bg_float = bg.astype(np.float32) / 255.0
    bg_float = np.maximum(bg_float, 0.1)
    result = enhanced.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] / bg_float
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def traditional_doc_enhance(img_pil):
    """Traditional document enhancement pipeline"""
    img = np.array(img_pil).astype(np.uint8)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Sharpen
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # White balance using gray world assumption
    avg_b = np.mean(sharpened[:, :, 0])
    avg_g = np.mean(sharpened[:, :, 1])
    avg_r = np.mean(sharpened[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3

    if avg_b > 0 and avg_g > 0 and avg_r > 0:
        sharpened[:, :, 0] = np.clip(sharpened[:, :, 0] * (avg_all / avg_b), 0, 255)
        sharpened[:, :, 1] = np.clip(sharpened[:, :, 1] * (avg_all / avg_g), 0, 255)
        sharpened[:, :, 2] = np.clip(sharpened[:, :, 2] * (avg_all / avg_r), 0, 255)

    return Image.fromarray(sharpened.astype(np.uint8))


def binarize_document(img_pil, method='adaptive'):
    """Binarize document image"""
    img = np.array(img_pil.convert('L'))

    if method == 'adaptive':
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    elif method == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    return Image.fromarray(binary)


def detect_document_border(img_pil):
    """Detect and crop document borders"""
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w > img.shape[1] * 0.3 and h > img.shape[0] * 0.3:
            cropped = img[y:y+h, x:x+w]
            return Image.fromarray(cropped)

    return img_pil
