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
