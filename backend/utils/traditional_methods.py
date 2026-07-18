"""
Advanced Traditional CV Methods for Document Processing
No deep learning required - pure OpenCV implementations
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def deskew_document(img_pil):
    """Detect and correct document skew/rotation"""
    img = np.array(img_pil.convert('L'))
    
    # Edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return img_pil, 0
    
    angles = []
    for line in lines:
        if len(line.shape) > 1:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 30:  # Only consider near-horizontal lines
            angles.append(angle)
    
    if not angles:
        return img_pil, 0
    
    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.5:
        return img_pil, 0
    
    # Rotate
    h, w = np.array(img_pil).shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(np.array(img_pil), M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(rotated), median_angle


def remove_color_cast(img_pil, strength=1.0):
    """Remove color cast (yellowing, blue tint, etc.)"""
    img = np.array(img_pil).astype(np.float32)
    
    # Gray World assumption
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3
    
    # Correct each channel
    result = img.copy()
    if avg_b > 0:
        result[:, :, 0] = img[:, :, 0] * (avg_all / avg_b)
    if avg_g > 0:
        result[:, :, 1] = img[:, :, 1] * (avg_all / avg_g)
    if avg_r > 0:
        result[:, :, 2] = img[:, :, 2] * (avg_all / avg_r)
    
    # Apply with strength
    result = np.clip(result * strength + img * (1 - strength), 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))


def adaptive_threshold_document(img_pil, block_size=11, C=2):
    """Smart adaptive thresholding for documents"""
    img = np.array(img_pil.convert('L'))
    
    # Pre-process: slight blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    
    # Clean up small noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(binary)


def enhance_text_sharpness(img_pil, strength=1.5):
    """Enhance text sharpness using unsharp masking"""
    img = np.array(img_pil).astype(np.float32)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    
    # Unsharp mask
    sharpened = img + strength * (img - blurred)
    sharpened = np.clip(sharpened, 0, 255)
    
    return Image.fromarray(sharpened.astype(np.uint8))


def remove_background_noise(img_pil):
    """Remove background noise from document"""
    img = np.array(img_pil)
    
    # Convert to grayscale for mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create clean background estimate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel)
    
    # Normalize
    bg_float = bg.astype(np.float32) / 255.0
    bg_float = np.maximum(bg_float, 0.3)
    
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] / bg_float
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def enhance_contrast_clahe(img_pil, clip_limit=2.0, grid_size=8):
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    img = np.array(img_pil)
    
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(result)


def full_document_cleanup(img_pil):
    """Full document cleanup pipeline"""
    # 1. Deskew
    result, angle = deskew_document(img_pil)
    
    # 2. Remove color cast
    result = remove_color_cast(result)
    
    # 3. Enhance contrast
    result = enhance_contrast_clahe(result)
    
    # 4. Remove background noise
    result = remove_background_noise(result)
    
    # 5. Sharpen text
    result = enhance_text_sharpness(result)
    
    return result, {
        'deskew_angle': angle,
        'steps': ['deskew', 'color_cast_removal', 'clahe', 'noise_removal', 'sharpen']
    }
