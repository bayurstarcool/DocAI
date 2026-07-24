"""
Shadow removal using StackOverflow technique (score 87)
dilate -> medianBlur -> absdiff -> normalize per RGB channel
Ref: https://stackoverflow.com/questions/44752240
"""
import cv2
import numpy as np


def so_shadow_removal(img_rgb, kernel_size=7, blur_radius=21):
    """
    Remove shadow using SO technique.
    
    Args:
        img_rgb: numpy array (H, W, 3) RGB
        kernel_size: dilate kernel size (default 7)
        blur_radius: medianBlur radius (default 21, must be odd)
    
    Returns:
        result: shadow-removed RGB image
        result_norm: normalized version
    """
    if blur_radius % 2 == 0:
        blur_radius += 1
    
    rgb_planes = cv2.split(img_rgb)
    result_planes = []
    result_norm_planes = []
    
    for plane in rgb_planes:
        # Dilate to expand bright areas (estimate background)
        dilated_img = cv2.dilate(plane, np.ones((kernel_size, kernel_size), np.uint8))
        # Median blur to smooth the background estimate
        bg_img = cv2.medianBlur(dilated_img, blur_radius)
        # Difference: text = original - background
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        # Normalize to full range
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    return result, result_norm


def so_shadow_removal_enhanced(img_rgb, kernel_size=7, blur_radius=21,
                                clahe_clip=2.0, sharpen_strength=0.3):
    """
    Enhanced SO shadow removal with post-processing.
    
    Steps:
    1. SO shadow removal (dilate->medianBlur->absdiff)
    2. CLAHE contrast enhancement
    3. Gentle text sharpening
    4. White balance
    """
    # Step 1: SO shadow removal
    result, result_norm = so_shadow_removal(img_rgb, kernel_size, blur_radius)
    
    # Step 2: CLAHE on normalized version
    lab = cv2.cvtColor(result_norm, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Step 3: Gentle text sharpening
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    detail = enhanced.astype(np.float32) - blurred.astype(np.float32)
    sharpened = enhanced.astype(np.float32) + detail * sharpen_strength
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Step 4: White balance (gray world)
    img_float = sharpened.astype(np.float32) + 1e-6
    means = img_float.reshape(-1, 3).mean(axis=0)
    target = max(means.mean(), 130.0)
    gains = target / np.maximum(means, 1.0)
    balanced = np.clip(img_float * gains, 0, 255).astype(np.uint8)
    
    return balanced


def so_shadow_removal_aggressive(img_rgb, kernel_size=11, blur_radius=31):
    """
    Aggressive shadow removal for heavy shadows.
    Larger kernel and blur radius for stronger effect.
    """
    return so_shadow_removal_enhanced(
        img_rgb, 
        kernel_size=kernel_size, 
        blur_radius=blur_radius,
        clahe_clip=3.0,
        sharpen_strength=0.4
    )
