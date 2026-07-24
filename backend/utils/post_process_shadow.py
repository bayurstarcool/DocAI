#!/usr/bin/env python3
"""Post-processing pipeline v2: preserve text thickness."""
import cv2
import numpy as np
import sys

def post_process(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        print(f"ERROR: Cannot read {input_path}")
        return False
    
    # Step 1: LAB for better color handling
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 2: Background detection — use gentler threshold
    # Use adaptive threshold to preserve thin lines
    l_float = l.astype(np.float32)
    
    # Smooth background estimation
    bg_est = cv2.GaussianBlur(l_float, (51, 51), 0)
    
    # Background mask: areas close to background estimate
    diff = np.abs(l_float - bg_est)
    bg_mask = (diff < 20).astype(np.uint8) * 255
    
    # Dilate to fill gaps but preserve text edges
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bg_mask = cv2.erode(bg_mask, kernel_small, iterations=1)
    
    # Step 3: For background areas → clean white
    # For text areas → keep original thickness, enhance contrast
    
    # Background: smooth white
    l_bg = cv2.GaussianBlur(l, (31, 31), 0)
    l_bg = np.clip(l_bg.astype(np.float32) * 0.8 + 50, 0, 255).astype(np.uint8)
    
    # Text: enhance without thinning
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_text = clahe.apply(l)
    
    # Merge using soft mask
    bg_region = bg_mask > 200
    text_region = ~bg_region
    
    l_enhanced = l.copy()
    l_enhanced[bg_region] = l_bg[bg_region]
    l_enhanced[text_region] = l_text[text_region]
    
    # Step 4: Morphological check — dilate text slightly to restore thickness
    text_mask = (l < 128).astype(np.uint8) * 255
    text_mask = cv2.dilate(text_mask, kernel_small, iterations=1)
    
    # Where text was detected but got thinned, restore from original
    restored = l.copy()
    restored[text_mask > 0] = np.minimum(l[text_mask > 0], l_enhanced[text_mask > 0])
    
    # Step 5: Gentle denoise only on background
    denoised = cv2.fastNlMeansDenoising(restored, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Step 6: Keep text from enhanced, background from denoised
    final_l = denoised.copy()
    final_l[text_mask > 0] = restored[text_mask > 0]
    
    # Step 7: Reconstruct
    lab_enhanced = cv2.merge([final_l, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Step 8: Gentle sharpen (less aggressive than v1)
    blurred = cv2.GaussianBlur(result, (0, 0), 2)
    result = cv2.addWeighted(result, 1.3, blurred, -0.3, 0)
    
    cv2.imwrite(output_path, result)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 post_process.py <input> [output]")
        sys.exit(1)
    
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace(".png", "_clean.png").replace(".jpg", "_clean.png")
    
    if post_process(inp, out):
        print(f"OK: {out}")
    else:
        print("FAILED")
