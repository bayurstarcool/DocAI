"""OpenCV classical document dewarping.
Strategy: contour detect → perspective transform.
For curved pages: optional text-line curvature correction.
"""
import cv2
import numpy as np
from PIL import Image

def detect_document_contour(img):
    """Find largest quadrilateral contour = document boundary."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    # Morphological close to connect text blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Sort by area, take largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    # Fallback: bounding rect of largest contour
    c = contours[0]
    x, y, w, h = cv2.boundingRect(c)
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def four_point_transform(img, pts):
    """Perspective warp to rectangle."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))

def deskew_by_text_lines(img):
    """Optional: correct slight skew using text line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect horizontal lines via morphology
    kernel_len = gray.shape[1] // 4
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    detect = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 2)
    lines = cv2.morphologyEx(detect, cv2.MORPH_OPEN, kernel, iterations=2)
    coords = np.column_stack(np.where(lines > 0))
    if len(coords) < 100:
        return img, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return img, 0.0
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

def opencv_dewarp(pil_image):
    """Full OpenCV dewarp pipeline: contour → perspective → deskew."""
    img = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()  # RGB→BGR
    orig_h, orig_w = img.shape[:2]

    # Step 1: Detect document boundary
    pts = detect_document_contour(img)
    if pts is not None:
        # Step 2: Perspective transform
        warped = four_point_transform(img, pts)
    else:
        # Fallback: just deskew
        warped = img.copy()

    # Step 3: Deskew by text lines
    deskewed, angle = deskew_by_text_lines(warped)

    result = deskewed[:, :, ::-1]  # BGR→RGB
    return Image.fromarray(result), pts, angle
