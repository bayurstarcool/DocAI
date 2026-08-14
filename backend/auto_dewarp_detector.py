"""Auto Dewarp Detector v8: perspective vs fold distinction.
Key insight:
  - low std + high range = perspective effect (tilt)
  - high std + high range = actual fold/curve (curl)
"""
import cv2
import numpy as np


def detect_distortion_type(image_bgr):
    if image_bgr is None:
        return 'curl', 50

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=w // 8, maxLineGap=15)

    angles = []
    if lines is not None and len(lines) > 5:
        for l in lines:
            coords = l.reshape(-1)
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(a) < 45:
                    angles.append(a)

    if not angles:
        return 'curl', 30

    median_angle = abs(np.median(angles))
    angle_std = np.std(angles)
    angle_range = max(angles) - min(angles)

    # --- TILT: flat doc, just rotated ---
    # Low std = lines are parallel = flat doc with perspective
    # High median = clear rotation angle
    is_perspective = angle_std < 3 and median_angle > 1
    is_consistent = angle_std < 2.5

    # --- CURL: actual fold/curve ---
    # High std + broad angle range = actual warp.
    # Border/noise lines can push std above 3 on flat pages, so require stronger evidence.
    is_curved = angle_std > 4.2 and angle_range > 12

    # --- Decision ---
    if is_perspective and not is_curved:
        # Flat doc with perspective rotation → tilt
        return 'tilt', round(float(min(100, median_angle * 30 + (3 - angle_std) * 10)), 1)
    elif is_curved:
        # Lines not parallel = actual fold/curve → curl
        curl_score = min(100, angle_std * 15 + angle_range * 3)
        return 'curl', round(float(curl_score), 1)
    elif median_angle > 1:
        # Some rotation, unclear → default tilt
        return 'tilt', round(float(min(100, median_angle * 25)), 1)
    else:
        # Mostly straight/level text lines. Treat as flat tilt/perspective; avoid unnecessary GeoTr warp.
        return 'tilt', round(float(min(100, max(30, 100 - angle_std * 20))), 1)
