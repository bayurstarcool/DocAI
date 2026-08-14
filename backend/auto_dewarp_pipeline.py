"""Auto Dewarp pipeline v6: tilt → DocRes E2E, curl → GeoTr+DocRes."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from backend.auto_dewarp_detector import detect_distortion_type


def _pil_to_bgr(image):
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        return image
    return np.array(image)


def auto_dewarp(image, docres_task_infer, geotr_doc3d_dewarp):
    img_bgr = _pil_to_bgr(image)
    distortion_type, score = detect_distortion_type(img_bgr)

    if distortion_type == 'curl':
        result = geotr_doc3d_dewarp(image)
        result = docres_task_infer(result, 'deshadowing')
        result = docres_task_infer(result, 'appearance')
        pipeline = 'curl_geotr_docres'
    else:
        # Flat tilt/perspective: avoid OpenCV perspective here because it can crop too aggressively.
        # Original DocRes end2end preserves more context and matches prior best tilt result.
        result = docres_task_infer(image, 'end2end')
        pipeline = 'tilt_docres_end2end'

    return result, pipeline, score
