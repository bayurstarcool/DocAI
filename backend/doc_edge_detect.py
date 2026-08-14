"""Document edge detector for DocAI crop.
Returns 4 corner points (tl,tr,br,bl) in ORIGINAL image coords.
Strategy:
- downscale, multi-mask candidate quads (Canny fixed/auto, OTSU +/-)
- score by rectangularity + area; REJECT frame-hugging quads (photo border)
- line-fit refine corners on full-res edges
- if no confident inset quad -> is_full=True (already-cropped scan; do not force crop)
"""
import cv2
import numpy as np


def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    r = np.zeros((4, 2), np.float32)
    r[0] = pts[np.argmin(s)]
    r[2] = pts[np.argmax(s)]
    d = np.diff(pts, 1)
    r[1] = pts[np.argmin(d)]
    r[3] = pts[np.argmax(d)]
    return r


def _is_frame(pts, sw, sh, band=0.012):
    bx = sw * band
    by = sh * band
    n = 0
    for x, y in pts.reshape(4, 2):
        if x < bx or x > sw - bx or y < by or y > sh - by:
            n += 1
    return n >= 3


def _score_quad(pts, sw, sh):
    area = cv2.contourArea(pts)
    ar = area / (sw * sh)
    if ar < 0.20 or ar > 0.985:
        return -1, ar
    if _is_frame(pts, sw, sh):
        return -1, ar
    o = order_points(pts)
    w1 = np.linalg.norm(o[1] - o[0])
    w2 = np.linalg.norm(o[2] - o[3])
    h1 = np.linalg.norm(o[3] - o[0])
    h2 = np.linalg.norm(o[2] - o[1])
    if min(w1, w2, h1, h2) < 10:
        return -1, ar
    rect = 1 - (abs(w1 - w2) / max(w1, w2) + abs(h1 - h2) / max(h1, h2)) / 2
    if rect < 0.55:
        return -1, ar
    return 0.55 * rect + 0.45 * min(ar / 0.80, 1.0), ar


def _candidates(small):
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gb = cv2.GaussianBlur(g, (5, 5), 0)
    masks = [cv2.Canny(gb, 75, 200)]
    v = np.median(gb)
    masks.append(cv2.Canny(gb, int(max(0, 0.67 * v)), int(min(255, 1.33 * v))))
    _, ot = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks += [ot, 255 - ot]
    quads = []
    for m in masks:
        mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
            cnts, _ = cv2.findContours(mc, mode, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
                peri = cv2.arcLength(c, True)
                if peri < 80:
                    continue
                for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
                    ap = cv2.approxPolyDP(c, eps * peri, True)
                    if len(ap) == 4 and cv2.isContourConvex(ap):
                        quads.append(ap.reshape(4, 2).astype(np.float32))
                        break
    return quads


def _fit_line_near(edge, p1, p2, band):
    H, W = edge.shape
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1:
        return None
    d /= L
    n = np.array([-d[1], d[0]])
    xmin = int(max(0, min(p1[0], p2[0]) - band))
    xmax = int(min(W, max(p1[0], p2[0]) + band))
    ymin = int(max(0, min(p1[1], p2[1]) - band))
    ymax = int(min(H, max(p1[1], p2[1]) + band))
    sub = edge[ymin:ymax, xmin:xmax]
    ys, xs = np.where(sub > 0)
    if len(xs) < 10:
        return None
    xs = xs + xmin
    ys = ys + ymin
    pts = np.stack([xs, ys], 1).astype(float)
    rel = pts - p1
    perp = np.abs(rel @ n)
    along = rel @ d
    keep = (perp < band) & (along > -band) & (along < L + band)
    pts = pts[keep]
    if len(pts) < 10:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    a = vy
    b = -vx
    c = -(a * x0 + b * y0)
    return (a, b, c)


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return np.array([x, y], np.float32)


def _refine(img, quad):
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    v = np.median(g)
    edge = cv2.Canny(g, int(max(0, 0.67 * v)), int(min(255, 1.33 * v)))
    edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), 1)
    tl, tr, br, bl = quad
    band = max(15, int(0.03 * max(W, H)))
    top = _fit_line_near(edge, tl, tr, band)
    right = _fit_line_near(edge, tr, br, band)
    bot = _fit_line_near(edge, br, bl, band)
    left = _fit_line_near(edge, bl, tl, band)
    if None in (top, right, bot, left):
        return quad, False
    ntl = _intersect(left, top)
    ntr = _intersect(top, right)
    nbr = _intersect(right, bot)
    nbl = _intersect(bot, left)
    if any(p is None for p in (ntl, ntr, nbr, nbl)):
        return quad, False
    nq = np.array([ntl, ntr, nbr, nbl], np.float32)
    tol = 0.08 * max(W, H)
    if np.max(np.linalg.norm(nq - quad, axis=1)) > tol:
        return quad, False
    nq[:, 0] = np.clip(nq[:, 0], 0, W - 1)
    nq[:, 1] = np.clip(nq[:, 1], 0, H - 1)
    return nq, True


def detect_corners(img_bgr):
    """Return (corners_4x2_float or None, confidence_float, is_full_bool).
    corners order: tl, tr, br, bl in ORIGINAL image coords."""
    H, W = img_bgr.shape[:2]
    ratio = max(H, W) / 700.0
    small = cv2.resize(img_bgr, (int(W / ratio), int(H / ratio))) if ratio > 1 else img_bgr.copy()
    sh, sw = small.shape[:2]
    best = None
    best_s = -1
    for q in _candidates(small):
        s, ar = _score_quad(q, sw, sh)
        if s > best_s:
            best_s = s
            best = q
    if best is None or best_s < 0.55:
        return None, float(max(best_s, 0.0)), True
    q = (order_points(best) * ratio).astype(np.float32)
    q, _ok = _refine(img_bgr, q)
    return q, float(best_s), False


def _approx_quad(mask):
    m = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in [0.015, 0.02, 0.03, 0.04, 0.06]:
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            return ap.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def detect_corners_grabcut(img_bgr):
    """GrabCut-based detector. Better for tilted document photos on textured bg.
    Returns (corners tl,tr,br,bl or None, confidence, is_full)."""
    H, W = img_bgr.shape[:2]
    ratio = max(H, W) / 900.0
    small = cv2.resize(img_bgr, (int(W / ratio), int(H / ratio))) if ratio > 1 else img_bgr.copy()
    sh, sw = small.shape[:2]
    mask = np.zeros((sh, sw), np.uint8)
    rect = (int(0.04 * sw), int(0.04 * sh), int(0.92 * sw), int(0.92 * sh))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None, 0.0, True
    m2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=1)
    q = _approx_quad(m2)
    if q is None:
        return None, 0.0, True
    o = order_points(q)
    ar = cv2.contourArea(o) / (sw * sh)
    # frame-hug or near-full -> already cropped; too small -> unreliable
    if ar > 0.93 or _is_frame(o, sw, sh) or ar < 0.20:
        return None, float(ar), True
    return (o * ratio).astype(np.float32), float(ar), False


def _tilt_of(corners):
    """Max deviation (deg) of the 4 edges from horizontal/vertical axes."""
    o = order_points(corners)
    tl, tr, br, bl = o
    import math
    def ang(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    top = abs(ang(tl, tr))            # ~0 if level
    bottom = abs(ang(bl, br))
    left = abs(abs(ang(tl, bl)) - 90) # ~0 if vertical
    right = abs(abs(ang(tr, br)) - 90)
    return max(top, bottom, left, right)


def detect_corners_auto(img_bgr):
    """Auto: run opencv first and RESPECT its is_full signal (already-cropped /
    axis-aligned scans stay full). Only consult grabcut to improve an opencv crop
    that exists but is low-confidence or clearly tilted.
    Returns (corners, confidence, is_full, chosen_method)."""
    oc_q, oc_conf, oc_full = detect_corners(img_bgr)
    # opencv found no confident inset quad -> already cropped / undetectable.
    # Trust it; do NOT let grabcut force a crop into an already-cropped doc.
    if oc_q is None or oc_full:
        return None, float(oc_conf), True, "opencv"
    # opencv crop is solid and roughly axis-aligned -> fast path
    if oc_conf >= 0.75 and _tilt_of(oc_q) < 8.0:
        return oc_q, oc_conf, False, "opencv"
    # opencv crop is poor/tilted -> consult grabcut, keep the better usable crop
    gc_q, gc_conf, gc_full = detect_corners_grabcut(img_bgr)
    candidates = [(oc_conf, oc_q, "opencv")]
    if gc_q is not None and not gc_full:
        candidates.append((gc_conf * 0.9, gc_q, "grabcut"))
    candidates.sort(key=lambda x: x[0], reverse=True)
    conf, q, method = candidates[0]
    return q, float(conf), False, method
