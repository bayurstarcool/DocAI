"""SAM document corner detector -> prints JSON corners (tl,tr,br,bl) in ORIGINAL coords.
CPU only. Usage: python sam_detect_corners.py <input_image>
stdout last line: JSON {"corners":[[x,y]*4],"area":float}"""
import sys, json, cv2, numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

inp = sys.argv[1]
img_bgr = cv2.imread(inp)
if img_bgr is None:
    print(json.dumps({"error": "unreadable"})); sys.exit(1)
image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

sam = sam_model_registry['vit_b'](checkpoint='/home/wahyu/models/sam/sam_vit_b_01ec64.pth')
sam.to('cpu'); sam.eval()

def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1); r = np.zeros((4, 2), np.float32)
    r[0] = pts[np.argmin(s)]; r[2] = pts[np.argmax(s)]
    d = np.diff(pts, 1); r[1] = pts[np.argmin(d)]; r[3] = pts[np.argmax(d)]
    return r

def approx_doc_from_mask(mask):
    mask = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in [0.01, 0.015, 0.02, 0.03, 0.04, 0.06]:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(c)
    return cv2.boxPoints(rect).astype(np.float32)

predictor = SamPredictor(sam)
predictor.set_image(image)
box = np.array([3, max(0, int(0.02 * H)), W - 3, H - 3])
points = np.array([[W // 2, H // 2], [20, 20], [W - 20, 20]])
labels = np.array([1, 0, 0])
masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box, multimask_output=True)
best = masks[int(np.argmax(scores))]
pts = approx_doc_from_mask(best)

if pts is None or (best.sum() / (W * H) < 0.20):
    mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=0.85,
                                         stability_score_thresh=0.90, min_mask_region_area=5000)
    auto = mask_gen.generate(image)
    cand = [m for m in auto if 0.20 < m['area'] / (W * H) < 0.98]
    cand = sorted(cand, key=lambda m: (m['predicted_iou'] + m['stability_score'], m['area']), reverse=True)
    if cand:
        pts = approx_doc_from_mask(cand[0]['segmentation'])

if pts is None:
    print(json.dumps({"error": "no_document"})); sys.exit(0)

o = order_points(pts)
o[:, 0] = np.clip(o[:, 0], 0, W - 1)
o[:, 1] = np.clip(o[:, 1], 0, H - 1)
area = float(cv2.contourArea(o) / (W * H))
print(json.dumps({"corners": [[float(x), float(y)] for x, y in o], "area": round(area, 3)}))
