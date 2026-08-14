"""Control Points dewarping inference for DocAI.
Predicts 31x31 fiducial points, then remaps using linear interpolation.
Preserves original aspect ratio by resize-pad to 992x992 then final resize to original.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import griddata

BASE = Path('/home/wahyu/controlpoints/Source')
sys.path.insert(0, str(BASE))
from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IM_SIZE = 992

def _strip_module(sd):
    if sd and all(k.startswith('module.') for k in sd.keys()):
        return {k[7:]: v for k, v in sd.items()}
    return sd

def _load_model():
    global _model
    if _model is not None:
        return _model
    model = FiducialPoints(n_classes=2, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)
    ckpt_path = Path('/home/wahyu/controlpoints/checkpoints/controlpoints_inv3d_best.pth')
    if not ckpt_path.exists():
        ckpt_path = Path('/home/wahyu/controlpoints/checkpoints/controlpoints_inv3d_smoke.pth')
    if not ckpt_path.exists():
        ckpt_path = BASE / 'checkpoint_pretrained.pkl'
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    sd = ckpt.get('model_state', ckpt)
    model.load_state_dict(_strip_module(sd), strict=False)
    model.eval().to(_device)
    _model = model
    return model

def _resize_pad(img_rgb):
    h, w = img_rgb.shape[:2]
    scale = IM_SIZE / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((IM_SIZE, IM_SIZE, 3), 255, dtype=np.uint8)
    x0 = (IM_SIZE - nw) // 2
    y0 = (IM_SIZE - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, (x0, y0, nw, nh)

def _remap_by_points(img_rgb_992, pts_31, seg):
    # pts_31: [31,31,2] in 992 coord. Source/control points on warped image.
    pts = pts_31.astype(np.float32)
    segment_h = max(float(seg[1]) if len(seg) > 1 else 32.0, 1.0)
    segment_w = max(float(seg[0]) if len(seg) > 0 else 32.0, 1.0)
    # Use fixed grid if segment prediction bad
    if segment_h < 10 or segment_w < 10 or segment_h > 80 or segment_w > 80:
        segment_h = IM_SIZE / 30.0
        segment_w = IM_SIZE / 30.0
    rows, cols = pts.shape[:2]
    gx, gy = np.mgrid[0:cols-1:complex(cols), 0:rows-1:complex(rows)]
    target = np.stack((gx, gy), axis=2) * [segment_w, segment_h]
    source = pts.transpose(1,0,2).reshape(-1, 2)
    target = target.reshape(-1, 2)
    out_w = int(round(segment_w * (cols - 1)))
    out_h = int(round(segment_h * (rows - 1)))
    out_w = max(64, min(out_w, IM_SIZE))
    out_h = max(64, min(out_h, IM_SIZE))
    grid_y, grid_x = np.mgrid[0:out_h-1:complex(out_h), 0:out_w-1:complex(out_w)]
    grid = griddata(target, source, (grid_x, grid_y), method='linear')
    if np.isnan(grid).any():
        grid_nearest = griddata(target, source, (grid_x, grid_y), method='nearest')
        grid = np.where(np.isnan(grid), grid_nearest, grid)
    grid = grid.astype(np.float32)
    bgr = img_rgb_992[:, :, ::-1]
    out = cv2.remap(bgr, grid[:, :, 0], grid[:, :, 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out[:, :, ::-1]

def controlpoints_dewarp(pil_image):
    model = _load_model()
    img_rgb = np.array(pil_image.convert('RGB'))
    oh, ow = img_rgb.shape[:2]
    padded, (x0, y0, nw, nh) = _resize_pad(img_rgb)
    x = torch.from_numpy(padded.transpose(2,0,1)).float().unsqueeze(0).to(_device) / 255.0
    with torch.no_grad():
        pred_pts, pred_seg = model(x, is_softmax=False)
    pts = pred_pts[0].detach().cpu().numpy().transpose(1,2,0)
    seg = pred_seg[0].detach().cpu().numpy()
    out = _remap_by_points(padded, pts, seg)
    # crop around original aspect from output center, resize to original
    h, w = out.shape[:2]
    crop_w = min(w, int(round(h * ow / oh)))
    crop_h = min(h, int(round(w * oh / ow)))
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_w // 2); y1 = max(0, cy - crop_h // 2)
    crop = out[y1:y1+crop_h, x1:x1+crop_w]
    final = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(final)
