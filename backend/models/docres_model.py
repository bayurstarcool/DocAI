"""
DocRes Restormer integration wrapper.
Loads pretrained DocRes model for shadow removal on documents.
Uses 6-channel input: image + deshadow prompt.
Supports multiple checkpoints: base (pretrained) and finetuned variants.
"""
import sys
import os
import time
import importlib.util
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import sys
from torch.nn import functional as F

DOCRES_DIR = Path('/home/wahyu/DocRes')

# Lazy MBD loader (import only when needed)
def _load_deeplab():
    sys.path.insert(0, str(DOCRES_DIR / 'data' / 'MBD'))
    from model.deep_lab_model.deeplab import DeepLab
    return DeepLab

_mbd_available = False
_mbd_deeplab_class = None
try:
    _mbd_deeplab_class = _load_deeplab()
    _mbd_available = True
except Exception:
    print('[WARN] MBD/Deeplab not available - dewarping disabled')



DOCRES_DIR = Path('/home/wahyu/DocRes')
sys.path.insert(0, str(DOCRES_DIR))

FINETUNE_CHECKPOINTS = {
    'finetune_v1': DOCRES_DIR / 'finetune_logs' / 'best.pth',
    'finetune_v2': DOCRES_DIR / 'finetune_logs' / 'deshadow_v2_20260726_065847' / 'best.pth',
    'finetune_v3': DOCRES_DIR / 'finetune_logs' / 'deshadow_v3_20260728_021703' / 'iter_45000.pth',
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_docres_utils = _load_module('docres_utils', DOCRES_DIR / 'utils.py')
_restormer_mod = _load_module('restormer_mod', DOCRES_DIR / 'models' / 'restormer_arch.py')
Restormer = _restormer_mod.Restormer


def smart_load_state_dict(model, checkpoint_path):
    """Load checkpoint with auto-detection of module. prefix."""
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt)
    keys = list(state.keys())
    if keys and all(k.startswith('module.') for k in keys):
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.load_state_dict(state)
    return model


def deshadow_prompt(img):
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (1024, 1024))
    bg_imgs = []
    for plane in cv2.split(img_r):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        bg_imgs.append(bg)
    return cv2.resize(cv2.merge(bg_imgs), (w, h))


def appearance_prompt(img):
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (1024, 1024))
    result_norm_planes = []
    for plane in cv2.split(img_r):
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm)
    result_norm = cv2.merge(result_norm_planes)
    result_norm = cv2.resize(result_norm, (w, h))
    return result_norm


def deblur_prompt(img):
    h, w = img.shape[:2]
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    hf = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    hf = cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)
    hf = cv2.resize(hf, (w, h))
    return hf


# MBD model singleton for dewarping
_mbd_model = None
_mbd_device = None

def _get_mbd_model(device='cuda'):
    global _mbd_model, _mbd_device
    if not _mbd_available:
        raise RuntimeError("MBD model (DeepLab) not available - torchvision missing or import failed")
    if _mbd_model is None or _mbd_device != device:
        if _mbd_deeplab_class is None:
            raise RuntimeError('MBD model not loaded')
        seg_model = _mbd_deeplab_class(num_classes=1, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
        seg_model = seg_model.to(device)
        ckpt = torch.load(str(DOCRES_DIR / 'data' / 'MBD' / 'checkpoint' / 'mbd.pkl'), map_location=device)
        seg_model.load_state_dict(ckpt['model_state'])
        seg_model.eval()
        _mbd_model = seg_model
        _mbd_device = device
        print(f"[OK] MBD model loaded, device={device}")
    return _mbd_model


def dewarp_prompt(img, device='cuda'):
    seg_model = _get_mbd_model(device)
    h_org, w_org = img.shape[:2]
    img_448 = cv2.resize(img, (448, 448))
    img_448 = cv2.GaussianBlur(img_448, (15, 15), 0, 0)
    img_448 = cv2.cvtColor(img_448, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_448.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        pred = seg_model(img_t)
        mask = pred[:, 0, :, :].unsqueeze(1)
        mask = F.interpolate(mask, (h_org, w_org), mode='bilinear')
        mask = mask.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask[mask > 100] = 255
        mask[mask < 100] = 0

    # Base coordinate grid for TPS
    base_coord = _docres_utils.getBasecoord(256, 256).numpy() / 256
    img_masked = img.copy()
    img_masked[mask == 0] = 0
    mask_r = cv2.resize(mask, (256, 256)) / 255.0
    prompt = np.concatenate([base_coord, np.expand_dims(mask_r, -1)], axis=-1)
    return img_masked, prompt


def _make_task_prompt(img_np, task, device='cuda'):
    if task == 'deshadowing':
        return deshadow_prompt(img_np), img_np
    elif task == 'appearance':
        return appearance_prompt(img_np), img_np
    elif task == 'deblurring':
        return deblur_prompt(img_np), img_np
    elif task == 'dewarping':
        im_masked, prompt = dewarp_prompt(img_np, device)
        return prompt, im_masked
    return deshadow_prompt(img_np), img_np


def build_model():
    return Restormer(
        inp_channels=6, out_channels=3, dim=48,
        num_blocks=[2, 3, 3, 4], num_refinement_blocks=4,
        heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
        bias=False, LayerNorm_type='WithBias', dual_pixel_task=True
    )


def load_pretrained(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = DOCRES_DIR / 'checkpoints' / 'docres.pkl'
    model = build_model()
    smart_load_state_dict(model, checkpoint_path)
    model.eval()
    return model





class DocResONNX:
    """ONNX-based DocRes inference (no PyTorch needed)."""
    
    def __init__(self, onnx_path=None, im_size=1280):
        if onnx_path is None:
            onnx_path = DOCRES_DIR / "checkpoints" / "docres_base.onnx"
        self.onnx_path = str(onnx_path)
        self.im_size = im_size
        self.session = None
        print(f"[OK] DocRes ONNX ready ({onnx_path})")
    
    def _ensure_loaded(self):
        if self.session is None:
            self.session = ort.InferenceSession(self.onnx_path)
    
    @staticmethod
    def _deshadow_prompt(img):
        h, w = img.shape[:2]
        work = cv2.resize(img, (1024, 1024))
        planes = cv2.split(work)
        bg = []
        for plane in planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg.append(cv2.medianBlur(dilated, 21))
        bg = cv2.merge(bg)
        return cv2.resize(bg, (w, h))
    
    def infer(self, img, im_size=None):
        self._ensure_loaded()
        
        if isinstance(img, Image.Image):
            img_np = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img_np = img
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported type: {type(img)}")
        
        res = im_size or self.im_size
        orig_h, orig_w = img_np.shape[:2]
        
        # Resize + prompt
        img_r = cv2.resize(img_np, (res, res))
        prompt = self._deshadow_prompt(img_r)
        
        # Normalize + concatenate
        img_n = img_r.astype(np.float32) / 255.0
        prompt_n = prompt.astype(np.float32) / 255.0
        in_6ch = np.concatenate([img_n, prompt_n], axis=-1)
        in_6ch = in_6ch.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        
        # Inference
        pred = self.session.run(None, {"input": in_6ch})[0]
        pred_np = (pred.squeeze(0).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        pred_full = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        return Image.fromarray(cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB))


class DocResModel:
    def __init__(self, checkpoint_path=None, device='cuda', im_size=1280, label='pretrained'):
        self.device = torch.device(device)
        self.im_size = im_size
        self.model = load_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)
        self.checkpoint_name = label
        print(f"[OK] DocRes loaded ({label}), im_size={im_size}, device={device}")

    @torch.inference_mode()
    def infer(self, img, im_size=None):
        if isinstance(img, Image.Image):
            img_np = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img_np = img
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported type: {type(img)}")

        res = im_size or self.im_size
        orig_h, orig_w = img_np.shape[:2]
        prompt = deshadow_prompt(img_np)

        img_r = cv2.resize(img_np, (res, res))
        prompt_r = cv2.resize(prompt, (res, res))
        img_n = img_r.astype(np.float32) / 255.0
        prompt_n = prompt_r.astype(np.float32) / 255.0

        in_6ch = torch.cat([
            torch.from_numpy(img_n.transpose(2, 0, 1)).unsqueeze(0),
            torch.from_numpy(prompt_n.transpose(2, 0, 1)).unsqueeze(0)
        ], dim=1).to(self.device).float()

        pred = self.model(in_6ch)
        pred_np = pred.squeeze(0).cpu().clamp(0, 1).numpy()
        pred_np = (pred_np.transpose(1, 2, 0) * 255).astype(np.uint8)
        pred_full = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB))

    def infer_task(self, img, task='deshadowing', im_size=None):
        """Infer with specific task prompt. task: deshadowing, appearance, deblurring, dewarping"""
        if isinstance(img, Image.Image):
            img_np = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            img_np = img
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            raise ValueError(f'Unsupported type: {type(img)}')

        res = im_size or self.im_size
        orig_h, orig_w = img_np.shape[:2]

        # Generate task-specific prompt
        if task == 'dewarping':
            prompt, img_input = _make_task_prompt(img_np, 'dewarping', device=str(self.device))
            # For dewarping, prompt is (256,256,3) and img_input is masked
            img_r = cv2.resize(img_input, (res, res))
            prompt_r = cv2.resize(prompt.astype(np.float32), (res, res))
        else:
            prompt, img_input = _make_task_prompt(img_np, task, device=str(self.device))
            img_r = cv2.resize(img_np, (res, res))
            prompt_r = cv2.resize(prompt.astype(np.float32), (res, res))

        img_n = img_r.astype(np.float32) / 255.0
        prompt_n = prompt_r.astype(np.float32) / 255.0

        in_6ch = torch.cat([
            torch.from_numpy(img_n.transpose(2, 0, 1)).unsqueeze(0),
            torch.from_numpy(prompt_n.transpose(2, 0, 1)).unsqueeze(0)
        ], dim=1).to(self.device).float()

        with torch.no_grad():
            pred = self.model(in_6ch)
            pred_np = pred.squeeze(0).cpu().clamp(0, 1).numpy()
            pred_np = (pred_np.transpose(1, 2, 0) * 255).astype(np.uint8)

            if task == 'dewarping':
                # Dewarping output needs remapping
                base_coord = _docres_utils.getBasecoord(res, res).numpy() / res
                pred_flow = pred_np[:, :, :2].astype(np.float32) + base_coord
                for _ in range(15):
                    pred_flow = cv2.blur(pred_flow, (3, 3), borderType=cv2.BORDER_REPLICATE)
                pred_flow = cv2.resize(pred_flow, (orig_w, orig_h)) * (orig_w, orig_h)
                orig_bgr = np.array(img.convert('RGB'))[:, :, ::-1] if isinstance(img, Image.Image) else img_np
                pred_full = cv2.remap(orig_bgr, pred_flow[:, :, 0].astype(np.float32),
                                      pred_flow[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
            else:
                pred_full = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB))

    def infer_dewarping(self, img, im_size=None):
        return self.infer_task(img, 'dewarping', im_size)

    def infer_appearance(self, img, im_size=None):
        return self.infer_task(img, 'appearance', im_size)

    def infer_deblurring(self, img, im_size=None):
        return self.infer_task(img, 'deblurring', im_size)

    def get_model_info(self):
        params = sum(p.numel() for p in self.model.parameters())
        return {
            'name': f'DocRes Restormer ({self.checkpoint_name})',
            'checkpoint': self.checkpoint_name,
            'im_size': self.im_size,
            'total_params': params,
        }


_instances = {}


def _get_or_create(key, checkpoint_path, device, label):
    if key not in _instances:
        try:
            _instances[key] = DocResModel(checkpoint_path, device=device, label=label)
        except Exception as e:
            print(f"[WARN] Failed to load DocRes ({label}): {e}")
            return None
    return _instances[key]


def get_docres(device='cuda', im_size=1280):
    return _get_or_create(
        'base', DOCRES_DIR / 'checkpoints' / 'docres.pkl', device, 'base'
    )


def get_docres_base(device='cuda', im_size=1280):
    return get_docres(device, im_size)


def get_docres_finetune(variant='finetune_v1', device='cuda', im_size=1280):
    cp = FINETUNE_CHECKPOINTS.get(variant)
    if cp is None:
        print(f"[WARN] Unknown finetune variant: {variant}")
        return None
    if not cp.exists():
        print(f"[WARN] Finetune checkpoint not found: {cp}")
        return None
    return _get_or_create(f'finetune_{variant}', cp, device, f'finetune_{variant}')
