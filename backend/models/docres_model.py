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


DOCRES_DIR = Path('/home/wahyu/DocRes')

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
