"""
DocRes Restormer integration wrapper.
Loads pretrained DocRes model for shadow removal on documents.
Uses 6-channel input: image + deshadow prompt.
"""
import sys
import os
import time
import importlib.util
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image


DOCRES_DIR = Path('/home/wahyu/DocRes')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_docres_utils = _load_module('docres_utils', DOCRES_DIR / 'utils.py')
convert_state_dict = _docres_utils.convert_state_dict
_restormer_mod = _load_module('restormer_mod', DOCRES_DIR / 'models' / 'restormer_arch.py')
Restormer = _restormer_mod.Restormer


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


def load_pretrained(checkpoint_path='checkpoints/docres.pkl'):
    if checkpoint_path is None:
        checkpoint_path = DOCRES_DIR / 'checkpoints' / 'docres.pkl'
    model = build_model()
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    state = ckpt['model_state']
    state = convert_state_dict(state)
    model.load_state_dict(state)
    model.eval()
    return model


class DocResModel:
    def __init__(self, checkpoint_path=None, device='cuda', im_size=1280):
        self.device = torch.device(device)
        self.im_size = im_size
        self.model = load_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)
        self.checkpoint_name = 'pretrained'
        print(f"[OK] DocRes model loaded (pretrained), im_size={im_size}")

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
            'name': 'DocRes Restormer',
            'checkpoint': self.checkpoint_name,
            'im_size': self.im_size,
            'total_params': params,
        }


_docres_instance = None


def get_docres(device='cuda', im_size=1280):
    global _docres_instance
    if _docres_instance is None:
        _docres_instance = DocResModel(device=device, im_size=im_size)
    return _docres_instance
