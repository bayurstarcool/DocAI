"""
DocRes Inference Wrapper
Calls original inference functions with proper device handling.
"""
import sys, os, cv2, numpy as np, torch, gc
from pathlib import Path
from PIL import Image
from torch.nn import functional as F

DOCRES_DIR = Path("/home/wahyu/DocRes")
sys.path.insert(0, str(DOCRES_DIR))
sys.path.insert(0, str(DOCRES_DIR / "data" / "MBD"))

import importlib.util
_docres_utils_spec = importlib.util.spec_from_file_location("docres_utils", str(DOCRES_DIR / "utils.py"))
docres_utils = importlib.util.module_from_spec(_docres_utils_spec)
_docres_utils_spec.loader.exec_module(docres_utils)
from models import restormer_arch
_convert_state_dict_spec = importlib.util.spec_from_file_location("convert_mod", str(DOCRES_DIR / "utils.py"))
_convert_mod = importlib.util.module_from_spec(_convert_state_dict_spec)
_convert_state_dict_spec.loader.exec_module(_convert_mod)
convert_state_dict = _convert_mod.convert_state_dict
from data.preprocess.crop_merge_image import stride_integral

_model = None
_device = None
_mbd_loaded = False

def load_mbd():
    global _mbd_loaded
    if not _mbd_loaded:
        from infer import net1_net2_infer_single_im
        import importlib
        globals()["net1_net2_infer_single_im"] = net1_net2_infer_single_im
        _mbd_loaded = True

def ensure_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = restormer_arch.Restormer(
            inp_channels=6, out_channels=3, dim=48,
            num_blocks=[2,3,3,4], num_refinement_blocks=4,
            heads=[1,2,4,8], ffn_expansion_factor=2.66,
            bias=False, LayerNorm_type="WithBias", dual_pixel_task=True
        )
        state = convert_state_dict(torch.load(
            str(DOCRES_DIR / "checkpoints" / "docres.pkl"), map_location="cpu"
        )["model_state"])
        model.load_state_dict(state)
        model.eval().to(_device)
        _model = model
    return _model, _device


def run_task(img_bgr, task):
    """Run inference using same logic as inference.py but with explicit device."""
    model, device = ensure_model()
    orig_h, orig_w = img_bgr.shape[:2]
    
    if task == "dewarping":
        load_mbd()
        im_masked, prompt_org = dewarp_prompt(img_bgr)
        INPUT_SIZE = 256
        im_input = cv2.resize(im_masked, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        prompt = torch.from_numpy(prompt_org.transpose(2,0,1)).unsqueeze(0).float().to(device)
        im_t = torch.from_numpy(im_input.transpose(2,0,1)).unsqueeze(0).float().to(device)
        in_im = torch.cat([im_t, prompt], dim=1)
        
        base_coord = docres_utils.getBasecoord(INPUT_SIZE, INPUT_SIZE) / INPUT_SIZE
        with torch.no_grad():
            pred = model(in_im.float())
            flow = pred[0,:2].cpu().numpy().transpose(1,2,0).astype(np.float32) + base_coord
            for _ in range(15):
                flow = cv2.blur(flow, (3,3), borderType=cv2.BORDER_REPLICATE)
            flow = cv2.resize(flow, (orig_w, orig_h)).astype(np.float32)
            flow[:,:,0] *= orig_w; flow[:,:,1] *= orig_h
            np.clip(flow[:,:,0], 0, orig_w-1, out=flow[:,:,0])
            np.clip(flow[:,:,1], 0, orig_h-1, out=flow[:,:,1])
            result = cv2.remap(img_bgr, flow[:,:,0], flow[:,:,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        release_mbd()
        
    elif task == "deshadowing":
        MAX_SIZE = 1600
        prompt = deshadow_prompt_func(img_bgr)
        in_im = np.concatenate([img_bgr, prompt], axis=-1)
        if max(orig_w, orig_h) < MAX_SIZE:
            in_im, pad_h, pad_w = stride_integral(in_im, 8)
        else:
            in_im = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))
        in_im = torch.from_numpy((in_im / 255.0).transpose(2,0,1)).unsqueeze(0).half().to(device)
        model_h = model.half()
        with torch.no_grad():
            pred = model_h(in_im)
            pred = torch.clamp(pred, 0, 1)[0].permute(1,2,0).cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            if max(orig_w, orig_h) < MAX_SIZE:
                result = pred[pad_h:, pad_w:]
            else:
                pred[pred==0] = 1
                sm = cv2.resize(img_bgr, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
                sm = cv2.resize(sm, (orig_w, orig_h))
                sm[sm==0] = 1e-5
                result = np.clip(img_bgr.astype(float) / sm, 0, 255).astype(np.uint8)
                
    elif task == "appearance":
        MAX_SIZE = 1600
        prompt = appearance_prompt_func(img_bgr)
        in_im = np.concatenate([img_bgr, prompt], axis=-1)
        if max(orig_w, orig_h) < MAX_SIZE:
            in_im, pad_h, pad_w = stride_integral(in_im, 8)
        else:
            in_im = cv2.resize(in_im, (MAX_SIZE, MAX_SIZE))
        in_im = torch.from_numpy((in_im / 255.0).transpose(2,0,1)).unsqueeze(0).half().to(device)
        model_h = model.half()
        with torch.no_grad():
            pred = model_h(in_im)
            pred = torch.clamp(pred, 0, 1)[0].permute(1,2,0).cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            if max(orig_w, orig_h) < MAX_SIZE:
                result = pred[pad_h:, pad_w:]
            else:
                pred[pred==0] = 1
                sm = cv2.resize(img_bgr, (MAX_SIZE, MAX_SIZE)).astype(float) / pred.astype(float)
                sm = cv2.resize(sm, (orig_w, orig_h))
                sm[sm==0] = 1e-5
                result = np.clip(img_bgr.astype(float) / sm, 0, 255).astype(np.uint8)
                
    elif task == "deblurring":
        in_im, pad_h, pad_w = stride_integral(img_bgr, 8)
        prompt = deblur_prompt_func(in_im)
        in_im = np.concatenate([in_im, prompt], axis=-1)
        in_im = torch.from_numpy((in_im / 255.0).transpose(2,0,1)).unsqueeze(0).half().to(device)
        model_h = model.half()
        with torch.no_grad():
            pred = model_h(in_im)
            pred = torch.clamp(pred, 0, 1)[0].permute(1,2,0).cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            result = pred[pad_h:, pad_w:]
    
    elif task == "end2end":
        # Dewarp -> deshadow -> appearance
        result = run_task(img_bgr, "dewarping")
        release_mbd()
        result = run_task(result, "deshadowing")
        result = run_task(result, "appearance")
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return result


def dewarp_prompt(img):
    load_mbd()
    mask = net1_net2_infer_single_im(img, str(DOCRES_DIR / "data" / "MBD" / "checkpoint" / "mbd.pkl"))
    base_coord = docres_utils.getBasecoord(256, 256) / 256
    img_masked = img.copy(); img_masked[mask == 0] = 0
    mask_r = cv2.resize(mask, (256, 256)) / 255.0
    prompt = np.concatenate([base_coord, np.expand_dims(mask_r, -1)], axis=-1)
    return img_masked, prompt.astype(np.float32)

def release_mbd():
    global _mbd_loaded
    for name in list(globals().keys()):
        if name == "net1_net2_infer_single_im":
            del globals()[name]
    _mbd_loaded = False
    gc.collect()
    torch.cuda.empty_cache()

def deshadow_prompt_func(img):
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (1024, 1024))
    bg = []
    for plane in cv2.split(img_r):
        d = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg.append(cv2.medianBlur(d, 21))
    return cv2.resize(cv2.merge(bg), (w, h))

def appearance_prompt_func(img):
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (1024, 1024))
    result = []
    for plane in cv2.split(img_r):
        d = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(d, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result.append(norm)
    return cv2.resize(cv2.merge(result), (w, h))

def deblur_prompt_func(img):
    h, w = img.shape[:2]
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    hf = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
    hf = cv2.cvtColor(hf, cv2.COLOR_BGR2GRAY)
    hf = cv2.cvtColor(hf, cv2.COLOR_GRAY2BGR)
    return hf


def docres_infer(img, task="deshadowing"):
    if isinstance(img, Image.Image):
        img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    elif isinstance(img, np.ndarray):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
    else:
        raise ValueError(f"Unsupported: {type(img)}")
    
    result_bgr = run_task(img_bgr, task)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
