from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from src.model import DocEnhancerNet
from src.enhance import magic_document_enhance, adaptive_binarize
from src.utils import imread_rgb, imwrite_rgb, pad_to_multiple, unpad

def to_tensor(img): return torch.from_numpy(img.transpose(2,0,1)).float()[None]/255.
def to_img(t): return (t[0].clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)

def load_model(path, device='cuda', base=48):
    if str(device).startswith('cuda') and getattr(torch.version,'hip',None):
        torch.backends.cudnn.enabled=False
    ck=torch.load(path,map_location='cpu',weights_only=False)
    base=ck.get('args',{}).get('base',base) if isinstance(ck,dict) else base
    m=DocEnhancerNet(base).to(device).eval(); m.load_state_dict(ck.get('model',ck)); return m

def run_ai(img, model, device):
    padded,hw=pad_to_multiple(img,32)
    x=to_tensor(padded).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.startswith('cuda')):
        y,mask=model(x)
    out=unpad(to_img(y),hw); m=unpad((mask[0,0].clamp(0,1).cpu().numpy()*255).astype(np.uint8),hw)
    return out,m

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--weights',default='runs/doc_enhancer/best.pth'); p.add_argument('--no-ai',action='store_true')
    p.add_argument('--mode',choices=['magic','gray','bw','ai','ai_magic','ocr'],default='ai_magic')
    p.add_argument('--save-mask'); p.add_argument('--save-json'); p.add_argument('--device',default='cuda')
    a=p.parse_args(); img=imread_rgb(a.input); mask=None; enhanced=img
    if not a.no_ai and a.mode in ['ai','ai_magic','ocr'] and Path(a.weights).exists():
        dev=a.device if torch.cuda.is_available() else 'cpu'; model=load_model(a.weights,dev); enhanced,mask=run_ai(img,model,dev)
    elif a.mode=='ai':
        print('WARNING: weights not found, fallback to magic filter')
    if a.mode in ['magic','ai_magic','ocr'] or a.no_ai:
        enhanced=magic_document_enhance(enhanced)
    if a.mode=='bw': enhanced=adaptive_binarize(enhanced)
    imwrite_rgb(a.output, enhanced)
    if a.save_mask and mask is not None: cv2.imwrite(a.save_mask,mask)
    if a.save_json:
        meta={'input':a.input,'output':a.output,'ai_used':mask is not None,'shadow_score':float(mask.mean()/255) if mask is not None else None,'mode':a.mode}
        Path(a.save_json).write_text(json.dumps(meta,indent=2))
if __name__=='__main__': main()
