from __future__ import annotations
import io, os, base64, secrets, time
from pathlib import Path
import cv2, numpy as np, torch
import psutil
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import Response, JSONResponse, FileResponse
from PIL import Image
from src.model import DocEnhancerNet
from src.enhance import magic_document_enhance, adaptive_binarize, ai_shadow_postprocess
from src.utils import pad_to_multiple, unpad
from api.train_manager import train_job

app=FastAPI(title='Doc Shadow AI', version='2.0')
STATIC_DIR=Path(__file__).parent/'static'
MODEL_PATH=os.getenv('MODEL_PATH','runs/prod_v1/best.pth'); DEVICE='cuda' if torch.cuda.is_available() else 'cpu'; model=None; current_model_path=None
if DEVICE=='cuda' and getattr(torch.version,'hip',None):
    torch.backends.cudnn.enabled=False
security=HTTPBasic()
ADMIN_USER=os.getenv('ADMIN_USER')
ADMIN_PASSWORD=os.getenv('ADMIN_PASSWORD')

def require_admin(credentials:HTTPBasicCredentials=Depends(security)):
    if not ADMIN_USER or not ADMIN_PASSWORD:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='admin auth not configured')
    ok_user=secrets.compare_digest(credentials.username, ADMIN_USER)
    ok_pass=secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (ok_user and ok_pass):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='invalid credentials', headers={'WWW-Authenticate':'Basic'})
    return True

def load_once():
    global model, current_model_path
    if model is None or current_model_path != MODEL_PATH:
        if not Path(MODEL_PATH).exists(): return None
        ck=torch.load(MODEL_PATH,map_location='cpu',weights_only=False)
        base=ck.get('args',{}).get('base',48) if isinstance(ck,dict) else 48
        model=DocEnhancerNet(base).to(DEVICE).eval(); model.load_state_dict(ck.get('model',ck))
        current_model_path=MODEL_PATH
    return model

def read_upload(data):
    pil=Image.open(io.BytesIO(data)).convert('RGB')
    return np.array(pil)

def encode_jpg(img, q=95):
    ok,buf=cv2.imencode('.jpg',cv2.cvtColor(img,cv2.COLOR_RGB2BGR),[int(cv2.IMWRITE_JPEG_QUALITY),q])
    return buf.tobytes()

def ai_enhance(img):
    m=load_once()
    if m is None: return img, None, False
    padded,hw=pad_to_multiple(img,32); x=torch.from_numpy(padded.transpose(2,0,1)).float()[None].to(DEVICE)/255.
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=DEVICE=='cuda'):
        y,mask=m(x)
    out=unpad((y[0].clamp(0,1).cpu().numpy().transpose(1,2,0)*255).astype(np.uint8),hw)
    sm=unpad((mask[0,0].clamp(0,1).cpu().numpy()*255).astype(np.uint8),hw)
    out=ai_shadow_postprocess(img,out,sm)
    return out, sm, True

@app.get('/health')
def health(): return {'ok':True,'device':DEVICE,'model_loaded':Path(MODEL_PATH).exists(),'active_model':MODEL_PATH}

@app.get('/models', dependencies=[Depends(require_admin)])
def list_models():
    items=[]
    for p in sorted(Path('runs').rglob('*.pth')):
        try: items.append({'path':str(p),'size':p.stat().st_size,'mtime':p.stat().st_mtime})
        except OSError: pass
    items.sort(key=lambda x:x['mtime'], reverse=True)
    return {'active':MODEL_PATH,'models':items}

@app.post('/models/active', dependencies=[Depends(require_admin)])
async def set_active_model(path:str=Form(...)):
    global MODEL_PATH
    if not Path(path).exists(): raise HTTPException(status_code=404, detail='model not found')
    MODEL_PATH=path
    load_once()
    return {'ok':True,'active':MODEL_PATH}

def _gpu_busy_percent():
    for path in sorted(Path('/sys/class/drm').glob('card*/device/gpu_busy_percent')):
        try:
            return int(path.read_text().strip())
        except Exception:
            continue
    return None

@app.get('/system/status', dependencies=[Depends(require_admin)])
def system_status():
    mem=psutil.virtual_memory(); disk=psutil.disk_usage('/')
    status={
        'time':time.time(),
        'cpu':{'percent':psutil.cpu_percent(interval=0.05),'count':psutil.cpu_count()},
        'ram':{'total_gb':round(mem.total/1e9,2),'used_gb':round(mem.used/1e9,2),'available_gb':round(mem.available/1e9,2),'percent':mem.percent},
        'disk':{'total_gb':round(disk.total/1e9,2),'used_gb':round(disk.used/1e9,2),'free_gb':round(disk.free/1e9,2),'percent':disk.percent},
        'gpu':{'available':torch.cuda.is_available(),'count':torch.cuda.device_count() if torch.cuda.is_available() else 0,'busy_percent':_gpu_busy_percent(),'items':[]},
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free,total=torch.cuda.mem_get_info(i); used=total-free
            status['gpu']['items'].append({'index':i,'name':torch.cuda.get_device_name(i),'vram_total_gb':round(total/1e9,2),'vram_used_gb':round(used/1e9,2),'vram_free_gb':round(free/1e9,2),'vram_percent':round((used/total)*100,2) if total else 0})
    return status

@app.get('/train-ui', dependencies=[Depends(require_admin)])
@app.get('/train-ui/', dependencies=[Depends(require_admin)])
def train_ui(): return FileResponse(STATIC_DIR/'index.html')

@app.get('/studio', dependencies=[Depends(require_admin)])
@app.get('/studio/', dependencies=[Depends(require_admin)])
def studio_ui(): return FileResponse(STATIC_DIR/'studio.html')

@app.get('/samples', dependencies=[Depends(require_admin)])
def samples():
    root=Path('samples/studio')
    items=[]
    if root.exists():
        for path in sorted(root.glob('*')):
            if path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                items.append({'name':path.name,'url':f'/sample-file/{path.name}'})
    return {'samples':items}

@app.get('/sample-file/{name}', dependencies=[Depends(require_admin)])
def sample_file(name:str):
    path=Path('samples/studio')/Path(name).name
    if not path.exists():
        raise HTTPException(status_code=404, detail='sample not found')
    return FileResponse(path)

def _scan_dataset(base:Path):
    paired={'shadow':base/'shadow','clean':base/'clean','mask':base/'mask'}
    if all(p.is_dir() for p in paired.values()):
        names=sorted(p.name for p in paired['shadow'].glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        return 'paired', names
    if base.is_dir():
        names=sorted(p.name for p in base.glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
        return 'flat', names
    return None, []

@app.get('/dataset/list', dependencies=[Depends(require_admin)])
def dataset_list():
    roots=[Path('data/open_pairs'),Path('data/open_docs_clean'),Path('data/clean_docs'),Path('data/synth_pairs')]
    out=[]
    for base in roots:
        kind,names=_scan_dataset(base)
        if kind: out.append({'name':str(base),'kind':kind,'count':len(names)})
    return {'datasets':out}

@app.get('/dataset/items', dependencies=[Depends(require_admin)])
def dataset_items(name:str, offset:int=0, limit:int=24):
    base=Path(name)
    if '..' in base.parts or base.parts[0]!='data':
        raise HTTPException(status_code=400, detail='invalid dataset path')
    kind,names=_scan_dataset(base)
    if not kind:
        raise HTTPException(status_code=404, detail='dataset not found')
    page=names[offset:offset+limit]
    items=[]
    for n in page:
        if kind=='paired':
            items.append({'name':n,
                'shadow':f'/dataset/file?name={name}/shadow/{n}',
                'clean':f'/dataset/file?name={name}/clean/{n}',
                'mask':f'/dataset/file?name={name}/mask/{n}'})
        else:
            items.append({'name':n,'image':f'/dataset/file?name={name}/{n}'})
    return {'name':name,'kind':kind,'count':len(names),'offset':offset,'limit':limit,'items':items}

@app.get('/dataset/file', dependencies=[Depends(require_admin)])
def dataset_file(name:str):
    path=Path(name)
    if '..' in path.parts or path.parts[0]!='data' or not path.is_file():
        raise HTTPException(status_code=404, detail='file not found')
    return FileResponse(path)

@app.get('/dataset-ui', dependencies=[Depends(require_admin)])
@app.get('/dataset-ui/', dependencies=[Depends(require_admin)])
def dataset_ui(): return FileResponse(STATIC_DIR/'dataset.html')

@app.get('/train/status', dependencies=[Depends(require_admin)])
def train_status(): return train_job.snapshot()

@app.post('/train/start', dependencies=[Depends(require_admin)])
async def train_start(payload:dict):
    try:
        return train_job.start(payload)
    except RuntimeError as e:
        return JSONResponse({'error':str(e),'status':train_job.snapshot()}, status_code=409)

@app.post('/train/stop', dependencies=[Depends(require_admin)])
def train_stop(): return train_job.stop()

@app.post('/enhance', dependencies=[Depends(require_admin)])
async def enhance(file:UploadFile=File(...), mode:str=Form('ai_magic'), return_json:bool=Form(False)):
    img=read_upload(await file.read()); out=img; mask=None; used=False
    if mode in ['ai','ai_magic','ocr']:
        out,mask,used=ai_enhance(img)
    if mode in ['magic','ai_magic','ocr'] or not used:
        out=magic_document_enhance(out)
    if mode=='bw': out=adaptive_binarize(out)
    if return_json:
        mask_b64=None
        if mask is not None:
            ok,buf=cv2.imencode('.png',mask); mask_b64=base64.b64encode(buf).decode()
        return JSONResponse({'image_base64':base64.b64encode(encode_jpg(out)).decode(),'mask_base64':mask_b64,'ai_used':used,'shadow_score':float(mask.mean()/255) if mask is not None else None})
    return Response(encode_jpg(out), media_type='image/jpeg')
