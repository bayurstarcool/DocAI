"""
DocAI - Full-Featured Web Application
FastAPI backend for document restoration AI with multiple processing modes.
"""
import io
import json
import os
import signal
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

import time
from datetime import datetime, timedelta, timezone
import numpy as np
import cv2
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.image_utils import run_document_restoration_pipeline, run_tiled_restoration
from backend.models.document_restorer import DocumentRestorerNet
from backend.auth import ADMIN_CREDENTIALS, create_token, is_authenticated
from backend.workspace import WorkspaceError, resolve_workspace_path, safe_workspace_name
from backend.utils.traditional_methods import (
    deskew_document, remove_color_cast, adaptive_threshold_document,
    enhance_text_sharpness, remove_background_noise, enhance_contrast_clahe,
    full_document_cleanup,
)
from backend.utils.shadowremove_enhance import (
    magic_document_enhance, adaptive_binarize, ai_shadow_postprocess,
)
from backend.models.shadow_remover import ShadowRemoverNet
from backend.models.doc_enhancer import DocEnhancerNet

# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
EVALUATION_DIR = BASE_DIR / 'evaluation'
MODEL_CHECKPOINT_PATH = CHECKPOINT_DIR / 'document_restorer' / 'best.pth'

RESTORATION_TILE_SIZE = 1024
RESTORATION_TILE_OVERLAP = 96
IMAGE_TEST_ROOT = BASE_DIR / 'data' / 'test'
DOWNLOAD_ROOT = BASE_DIR / 'datasets' / 'ShadowDocument7K'
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

# --- Global Objects ---
app = FastAPI(title="DocAI")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- Model Loading ---
def _pil_to_b64(img: Image.Image) -> str:
    import base64
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _load_checkpoint(model_cls, checkpoint_path, map_location, **kwargs):
    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found: {checkpoint_path}")
        return None
    try:
        model = model_cls(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        strict = model_cls is DocumentRestorerNet
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"  [INFO] Missing keys (initialized randomly): {len(missing)}")
        if unexpected:
            print(f"  [WARN] Unexpected keys ignored: {len(unexpected)}")
        model = model.to(map_location)
        model.eval()
        print(f"[OK] Loaded {model_cls.__name__} from {checkpoint_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load {model_cls.__name__}: {e}")
        return None


doc_restorer_model = _load_checkpoint(DocumentRestorerNet, MODEL_CHECKPOINT_PATH, device)
doc_restorer_checkpoint = str(MODEL_CHECKPOINT_PATH.relative_to(BASE_DIR)) if doc_restorer_model is not None else None
shadow_remover_model = _load_checkpoint(ShadowRemoverNet, CHECKPOINT_DIR / 'shadow_remover' / 'best.pth', device)
doc_enhancer_model = _load_checkpoint(DocEnhancerNet, CHECKPOINT_DIR / 'doc_enhancer' / 'best.pth', device)

# Lazy-load DocShadow SD7K
_docshadow = None
def get_docshadow():
    global _docshadow
    if _docshadow is None:
        try:
            from backend.models.docshadow_sd7k_model import get_docshadow_model
            _docshadow = get_docshadow_model(device=str(device))
        except Exception as e:
            print(f"[WARN] DocShadow SD7K not available: {e}")
    return _docshadow


# --- FastAPI Setup ---
spa_dist = BASE_DIR / 'frontend' / 'dist'
login_html = BASE_DIR / 'frontend' / 'templates_bak' / 'login.html'
app.mount("/assets", StaticFiles(directory=str(spa_dist / 'assets')), name="spa-assets")
fonts_dir = BASE_DIR / 'frontend' / 'public' / 'fonts'
app.mount("/fonts", StaticFiles(directory=str(fonts_dir)), name="local-fonts")


def require_page_auth(request: Request):
    if not is_authenticated(request):
        return RedirectResponse('/login', status_code=303)
    return None


def require_api_auth(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail='Unauthorized')


def render_page(request: Request, template_name: str = None):
    redirect = require_page_auth(request)
    if redirect:
        return redirect
    return FileResponse(str(spa_dist / 'index.html'))


def _validate_image_upload(file: UploadFile, contents: bytes | None = None) -> bytes:
    filename = Path(file.filename or '').name
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image file type.")
    if contents is None:
        contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image file too large.")
    return contents


def _read_image(file: UploadFile) -> Image.Image:
    try:
        contents = _validate_image_upload(file)
        image = Image.open(io.BytesIO(contents))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions too large.")
        return image.convert('RGB')
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file provided.")


def _safe_training_output(value: str) -> str:
    output_path = (BASE_DIR / value).resolve()
    checkpoint_root = (CHECKPOINT_DIR).resolve()
    if output_path != checkpoint_root and checkpoint_root not in output_path.parents:
        raise HTTPException(status_code=400, detail="Training output must be inside checkpoints/.")
    return str(output_path.relative_to(BASE_DIR))

def _safe_evaluation_output(value: str) -> str:
    output_path = (BASE_DIR / value).resolve()
    evaluation_root = EVALUATION_DIR.resolve()
    if output_path != evaluation_root and evaluation_root not in output_path.parents:
        raise HTTPException(status_code=400, detail="Evaluation output must be inside evaluation/.")
    return str(output_path.relative_to(BASE_DIR))

def _safe_checkpoint_path(value: str) -> str:
    checkpoint_path = (BASE_DIR / value).resolve()
    checkpoint_root = CHECKPOINT_DIR.resolve()
    if checkpoint_path != checkpoint_root and checkpoint_root not in checkpoint_path.parents:
        raise HTTPException(status_code=400, detail="Checkpoint must be inside checkpoints/.")
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return str(checkpoint_path.relative_to(BASE_DIR))


def _validate_training_params(epochs: int, batch_size: int, size: int, lr: float, base_channels: int, workers: int):
    if not 1 <= epochs <= 1000:
        raise HTTPException(status_code=400, detail="epochs must be between 1 and 1000")
    if not 1 <= batch_size <= 64:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 64")
    if size < 64 or size > 2048 or size % 32 != 0:
        raise HTTPException(status_code=400, detail="size must be a multiple of 32 between 64 and 2048")
    if not 0 < lr <= 1:
        raise HTTPException(status_code=400, detail="lr must be between 0 and 1")
    if not 8 <= base_channels <= 256:
        raise HTTPException(status_code=400, detail="base_channels must be between 8 and 256")
    if not 0 <= workers <= 16:
        raise HTTPException(status_code=400, detail="workers must be between 0 and 16")


def _validate_loss_weights(**weights: float):
    for name, value in weights.items():
        if value < 0 or value > 10:
            raise HTTPException(status_code=400, detail=f"{name} must be between 0 and 10")

def _validate_training_controls(early_stop_patience: int, min_delta: float, grad_clip_norm: float):
    if not 0 <= early_stop_patience <= 1000:
        raise HTTPException(status_code=400, detail="early_stop_patience must be between 0 and 1000")
    if min_delta < 0 or min_delta > 1:
        raise HTTPException(status_code=400, detail="min_delta must be between 0 and 1")
    if grad_clip_norm < 0 or grad_clip_norm > 100:
        raise HTTPException(status_code=400, detail="grad_clip_norm must be between 0 and 100")


def _pil_to_response(img: Image.Image, filename: str = "result.png"):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# =====================================================================
#  PAGE ROUTES
# =====================================================================
@app.get("/")
async def get_index_page(request: Request):
    return render_page(request, "index.html")


from starlette.responses import HTMLResponse

@app.get("/login")
async def login_page(request: Request):
    if is_authenticated(request):
        return RedirectResponse('/', status_code=303)
    return HTMLResponse(login_html.read_text(encoding='utf-8'))


@app.get("/test")
async def test_model_page(request: Request):
    return render_page(request, "test.html")


@app.get("/train")
async def train_page(request: Request):
    return render_page(request, "train.html")


@app.get("/image-tests")
async def image_tests_page(request: Request):
    return render_page(request, "image_tests.html")


@app.get("/download")
async def download_page(request: Request):
    return render_page(request, "download.html")

@app.get("/datasets")
async def datasets_page(request: Request):
    return render_page(request, "datasets.html")


# =====================================================================
#  AUTH API
# =====================================================================
@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username != ADMIN_CREDENTIALS['username'] or password != ADMIN_CREDENTIALS['password']:
        return JSONResponse({'success': False, 'message': 'Username atau password salah'}, status_code=401)
    token = create_token(username)
    response = JSONResponse({'success': True, 'token': token, 'user': username})
    response.set_cookie('docai_token', token, httponly=True, samesite='lax', max_age=24 * 60 * 60)
    return response


@app.get("/api/auth/logout")
async def logout():
    response = JSONResponse({'success': True})
    response.delete_cookie('docai_token')
    return response


@app.get("/api/auth/check")
async def auth_check(request: Request):
    return {'authenticated': is_authenticated(request)}


# =====================================================================
#  HEALTH / INFO
# =====================================================================
@app.get("/api/health")
def health_check():
    docshadow = get_docshadow()
    return {
        "status": "ok",
        "device": str(device),
        "models": {
            "document_restorer": doc_restorer_model is not None,
            "shadow_remover": shadow_remover_model is not None,
            "doc_enhancer": doc_enhancer_model is not None,
            "docshadow_sd7k": docshadow is not None,
        },
    }


@app.get("/api/models/info")
async def models_info(request: Request):
    require_api_auth(request)
    info = {}
    for name, model in [
        ("document_restorer", doc_restorer_model),
        ("shadow_remover", shadow_remover_model),
        ("doc_enhancer", doc_enhancer_model),
    ]:
        if model is not None and hasattr(model, 'get_model_info'):
            info[name] = model.get_model_info()
        else:
            info[name] = {"loaded": model is not None}
    docshadow = get_docshadow()
    if docshadow:
        info["docshadow_sd7k"] = {"loaded": True, "weights": docshadow.list_available_weights()}
    else:
        info["docshadow_sd7k"] = {"loaded": False}
    return info


@app.get("/api/system/status")
async def system_status(request: Request):
    require_api_auth(request)
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(BASE_DIR))
        load_average = os.getloadavg() if hasattr(os, 'getloadavg') else None
        status = {
            'time': time.time(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.05),
                'count': psutil.cpu_count(),
                'load_average': list(load_average) if load_average else None,
            },
            'ram': {
                'total_gb': round(memory.total / 1e9, 2),
                'used_gb': round(memory.used / 1e9, 2),
                'available_gb': round(memory.available / 1e9, 2),
                'percent': memory.percent,
            },
            'disk': {
                'total_gb': round(disk.total / 1e9, 2),
                'used_gb': round(disk.used / 1e9, 2),
                'free_gb': round(disk.free / 1e9, 2),
                'percent': disk.percent,
            },
            'gpu': {
                'available': torch.cuda.is_available(),
                'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'items': [],
            },
        }
        if torch.cuda.is_available():
            smi_stats = _nvidia_smi_stats()
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                used = total - free
                runtime = smi_stats.get(index, {})
                used_mb = runtime.get('memory_used_mb')
                total_mb = runtime.get('memory_total_mb')
                free_mb = runtime.get('memory_free_mb')
                if used_mb is None:
                    used_mb = used / (1024 ** 2)
                if total_mb is None:
                    total_mb = total / (1024 ** 2)
                if free_mb is None:
                    free_mb = free / (1024 ** 2)
                status['gpu']['items'].append({
                    'index': index,
                    'name': torch.cuda.get_device_name(index),
                    'vram_total_mb': round(total_mb),
                    'vram_used_mb': round(used_mb),
                    'vram_free_mb': round(free_mb),
                    'vram_total_gb': round(total_mb / 1024, 2),
                    'vram_used_gb': round(used_mb / 1024, 2),
                    'vram_free_gb': round(free_mb / 1024, 2),
                    'vram_percent': round((used_mb / total_mb) * 100, 2) if total_mb else 0,
                    'utilization_percent': runtime.get('utilization_percent'),
                    'temperature_c': runtime.get('temperature_c'),
                    'power_draw_w': runtime.get('power_draw_w'),
                    'power_limit_w': runtime.get('power_limit_w'),
                    'stats_source': runtime.get('source', 'torch'),
                })
        return status
    except Exception as error:
        raise HTTPException(status_code=500, detail=f'System status failed: {error}')


# =====================================================================
#  CORE SCANNING ENDPOINTS
# =====================================================================
@app.post("/api/scan")
async def scan_document(request: Request, file: UploadFile = File(...), mode: str = Form("restore")):
    require_api_auth(request)
    image = _read_image(file)

    if mode == "restore" and doc_restorer_model is not None:
        start = time.time()
        with torch.inference_mode():
            restored, mask, info = run_document_restoration_pipeline(
                doc_restorer_model, image, device,
                tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
            )
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(restored)

    elif mode == "shadow_remove" and shadow_remover_model is not None:
        start = time.time()
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = shadow_remover_model(img_tensor)
        result = _np_to_pil(output[0].cpu().numpy().transpose(1, 2, 0) * 255)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "enhance" and doc_enhancer_model is not None:
        start = time.time()
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = doc_enhancer_model(img_tensor)
        result_img = output.get('enhanced', output[0] if isinstance(output, (tuple, list)) else output)
        if isinstance(result_img, torch.Tensor):
            result_img = result_img[0].cpu().numpy().transpose(1, 2, 0) * 255
        result = _np_to_pil(result_img)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "magic_enhance":
        start = time.time()
        result_np = magic_document_enhance(np.array(image))
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(_np_to_pil(result_np))

    elif mode == "binarize":
        start = time.time()
        result_np = adaptive_binarize(np.array(image))
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(_np_to_pil(result_np))

    elif mode == "deskew":
        start = time.time()
        result, angle = deskew_document(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "cleanup":
        start = time.time()
        result, info = full_document_cleanup(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "clahe":
        start = time.time()
        result = enhance_contrast_clahe(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "denoise":
        start = time.time()
        result = remove_background_noise(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "sharpen":
        start = time.time()
        result = enhance_text_sharpness(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    else:
        available = ["restore", "shadow_remove", "enhance", "magic_enhance", "binarize",
                      "deskew", "cleanup", "clahe", "denoise", "sharpen"]
        raise HTTPException(status_code=400,
                            detail=f"Mode '{mode}' not available. Use one of: {available}")


@app.post("/api/scan/json")
async def scan_document_json(request: Request, file: UploadFile = File(...), mode: str = Form("restore")):
    """Return processing info alongside the scan result."""
    require_api_auth(request)
    image = _read_image(file)
    start = time.time()

    if mode == "restore" and doc_restorer_model is not None:
        with torch.inference_mode():
            restored, mask, info = run_document_restoration_pipeline(
                doc_restorer_model, image, device,
                tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
            )
        elapsed = round((time.time() - start) * 1000, 1)
        uid = uuid.uuid4().hex[:8]
        out_path = UPLOAD_DIR / f"scan_{uid}.png"
        restored.save(str(out_path))
        return {"success": True, "mode": mode, "time_ms": elapsed,
                "size": list(restored.size), "url": f"/api/uploads/{out_path.name}",
                "pipeline": info['pipeline'], "mask_mean": info['mask_mean'],
                "illumination_mean": info['illumination_mean']}

    # For non-restore modes, fall back to regular scan
    return await scan_document(request, file, mode)


@app.get("/api/uploads/{filename}")
async def serve_upload(filename: str, request: Request):
    require_api_auth(request)
    file_path = UPLOAD_DIR / Path(filename).name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="image/png")


# =====================================================================
#  DOCSHADOW SD7K INFERENCE
# =====================================================================
@app.post("/api/docshadow/infer")
async def docshadow_infer(request: Request, file: UploadFile = File(...), weight: str = Form("SD7K")):
    require_api_auth(request)
    docshadow = get_docshadow()
    if docshadow is None:
        raise HTTPException(status_code=503, detail="DocShadow SD7K model not loaded")
    image = _read_image(file)
    try:
        start = time.time()
        result, metrics = docshadow.infer(image, weight_name=weight)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/api/docshadow/weights")
async def docshadow_weights(request: Request):
    require_api_auth(request)
    docshadow = get_docshadow()
    if docshadow is None:
        return {"loaded": False, "weights": []}
    return {"loaded": True, "weights": docshadow.list_available_weights()}


# =====================================================================
#  AI SHADOW POST-PROCESSING
# =====================================================================
@app.post("/api/ai-postprocess")
async def ai_postprocess(request: Request,
                         original: UploadFile = File(...),
                         ai_result: UploadFile = File(...),
                         mask: UploadFile = File(None)):
    require_api_auth(request)
    original_img = _np_to_pil(np.array(_read_image(original)))
    ai_img = _np_to_pil(np.array(_read_image(ai_result)))
    mask_np = None
    if mask:
        mask_pil = _read_image(mask).convert('L')
        mask_np = np.array(mask_pil)
    result_np = ai_shadow_postprocess(np.array(original_img), np.array(ai_img), mask_np)
    return _pil_to_response(_np_to_pil(result_np))


# =====================================================================
#  BATCH PROCESSING
# =====================================================================
@app.post("/api/batch/process")
async def batch_process(request: Request,
                        files: list[UploadFile] = File(...),
                        mode: str = Form("restore")):
    require_api_auth(request)
    results = []
    for f in files:
        uid = uuid.uuid4().hex[:8]
        try:
            image = _read_image(f)
            start = time.time()
            if mode == "restore" and doc_restorer_model is not None:
                with torch.inference_mode():
                    out, _, _ = run_document_restoration_pipeline(
                        doc_restorer_model, image, device,
                        tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
                    )
            elif mode == "magic_enhance":
                out = _np_to_pil(magic_document_enhance(np.array(image)))
            elif mode == "binarize":
                out = _np_to_pil(adaptive_binarize(np.array(image)))
            elif mode == "cleanup":
                out, _ = full_document_cleanup(image)
            elif mode == "clahe":
                out = enhance_contrast_clahe(image)
            elif mode == "denoise":
                out = remove_background_noise(image)
            elif mode == "sharpen":
                out = enhance_text_sharpness(image)
            elif mode == "deskew":
                out, _ = deskew_document(image)
            else:
                results.append({"name": f.filename, "success": False, "error": f"Unknown mode: {mode}"})
                continue

            elapsed = round((time.time() - start) * 1000, 1)
            out_name = f"{uid}_{f.filename or 'image.png'}"
            out_path = UPLOAD_DIR / out_name
            out.save(str(out_path))
            results.append({
                "name": f.filename, "success": True,
                "time_ms": elapsed, "url": f"/api/uploads/{out_name}",
            })
        except Exception as e:
            results.append({"name": f.filename, "success": False, "error": str(e)})

    return {"count": len(results), "results": results}


# =====================================================================
#  IMAGE TESTS WORKSPACE
# =====================================================================
@app.get("/api/image-tests")
async def list_image_tests(request: Request):
    require_api_auth(request)
    IMAGE_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    tests = []
    for path in sorted(IMAGE_TEST_ROOT.iterdir()):
        if path.is_dir():
            image_count = sum(1 for item in path.rglob('*') if item.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
            tests.append({'name': path.name, 'image_count': image_count})
    return {'tests': tests}


@app.get("/api/image-tests/contact-sheet-analysis")
async def contact_sheet_analysis(request: Request):
    require_api_auth(request)
    candidates = [
        IMAGE_TEST_ROOT / 'contact_sheet_analysis_latest2.png',
        IMAGE_TEST_ROOT / 'contact_sheet_analysis_light.png',
        IMAGE_TEST_ROOT / 'contact_sheet_analysis.png',
    ]
    file_path = next((path for path in candidates if path.is_file()), IMAGE_TEST_ROOT / 'contact_sheet_analysis.png')
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail='Contact sheet analysis not found')
    return FileResponse(file_path)

@app.get("/api/image-tests/contact-sheets")
async def list_contact_sheets(request: Request):
    require_api_auth(request)
    IMAGE_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    sheets = []
    for path in sorted(IMAGE_TEST_ROOT.glob('contact_sheet*'), key=lambda value: value.stat().st_mtime, reverse=True):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            sheets.append({
                'name': path.name,
                'path': f'data/test/{path.name}',
                'size': path.stat().st_size,
                'url': f'/api/image-tests/contact-sheets/{path.name}',
            })
    return {'sheets': sheets}

@app.get("/api/image-tests/contact-sheets/{name}")
async def preview_contact_sheet(request: Request, name: str):
    require_api_auth(request)
    safe_name = Path(name).name
    file_path = IMAGE_TEST_ROOT / safe_name
    if not file_path.is_file() or not safe_name.startswith('contact_sheet') or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail='Contact sheet not found')
    return FileResponse(file_path)

@app.get("/api/image-tests/files")
async def list_image_test_files(request: Request, test: str, path: str = ''):
    require_api_auth(request)
    try:
        test_root = IMAGE_TEST_ROOT / safe_workspace_name(test)
        folder = resolve_workspace_path(test_root, path or '.')
    except WorkspaceError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail='Path is not a folder')
    items = []
    for item in sorted(folder.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower())):
        relative = item.relative_to(test_root).as_posix()
        is_image = item.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
        items.append({
            'name': item.name, 'path': relative,
            'type': 'folder' if item.is_dir() else 'file',
            'size': item.stat().st_size if item.is_file() else None,
            'is_image': is_image,
        })
    return {'items': items}


@app.get("/api/image-tests/preview")
async def preview_image_test(request: Request, test: str, path: str):
    require_api_auth(request)
    try:
        test_root = IMAGE_TEST_ROOT / safe_workspace_name(test)
        file_path = resolve_workspace_path(test_root, path)
    except WorkspaceError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail='File not found')
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail='File is not a supported image')
    return FileResponse(file_path)


@app.post("/api/image-tests/upload")
async def upload_image_test(request: Request, test: str = Form(...), destination: str = Form('input'), file: UploadFile = File(...)):
    require_api_auth(request)
    if destination not in {'input', 'output'}:
        raise HTTPException(status_code=400, detail='Invalid destination')
    test_root = IMAGE_TEST_ROOT / safe_workspace_name(test)
    target_dir = test_root / destination
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or 'upload.png').name
    _validate_image_upload(file)
    file.file.seek(0)
    target_path = target_dir / filename
    with target_path.open('wb') as output:
        shutil.copyfileobj(file.file, output)
    return {'success': True, 'path': target_path.relative_to(test_root).as_posix()}

@app.post("/api/image-tests/upload-pair")
async def upload_image_test_pair(request: Request,
                                 test: str = Form(...),
                                 input_file: UploadFile = File(...),
                                 output_file: UploadFile = File(...)):
    require_api_auth(request)
    clean_name = (test or '').strip() or f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    test_root = IMAGE_TEST_ROOT / safe_workspace_name(clean_name)
    input_dir = test_root / 'input'
    output_dir = test_root / 'output'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for destination, upload, target_dir in (
        ('input', input_file, input_dir),
        ('output', output_file, output_dir),
    ):
        _validate_image_upload(upload)
        upload.file.seek(0)
        filename = Path(upload.filename or f'{destination}.png').name
        target_path = target_dir / filename
        with target_path.open('wb') as output:
            shutil.copyfileobj(upload.file, output)
        saved[destination] = target_path.relative_to(test_root).as_posix()

    return {'success': True, 'test': test_root.name, 'paths': saved}

@app.delete("/api/image-tests/item")
async def delete_image_test_item(request: Request, test: str, path: str = ''):
    require_api_auth(request)
    try:
        test_root = IMAGE_TEST_ROOT / safe_workspace_name(test)
        target_path = resolve_workspace_path(test_root, path or '.')
    except WorkspaceError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if target_path == test_root:
        shutil.rmtree(target_path, ignore_errors=True)
        return {'success': True, 'deleted': test_root.name, 'test_deleted': True}
    if not target_path.exists():
        raise HTTPException(status_code=404, detail='File or folder not found')
    relative = target_path.relative_to(test_root).as_posix()
    if target_path.is_dir():
        shutil.rmtree(target_path)
    else:
        target_path.unlink()
    return {'success': True, 'deleted': relative, 'test_deleted': False}


# =====================================================================
#  DATASET MANAGEMENT
# =====================================================================
@app.get("/api/datasets/shadow7k/progress")
async def shadow7k_progress(request: Request):
    require_api_auth(request)
    image_count = sum(1 for path in DOWNLOAD_ROOT.rglob('*') if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}) if DOWNLOAD_ROOT.exists() else 0
    size_mb = round(sum(path.stat().st_size for path in DOWNLOAD_ROOT.rglob('*') if path.is_file()) / (1024 * 1024), 2) if DOWNLOAD_ROOT.exists() else 0
    status = 'completed' if image_count >= 7000 else 'partial' if image_count else 'idle'
    progress = min(100, round(image_count / 7000 * 100)) if image_count else 0
    return {'status': status, 'download_active': False, 'progress': progress, 'total': image_count, 'size_mb': size_mb}


@app.post("/api/datasets/download")
async def start_dataset_download(request: Request):
    require_api_auth(request)
    return {'success': False, 'error': 'Download otomatis dinonaktifkan. Gunakan instruksi download manual di halaman ini.'}


@app.post("/api/datasets/download/stop")
async def stop_dataset_download(request: Request):
    require_api_auth(request)
    return {'success': True, 'message': 'Tidak ada download aktif.'}


# =====================================================================
#  TRAINING STATUS
# =====================================================================
@app.get("/api/training/status")
async def training_status(request: Request):
    require_api_auth(request)
    running = _training_is_running()
    meta = _stored_training_meta()
    active_output = meta.get('run_output')
    active_output_path = (BASE_DIR / active_output).resolve() if active_output else CHECKPOINT_DIR / 'document_restorer'
    history_file = active_output_path / 'training_history.json'
    history = {}
    if history_file.exists():
        try:
            import json
            history = json.loads(history_file.read_text())
        except Exception:
            pass
    best_path = CHECKPOINT_DIR / 'document_restorer' / 'best.pth'
    last_path = CHECKPOINT_DIR / 'document_restorer' / 'last.pth'
    preview_dir = active_output_path / 'previews'
    latest_preview = None
    if preview_dir.exists():
        previews = sorted(preview_dir.glob('epoch_*.png'), key=lambda path: path.stat().st_mtime, reverse=True)
        if previews:
            latest_preview = previews[0].name

    # Parse ETA and progress from training log
    eta = None
    current_epoch = None
    total_epochs = None
    current_batch = None
    total_batches = None
    last_train_loss = None
    last_val_loss = None
    last_val_psnr = None
    last_val_ssim = None
    with _training_lock:
        for line in reversed(_training_log):
            # Parse eta=1h 23m 45s from epoch log line
            import re as _re
            m = _re.search(r'eta=(\S+\s+\S+\s+\S+)', line)
            if m:
                eta = m.group(1)
            m = _re.search(r'epoch=(\d+)/(\d+)', line)
            if m:
                current_epoch = int(m.group(1))
                total_epochs = int(m.group(2))
            m = _re.search(r'batch=(\d+)/(\d+)', line)
            if m:
                current_batch = int(m.group(1))
                total_batches = int(m.group(2))
            m = _re.search(r'train_loss=(\S+)', line)
            if m:
                last_train_loss = m.group(1)
            m = _re.search(r'val_loss=(\S+)', line)
            if m:
                last_val_loss = m.group(1)
            m = _re.search(r'val_psnr=(\S+)', line)
            if m:
                last_val_psnr = m.group(1)
            m = _re.search(r'val_ssim=(\S+)', line)
            if m:
                last_val_ssim = m.group(1)

    # Check if model is available for testing (even during training)
    model_available = best_path.exists()
    model_mtime = None
    if model_available:
        model_mtime = int(best_path.stat().st_mtime)

    started_at = _training_started_at or meta.get('started_at')
    if running and started_at is None:
        try:
            started_at = psutil.Process(_training_process.pid).create_time()
        except Exception:
            started_at = None
    eta_seconds = _eta_to_seconds(eta)
    estimated_finish_at = time.time() + eta_seconds if running and eta_seconds is not None else (_training_estimated_finish_at or meta.get('estimated_finish_at') if running else None)
    progress_percent = None
    if current_epoch is not None and total_epochs:
        completed_epochs = max(current_epoch - 1, 0)
        batch_fraction = (current_batch / total_batches) if current_batch is not None and total_batches else 0
        progress_percent = round(((completed_epochs + batch_fraction) / total_epochs) * 100, 2)

    return {
        'running': running,
        'process_kind': _training_kind or meta.get('kind'),
        'started_at': started_at,
        'started_at_wib': _format_wib(started_at),
        'estimated_finish_at': estimated_finish_at,
        'estimated_finish_wib': _format_wib(estimated_finish_at),
        'run_command': meta.get('cmd'),
        'has_history': bool(history),
        'epochs': len(history.get('train_loss', [])),
        'best_loss': min(history.get('val_loss', [float('inf')])),
        'best_exists': best_path.exists(),
        'last_exists': last_path.exists(),
        'history': history,
        'eta': eta,
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'current_batch': current_batch,
        'total_batches': total_batches,
        'progress_percent': progress_percent,
        'last_train_loss': last_train_loss,
        'last_val_loss': last_val_loss,
        'last_val_psnr': last_val_psnr,
        'last_val_ssim': last_val_ssim,
        'model_available_for_test': model_available,
        'model_mtime': model_mtime,
        'active_run_id': meta.get('run_id'),
        'active_run_output': active_output,
        'latest_preview_url': f'/api/training/preview/{latest_preview}?run_id={meta.get("run_id", "")}&v={int((preview_dir / latest_preview).stat().st_mtime)}' if latest_preview else None,
    }


@app.get("/api/training/preview/{filename}")
async def training_preview(request: Request, filename: str, run_id: str = ''):
    require_api_auth(request)
    preview_root = CHECKPOINT_DIR / 'document_restorer' / 'runs' / Path(run_id).name if run_id else CHECKPOINT_DIR / 'document_restorer'
    preview_path = preview_root / 'previews' / Path(filename).name
    if not preview_path.exists() or preview_path.suffix.lower() != '.png':
        raise HTTPException(status_code=404, detail='Preview not found')
    return FileResponse(preview_path, media_type='image/png')

@app.get("/api/training/runs")
async def training_runs(request: Request):
    require_api_auth(request)
    runs_root = CHECKPOINT_DIR / 'document_restorer' / 'runs'
    runs = []
    if runs_root.exists():
        for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
            manifest = {}
            manifest_path = run_dir / 'run_manifest.json'
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                except Exception:
                    pass
            if manifest.get('status') == 'running' and run_dir.name != _stored_training_meta().get('run_id'):
                manifest['status'] = 'interrupted'
            checkpoints = []
            for name in ('best.pth', 'best_loss.pth', 'best_psnr.pth', 'best_ssim.pth', 'last.pth'):
                checkpoint_path = run_dir / name
                if checkpoint_path.exists():
                    checkpoints.append({'name': name, 'path': checkpoint_path.relative_to(BASE_DIR).as_posix(), 'mtime': int(checkpoint_path.stat().st_mtime)})
            runs.append({**manifest, 'run_id': run_dir.name, 'checkpoints': checkpoints})
    return {'runs': runs}

@app.get("/api/training/evaluation/status")
async def evaluation_status(request: Request, output: str = 'evaluation/document_restorer'):
    require_api_auth(request)
    output = _safe_evaluation_output(output)
    output_path = BASE_DIR / output
    summary_path = output_path / 'summary.json'
    summary = None
    if summary_path.exists():
        try:
            import json
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception:
            summary = None
    preview_path = output_path / 'preview_grid.png'
    return {
        'exists': summary_path.exists(),
        'summary': summary,
        'preview_url': f'/api/training/evaluation/preview?output={output}' if preview_path.exists() else None,
        'metrics_url': f'/api/training/evaluation/metrics?output={output}' if (output_path / 'metrics.csv').exists() else None,
    }

@app.get("/api/training/evaluation/preview")
async def evaluation_preview(request: Request, output: str = 'evaluation/document_restorer'):
    require_api_auth(request)
    output = _safe_evaluation_output(output)
    preview_path = BASE_DIR / output / 'preview_grid.png'
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail='Evaluation preview not found')
    return FileResponse(preview_path, media_type='image/png')

@app.get("/api/training/evaluation/metrics")
async def evaluation_metrics(request: Request, output: str = 'evaluation/document_restorer'):
    require_api_auth(request)
    output = _safe_evaluation_output(output)
    metrics_path = BASE_DIR / output / 'metrics.csv'
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail='Evaluation metrics not found')
    return FileResponse(metrics_path, media_type='text/csv', filename='metrics.csv')


# =====================================================================
#  DATASET BROWSING & VALIDATION
# =====================================================================
@app.get("/api/datasets")
async def list_datasets(request: Request):
    """List all available datasets in data/datasets and datasets/"""
    require_api_auth(request)
    from backend.datasets.manager import RestorationDatasetManager
    import os

    search_dirs = [
        BASE_DIR / 'data' / 'datasets',
        BASE_DIR / 'datasets',
    ]

    all_datasets = []
    seen_dataset_paths = set()

    def add_dataset(info: dict):
        path = info.get('path')
        if path in seen_dataset_paths:
            return
        seen_dataset_paths.add(path)
        all_datasets.append(info)
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Clean datasets
        clean_root = search_dir / 'clean'
        if clean_root.exists():
            for d in sorted(clean_root.iterdir()):
                if d.is_dir():
                    imgs = [f for f in d.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
                    add_dataset({
                        'name': d.name,
                        'kind': 'clean',
                        'path': str(d.relative_to(BASE_DIR)),
                        'abs_path': str(d),
                        'image_count': len(imgs),
                        'source': str(search_dir.relative_to(BASE_DIR)),
                    })
        # Paired datasets
        paired_root = search_dir / 'paired'
        if paired_root.exists():
            for d in sorted(paired_root.iterdir()):
                if d.is_dir():
                    info = _scan_paired_dataset(d)
                    info['source'] = str(search_dir.relative_to(BASE_DIR))
                    add_dataset(info)
        # Identity datasets
        identity_root = search_dir / 'identity'
        if identity_root.exists():
            direct_imgs = [f for f in identity_root.iterdir() if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
            if direct_imgs:
                add_dataset({
                    'name': identity_root.name,
                    'kind': 'identity',
                    'path': str(identity_root.relative_to(BASE_DIR)),
                    'abs_path': str(identity_root),
                    'image_count': len(direct_imgs),
                    'source': str(search_dir.relative_to(BASE_DIR)),
                    'ready': True,
                    'note': 'Identity input=target preservation dataset',
                })
            for d in sorted(path for path in identity_root.iterdir() if path.is_dir()):
                imgs = [f for f in d.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
                add_dataset({
                    'name': d.name,
                    'kind': 'identity',
                    'path': str(d.relative_to(BASE_DIR)),
                    'abs_path': str(d),
                    'image_count': len(imgs),
                    'source': str(search_dir.relative_to(BASE_DIR)),
                    'ready': len(imgs) > 0,
                    'note': 'Identity input=target preservation dataset',
                })
        data_identity_root = BASE_DIR / 'data' / 'identity'
        if search_dir == BASE_DIR / 'datasets' and data_identity_root.exists():
            direct_imgs = [f for f in data_identity_root.iterdir() if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
            if direct_imgs:
                add_dataset({
                    'name': data_identity_root.name,
                    'kind': 'identity',
                    'path': str(data_identity_root.relative_to(BASE_DIR)),
                    'abs_path': str(data_identity_root),
                    'image_count': len(direct_imgs),
                    'source': 'data',
                    'ready': True,
                    'note': 'Identity input=target preservation dataset',
                })
            for d in sorted(path for path in data_identity_root.iterdir() if path.is_dir()):
                imgs = [f for f in d.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
                add_dataset({
                    'name': d.name,
                    'kind': 'identity',
                    'path': str(d.relative_to(BASE_DIR)),
                    'abs_path': str(d),
                    'image_count': len(imgs),
                    'source': 'data',
                    'ready': len(imgs) > 0,
                    'note': 'Identity input=target preservation dataset',
                })
        # Direct dataset folders (e.g. ShadowDocument7K or Document_Enhancement/train)
        for d in sorted(search_dir.iterdir()):
            if d.is_dir() and d.name not in {'clean', 'paired', 'test_download'}:
                if (d / 'train' / 'input').is_dir() and (d / 'train' / 'target').is_dir():
                    info = _scan_paired_dataset(d)
                    info['source'] = str(search_dir.relative_to(BASE_DIR))
                    add_dataset(info)
                elif (d / 'train').is_dir():
                    # Clean dataset folder with images under train/.
                    train_dir = d / 'train'
                    imgs = [f for f in train_dir.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}]
                    add_dataset({
                        'name': d.name,
                        'kind': 'clean',
                        'path': str(d.relative_to(BASE_DIR)),
                        'abs_path': str(train_dir),
                        'image_count': len(imgs),
                        'source': str(search_dir.relative_to(BASE_DIR)),
                        'note': 'Clean dataset (synthetic degradation)',
                    })
    return {'datasets': all_datasets}


def _scan_paired_dataset(d: Path) -> dict:
    """Scan a paired dataset directory for input/target pairs."""
    IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    input_dir = d / 'input'
    target_dir = d / 'target'
    train_input = d / 'train' / 'input'
    train_target = d / 'train' / 'target'
    test_input = d / 'test' / 'input'
    test_target = d / 'test' / 'target'

    if train_input.exists() and train_target.exists():
        inputs = {p.stem for p in train_input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        targets = {p.stem for p in train_target.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        matched = len(inputs & targets)
        test_count = 0
        if test_input.exists():
            test_count = sum(1 for p in test_input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        return {
            'name': d.name,
            'kind': 'paired',
            'path': str(d.relative_to(BASE_DIR)),
            'abs_path': str(d),
            'pair_count': matched,
            'train_pairs': matched,
            'test_count': test_count,
            'has_train_test': True,
            'ready': matched > 0,
        }
    elif input_dir.exists() and target_dir.exists():
        inputs = {p.stem for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        targets = {p.stem for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
        matched = len(inputs & targets)
        return {
            'name': d.name,
            'kind': 'paired',
            'path': str(d.relative_to(BASE_DIR)),
            'abs_path': str(d),
            'pair_count': matched,
            'train_pairs': matched,
            'test_count': 0,
            'has_train_test': False,
            'ready': matched > 0,
        }
    else:
        imgs = [f for f in d.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]
        return {
            'name': d.name,
            'kind': 'unknown',
            'path': str(d.relative_to(BASE_DIR)),
            'abs_path': str(d),
            'image_count': len(imgs),
            'ready': False,
            'note': 'Structure not recognized',
        }


def _dataset_root(path: str) -> Path:
    root = (BASE_DIR / path).resolve()
    allowed_roots = [(BASE_DIR / 'data' / 'datasets').resolve(), (BASE_DIR / 'datasets').resolve()]
    if not any(root == allowed or allowed in root.parents for allowed in allowed_roots):
        raise HTTPException(status_code=400, detail='Dataset path outside allowed roots')
    if not root.is_dir():
        raise HTTPException(status_code=404, detail='Dataset folder not found')
    return root


@app.get('/api/datasets/explorer')
async def browse_dataset(request: Request, dataset: str, path: str = '', offset: int = 0, limit: int = 60):
    """Browse one registered dataset without allowing filesystem traversal."""
    require_api_auth(request)
    root = _dataset_root(dataset)
    current = (root / path).resolve()
    if current != root and root not in current.parents:
        raise HTTPException(status_code=400, detail='Invalid dataset path')
    if not current.is_dir():
        raise HTTPException(status_code=404, detail='Folder not found')
    limit = min(max(limit, 1), 120)
    image_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    entries = []
    for item in sorted(current.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
        relative = item.relative_to(root).as_posix()
        is_image = item.is_file() and item.suffix.lower() in image_suffixes
        if item.is_dir() or is_image:
            entries.append({'name': item.name, 'path': relative, 'kind': 'folder' if item.is_dir() else 'image', 'size': item.stat().st_size if item.is_file() else None})
    page = entries[offset:offset + limit]
    return {'dataset': dataset, 'path': current.relative_to(root).as_posix() if current != root else '', 'total': len(entries), 'offset': offset, 'limit': limit, 'entries': page}


@app.get('/api/datasets/explorer/image')
async def dataset_explorer_image(request: Request, dataset: str, path: str):
    require_api_auth(request)
    root = _dataset_root(dataset)
    image_path = (root / path).resolve()
    image_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    if root not in image_path.parents or not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
        raise HTTPException(status_code=404, detail='Dataset image not found')
    return FileResponse(image_path)


@app.get("/api/datasets/validate")
async def validate_dataset(request: Request, path: str):
    """Validate a specific dataset path."""
    require_api_auth(request)
    from backend.datasets.manager import RestorationDatasetManager
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        raise HTTPException(status_code=400, detail='Path outside project')
    if not target.exists():
        raise HTTPException(status_code=404, detail='Dataset not found')

    # Check if it's paired or clean
    has_input = (target / 'input').exists() or (target / 'train' / 'input').exists()
    has_target = (target / 'target').exists() or (target / 'train' / 'target').exists()
    has_clean = target.suffix == '' and any(f.suffix.lower() in {'.png','.jpg','.jpeg'} for f in target.iterdir() if f.is_file())

    if has_input and has_target:
        info = _scan_paired_dataset(target)
        info['validation'] = 'paired'
    elif has_clean:
        imgs = [f for f in target.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg'}]
        info = {'name': target.name, 'kind': 'clean', 'path': path, 'image_count': len(imgs), 'ready': len(imgs) > 0, 'validation': 'clean'}
    else:
        imgs = [f for f in target.rglob('*') if f.is_file() and f.suffix.lower() in {'.png','.jpg','.jpeg'}]
        info = {'name': target.name, 'kind': 'unknown', 'path': path, 'image_count': len(imgs), 'ready': False, 'validation': 'unknown'}
    return info



# =====================================================================
#  LIVE MODEL RELOAD (test during training)
# =====================================================================
@app.post("/api/model/reload")
async def reload_model(request: Request):
    """Reload best.pth from disk — allows testing model while training continues."""
    require_api_auth(request)
    global doc_restorer_model, doc_restorer_checkpoint

    payload = await request.json() if request.headers.get('content-type', '').startswith('application/json') else {}
    checkpoint_value = payload.get('checkpoint', 'checkpoints/document_restorer/best.pth')
    safe_checkpoint = _safe_checkpoint_path(checkpoint_value)
    target_path = BASE_DIR / safe_checkpoint
    if not target_path.exists():
        raise HTTPException(status_code=404, detail='Model checkpoint not found')

    try:
        new_model = _load_checkpoint(DocumentRestorerNet, target_path, device)
        if new_model is None:
            raise HTTPException(status_code=500, detail='Failed to load model')
        doc_restorer_model = new_model
        doc_restorer_checkpoint = safe_checkpoint
        mtime = int(target_path.stat().st_mtime)
        return {
            'success': True,
            'message': f'Model reloaded from {safe_checkpoint}',
            'checkpoint': safe_checkpoint,
            'model_mtime': mtime,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Reload failed: {str(e)}')

@app.get("/api/model/status")
async def model_status(request: Request):
    """Check if model is loaded and ready for inference."""
    require_api_auth(request)
    best_path = CHECKPOINT_DIR / 'document_restorer' / 'best.pth'
    return {
        'loaded': doc_restorer_model is not None,
        'checkpoint_exists': best_path.exists(),
        'checkpoint_mtime': int(best_path.stat().st_mtime) if best_path.exists() else None,
        'loaded_checkpoint': doc_restorer_checkpoint,
    }

# =====================================================================
#  TRAINING PROCESS MANAGEMENT
# =====================================================================
import subprocess
import threading

_training_process = None
_training_log = []
_training_lock = threading.Lock()
_training_started_at = None
_training_estimated_finish_at = None
_training_kind = None
TRAINING_LOG_PATH = CHECKPOINT_DIR / 'document_restorer' / 'run.log'
TRAINING_PID_PATH = CHECKPOINT_DIR / 'document_restorer' / 'run.pid'
TRAINING_META_PATH = CHECKPOINT_DIR / 'document_restorer' / 'run_meta.json'
WIB = timezone(timedelta(hours=7))

def _format_wib(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=WIB).strftime('%d %b %Y %H:%M:%S WIB')

def _eta_to_seconds(eta: str | None) -> int | None:
    if not eta:
        return None

def _nvidia_smi_stats() -> dict[int, dict]:
    def number(value: str):
        try:
            if value in {'N/A', '[N/A]', ''}:
                return None
            return float(value)
        except Exception:
            return None

    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode != 0:
            return {}
        stats = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 8:
                continue
            index = int(parts[0])
            stats[index] = {
                'utilization_percent': number(parts[1]),
                'memory_used_mb': number(parts[2]),
                'memory_total_mb': number(parts[3]),
                'memory_free_mb': number(parts[4]),
                'temperature_c': number(parts[5]),
                'power_draw_w': number(parts[6]),
                'power_limit_w': number(parts[7]),
                'source': 'nvidia-smi',
            }
        return stats
    except Exception:
        return {}

def _count_training_samples(paths: list[str]) -> int:
    total = 0
    for value in paths:
        root = (BASE_DIR / value).resolve()
        if not root.exists() or not str(root).startswith(str(BASE_DIR.resolve())):
            continue
        input_dir = root / 'train' / 'input' if (root / 'train' / 'input').exists() else root / 'input'
        scan_root = input_dir if input_dir.exists() else root
        total += sum(1 for path in scan_root.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)
    return total

def _estimate_training_finish(started_at: float, paired_data: str, epochs: int, batch_size: int, size: int, max_train_samples: int) -> float | None:
    paths = [path for path in paired_data.split(',') if path]
    samples = _count_training_samples(paths)
    if max_train_samples > 0:
        samples = min(samples, max_train_samples)
    if samples <= 0:
        return None
    batches_per_epoch = max(1, (samples + batch_size - 1) // batch_size)
    seconds_per_batch = 0.12 * max(1, batch_size) * ((size / 512) ** 2)
    validation_padding = max(120, batches_per_epoch * 0.08 * seconds_per_batch)
    return started_at + epochs * batches_per_epoch * seconds_per_batch + epochs * validation_padding

def _estimate_evaluation_finish(started_at: float, paired_data: str, batch_size: int, size: int, max_samples: int) -> float | None:
    samples = _count_training_samples([paired_data])
    if max_samples > 0:
        samples = min(samples, max_samples)
    if samples <= 0:
        return None
    batches = max(1, (samples + batch_size - 1) // batch_size)
    seconds_per_batch = 0.035 * max(1, batch_size) * ((size / 512) ** 2)
    return started_at + batches * seconds_per_batch + 60
    try:
        import re as _re
        match = _re.search(r'(\d+)h\s+(\d+)m\s+(\d+)s', eta)
        if not match:
            return None
        hours, minutes, seconds = (int(value) for value in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return None

def _set_training_log(lines: list[str]):
    global _training_log
    _training_log = lines[-1500:]
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_LOG_PATH.write_text('\n'.join(_training_log) + ('\n' if _training_log else ''), encoding='utf-8')

def _load_training_log(lines: list[str]):
    global _training_log
    _training_log = lines[-1500:]

def _append_training_log(line: str):
    global _training_log
    _training_log.append(line)
    if len(_training_log) > 2000:
        _training_log = _training_log[-1500:]
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG_PATH.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')

def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def _record_training_pid(pid: int):
    TRAINING_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_PID_PATH.write_text(str(pid), encoding='utf-8')

def _record_training_meta(kind: str, started_at: float, estimated_finish_at: float | None, cmd: list[str]):
    TRAINING_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_META_PATH.write_text(json.dumps({
        'kind': kind,
        'started_at': started_at,
        'started_at_wib': _format_wib(started_at),
        'estimated_finish_at': estimated_finish_at,
        'estimated_finish_wib': _format_wib(estimated_finish_at),
        'cmd': cmd,
    }, indent=2), encoding='utf-8')

def _stored_training_meta() -> dict:
    try:
        if TRAINING_META_PATH.exists():
            return json.loads(TRAINING_META_PATH.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return {}

def _stored_training_pid() -> int | None:
    try:
        if TRAINING_PID_PATH.exists():
            return int(TRAINING_PID_PATH.read_text(encoding='utf-8').strip())
    except Exception:
        return None
    return None

def _training_is_running() -> bool:
    if _training_process is not None and _training_process.poll() is None:
        return True
    pid = _stored_training_pid()
    if _pid_running(pid):
        return True
    try:
        TRAINING_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    return False

def _monitor_training_process(proc):
    global _training_process
    proc.wait()
    with _training_lock:
        _append_training_log(f'[INFO] Training process exited with code {proc.returncode}')
    try:
        TRAINING_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    _training_process = None


# =====================================================================
#  OCR / TEXT DETECTION
# =====================================================================
@app.post("/api/ocr/detect")
async def ocr_detect(request: Request, file: UploadFile = File(...)):
    """Extract text from an uploaded document image using Tesseract OCR."""
    require_api_auth(request)
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img)
        # Run OCR with multiple PSM modes for best results
        import pytesseract
        data = pytesseract.image_to_data(img_np, lang='eng+ind', output_type=pytesseract.Output.DICT,
                                          config='--psm 6 --oem 3')
        words = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                words.append({
                    'text': data['text'][i],
                    'conf': int(data['conf'][i]),
                    'x': int(data['left'][i]),
                    'y': int(data['top'][i]),
                    'w': int(data['width'][i]),
                    'h': int(data['height'][i]),
                })
        full_text = pytesseract.image_to_string(img_np, lang='eng+ind', config='--psm 6 --oem 3')
        return {
            'success': True,
            'text': full_text.strip(),
            'words': words,
            'word_count': len(words),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'OCR failed: {str(e)}')

@app.post("/api/ocr/detect-from-result")
async def ocr_detect_from_result(request: Request):
    """Extract text from the last processed image blob."""
    require_api_auth(request)
    try:
        body = await request.json()
        image_b64 = body.get('image')
        if not image_b64:
            raise HTTPException(status_code=400, detail='No image provided')
        import base64
        img_bytes = base64.b64decode(image_b64.split(',')[-1] if ',' in image_b64 else image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        import pytesseract
        text = pytesseract.image_to_string(np.array(img), lang='eng+ind', config='--psm 6 --oem 3')
        return {'success': True, 'text': text.strip()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'OCR failed: {str(e)}')


# =====================================================================
#  FULL DOCUMENT PROCESSING PIPELINE (multi-step)
# =====================================================================
@app.post("/api/pipeline/process")
async def pipeline_process(request: Request, file: UploadFile = File(...)):
    """Run the document restoration pipeline in the production order."""
    require_api_auth(request)
    if doc_restorer_model is None:
        raise HTTPException(status_code=503, detail='DocumentRestorerNet checkpoint not loaded')
    image = _read_image(file)
    with torch.inference_mode():
        final, _, _ = run_document_restoration_pipeline(
            doc_restorer_model, image, device,
            tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
        )
    return _pil_to_response(final, filename='pipeline_result.png')

@app.post("/api/pipeline/process-json")
async def pipeline_process_json(request: Request, file: UploadFile = File(...)):
    """Run the production pipeline and return ordered step metadata."""
    require_api_auth(request)
    if doc_restorer_model is None:
        raise HTTPException(status_code=503, detail='DocumentRestorerNet checkpoint not loaded')
    image = _read_image(file)
    start_total = time.time()
    with torch.inference_mode():
        final, _, info = run_document_restoration_pipeline(
            doc_restorer_model, image, device,
            tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
        )
    total_ms = round((time.time() - start_total) * 1000)
    steps = [
        {'name': 'input', 'label': 'Input', 'status': 'done', 'time_ms': 0, 'info': f'Original size: {image.size[0]}x{image.size[1]}'},
        {'name': 'document_segmentation', 'label': 'Document Segmentation', 'status': 'done', 'time_ms': 0, 'info': f'Segmented size: {info["segmented_size"][0]}x{info["segmented_size"][1]}'},
        {'name': 'shadow_mask_prediction', 'label': 'Shadow Mask / Shadow Matte Prediction', 'status': 'done', 'time_ms': 0, 'info': f'Mask mean: {info["mask_mean"]:.4f}'},
        {'name': 'illumination_map_estimation', 'label': 'Illumination Map Estimation', 'status': 'done', 'time_ms': 0, 'info': f'Illumination mean: {info["illumination_mean"]:.4f}'},
        {'name': 'shadow_correction', 'label': 'Shadow Correction', 'status': 'done', 'time_ms': 0, 'info': 'AI restoration blended with illumination gain'},
        {'name': 'white_balance', 'label': 'White Balance', 'status': 'done', 'time_ms': 0, 'info': 'Gray-world white balance'},
        {'name': 'local_contrast_enhancement', 'label': 'Local Contrast Enhancement', 'status': 'done', 'time_ms': 0, 'info': 'CLAHE on L channel'},
        {'name': 'final_document', 'label': 'Final Document', 'status': 'done', 'time_ms': total_ms, 'info': f'Total pipeline time: {total_ms}ms'},
    ]
    return {
        'success': True,
        'total_ms': total_ms,
        'steps': steps,
        'pipeline': info['pipeline'],
        'image': f'data:image/png;base64,{_pil_to_b64(final)}',
    }

def _training_log_reader(proc):
    global _training_log, _training_process
    for line in iter(proc.stdout.readline, ''):
        if not line:
            break
        with _training_lock:
            _append_training_log(line.rstrip())
    proc.wait()
    with _training_lock:
        _append_training_log(f'[INFO] Training process exited with code {proc.returncode}')
    _training_process = None


@app.post("/api/training/start")
async def start_training(request: Request,
                         clean_data: str = Form(''),
                         paired_data: str = Form(''),
                         validation_paired_data: str = Form(''),
                         identity_data: str = Form(''),
                         output: str = Form('checkpoints/document_restorer'),
                         epochs: int = Form(40),
                         batch_size: int = Form(8),
                         size: int = Form(512),
                         lr: float = Form(1e-4),
                         base_channels: int = Form(32),
                         workers: int = Form(4),
                         device: str = Form('auto'),
                         resume: str = Form(''),
                         resume_weights_only: bool = Form(False),
                         early_stop_patience: int = Form(7),
                         min_delta: float = Form(1e-4),
                         grad_clip_norm: float = Form(1.0),
                         perceptual_weight: float = Form(0.05),
                         ssim_weight: float = Form(0.1),
                         shadow_loss_weight: float = Form(1.25),
                         illumination_weight: float = Form(0.35),
                         mask_loss_weight: float = Form(0.2),
                         gradient_weight: float = Form(0.15),
                         color_weight: float = Form(0.08),
                         identity_weight: float = Form(0.25),
                         max_train_samples: int = Form(0),
                         max_val_samples: int = Form(0)):
    require_api_auth(request)
    global _training_process, _training_log, _training_started_at, _training_estimated_finish_at, _training_kind

    if _training_process is not None and _training_process.poll() is None:
        raise HTTPException(status_code=409, detail='Training already running')

    _validate_training_params(epochs, batch_size, size, lr, base_channels, workers)
    _validate_training_controls(early_stop_patience, min_delta, grad_clip_norm)
    _validate_loss_weights(
        perceptual_weight=perceptual_weight,
        ssim_weight=ssim_weight,
        shadow_loss_weight=shadow_loss_weight,
        illumination_weight=illumination_weight,
        mask_loss_weight=mask_loss_weight,
        gradient_weight=gradient_weight,
        color_weight=color_weight,
        identity_weight=identity_weight,
    )
    output = _safe_training_output(output)
    publish_output = output
    run_id = datetime.now(tz=WIB).strftime('%Y%m%d-%H%M%S-%f')
    run_output = f'{output}/runs/{run_id}'

    cmd = [
        sys.executable, str(BASE_DIR / 'train.py'),
        '--output', run_output,
        '--publish-output', publish_output,
        '--run-id', run_id,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--size', str(size),
        '--lr', str(lr),
        '--base-channels', str(base_channels),
        '--workers', str(workers),
        '--device', device,
        '--early-stop-patience', str(early_stop_patience),
        '--min-delta', str(min_delta),
        '--grad-clip-norm', str(grad_clip_norm),
        '--perceptual-weight', str(perceptual_weight),
        '--ssim-weight', str(ssim_weight),
        '--shadow-loss-weight', str(shadow_loss_weight),
        '--illumination-weight', str(illumination_weight),
        '--mask-loss-weight', str(mask_loss_weight),
        '--gradient-weight', str(gradient_weight),
        '--color-weight', str(color_weight),
        '--identity-weight', str(identity_weight),
    ]
    if clean_data:
        cmd.extend(['--clean-data'] + clean_data.split(','))
    if paired_data:
        cmd.extend(['--paired-data'] + paired_data.split(','))
    if validation_paired_data:
        cmd.extend(['--validation-paired-data'] + validation_paired_data.split(','))
    if identity_data:
        cmd.extend(['--identity-data'] + identity_data.split(','))
    if resume:
        cmd.extend(['--resume', resume])
        if resume_weights_only:
            cmd.append('--resume-weights-only')
    if max_train_samples > 0:
        cmd.extend(['--max-train-samples', str(max_train_samples)])
    if max_val_samples > 0:
        cmd.extend(['--max-validation-samples', str(max_val_samples)])

    with _training_lock:
        _training_started_at = time.time()
        _training_estimated_finish_at = _estimate_training_finish(_training_started_at, paired_data, epochs, batch_size, size, max_train_samples)
        _training_kind = 'training'
        _set_training_log([f'[CMD] {" ".join(cmd)}'])
        _record_training_meta(_training_kind, _training_started_at, _training_estimated_finish_at, cmd)
        meta = _stored_training_meta()
        meta.update({'run_id': run_id, 'run_output': run_output, 'publish_output': publish_output})
        TRAINING_META_PATH.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    log_handle = TRAINING_LOG_PATH.open('a', encoding='utf-8')
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR), env={**os.environ, 'PYTHONPATH': str(BASE_DIR), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        text=True, bufsize=1, start_new_session=True,
    )
    log_handle.close()
    _training_process = proc
    _record_training_pid(proc.pid)
    threading.Thread(target=_monitor_training_process, args=(proc,), daemon=True).start()
    return {'success': True, 'pid': proc.pid, 'run_id': run_id, 'run_output': run_output}


@app.post("/api/training/stop")
async def stop_training(request: Request):
    require_api_auth(request)
    global _training_process
    pid = _training_process.pid if _training_process is not None and _training_process.poll() is None else _stored_training_pid()
    if not _pid_running(pid):
        return {'success': True, 'message': 'No training running'}
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if _training_process is not None:
        try:
            _training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _training_process.wait(timeout=5)
    try:
        TRAINING_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    _training_process = None
    return {'success': True, 'message': 'Training stopped'}


@app.post("/api/training/evaluate")
async def evaluate_training(request: Request,
                            paired_data: str = Form('datasets/ShadowDocument7K/test'),
                            checkpoint: str = Form('checkpoints/document_restorer/best.pth'),
                            output: str = Form('evaluation/document_restorer'),
                            size: int = Form(768),
                            batch_size: int = Form(1),
                            workers: int = Form(4),
                            device: str = Form('cuda'),
                            max_samples: int = Form(0),
                            pipeline: bool = Form(True)):
    require_api_auth(request)
    global _training_process, _training_log, _training_started_at, _training_estimated_finish_at, _training_kind

    if _training_process is not None and _training_process.poll() is None:
        raise HTTPException(status_code=409, detail='Training or evaluation already running')

    _validate_training_params(1, batch_size, size, 1e-4, 32, workers)
    checkpoint = _safe_checkpoint_path(checkpoint)
    output = _safe_evaluation_output(output)

    cmd = [
        sys.executable, str(BASE_DIR / 'evaluate_restorer.py'),
        '--paired-data', paired_data,
        '--checkpoint', checkpoint,
        '--output', output,
        '--size', str(size),
        '--batch-size', str(batch_size),
        '--workers', str(workers),
        '--device', device,
    ]
    if max_samples > 0:
        cmd.extend(['--max-samples', str(max_samples)])
    if pipeline:
        cmd.append('--pipeline')

    with _training_lock:
        _training_started_at = time.time()
        _training_estimated_finish_at = _estimate_evaluation_finish(_training_started_at, paired_data, batch_size, size, max_samples)
        _training_kind = 'evaluation'
        _set_training_log([f'[EVAL CMD] {" ".join(cmd)}'])
        _record_training_meta(_training_kind, _training_started_at, _training_estimated_finish_at, cmd)

    log_handle = TRAINING_LOG_PATH.open('a', encoding='utf-8')
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR), env={**os.environ, 'PYTHONPATH': str(BASE_DIR), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        text=True, bufsize=1, start_new_session=True,
    )
    log_handle.close()
    _training_process = proc
    _record_training_pid(proc.pid)
    threading.Thread(target=_monitor_training_process, args=(proc,), daemon=True).start()
    return {'success': True, 'pid': proc.pid}

@app.post("/api/models/export-mobile")
async def export_mobile_model(request: Request, checkpoint: str = Form(...)):
    require_api_auth(request)
    safe_checkpoint = _safe_checkpoint_path(checkpoint)
    checkpoint_path = BASE_DIR / safe_checkpoint
    run_name = checkpoint_path.parent.name
    export_dir = CHECKPOINT_DIR / 'document_restorer' / 'mobile' / run_name
    export_path = export_dir / f'{checkpoint_path.stem}.onnx'
    command = [
        sys.executable, str(BASE_DIR / 'export_mobile_model.py'),
        '--checkpoint', safe_checkpoint,
        '--output', str(export_path.relative_to(BASE_DIR)),
    ]
    result = subprocess.run(command, cwd=str(BASE_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=(result.stderr or result.stdout or 'Mobile export failed')[-2000:])
    return {
        'success': True,
        'model': export_path.relative_to(BASE_DIR).as_posix(),
        'metadata': export_path.with_suffix('.json').relative_to(BASE_DIR).as_posix(),
    }

@app.get("/api/training/log")
async def training_log(request: Request, offset: int = 0):
    require_api_auth(request)
    with _training_lock:
        running = _training_is_running()
        if TRAINING_LOG_PATH.exists():
            _load_training_log(TRAINING_LOG_PATH.read_text(encoding='utf-8').splitlines())
        lines = _training_log[offset:]
    return {'running': running, 'lines': lines, 'total': len(_training_log)}
