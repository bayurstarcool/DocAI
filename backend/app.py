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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse, Response

import time
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps, ImageEnhance

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.image_utils import run_document_restoration_pipeline, run_tiled_restoration
from backend.models.document_restorer import DocumentRestorerNet
from backend.auth import ADMIN_CREDENTIALS, create_token, is_authenticated, register_intern, validate_intern, validate_token_any, get_token_from_request, validate_token
from backend.workspace import WorkspaceError, resolve_workspace_path, safe_workspace_name
from backend.utils.traditional_methods import (
    deskew_document, remove_color_cast, adaptive_threshold_document,
    enhance_text_sharpness, remove_background_noise, enhance_contrast_clahe,
    full_document_cleanup,
)
from backend.gpu_manager_simple import unload_idle_models, mark_used
from backend.utils.shadowremove_enhance import (
    magic_document_enhance, adaptive_binarize, ai_shadow_postprocess,
)
from backend.utils.so_shadow_removal import (
    so_shadow_removal, so_shadow_removal_enhanced, so_shadow_removal_aggressive,
)
from backend.utils.traditional_shadow_methods import effective_bg_estimation, iterative_removal, color_binarize
from backend.models.shadow_remover import ShadowRemoverNet
from backend.models.doc_enhancer import DocEnhancerNet

from backend.models.docres_model import get_docres, get_docres_base, get_docres_finetune, DocResONNX
from backend.utils.image_adjust import apply_adjustments
import contextvars
_ADJUST_CTX = contextvars.ContextVar("adjust_ctx", default=None)


# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent
DOCRES_DIR = Path('/home/wahyu/DocRes')
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
def _get_inference_device():
    import json as _json
    st_path = BASE_DIR.parent / "DocRes" / "finetune_logs" / "docres_status.json"
    if st_path.exists():
        try:
            st = _json.loads(st_path.read_text())
            if st.get("running"):
                print("[INF] Training active -> CPU inference")
                return torch.device("cpu")
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = _get_inference_device()
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


# Lazy-load Document Restorer
doc_restorer_model = None
doc_restorer_checkpoint = None
_doc_restorer_loaded = False
def get_doc_restorer_model():
    global doc_restorer_model, doc_restorer_checkpoint, _doc_restorer_loaded
    if not _doc_restorer_loaded:
        _doc_restorer_loaded = True
        doc_restorer_model = _load_checkpoint(DocumentRestorerNet, MODEL_CHECKPOINT_PATH, device)
        doc_restorer_checkpoint = str(MODEL_CHECKPOINT_PATH.relative_to(BASE_DIR)) if doc_restorer_model is not None else None
    return doc_restorer_model

# Lazy-load Shadow Remover
shadow_remover_model = None
_shadow_remover_loaded = False
def get_shadow_remover_model():
    global shadow_remover_model, _shadow_remover_loaded
    if not _shadow_remover_loaded:
        _shadow_remover_loaded = True
        shadow_remover_model = _load_checkpoint(ShadowRemoverNet, CHECKPOINT_DIR / 'shadow_remover' / 'best.pth', device)
    return shadow_remover_model

# Lazy-load Doc Enhancer
doc_enhancer_model = None
_doc_enhancer_loaded = False
def get_doc_enhancer_model():
    global doc_enhancer_model, _doc_enhancer_loaded
    if not _doc_enhancer_loaded:
        _doc_enhancer_loaded = True
        doc_enhancer_model = _load_checkpoint(DocEnhancerNet, CHECKPOINT_DIR / 'doc_enhancer' / 'best.pth', device)
    return doc_enhancer_model

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
    mark_used("shadow")
    return _docshadow


# Lazy-load DocRes
_docres = None
def get_docres_model():
    global _docres
    if _docres is None:
        try:
            _docres = get_docres(device=str(device))
        except Exception as e:
            print(f"[WARN] DocRes not available: {e}")
    mark_used("docres")
    return _docres


# Lazy-load DocRes variants
_docres_base = None
def get_docres_base_model():
    global _docres_base
    if _docres_base is None:
        try: _docres_base = get_docres_base(device=str(device))
        except Exception as e: print(f"[WARN] DocRes base not available: {e}")
    mark_used("base")
    return _docres_base

_docres_ft = None
_docres_ft_variant = None
def get_docres_ft_model(variant='finetune_v1'):
    global _docres_ft, _docres_ft_variant
    if _docres_ft is None or _docres_ft_variant != variant:
        try: _docres_ft = get_docres_finetune(variant, device=str(device)); _docres_ft_variant = variant
        except Exception as e: print(f"[WARN] DocRes finetune not available: {e}")
    return _docres_ft


_docres_onnx = None
def get_docres_onnx_model():
    global _docres_onnx
    if _docres_onnx is None:
        try:
            _docres_onnx = DocResONNX(
                onnx_path=DOCRES_DIR / "onnx_exports" / "docres_base_appearance.onnx",
                im_size=1280,
            )
        except Exception as e:
            print(f"[WARN] DocRes ONNX load failed: {e}")
            return None
    return _docres_onnx


_appearance_mixed = None
_docres_task = None
def get_docres_task_model():
    global _docres_task
    if _docres_task is None:
        import glob as _g
        from pathlib import Path as _P
        from backend.models.docres_model import DocResModel as _DRM
        runs=sorted([d for d in (_P(BASE_DIR).parent/"DocRes"/"finetune_logs").glob("base_multitask_*") if (d/"best.pth").exists()],key=lambda x:x.stat().st_mtime,reverse=True)
        if runs:
            try:
                _dev = "cpu" if str(device)=="cpu" else str(device); _docres_task = _DRM(runs[0]/"best.pth", device=_dev, im_size=768, label="multitask_finetuned")
                print(f"[OK] DocRes task model: fine-tuned {runs[0].name}/best.pth")
            except Exception as e:
                print(f"[WARN] Fine-tuned task model failed: {e}")
        if _docres_task is None:
            base = get_docres_base_model()
            if base is None:
                try: base = get_docres(device=str(device))
                except Exception as e: print(f"[WARN] DocRes task model not available: {e}")
            _docres_task = base
    mark_used("task")
    return _docres_task


def docres_task_infer(img, task='deshadowing'):
    mdl = get_docres_task_model()
    if mdl is None:
        raise HTTPException(status_code=500, detail=f"DocRes not loaded")
    return mdl.infer_task(img, task)


# --- FastAPI Setup ---
spa_dist = BASE_DIR / 'frontend' / 'dist'
@app.get("/opencv.js")
async def serve_opencv():
    return FileResponse(str(spa_dist / "opencv.js"), media_type="application/javascript")
@app.get("/canny_edge.js")
async def serve_canny_edge():
    return FileResponse(str(spa_dist / "canny_edge.js"), media_type="application/javascript")
login_html = BASE_DIR / 'frontend' / 'templates_bak' / 'login.html'
app.mount("/assets", StaticFiles(directory=str(spa_dist / 'assets')), name="spa-assets")
# removed
fonts_dir = BASE_DIR / 'frontend' / 'public' / 'fonts'
app.mount("/fonts", StaticFiles(directory=str(fonts_dir)), name="local-fonts")


def require_page_auth(request: Request):
    if not is_authenticated(request):
        return RedirectResponse('/login', status_code=303)
    return None


def require_api_auth(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail='Unauthorized')


def require_admin_page_auth(request: Request):
    token = get_token_from_request(request)
    username = validate_token_any(token) if token else None
    if not username:
        return RedirectResponse('/login', status_code=303)
    if username != ADMIN_CREDENTIALS['username']:
        return RedirectResponse('/magang', status_code=303)
    return None


def require_admin_api_auth(request: Request):
    token = get_token_from_request(request)
    username = validate_token_any(token) if token else None
    if username != ADMIN_CREDENTIALS['username']:
        raise HTTPException(status_code=403, detail='Hanya admin')


def render_page(request: Request, template_name: str = None):
    redirect = require_page_auth(request)
    if redirect:
        return redirect
    html = (spa_dist / 'index.html').read_text()
    if 'canny_edge.js' not in html:
        html = html.replace('</head>', '<script src="/canny_edge.js"></script>\n</head>')
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


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
    _adj = _ADJUST_CTX.get()
    if _adj:
        try:
            img = apply_adjustments(img, **_adj)
        except Exception as _e:
            print(f"[WARN] adjustment failed: {_e}")
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

@app.get("/docres")
async def docres_page(request: Request):
    return render_page(request)


@app.get("/image-tests")
async def image_tests_page(request: Request):
    return render_page(request, "image_tests.html")


@app.get("/download")
async def download_page(request: Request):
    return render_page(request, "download.html")

@app.get("/datasets")
async def datasets_page(request: Request):
    return render_page(request, "datasets.html")


@app.get("/datasets/{name}")
async def datasets_detail_page(request: Request, name: str):
    return render_page(request, "datasets.html")


@app.get("/dataset-manager")
async def dataset_manager_page(request: Request):
    return render_page(request)


@app.get("/dataset-manager/{slug}")
async def dataset_manager_detail_page(request: Request, slug: str):
    return render_page(request)


@app.get("/synthetic-shadow")
async def synthetic_shadow_page(request: Request):
    return render_page(request)


@app.get("/synthetic-dewarp")
async def synthetic_dewarp_page(request: Request):
    redirect = require_admin_page_auth(request)
    if redirect:
        return redirect
    return FileResponse(str(spa_dist / 'index.html'))


@app.get("/magang")
async def magang_page(request: Request):
    return FileResponse(str(spa_dist / 'index.html'))


@app.get("/magang-review")
async def magang_review_page(request: Request):
    return FileResponse(str(spa_dist / 'index.html'))


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


@app.post("/api/auth/register")
async def register(username: str = Form(...), password: str = Form(...)):
    result = register_intern(username, password)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    # Auto-login after register
    token = create_token(username)
    response = JSONResponse({"success": True, "token": token, "user": username, "message": "Registrasi berhasil. Sekarang kamu bisa upload dataset!"})
    response.set_cookie("docai_token", token, httponly=True, samesite="lax", max_age=24 * 60 * 60)
    return response


@app.post("/api/auth/login/intern")
async def login_intern(username: str = Form(...), password: str = Form(...)):
    if validate_intern(username, password):
        token = create_token(username)
        response = JSONResponse({"success": True, "token": token, "user": username})
        response.set_cookie("docai_token", token, httponly=True, samesite="lax", max_age=24 * 60 * 60)
        return response
    return JSONResponse({"success": False, "message": "Username atau password salah"}, status_code=401)


@app.get("/api/auth/logout")
async def logout():
    response = JSONResponse({'success': True})
    response.delete_cookie('docai_token')
    return response


@app.get("/api/auth/check")
async def auth_check(request: Request):
    return {'authenticated': is_authenticated(request)}



# =====================================================================
#  MAGANG (INTERN) DATASET APIs
# =====================================================================

@app.get("/api/magang/list")
async def magang_list(request: Request):
    """List datasets milik intern sendiri, atau semua (admin)."""
    from backend.auth import ADMIN_CREDENTIALS
    token = get_token_from_request(request)
    username = validate_token_any(token)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")
    is_admin = (username == ADMIN_CREDENTIALS["username"])
    base = BASE_DIR / "datasets" / "magang"
    base.mkdir(parents=True, exist_ok=True)

    result = []
    if is_admin:
        # Admin lihat semua
        for user_dir in sorted(base.iterdir()):
            if user_dir.is_dir():
                info = _scan_magang(user_dir)
                if info:
                    result.append(info)
    else:
        # Intern lihat milik sendiri
        user_dir = base / username
        if user_dir.exists():
            info = _scan_magang(user_dir)
            if info:
                result.append(info)
        else:
            user_dir.mkdir(parents=True, exist_ok=True)
    return {"datasets": result}


def _scan_magang(user_dir: Path) -> dict:
    input_dir = user_dir / "input"
    target_dir = user_dir / "target"
    IMG = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    inputs = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG] if input_dir.exists() else []
    targets = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in IMG] if target_dir.exists() else []
    input_stems = {f.stem for f in inputs}
    target_stems = {f.stem for f in targets}
    paired = len(input_stems & target_stems)
    return {
        "username": user_dir.name,
        "path": str(user_dir.relative_to(BASE_DIR)),
        "input_count": len(inputs),
        "target_count": len(targets),
        "paired_count": paired,
        "updated_at": user_dir.stat().st_mtime,
    }


@app.post("/api/magang/audit")
async def magang_audit(request: Request):
    """Audit pair tanpa menyimpan."""
    token = get_token_from_request(request)
    username = validate_token_any(token)
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized")

    form = await request.form()
    shadow_file = form.get("shadow")
    clean_file = form.get("clean")
    if not shadow_file or not clean_file:
        raise HTTPException(status_code=400, detail="Shadow dan clean file required")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        shadow_path = os.path.join(tmpdir, "shadow.jpg")
        clean_path = os.path.join(tmpdir, "clean.jpg")
        with open(shadow_path, "wb") as f:
            f.write(await shadow_file.read())
        with open(clean_path, "wb") as f:
            f.write(await clean_file.read())

        from backend.utils.dataset_audit import audit_pair
        result = audit_pair(shadow_path, clean_path)

    score = result.get("score", 0)
    return {
        "score": score,
        "grade": result.get("grade", "D"),
        "reasons": result.get("reasons", []),
        "passed": score >= 90,
        "min_score": 90,
        "dimensions": result.get("dimensions", {}),
        "luminance": result.get("luminance", {}),
    }

@app.post("/api/magang/upload")
async def magang_upload(request: Request, shadow: UploadFile = File(...), clean: UploadFile = File(...)):
    """Intern upload pair ke foldernya sendiri."""
    from backend.auth import ADMIN_CREDENTIALS
    token = get_token_from_request(request)
    username = validate_token_any(token)
    if not username or username == ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=401, detail="Hanya intern yang bisa upload")
    
    base = BASE_DIR / "datasets" / "magang" / username
    input_dir = base / "input"
    target_dir = base / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Validate images
    shadow_bytes = await shadow.read()
    clean_bytes = await clean.read()
    if len(shadow_bytes) == 0 or len(clean_bytes) == 0:
        raise HTTPException(status_code=400, detail="File tidak boleh kosong")

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    shadow_ext = Path(shadow.filename or "image.png").suffix.lower()
    clean_ext = Path(clean.filename or "image.png").suffix.lower()
    if shadow_ext not in IMAGE_EXTS or clean_ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Format file tidak didukung")
    if len(shadow_bytes) < 50 * 1024 or len(clean_bytes) < 50 * 1024:
        raise HTTPException(status_code=400, detail="File terlalu kecil (<50KB)")

    # Cek duplikat: bandingkan hash file shadow
    import hashlib
    shadow_hash = hashlib.md5(shadow_bytes).hexdigest()
    clean_hash = hashlib.md5(clean_bytes).hexdigest()

    if shadow_hash == clean_hash:
        raise HTTPException(status_code=400, detail="File shadow dan clean sama! Upload dua foto yang berbeda.")

    for existing_f in list(input_dir.glob("*.*")):
        if existing_f.suffix.lower() not in IMAGE_EXTS:
            continue
        with open(existing_f, "rb") as ef:
            existing_bytes = ef.read()
        if hashlib.md5(existing_bytes).hexdigest() == shadow_hash:
            raise HTTPException(status_code=400, detail=f"File shadow sudah pernah diupload ({existing_f.name}). Upload gambar baru.")
        if hashlib.md5(existing_bytes).hexdigest() == clean_hash:
            raise HTTPException(status_code=400, detail=f"File clean sudah pernah diupload ({existing_f.name}). Upload gambar baru.")

    # Find next index
    existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    idx = len(existing) + 1
    stem = f"pair_{idx:04d}"

    shadow_path = input_dir / f"{stem}{shadow_ext}"
    clean_path = target_dir / f"{stem}{clean_ext}"

    with open(shadow_path, "wb") as f:
        f.write(shadow_bytes)
    with open(clean_path, "wb") as f:
        f.write(clean_bytes)

    return {
        "success": True,
        "pair": stem,
        "input": str(shadow_path.relative_to(BASE_DIR)),
        "target": str(clean_path.relative_to(BASE_DIR)),
    }


def _find_magang_file(base: Path, username: str, kind: str, pair_name: str):
    folder = base / username / kind
    for ext in IMAGE_EXTENSIONS:
        cand = folder / f"{pair_name}{ext}"
        if cand.exists():
            return cand
    return None


def _validate_magang_pair(username: str, pair_name: str):
    base = BASE_DIR / "datasets" / "magang"
    src_input = _find_magang_file(base, username, "input", pair_name)
    src_target = _find_magang_file(base, username, "target", pair_name)
    issues = []
    info = {"valid": False, "issues": issues, "input": None, "target": None}
    if not src_input:
        issues.append("input missing")
    if not src_target:
        issues.append("target missing")
    if issues:
        return info
    try:
        with Image.open(src_input) as im:
            im.verify()
        with Image.open(src_target) as im:
            im.verify()
        with Image.open(src_input) as im_in, Image.open(src_target) as im_tg:
            in_size = im_in.size
            tg_size = im_tg.size
        info["input"] = {"path": str(src_input.relative_to(BASE_DIR / 'datasets' / 'magang')), "size": in_size, "bytes": src_input.stat().st_size}
        info["target"] = {"path": str(src_target.relative_to(BASE_DIR / 'datasets' / 'magang')), "size": tg_size, "bytes": src_target.stat().st_size}
        if min(in_size) < 64:
            issues.append("input too small")
        if min(tg_size) < 64:
            issues.append("target too small")
        if src_input.stat().st_size == 0:
            issues.append("input empty")
        if src_target.stat().st_size == 0:
            issues.append("target empty")
        try:
            from backend.utils.dataset_audit import audit_pair
            audit = audit_pair(str(src_input), str(src_target))
            score = float(audit.get("score", 0) or 0)
            info["audit"] = {
                "score": score,
                "grade": audit.get("grade", "D"),
                "reasons": audit.get("reasons", []),
                "dimensions": audit.get("dimensions", {}),
                "luminance": audit.get("luminance", {}),
                "min_score": 90,
                "passed": score >= 90,
            }
            if score < 90:
                issues.append(f"score below 90: {score:.1f}")
        except Exception as e:
            issues.append(f"audit failed: {e}")
    except Exception as e:
        issues.append(f"image invalid: {e}")
    info["valid"] = len(issues) == 0
    return info


@app.get("/api/magang/review")
async def magang_review(request: Request):
    """Admin review: lihat semua dataset magang + thumbnail."""
    from backend.auth import ADMIN_CREDENTIALS
    token = get_token_from_request(request)
    username = validate_token_any(token)
    if username != ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=403, detail="Hanya admin")
    
    base = BASE_DIR / "datasets" / "magang"
    if not base.exists():
        return {"users": [], "total_pairs": 0}
    
    users = []
    total_pairs = 0
    for user_dir in sorted(base.iterdir()):
        if not user_dir.is_dir():
            continue
        info = _scan_magang(user_dir)
        if not info:
            continue
        total_pairs += info["paired_count"]
        
        # List pairs
        input_dir = user_dir / "input"
        target_dir = user_dir / "target"
        IMG = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        inputs = {}
        if input_dir.exists():
            for f in input_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG:
                    inputs[f.stem] = str(f.relative_to(BASE_DIR / 'datasets' / 'magang'))
        targets = {}
        if target_dir.exists():
            for f in target_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMG:
                    targets[f.stem] = str(f.relative_to(BASE_DIR / 'datasets' / 'magang'))
        
        pairs_list = []
        for stem in sorted(set(list(inputs.keys()) + list(targets.keys()))):
            paired = stem in inputs and stem in targets
            # Fast list only. Heavy audit score runs lazily via /api/magang/validate-pair
            pairs_list.append({
                "name": stem,
                "input": inputs.get(stem),
                "target": targets.get(stem),
                "paired": paired,
                "validation": {"valid": paired, "issues": [] if paired else ["unpaired"]},
                "valid": paired,
            })
        
        users.append({
            "username": user_dir.name,
            "path": str(user_dir.relative_to(BASE_DIR)),
            "input_count": info["input_count"],
            "target_count": info["target_count"],
            "paired_count": info["paired_count"],
            "pairs": pairs_list,
        })
    
    return {"users": users, "total_pairs": total_pairs}


@app.post("/api/magang/validate-pair")
async def magang_validate_pair(request: Request):
    token = get_token_from_request(request)
    admin_user = validate_token_any(token)
    if admin_user != ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=403, detail="Hanya admin")
    body = await request.json()
    username = body.get("username", "")
    pair_name = body.get("pair_name", "")
    if not username or not pair_name:
        raise HTTPException(status_code=400, detail="username dan pair_name required")
    return _validate_magang_pair(username, pair_name)


def _move_magang_pair_to_custom(username: str, pair_name: str):
    """Validate + move a single magang pair to custom. Returns (ok, detail_dict)."""
    validation = _validate_magang_pair(username, pair_name)
    if not validation.get("valid"):
        return False, {"pair_name": pair_name, "moved": False,
                       "error": "invalid", "validation": validation}
    base = BASE_DIR / "datasets" / "magang"
    src_input = _find_magang_file(base, username, "input", pair_name)
    src_target = _find_magang_file(base, username, "target", pair_name)
    if not src_input or not src_target:
        return False, {"pair_name": pair_name, "moved": False, "error": "not_found"}

    custom_input = BASE_DIR / "datasets" / "paired" / "custom" / "custom" / "input"
    custom_target = BASE_DIR / "datasets" / "paired" / "custom" / "custom" / "target"
    custom_input.mkdir(parents=True, exist_ok=True)
    custom_target.mkdir(parents=True, exist_ok=True)

    existing = list(custom_input.glob("pair_*.png"))
    idx = len(existing) + 1
    new_stem = f"pair_{idx:04d}"
    # avoid collision if indices are sparse
    while (custom_input / f"{new_stem}.png").exists():
        idx += 1
        new_stem = f"pair_{idx:04d}"

    with Image.open(src_input) as im:
        ImageOps.exif_transpose(im).convert("RGB").save(custom_input / f"{new_stem}.png", "PNG")
    with Image.open(src_target) as im:
        ImageOps.exif_transpose(im).convert("RGB").save(custom_target / f"{new_stem}.png", "PNG")

    src_input.unlink()
    src_target.unlink()
    return True, {"pair_name": pair_name, "moved": True, "new_pair": new_stem}


@app.post("/api/magang/move")
async def magang_move(request: Request):
    """Admin memindahkan pair dari magang ke custom."""
    from backend.auth import ADMIN_CREDENTIALS
    token = get_token_from_request(request)
    admin = validate_token_any(token)
    if admin != ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=403, detail="Hanya admin")

    body = await request.json()
    username = body.get("username", "")
    pair_name = body.get("pair_name", "")
    if not username or not pair_name:
        raise HTTPException(status_code=400, detail="username dan pair_name required")

    ok, detail = _move_magang_pair_to_custom(username, pair_name)
    if not ok:
        raise HTTPException(status_code=400, detail={"message": "Gagal pindah", **detail})
    return {"success": True, "new_pair": detail["new_pair"],
            "from": f"magang/{username}", "to": "custom/custom"}


@app.post("/api/magang/move-bulk")
async def magang_move_bulk(request: Request):
    """Admin bulk-move: pindahkan banyak pair sekaligus dari magang ke custom.
    Body: {username, pair_names:[...]}. Returns per-pair result + summary."""
    from backend.auth import ADMIN_CREDENTIALS
    token = get_token_from_request(request)
    admin = validate_token_any(token)
    if admin != ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=403, detail="Hanya admin")

    body = await request.json()
    username = body.get("username", "")
    pair_names = body.get("pair_names", []) or []
    if not username or not isinstance(pair_names, list) or not pair_names:
        raise HTTPException(status_code=400, detail="username dan pair_names[] required")

    results = []
    moved = 0
    for pn in pair_names:
        try:
            ok, detail = _move_magang_pair_to_custom(username, str(pn))
        except Exception as e:
            ok, detail = False, {"pair_name": str(pn), "moved": False, "error": str(e)}
        if ok:
            moved += 1
        results.append(detail)
    return {"success": True, "moved": moved, "requested": len(pair_names), "results": results}



# =====================================================================
#  SYNTHETIC DEWARP DATASET CURATION (ADMIN ONLY) — realtime jobs
# =====================================================================
import threading as _threading

SYNTH_DEWARP_ROOT = BASE_DIR.parent / "controlpoints" / "web_synthetic"
SYNTH_DEWARP_CANDIDATES = SYNTH_DEWARP_ROOT / "candidates" / "color"
SYNTH_DEWARP_SELECTED = SYNTH_DEWARP_ROOT / "selected" / "color"
SYNTH_DEWARP_REJECTED = SYNTH_DEWARP_ROOT / "rejected" / "color"
SYNTH_DEWARP_PREVIEW = SYNTH_DEWARP_ROOT / "preview"
SYNTH_DEWARP_UPLOADS = SYNTH_DEWARP_ROOT / "uploads"
SYNTH_DEWARP_CUSTOM = BASE_DIR / "datasets" / "controlpoints_custom" / "color"
SYNTH_GEN_SCRIPT = "/home/wahyu/synthetic-controlpoints/perturbed_images_generation_multiProcess.py"
SYNTH_GEN_CWD = "/home/wahyu/synthetic-controlpoints"
SYNTH_GEN_BG = "/home/wahyu/synthetic-controlpoints/background/"
for _p in [SYNTH_DEWARP_CANDIDATES, SYNTH_DEWARP_SELECTED, SYNTH_DEWARP_REJECTED, SYNTH_DEWARP_PREVIEW, SYNTH_DEWARP_UPLOADS, SYNTH_DEWARP_CUSTOM]:
    _p.mkdir(parents=True, exist_ok=True)

# In-memory job registry for realtime synthetic generation
_synth_jobs = {}
_synth_jobs_lock = _threading.Lock()


def _admin_only(request: Request):
    token = get_token_from_request(request)
    username = validate_token_any(token) if token else None
    if username != ADMIN_CREDENTIALS["username"]:
        raise HTTPException(status_code=403, detail="Hanya admin")


def _synth_item_from_gw(path: Path):
    return {
        "name": path.name,
        "stem": path.stem,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "image_url": f"/api/synthetic-dewarp/preview/{path.stem}.jpg",
        "points_url": f"/api/synthetic-dewarp/preview/{path.stem}_points.jpg",
    }


def _render_gw_preview(gw_path: Path):
    import pickle
    with open(gw_path, 'rb') as f:
        d = pickle.load(f)
    img = d.get('image')
    pts = d.get('fiducial_points')
    if img is None:
        raise HTTPException(status_code=400, detail='gw image missing')
    jpg = SYNTH_DEWARP_PREVIEW / f"{gw_path.stem}.jpg"
    pts_jpg = SYNTH_DEWARP_PREVIEW / f"{gw_path.stem}_points.jpg"
    cv2.imwrite(str(jpg), img)
    mark = img.copy()
    if pts is not None:
        flat = pts.reshape(-1, 2)
        step = max(1, len(flat) // 160)
        for pt in flat[::step]:
            x, y = int(pt[1]), int(pt[0])
            if 0 <= x < mark.shape[1] and 0 <= y < mark.shape[0]:
                cv2.circle(mark, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite(str(pts_jpg), mark)


def _resolve_clean_input(input_name: str, source: str):
    """Return absolute path of a clean input from inv3d target or uploads."""
    safe = Path(input_name).name
    if source == "upload":
        cand = SYNTH_DEWARP_UPLOADS / safe
    else:
        cand = BASE_DIR / "datasets" / "paired" / "inv3d" / "target" / safe
    return cand if cand.exists() else None


def _synth_worker(job_id: str, clean_path: Path, count: int, input_stem: str):
    import shutil, subprocess, time, glob
    job = _synth_jobs[job_id]
    try:
        run_root = SYNTH_DEWARP_ROOT / "runs" / job_id
        scan_dir = run_root / "scan"
        out_dir = run_root / "out"
        scan_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clean_path, scan_dir / clean_path.name)
        cmd = [
            "/home/wahyu/miniconda3/bin/python3", SYNTH_GEN_SCRIPT,
            "--path", str(scan_dir) + "/",
            "--bg_path", SYNTH_GEN_BG,
            "--output_path", str(out_dir) + "/",
            "--sys_num", str(max(1, (count + 1) // 2)),
        ]
        proc = subprocess.Popen(cmd, cwd=SYNTH_GEN_CWD, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, text=True)
        with _synth_jobs_lock:
            job["pid"] = proc.pid
            job["status"] = "running"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        seen = set()
        deadline = time.time() + 600
        while time.time() < deadline:
            if job.get("cancel"):
                break
            done = len(job["items"])
            if done >= count:
                break
            for f in sorted((out_dir / "color").glob("*.gw")):
                if f.name in seen:
                    continue
                # ensure file finished writing (size stable)
                try:
                    sz1 = f.stat().st_size
                    time.sleep(0.4)
                    if f.stat().st_size != sz1:
                        continue
                except OSError:
                    continue
                seen.add(f.name)
                dst = SYNTH_DEWARP_CANDIDATES / f"{input_stem}_{ts}_{len(job['items'])+1}_{f.name}"
                shutil.copy2(f, dst)
                try:
                    _render_gw_preview(dst)
                except Exception as e:
                    print(f"[WARN] preview failed: {e}")
                with _synth_jobs_lock:
                    job["items"].append(_synth_item_from_gw(dst))
                if len(job["items"]) >= count:
                    break
            if proc.poll() is not None and not list((out_dir / "color").glob("*.gw")):
                # process ended without producing; small grace then stop
                time.sleep(2)
                if len(job["items"]) == 0 and proc.poll() is not None:
                    time.sleep(3)
            time.sleep(1.5)
        try:
            proc.terminate()
        except Exception:
            pass
        with _synth_jobs_lock:
            job["status"] = "cancelled" if job.get("cancel") else "done"
            job["finished_at"] = time.time()
    except Exception as e:
        with _synth_jobs_lock:
            job["status"] = "error"
            job["error"] = str(e)


@app.get("/api/synthetic-dewarp/clean-inputs")
async def synthetic_dewarp_clean_inputs(request: Request):
    _admin_only(request)
    items = []
    root = BASE_DIR / "datasets" / "paired" / "inv3d" / "target"
    for f in sorted(root.glob("*.png")):
        clean = f.stem.isdigit()
        items.append({"name": f.name, "source": "inv3d", "clean": clean, "url": f"/api/synthetic-dewarp/clean-preview/{f.name}"})
    uploads = []
    for f in sorted(SYNTH_DEWARP_UPLOADS.glob("*")):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            uploads.append({"name": f.name, "source": "upload", "clean": True, "url": f"/api/synthetic-dewarp/upload-preview/{f.name}"})
    return {"items": items, "uploads": uploads, "count": len(items) + len(uploads)}


SYNTH_DEWARP_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


@app.post("/api/synthetic-dewarp/upload")
async def synthetic_dewarp_upload(request: Request, file: UploadFile = File(...)):
    _admin_only(request)
    contents = await file.read()
    if len(contents) > SYNTH_DEWARP_MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"Ukuran gambar maksimal 5MB (file {len(contents)/1024/1024:.1f}MB)")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Tipe file gambar tidak didukung.")
    try:
        import io as _iov
        Image.open(_iov.BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar valid.")
    safe = Path(file.filename or "upload.png").name
    stem = Path(safe).stem
    dst = SYNTH_DEWARP_UPLOADS / f"{stem}.png"
    i = 1
    while dst.exists():
        dst = SYNTH_DEWARP_UPLOADS / f"{stem}_{i}.png"
        i += 1
    import io as _io
    img = Image.open(_io.BytesIO(contents)).convert("RGB")
    img.save(dst, "PNG")
    return {"success": True, "name": dst.name, "source": "upload", "url": f"/api/synthetic-dewarp/upload-preview/{dst.name}"}


@app.get("/api/synthetic-dewarp/upload-preview/{filename}")
async def synthetic_dewarp_upload_preview(request: Request, filename: str):
    _admin_only(request)
    safe = Path(filename).name
    path = SYNTH_DEWARP_UPLOADS / safe
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))


@app.get("/api/synthetic-dewarp/clean-preview/{filename}")
async def synthetic_dewarp_clean_preview(request: Request, filename: str):
    _admin_only(request)
    safe = Path(filename).name
    path = BASE_DIR / "datasets" / "paired" / "inv3d" / "target" / safe
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))


@app.get("/api/synthetic-dewarp/preview/{filename}")
async def synthetic_dewarp_preview(request: Request, filename: str):
    _admin_only(request)
    safe = Path(filename).name
    path = SYNTH_DEWARP_PREVIEW / safe
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(path))


@app.get("/api/synthetic-dewarp/candidates")
async def synthetic_dewarp_candidates(request: Request):
    _admin_only(request)
    for f in sorted(SYNTH_DEWARP_CANDIDATES.glob("*.gw")):
        if not (SYNTH_DEWARP_PREVIEW / f"{f.stem}.jpg").exists():
            try: _render_gw_preview(f)
            except Exception as e: print(f"[WARN] preview failed {f}: {e}")
    return {
        "candidates": [_synth_item_from_gw(f) for f in sorted(SYNTH_DEWARP_CANDIDATES.glob("*.gw"))],
        "selected_count": len(list(SYNTH_DEWARP_SELECTED.glob("*.gw"))),
        "rejected_count": len(list(SYNTH_DEWARP_REJECTED.glob("*.gw"))),
        "custom_count": len(list(SYNTH_DEWARP_CUSTOM.glob("*.gw"))),
    }


@app.post("/api/synthetic-dewarp/generate")
async def synthetic_dewarp_generate(request: Request):
    """Start async generation job. Returns job_id immediately; poll /job/{id}."""
    _admin_only(request)
    body = await request.json()
    input_name = Path(body.get("input", "")).name
    source = body.get("source", "inv3d")
    count = max(1, min(int(body.get("count", 5)), 20))
    clean_path = _resolve_clean_input(input_name, source)
    if not clean_path:
        raise HTTPException(status_code=404, detail="clean input not found")
    job_id = uuid.uuid4().hex[:12]
    with _synth_jobs_lock:
        _synth_jobs[job_id] = {
            "job_id": job_id, "status": "starting", "input": input_name, "source": source,
            "count": count, "items": [], "cancel": False, "error": None,
            "started_at": time.time(),
        }
    t = _threading.Thread(target=_synth_worker, args=(job_id, clean_path, count, Path(input_name).stem), daemon=True)
    t.start()
    return {"success": True, "job_id": job_id, "count": count}


@app.get("/api/synthetic-dewarp/job/{job_id}")
async def synthetic_dewarp_job(request: Request, job_id: str):
    _admin_only(request)
    with _synth_jobs_lock:
        job = _synth_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "job_id": job_id,
            "status": job["status"],
            "count": job["count"],
            "done": len(job["items"]),
            "items": list(job["items"]),
            "error": job.get("error"),
        }


@app.post("/api/synthetic-dewarp/job/{job_id}/cancel")
async def synthetic_dewarp_job_cancel(request: Request, job_id: str):
    _admin_only(request)
    with _synth_jobs_lock:
        job = _synth_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        job["cancel"] = True
    return {"success": True}


@app.post("/api/synthetic-dewarp/mark")
async def synthetic_dewarp_mark(request: Request):
    _admin_only(request)
    body = await request.json()
    names = [Path(n).name for n in body.get("names", [])]
    action = body.get("action", "")
    if action not in {"select", "reject"}:
        raise HTTPException(status_code=400, detail="action must select/reject")
    dst_dir = SYNTH_DEWARP_SELECTED if action == "select" else SYNTH_DEWARP_REJECTED
    moved = []
    import shutil
    for name in names:
        src = SYNTH_DEWARP_CANDIDATES / name
        if src.exists():
            shutil.move(str(src), str(dst_dir / name))
            moved.append(name)
    return {"success": True, "moved": moved, "action": action}


@app.post("/api/synthetic-dewarp/commit-custom")
async def synthetic_dewarp_commit_custom(request: Request):
    """Move selected .gw samples into the custom dewarp training dataset."""
    _admin_only(request)
    body = await request.json()
    names = body.get("names")
    import shutil
    if names:
        srcs = [SYNTH_DEWARP_SELECTED / Path(n).name for n in names]
    else:
        srcs = list(SYNTH_DEWARP_SELECTED.glob("*.gw"))
    committed = []
    for src in srcs:
        if src.exists() and src.suffix == ".gw":
            shutil.move(str(src), str(SYNTH_DEWARP_CUSTOM / src.name))
            committed.append(src.name)
    return {
        "success": True,
        "committed": committed,
        "custom_count": len(list(SYNTH_DEWARP_CUSTOM.glob("*.gw"))),
        "custom_path": str(SYNTH_DEWARP_CUSTOM),
    }


@app.get("/api/synthetic-dewarp/custom")
async def synthetic_dewarp_custom_list(request: Request):
    _admin_only(request)
    for f in sorted(SYNTH_DEWARP_CUSTOM.glob("*.gw")):
        if not (SYNTH_DEWARP_PREVIEW / f"{f.stem}.jpg").exists():
            try: _render_gw_preview(f)
            except Exception as e: print(f"[WARN] custom preview failed {f}: {e}")
    items = [_synth_item_from_gw(f) for f in sorted(SYNTH_DEWARP_CUSTOM.glob("*.gw"))]
    return {"items": items, "count": len(items), "path": str(SYNTH_DEWARP_CUSTOM)}


@app.post("/api/synthetic-dewarp/custom/remove")
async def synthetic_dewarp_custom_remove(request: Request):
    _admin_only(request)
    body = await request.json()
    names = [Path(n).name for n in body.get("names", [])]
    to_selected = bool(body.get("to_selected", False))
    import shutil
    removed = []
    for name in names:
        src = SYNTH_DEWARP_CUSTOM / name
        if src.exists() and src.suffix == ".gw":
            if to_selected:
                shutil.move(str(src), str(SYNTH_DEWARP_SELECTED / name))
            else:
                src.unlink()
            removed.append(name)
    return {"success": True, "removed": removed, "to_selected": to_selected, "custom_count": len(list(SYNTH_DEWARP_CUSTOM.glob("*.gw")))}


@app.get("/api/synthetic-dewarp/selected")
async def synthetic_dewarp_selected(request: Request):
    _admin_only(request)
    for f in sorted(SYNTH_DEWARP_SELECTED.glob("*.gw")):
        if not (SYNTH_DEWARP_PREVIEW / f"{f.stem}.jpg").exists():
            try: _render_gw_preview(f)
            except Exception: pass
    return {
        "selected": [_synth_item_from_gw(f) for f in sorted(SYNTH_DEWARP_SELECTED.glob("*.gw"))],
        "custom_count": len(list(SYNTH_DEWARP_CUSTOM.glob("*.gw"))),
    }


# =====================================================================
#  HEALTH / INFO
# =====================================================================
@app.get("/api/health")
def health_check():
    # Read-only status. Do NOT call lazy loaders here; they allocate GPU VRAM.
    return {
        "status": "ok",
        "device": str(device),
        "models": {
            "document_restorer": get_doc_restorer_model() is not None,
            "shadow_remover": get_shadow_remover_model() is not None,
            "doc_enhancer": get_doc_enhancer_model() is not None,
            "docshadow_sd7k": _docshadow is not None,
            "docres": _docres is not None,
            "docres_base": _docres_base is not None,
            "docres_task": _docres_task is not None,
            "docres_onnx": _docres_onnx is not None,
        },
    }


@app.get("/api/models/info")
async def models_info(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    info = {}
    for name, model in [
        ("document_restorer", get_doc_restorer_model()),
        ("shadow_remover", get_shadow_remover_model()),
        ("doc_enhancer", get_doc_enhancer_model()),
    ]:
        if model is not None and hasattr(model, 'get_model_info'):
            info[name] = model.get_model_info()
        else:
            info[name] = {"loaded": model is not None}
    # Read-only info. Avoid lazy-loading DocRes/DocShadow because status page must not allocate VRAM.
    if _docres is not None:
        info["docres"] = _docres.get_model_info()
    else:
        info["docres"] = {"loaded": False}
    if _docshadow is not None:
        info["docshadow_sd7k"] = {"loaded": True, "weights": _docshadow.list_available_weights()}
    else:
        info["docshadow_sd7k"] = {"loaded": False}
    info["docres_base"] = {"loaded": _docres_base is not None}
    info["docres_task"] = {"loaded": _docres_task is not None}
    info["docres_onnx"] = {"loaded": _docres_onnx is not None}
    return info


# DocRes system monitoring + training history

@app.get("/api/training/docres/system")
async def docres_system_status(request: Request):
    import psutil, subprocess
    result = {"cpu_percent": psutil.cpu_percent(interval=0.1), "cpu_count": psutil.cpu_count()}
    mem = psutil.virtual_memory()
    result["ram_used_gb"] = round(mem.used / 1024**3, 1)
    result["ram_total_gb"] = round(mem.total / 1024**3, 1)
    result["ram_percent"] = mem.percent
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            if len(parts) >= 5:
                result["gpu"] = {"name": parts[0].strip(), "vram_used": int(parts[1]), "vram_total": int(parts[2]), "utilization": int(parts[3]), "temperature": int(parts[4])}
    except Exception:
        pass
    return result

@app.get("/api/training/docres/history")
async def docres_training_history(request: Request):
    import re, json as _json_mod
    log_dir = DOCRES_STATUS_PATH.parent
    points = []
    if log_dir.exists():
        runs = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("deshadow_v3_")], reverse=True)
        if runs:
            log_path = runs[0] / "train.log"
            if log_path.exists():
                for line in log_path.read_text().splitlines():
                    m = re.search(r"iter=(\d+)\s+loss=([\d.]+)\s+lr=([\d.e\-]+)", line)
                    if m:
                        points.append({"iter": int(m.group(1)), "loss": float(m.group(2)), "lr": float(m.group(3))})
    return {"points": points[-500:]}

# DocRes fine-tuning training status
DOCRES_STATUS_PATH = BASE_DIR.parent / "DocRes" / "finetune_logs" / "docres_status.json"

@app.get("/api/training/docres/status")
async def docres_training_status(request: Request):
    import json as _json
    if DOCRES_STATUS_PATH.exists():
        try:
            import math
            data = _json.loads(DOCRES_STATUS_PATH.read_text())
            # Fix inf/nan values
            for k, v in data.items():
                if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                    data[k] = None
            # Read config.json from log_dir if available
            log_dir = data.get("log_dir")
            if log_dir:
                cfg_path = BASE_DIR.parent / "DocRes" / log_dir.lstrip("./") / "config.json"
                if cfg_path.exists():
                    try:
                        data["config"] = _json.loads(cfg_path.read_text())
                    except Exception:
                        pass
                # Read train.log for loss history
                log_file = BASE_DIR.parent / "DocRes" / log_dir.lstrip("./") / "train.log"
                if log_file.exists():
                    try:
                        lines = log_file.read_text().strip().split("\n")
                        loss_history = []
                        for line in lines:
                            if "iter=" in line and "loss=" in line:
                                parts = {}
                                for kv in line.split():
                                    if "=" in kv:
                                        k, v = kv.split("=", 1)
                                        parts[k] = float(v)
                                if "iter" in parts and "loss" in parts:
                                    loss_history.append({"iter": int(parts["iter"]), "loss": parts["loss"]})
                        data["loss_history"] = loss_history
                    except Exception:
                        pass
            # Reconcile stale external status. Trainer may exit without flipping
            # docres_status.json; completion is authoritative when iter reaches
            # total_iter, and no matching training process is running.
            if data.get("iter") is not None and data.get("total_iter"):
                try:
                    completed = int(data["iter"]) >= int(data["total_iter"])
                except (TypeError, ValueError):
                    completed = False
                if completed:
                    data["running"] = False
                    data["status"] = "completed"
                    data["progress"] = 100.0
                    data["eta_seconds"] = 0
                    data["eta_minutes"] = 0.0

            # Derive monitoring fields when external trainer omits them
            if data.get("running") and data.get("iter") is not None and data.get("total_iter"):
                import time as _time
                speed = float(data.get("speed_it_per_sec") or 0.72)
                data["speed_it_per_sec"] = speed
                data["eta_seconds"] = int((data["total_iter"] - data["iter"]) / max(speed, 1e-6))
                if not data.get("started_at"):
                    data["started_at"] = _time.strftime("%Y-%m-%dT%H:%M:%S+00:00", _time.gmtime(_time.time() - data["iter"] / speed))
                data["estimated_finish_at"] = _time.strftime("%Y-%m-%dT%H:%M:%S+00:00", _time.gmtime(_time.time() + data["eta_seconds"]))
            if data.get("best_score") is None and log_dir:
                import re as _re
                log_file = BASE_DIR.parent / "DocRes" / log_dir.lstrip("./") / "train.log"
                if log_file.exists():
                    vals=[]
                    for line in log_file.read_text().splitlines():
                        m=_re.search(r"iter=(\d+)\s+loss=([\d.]+)",line)
                        if m: vals.append((float(m.group(2)),int(m.group(1))))
                    if vals:
                        best_iter=min(vals)[1]; data["best_score"]=min(vals)[0]; data["best_checkpoint"]=f"iter_{best_iter}.pth"

            # Normalize external DocRes status aliases for frontend monitor
            if data.get("speed_it_per_sec") is not None:
                data["speed"] = data["speed_it_per_sec"]
            if data.get("eta_seconds") is not None:
                data["eta"] = data["eta_seconds"]
                data["eta_minutes"] = round(data["eta_seconds"] / 60, 1)
            if data.get("started_at"):
                data["start_time"] = data["started_at"]
            if data.get("estimated_finish_at"):
                data["finish_time"] = data["estimated_finish_at"]
            if data.get("best_score") is None:
                data["best_score"] = data.get("best_loss")

            # Add WIB timestamps
            if data.get("started_at"):
                from datetime import datetime, timezone, timedelta
                wib = timezone(timedelta(hours=7))
                try:
                    started = datetime.fromisoformat(data["started_at"]).replace(tzinfo=timezone.utc)
                    data["started_at_wib"] = started.astimezone(wib).strftime("%d %b %Y %H:%M WIB")
                    if data.get("eta_seconds") and data["eta_seconds"] > 0:
                        finish_utc = started + timedelta(seconds=data["eta_seconds"])
                        data["finish_at_wib"] = finish_utc.astimezone(wib).strftime("%d %b %Y %H:%M WIB")
                        h = int(data["eta_seconds"] // 3600)
                        m = int((data["eta_seconds"] % 3600) // 60)
                        data["eta_human"] = f"{h}h {m}m"
                except Exception:
                    pass
            return data
        except Exception:
            pass
    return {"running": False, "status": "no data"}

@app.get("/api/system/status")
async def system_status(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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

@app.post("/api/detect_edges")
async def detect_edges(request: Request, file: UploadFile = File(...), method: str = Form("opencv")):
    """Detect document corners, return JSON points in ORIGINAL image coords.
    Does NOT crop -- frontend draws draggable points, manual panel stays visible.
    method: 'opencv' (fast) or 'grabcut' (slower, better on tilted photos).
    Response: {corners:[[x,y]*4], confidence, is_full, width, height, method}
    corners order: tl,tr,br,bl. If is_full=true, corners = near-full inset."""
    try:
        image = _read_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image read error: {e}")
    import cv2 as _cv2
    import numpy as _np
    from backend.doc_edge_detect import detect_corners, detect_corners_grabcut, detect_corners_auto
    img_np = _cv2.cvtColor(_np.array(image.convert("RGB")), _cv2.COLOR_RGB2BGR)
    H, W = img_np.shape[:2]
    chosen = method
    if method == "grabcut":
        corners, conf, is_full = detect_corners_grabcut(img_np)
    elif method == "auto":
        corners, conf, is_full, chosen = detect_corners_auto(img_np)
    else:
        method = "opencv"
        corners, conf, is_full = detect_corners(img_np)
    method = chosen
    if corners is None or is_full:
        m = 0.01
        corners = _np.array([[W * m, H * m], [W * (1 - m), H * m],
                             [W * (1 - m), H * (1 - m)], [W * m, H * (1 - m)]], _np.float32)
        is_full = True
    return JSONResponse({
        "corners": [[float(x), float(y)] for x, y in corners],
        "confidence": round(float(conf), 3),
        "is_full": bool(is_full),
        "width": int(W),
        "height": int(H),
        "method": method,
    })


@app.post("/api/crop")
async def crop_document(request: Request, file: UploadFile = File(...), method: str = Form("opencv")):
    # Crop document: opencv (fast) or sam (accurate)
    try:
        image = _read_image(file)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Image read error: {e}")

    import tempfile, subprocess, os

    if method == "sam":
        # SAM crop (accurate, ~10s CPU)
        tmp_in = tempfile.mktemp(suffix=".jpg")
        tmp_out = tempfile.mktemp(suffix=".jpg")
        try:
            image.save(tmp_in)
            result = subprocess.run(
                ["/home/wahyu/miniconda3/bin/python3", "backend/sam_document_crop.py", tmp_in, tmp_out],
                capture_output=True, text=True, cwd=str(BASE_DIR), timeout=120)
            if os.path.exists(tmp_out):
                with open(tmp_out, "rb") as f:
                    return StreamingResponse(iter([f.read()]), media_type="image/jpeg")
            else:
                raise HTTPException(status_code=500, detail=f"SAM crop failed: {result.stderr[:200]}")
        finally:
            for fp in [tmp_in, tmp_out]:
                if os.path.exists(fp): os.unlink(fp)
    else:
        # Robust 8-point crop: multi-pass + midpoint detection
        import cv2, numpy as np
        img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h_orig, w_orig = img_np.shape[:2]
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        def order_points(pts):
            pts = pts.reshape(4, 2).astype(np.float32)
            s = pts.sum(axis=1); r = np.zeros((4,2), dtype=np.float32)
            r[0] = pts[np.argmin(s)]; r[2] = pts[np.argmax(s)]
            d = np.diff(pts, axis=1); r[1] = pts[np.argmin(d)]; r[3] = pts[np.argmax(d)]
            return r

        def find_quad(cnts, min_area_ratio=0.05):
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
                area = cv2.contourArea(c)
                if area < h_orig * w_orig * min_area_ratio: continue
                peri = cv2.arcLength(c, True)
                for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                    approx = cv2.approxPolyDP(c, eps*peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype(np.float32)
                        if cv2.contourArea(pts) > h_orig * w_orig * min_area_ratio:
                            return pts
            return None

        def refine_corners(gray, corners, radius=15):
            """Refine corner positions by finding strongest edge nearby"""
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            refined = []
            for cx, cy in corners:
                x1 = max(0, int(cx) - radius)
                y1 = max(0, int(cy) - radius)
                x2 = min(gray.shape[1], int(cx) + radius)
                y2 = min(gray.shape[0], int(cy) + radius)
                roi = edges[y1:y2, x1:x2]
                if roi.size == 0:
                    refined.append([cx, cy])
                    continue
                # Find point with max gradient in ROI
                ys, xs = np.where(roi > 0)
                if len(xs) > 0:
                    best_i = np.argmax(roi[ys, xs])
                    refined.append([x1 + xs[best_i], y1 + ys[best_i]])
                else:
                    refined.append([cx, cy])
            return np.array(refined, dtype=np.float32)

        def get_midpoints(corners):
            """Calculate 4 midpoints between consecutive corners"""
            midpts = []
            for i in range(4):
                j = (i + 1) % 4
                mx = (corners[i][0] + corners[j][0]) / 2
                my = (corners[i][1] + corners[j][1]) / 2
                midpts.append([mx, my])
            return np.array(midpts, dtype=np.float32)

        def detect_from_threshold(gray, method_params):
            """Run detection with specific parameters, return 4 corners or None"""
            block_size, C, morph_op, morph_iter = method_params
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, block_size, C)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            if morph_op == 'close':
                processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
            else:
                processed = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
            cnts, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return find_quad(cnts)

        def detect_from_otsu(gray):
            """OTSU-based detection"""
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return find_quad(cnts)

        def detect_from_canny(gray):
            """Canny-based detection"""
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
            cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return find_quad(cnts)

        # Multi-pass detection with different parameters
        all_corners = []
        
        # Pass 1: OTSU
        q = detect_from_otsu(gray)
        if q is not None: all_corners.append(q)
        
        # Pass 2: OTSU on inverted
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts_inv, _ = cv2.findContours(otsu_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        q = find_quad(cnts_inv)
        if q is not None: all_corners.append(q)
        
        # Pass 3: Canny
        q = detect_from_canny(gray)
        if q is not None: all_corners.append(q)
        
        # Pass 4-6: Adaptive with different block sizes
        for bs in [11, 15, 21]:
            for C in [2, 5]:
                q = detect_from_threshold(gray, (bs, C, 'close', 2))
                if q is not None: all_corners.append(q)
        
        # Pass 7: Adaptive with morph open
        q = detect_from_threshold(gray, (15, 5, 'open', 1))
        if q is not None: all_corners.append(q)

        if not all_corners:
            # Final fallback: bounding box
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cropped = img_np[y:y+h, x:x+w]
                _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
                return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")
            raise HTTPException(status_code=400, detail="Could not detect document edges")

        # Filter out quads too close to full image (likely wrong detection)
        filtered = [q for q in all_corners if cv2.contourArea(q) / (h_orig * w_orig) < 0.90]
        if not filtered:
            filtered = all_corners  # Fallback: use all

        # Average filtered corners
        avg_corners = np.mean(filtered, axis=0).astype(np.float32)
        avg_corners = order_points(avg_corners)

        # Refine with edge detection
        avg_corners = refine_corners(gray, avg_corners)

        # Get 8 points (4 corners + 4 midpoints)
        midpts = get_midpoints(avg_corners)
        eight_points = np.vstack([avg_corners, midpts])

        # Warp using 4 corners
        tl, tr, br, bl = avg_corners
        ww = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
        hh = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
        ww, hh = max(ww, 64), max(hh, 64)
        dst = np.array([[0,0],[ww-1,0],[ww-1,hh-1],[0,hh-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(avg_corners, dst)
        warped = cv2.warpPerspective(img_np, M, (ww, hh))

        _, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")#  CORE SCANNING ENDPOINTS
# =====================================================================
@app.post("/api/scan")
async def scan_document(request: Request, file: UploadFile = File(...), mode: str = Form("restore"),
                         pipeline_mode: str = Form("full"), shadow_strength: float = Form(1.0),
                         adj_brightness: float = Form(1.0), adj_contrast: float = Form(1.0),
                         adj_saturation: float = Form(1.0), adj_sharpness: float = Form(1.0),
                         adj_gamma: float = Form(1.0), adj_white_balance: bool = Form(False),
                         adj_clahe: float = Form(0.0), variant: str = Form("finetune_v3")):
    global _appearance_mixed
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    _ADJUST_CTX.set({
        "brightness": adj_brightness, "contrast": adj_contrast,
        "saturation": adj_saturation, "sharpness": adj_sharpness,
        "gamma": adj_gamma, "white_balance": adj_white_balance,
        "clahe_clip": adj_clahe,
    })
    image = _read_image(file)

    if mode == "restore" and get_doc_restorer_model() is not None:
        start = time.time()
        with torch.inference_mode():
            restored, mask, info = run_document_restoration_pipeline(
                get_doc_restorer_model(), image, device,
                tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
                mode=pipeline_mode, shadow_strength=shadow_strength,
            )
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(restored)

    elif mode == "shadow_remove" and get_shadow_remover_model() is not None:
        start = time.time()
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = get_shadow_remover_model()(img_tensor)
        out_np = output[0].cpu().numpy().transpose(1, 2, 0)
        out_min, out_max = out_np.min(), out_np.max()
        if out_max - out_min > 1e-6:
            out_np = (out_np - out_min) / (out_max - out_min)
        result = _np_to_pil(out_np * 255)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "enhance" and get_doc_enhancer_model() is not None:
        start = time.time()
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = get_doc_enhancer_model()(img_tensor)
        result_img = output.get('enhanced', output[0] if isinstance(output, (tuple, list)) else output)
        if isinstance(result_img, torch.Tensor):
            result_img = result_img[0].cpu().numpy().transpose(1, 2, 0) * 255
        result = _np_to_pil(result_img)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "shadow_so":
        start = time.time()
        img_np = np.array(image.convert('RGB'))
        result_np = so_shadow_removal_enhanced(img_np)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(Image.fromarray(result_np))

    elif mode == "shadow_so_aggressive":
        start = time.time()
        img_np = np.array(image.convert('RGB'))
        result_np = so_shadow_removal_aggressive(img_np)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(Image.fromarray(result_np))


    elif mode == "shadow_effective_bg":
        start = time.time()
        img_np = np.array(image.convert("RGB"))
        result_np = effective_bg_estimation(img_np)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(Image.fromarray(result_np))

    elif mode == "shadow_iterative":
        start = time.time()
        img_np = np.array(image.convert("RGB"))
        result_np = iterative_removal(img_np)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(Image.fromarray(result_np))

    elif mode == "docres_base":
        start = time.time()
        mdl = get_docres_base_model()
        if mdl is None:
            # Force retry load
            try:
                from backend.models.docres_model import get_docres_base as _grb
                mdl = _grb(device=str(device))
                import backend.app as _app
                _app._docres_base = mdl
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"DocRes base load failed: {e}")
        if mdl is None:
            raise HTTPException(status_code=500, detail="DocRes base model not loaded")
        result = mdl.infer(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_finetune":
        variant = variant
        start = time.time()
        mdl = get_docres_ft_model(variant)
        if mdl is None:
            raise HTTPException(status_code=500, detail=f"DocRes finetune ({variant}) not loaded")
        result = mdl.infer(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_onnx":
        start = time.time()
        mdl = get_docres_onnx_model()
        if mdl is None:
            raise HTTPException(status_code=500, detail="DocRes ONNX model not loaded")
        result = mdl.infer(image, task="appearance")
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_dewarping":
        start = time.time()
        result = docres_task_infer(image, 'dewarping')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_appearance_mixed_v5":
        try:
            import tempfile, os, importlib.util
            if _appearance_mixed is None:
                spec = importlib.util.spec_from_file_location("appearance_test", "/home/wahyu/DocRes/test_appearance.py")
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                _appearance_mixed = mod.load_model("/home/wahyu/DocRes/finetune_logs/appearance_mixed_custom_v2_pilot_20260810/run1/best.pth")
                _appearance_mixed._appearance_infer = mod.inference_appearance
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                image.save(tf.name); tmp = tf.name
            mixed = _appearance_mixed._appearance_infer(_appearance_mixed, tmp, device=str(device)); os.unlink(tmp)
            v5 = get_docres_ft_model("finetune_v5")
            if v5 is None: raise RuntimeError("V5 model not loaded")
            v5out = np.array(v5.infer(Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))))
            result = Image.fromarray(np.clip(0.75 * np.array(v5out, dtype=np.float32) + 0.25 * cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB), 0, 255).astype(np.uint8))
            arr=np.array(result).astype(np.float32); gray=cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32); bg=cv2.GaussianBlur(gray,(0,0),45); mask=cv2.GaussianBlur(np.clip((bg-gray-3)/55,0,1),(0,0),35); edge=np.clip(np.abs(cv2.Laplacian(gray,cv2.CV_32F))/45,0,1); mask*=1-.5*edge; arr=np.clip(arr*(1+mask[...,None]*.15),0,255).astype(np.uint8); return _pil_to_response(Image.fromarray(arr))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mixed V5 load/infer failed: {e}")

    elif mode in ("docres_appearance_mixed_v5_tiled_lift12", "docres_appearance_mixed_v5_tiled_lift12_base_detail"):
        try:
            import tempfile, os, importlib.util
            if _appearance_mixed is None:
                spec = importlib.util.spec_from_file_location("appearance_test", "/home/wahyu/DocRes/test_appearance.py")
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                _appearance_mixed = mod.load_model("/home/wahyu/DocRes/finetune_logs/appearance_mixed_custom_v2_pilot_20260810/run1/best.pth")
                _appearance_mixed._appearance_infer = mod.inference_appearance
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                image.save(tf.name); tmp = tf.name
            mixed_bgr = _appearance_mixed._appearance_infer(_appearance_mixed, tmp, device=str(device)); os.unlink(tmp)
            v5 = get_docres_ft_model("finetune_v5")
            if v5 is None: raise RuntimeError("V5 model not loaded")
            mixed_rgb = cv2.cvtColor(mixed_bgr, cv2.COLOR_BGR2RGB)
            v5_rgb = _v5_tiled_infer(v5, mixed_bgr)
            arr = np.clip(0.75 * v5_rgb.astype(np.float32) + 0.25 * mixed_rgb.astype(np.float32), 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            luma, aa, bb = cv2.split(lab)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            bg = cv2.GaussianBlur(gray, (0, 0), 45)
            mask = cv2.GaussianBlur(np.clip((bg - gray - 3.0) / 55.0, 0.0, 1.0), (0, 0), 35)
            edge = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 45.0, 0.0, 1.0)
            mask *= 1.0 - 0.5 * edge
            luma = np.clip(luma + np.minimum(luma * 0.12 * mask, 22.0), 0.0, 255.0)
            result_rgb = cv2.cvtColor(cv2.merge([luma, aa, bb]).astype(np.uint8), cv2.COLOR_LAB2RGB)
            if mode == "docres_appearance_mixed_v5_tiled_lift12_base_detail":
                base_onnx = get_docres_onnx_model()
                if base_onnx is None:
                    raise RuntimeError("Base Appearance ONNX not loaded")
                base_rgb = np.array(base_onnx.infer(image, task="appearance"))
                result_rgb = _base_appearance_detail_fuse(base_rgb, result_rgb, alpha=1.0, sigma=1.0)
            return _pil_to_response(Image.fromarray(result_rgb))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mixed V5 tiled LIFT12 load/infer failed: {e}")

    elif mode == "docres_appearance_mixed_v5_lift12":
        try:
            import tempfile, os, importlib.util
            if _appearance_mixed is None:
                spec = importlib.util.spec_from_file_location("appearance_test", "/home/wahyu/DocRes/test_appearance.py")
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                _appearance_mixed = mod.load_model("/home/wahyu/DocRes/finetune_logs/appearance_mixed_custom_v2_pilot_20260810/run1/best.pth")
                _appearance_mixed._appearance_infer = mod.inference_appearance
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                image.save(tf.name); tmp = tf.name
            mixed_bgr = _appearance_mixed._appearance_infer(_appearance_mixed, tmp, device=str(device)); os.unlink(tmp)
            v5 = get_docres_ft_model("finetune_v5")
            if v5 is None:
                raise RuntimeError("V5 model not loaded")
            mixed_rgb = cv2.cvtColor(mixed_bgr, cv2.COLOR_BGR2RGB)
            v5_rgb = np.array(v5.infer(Image.fromarray(mixed_rgb)))
            arr = np.clip(0.75 * v5_rgb.astype(np.float32) + 0.25 * mixed_rgb.astype(np.float32), 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            luma, aa, bb = cv2.split(lab)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            bg = cv2.GaussianBlur(gray, (0, 0), 45)
            mask = np.clip((bg - gray - 3.0) / 55.0, 0.0, 1.0)
            mask = cv2.GaussianBlur(mask, (0, 0), 35)
            edge = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 45.0, 0.0, 1.0)
            mask *= 1.0 - 0.5 * edge
            luma = np.clip(luma + np.minimum(luma * 0.12 * mask, 22.0), 0.0, 255.0)
            result = Image.fromarray(cv2.cvtColor(cv2.merge([luma, aa, bb]).astype(np.uint8), cv2.COLOR_LAB2RGB))
            return _pil_to_response(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Mixed V5 LIFT12 load/infer failed: {e}")

    elif mode == "docres_appearance_mixed":
        try:
            import tempfile, os, importlib.util
            if _appearance_mixed is None:
                spec = importlib.util.spec_from_file_location("appearance_test", "/home/wahyu/DocRes/test_appearance.py")
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                _appearance_mixed = mod.load_model("/home/wahyu/DocRes/finetune_logs/appearance_mixed_custom_v2_pilot_20260810/run1/best.pth")
                _appearance_mixed._appearance_infer = mod.inference_appearance
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                image.save(tf.name); tmp = tf.name
            result = _appearance_mixed._appearance_infer(_appearance_mixed, tmp, device=str(device)); os.unlink(tmp)
            return _pil_to_response(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Appearance mixed load/infer failed: {e}")

    elif mode == "docres_appearance":
        start = time.time()
        result = docres_task_infer(image, 'appearance')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_deblurring":
        start = time.time()
        result = docres_task_infer(image, 'deblurring')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "docres_end2end":
        from backend.docres_infer import docres_infer
        start = time.time()
        result = docres_infer(image, 'end2end')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "dewarpnet":
        from backend.dewarpnet_infer import dewarpnet_infer
        start = time.time()
        result = dewarpnet_infer(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "opencv_dewarp":
        from backend.opencv_dewarp import opencv_dewarp
        start = time.time()
        result, pts, angle = opencv_dewarp(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "controlpoints_dewarp":
        from backend.controlpoints_infer import controlpoints_dewarp
        start = time.time()
        result = controlpoints_dewarp(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "geotr_doc3d_dewarp":
        from backend.geotr_doc3d_infer import geotr_doc3d_dewarp
        start = time.time()
        result = geotr_doc3d_dewarp(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "geotr_doc3d_docres":
        from backend.geotr_doc3d_infer import geotr_doc3d_dewarp
        start = time.time()
        result = geotr_doc3d_dewarp(image)
        result = docres_task_infer(result, 'deshadowing')
        result = docres_task_infer(result, 'appearance')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "sam_geotr_docres":
        from backend.sam_geotr_pipeline import sam_document_crop
        from backend.geotr_doc3d_infer import geotr_doc3d_dewarp
        start = time.time()
        result = sam_document_crop(image)
        result = geotr_doc3d_dewarp(result)
        result = docres_task_infer(result, 'deshadowing')
        result = docres_task_infer(result, 'appearance')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "sam_perspective_appearance":
        from backend.sam_geotr_pipeline import sam_document_crop
        start = time.time()
        result = sam_document_crop(image)
        result = docres_task_infer(result, 'appearance')
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "textline_refine_dewarp":
        from backend.geotr_doc3d_infer import geotr_doc3d_dewarp
        from backend.textline_refine import textline_refine_dewarp
        start = time.time()
        result = textline_refine_dewarp(image, geotr_doc3d_dewarp, docres_task_infer)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "auto_dewarp":
        from backend.auto_dewarp_pipeline import auto_dewarp
        from backend.geotr_doc3d_infer import geotr_doc3d_dewarp
        from backend.docres_infer import docres_infer as _docres_infer_fn
        start = time.time()
        result, pipeline, score = auto_dewarp(image, _docres_infer_fn, geotr_doc3d_dewarp)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    elif mode == "color_binarize":
        start = time.time()
        img_np = np.array(image.convert("RGB"))
        result_np = color_binarize(img_np)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(Image.fromarray(result_np))
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

    elif mode == "docres":
        start = time.time()
        docres = get_docres_model()
        if docres is None:
            raise HTTPException(status_code=503, detail="DocRes model not loaded")
        result = docres.infer(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)

    else:
        available = ["restore", "shadow_remove", "shadow_so", "shadow_so_aggressive", "shadow_effective_bg", "shadow_iterative", "docres_base", "docres_finetune", "docres_appearance_mixed_v5_lift12", "docres_appearance_mixed_v5_tiled_lift12", "docres_appearance_mixed_v5_tiled_lift12_base_detail", "color_binarize", "enhance", "magic_enhance", "binarize",
                      "deskew", "cleanup", "clahe", "denoise", "sharpen", "docres"]
        raise HTTPException(status_code=400,
                            detail=f"Mode '{mode}' not available. Use one of: {available}")



def _base_appearance_detail_fuse(base_rgb, candidate_rgb, alpha=1.0, sigma=1.0):
    """Keep candidate illumination while restoring Base Appearance high-frequency detail."""
    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    cand_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_l = base_lab[:, :, 0]
    cand_l = cand_lab[:, :, 0]
    detail = base_l - cv2.GaussianBlur(base_l, (0, 0), sigma)
    smooth = cv2.GaussianBlur(cand_l, (0, 0), sigma)
    cand_lab[:, :, 0] = np.clip(smooth + alpha * detail, 0, 255)
    return cv2.cvtColor(cand_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _v5_tiled_infer(v5, bgr, tile=576, overlap=96):
    """Run 576-trained V5 with tiled inference and feathered overlap."""
    from backend.models.docres_model import deshadow_prompt
    h, w = bgr.shape[:2]
    prompt = deshadow_prompt(bgr)
    full = np.concatenate([bgr, prompt], axis=2).astype(np.float32) / 255.0
    stride = tile - overlap
    acc = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w, 1), np.float32)
    win = np.ones((tile, tile), np.float32)
    fade = np.linspace(0, 1, overlap)
    win[:overlap, :] *= fade[:, None]
    win[-overlap:, :] *= fade[::-1, None]
    win[:, :overlap] *= fade[None, :]
    win[:, -overlap:] *= fade[None, ::-1]
    win = win[..., None]
    ys = [0] if h <= tile else list(range(0, h - tile + 1, stride))
    xs = [0] if w <= tile else list(range(0, w - tile + 1, stride))
    if h > tile and ys[-1] != h - tile: ys.append(h - tile)
    if w > tile and xs[-1] != w - tile: xs.append(w - tile)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                ph, pw = min(tile, h - y), min(tile, w - x)
                patch = cv2.copyMakeBorder(full[y:y+ph, x:x+pw], 0, tile-ph, 0, tile-pw, cv2.BORDER_REFLECT)
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(v5.device).float()
                out = torch.clamp(v5.model(tensor), 0, 1)[0].cpu().numpy().transpose(1, 2, 0)[:ph, :pw]
                ww = win[:ph, :pw]
                acc[y:y+ph, x:x+pw] += out * ww
                weights[y:y+ph, x:x+pw] += ww
    weights[weights == 0] = 1
    return cv2.cvtColor(np.clip(acc / weights * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _compare_mixed_chain(image, selected_mode):
    """Run canonical Mixed/V5/LIFT chain used by /api/scan."""
    global _appearance_mixed
    import importlib.util
    if _appearance_mixed is None:
        spec = importlib.util.spec_from_file_location("appearance_test_compare", "/home/wahyu/DocRes/test_appearance.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _appearance_mixed = mod.load_model(
            "/home/wahyu/DocRes/finetune_logs/appearance_mixed_custom_v2_pilot_20260810/run1/best.pth"
        )
        _appearance_mixed._appearance_infer = mod.inference_appearance
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        image.save(tf.name)
        temp_path = tf.name
    try:
        mixed_bgr = _appearance_mixed._appearance_infer(_appearance_mixed, temp_path, device=str(device))
    finally:
        os.unlink(temp_path)
    mixed_rgb = cv2.cvtColor(mixed_bgr, cv2.COLOR_BGR2RGB)
    if selected_mode == "docres_appearance_mixed":
        return Image.fromarray(mixed_rgb), "Mixed best iter1500"
    v5 = get_docres_ft_model("finetune_v5")
    if v5 is None:
        raise RuntimeError("V5 model not loaded")
    if selected_mode in ("docres_appearance_mixed_v5_tiled_lift12", "docres_appearance_mixed_v5_tiled_lift12_base_detail"):
        v5_rgb = _v5_tiled_infer(v5, mixed_bgr)
    else:
        v5_rgb = np.array(v5.infer(Image.fromarray(mixed_rgb)))
    arr = np.clip(0.75 * v5_rgb.astype(np.float32) + 0.25 * mixed_rgb.astype(np.float32), 0, 255).astype(np.uint8)
    if selected_mode == "docres_appearance_mixed_v5":
        return Image.fromarray(arr), "Mixed best iter1500 + V5 Blend75"
    if selected_mode == "docres_appearance_mixed_v5_tiled_lift12_base_detail":
        tiled_label = "Mixed best iter1500 + V5 tiled + LIFT12 + Base Appearance detail"
    elif selected_mode == "docres_appearance_mixed_v5_tiled_lift12":
        tiled_label = "Mixed best iter1500 + V5 tiled 576 overlap96 + Blend75 + inline LIFT12"
    else:
        tiled_label = "Mixed best iter1500 + V5 Blend75 + inline LIFT12"
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    luma, aa, bb = cv2.split(lab)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), 45)
    mask = cv2.GaussianBlur(np.clip((bg - gray - 3.0) / 55.0, 0.0, 1.0), (0, 0), 35)
    edge = np.clip(np.abs(cv2.Laplacian(gray, cv2.CV_32F)) / 45.0, 0.0, 1.0)
    luma = np.clip(luma + np.minimum(luma * 0.12 * mask * (1.0 - 0.5 * edge), 22.0), 0.0, 255.0)
    result = cv2.cvtColor(cv2.merge([luma, aa, bb]).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if selected_mode == "docres_appearance_mixed_v5_tiled_lift12_base_detail":
        base_onnx = get_docres_onnx_model()
        if base_onnx is None:
            raise RuntimeError("Base Appearance ONNX not loaded")
        base_rgb = np.array(base_onnx.infer(image, task="appearance"))
        result = _base_appearance_detail_fuse(base_rgb, result, alpha=1.0, sigma=1.0)
    return Image.fromarray(result), tiled_label


@app.post("/api/docres/compare")
async def docres_compare(request: Request, file: UploadFile = File(...), task: str = Form("deshadowing"),
                         selected_mode: str = Form("docres_appearance_mixed")):
    """Compare Base against explicitly selected DocAI model/pipeline on same input."""
    allowed = {"docres_base", "docres_appearance_mixed", "docres_appearance_mixed_v5", "docres_appearance_mixed_v5_lift12", "docres_appearance_mixed_v5_tiled_lift12", "docres_appearance_mixed_v5_tiled_lift12_base_detail"}
    if selected_mode not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported compare mode: {selected_mode}")
    unload_idle_models()
    image = _read_image(file)
    start_total = time.time()
    base_mdl = get_docres_base_model()
    if base_mdl is None:
        raise HTTPException(status_code=503, detail="Base DocRes model not loaded")
    t0 = time.time()
    base_result = base_mdl.infer_task(image, task)
    base_ms = round((time.time() - t0) * 1000)
    t0 = time.time()
    if selected_mode == "docres_base":
        selected_result, selected_label = base_result, "Base production"
    elif selected_mode.startswith("docres_appearance_mixed"):
        selected_result, selected_label = _compare_mixed_chain(image, selected_mode)
    else:
        selected_result, selected_label = get_docres_task_model().infer_task(image, task), "Task model"
    selected_ms = round((time.time() - t0) * 1000)
    total_ms = round((time.time() - start_total) * 1000)
    selected = {"image": f"data:image/png;base64,{_pil_to_b64(selected_result)}", "ms": selected_ms,
                "label": selected_label, "mode": selected_mode}
    return {"success": True, "task": task, "selected_mode": selected_mode, "total_ms": total_ms,
            "base": {"image": f"data:image/png;base64,{_pil_to_b64(base_result)}", "ms": base_ms,
                     "label": "Base production", "mode": "docres_base"},
            "selected": selected, "finetuned": selected}

@app.post("/api/scan/json")
async def scan_document_json(request: Request, file: UploadFile = File(...), mode: str = Form("restore"),
                              pipeline_mode: str = Form("full"), shadow_strength: float = Form(1.0)):
    """Return processing info alongside the scan result."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    image = _read_image(file)
    start = time.time()

    if mode == "restore" and get_doc_restorer_model() is not None:
        with torch.inference_mode():
            restored, mask, info = run_document_restoration_pipeline(
                get_doc_restorer_model(), image, device,
                tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
                mode=pipeline_mode, shadow_strength=shadow_strength,
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    file_path = UPLOAD_DIR / Path(filename).name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), media_type="image/png")


# =====================================================================
#  DOCSHADOW SD7K INFERENCE
# =====================================================================
@app.post("/api/docshadow/infer")
async def docshadow_infer(request: Request, file: UploadFile = File(...), weight: str = Form("SD7K")):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    docres_mdl = get_docres_model()
    if docres_mdl:
        info["docres"] = docres_mdl.get_model_info()
    else:
        info["docres"] = {"loaded": False}
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    docres_mdl = get_docres_model()
    if docres_mdl:
        info["docres"] = docres_mdl.get_model_info()
    else:
        info["docres"] = {"loaded": False}
    docshadow = get_docshadow()
    if docshadow is None:
        return {"loaded": False, "weights": []}
    return {"loaded": True, "weights": docshadow.list_available_weights()}


# =====================================================================
#  DOCRES RESTORMER INFERENCE
# =====================================================================
@app.post("/api/docres/infer")
async def docres_infer(request: Request, file: UploadFile = File(...)):
    """Run DocRes Restormer shadow removal."""
    global _docres
    if _docres is None:
        try:
            _docres = get_docres(device=str(device))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"DocRes model not loaded: {e}")
    image = _read_image(file)
    try:
        start = time.time()
        result = _docres.infer(image)
        elapsed = round((time.time() - start) * 1000, 1)
        return _pil_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.get("/api/docres/status")
async def docres_status(request: Request):
    global _docres
    return {
        "loaded": _docres is not None,
        "checkpoint": _docres.checkpoint_name if _docres else None,
        "im_size": _docres.im_size if _docres else None,
    }


# =====================================================================
#  AI SHADOW POST-PROCESSING
# =====================================================================
@app.post("/api/ai-postprocess")
async def ai_postprocess(request: Request,
                         original: UploadFile = File(...),
                         ai_result: UploadFile = File(...),
                         mask: UploadFile = File(None)):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    results = []
    for f in files:
        uid = uuid.uuid4().hex[:8]
        try:
            image = _read_image(f)
            start = time.time()
            if mode == "restore" and get_doc_restorer_model() is not None:
                with torch.inference_mode():
                    out, _, _ = run_document_restoration_pipeline(
                        get_doc_restorer_model(), image, device,
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    IMAGE_TEST_ROOT.mkdir(parents=True, exist_ok=True)
    tests = []
    for path in sorted(IMAGE_TEST_ROOT.iterdir()):
        if path.is_dir():
            image_count = sum(1 for item in path.rglob('*') if item.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'})
            tests.append({'name': path.name, 'image_count': image_count})
    return {'tests': tests}


@app.get("/api/image-tests/contact-sheet-analysis")
async def contact_sheet_analysis(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    safe_name = Path(name).name
    file_path = IMAGE_TEST_ROOT / safe_name
    if not file_path.is_file() or not safe_name.startswith('contact_sheet') or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=404, detail='Contact sheet not found')
    return FileResponse(file_path)

@app.get("/api/image-tests/files")
async def list_image_test_files(request: Request, test: str, path: str = ''):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    image_count = sum(1 for path in DOWNLOAD_ROOT.rglob('*') if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}) if DOWNLOAD_ROOT.exists() else 0
    size_mb = round(sum(path.stat().st_size for path in DOWNLOAD_ROOT.rglob('*') if path.is_file()) / (1024 * 1024), 2) if DOWNLOAD_ROOT.exists() else 0
    status = 'completed' if image_count >= 7000 else 'partial' if image_count else 'idle'
    progress = min(100, round(image_count / 7000 * 100)) if image_count else 0
    return {'status': status, 'download_active': False, 'progress': progress, 'total': image_count, 'size_mb': size_mb}


@app.post("/api/datasets/download")
async def start_dataset_download(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    return {'success': False, 'error': 'Download otomatis dinonaktifkan. Gunakan instruksi download manual di halaman ini.'}


@app.post("/api/datasets/download/stop")
async def stop_dataset_download(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    return {'success': True, 'message': 'Tidak ada download aktif.'}


# =====================================================================
#  TRAINING STATUS
# =====================================================================
@app.get("/api/training/status")
async def training_status(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    latest_previews = []
    if preview_dir.exists():
        previews = sorted(preview_dir.glob('epoch_*.png'), key=lambda path: path.stat().st_mtime, reverse=True)
        if previews:
            latest_preview = previews[0].name
            for preview in previews[:12]:
                latest_previews.append({
                    'name': preview.name,
                    'epoch': preview.stem.replace('epoch_', ''),
                    'mtime': int(preview.stat().st_mtime),
                    'url': f'/api/training/preview/{preview.name}?run_id={meta.get("run_id", "")}&v={int(preview.stat().st_mtime)}',
                })

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
        if TRAINING_LOG_PATH.exists():
            _load_training_log(TRAINING_LOG_PATH.read_text(encoding='utf-8').splitlines())
        for line in reversed(_training_log):
            # Parse eta=1h 23m 45s from epoch log line
            import re as _re
            m = _re.search(r'eta=(\S+\s+\S+\s+\S+)', line)
            if m:
                eta = m.group(1)
            m = _re.search(r'epoch=(\d+)/(\d+)', line)
            if m and current_epoch is None:
                current_epoch = int(m.group(1))
                total_epochs = int(m.group(2))
            m = _re.search(r'batch=(\d+)/(\d+)', line)
            if m and current_batch is None:
                current_batch = int(m.group(1))
                total_batches = int(m.group(2))
            m = _re.search(r'train_loss=(\S+)', line)
            if m and last_train_loss is None:
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
        'best_loss': (min(history.get('val_loss', [])) if history.get('val_loss') else None),
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
        'training_mode': meta.get('training_mode'),
        'run_config': meta.get('run_config'),
        'log_path': meta.get('log_path'),
        'latest_preview_url': f'/api/training/preview/{latest_preview}?run_id={meta.get("run_id", "")}&v={int((preview_dir / latest_preview).stat().st_mtime)}' if latest_preview else None,
        'latest_previews': latest_previews,
    }


@app.get("/api/training/preview/{filename}")
async def training_preview(request: Request, filename: str, run_id: str = ''):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    preview_root = CHECKPOINT_DIR / 'document_restorer' / 'runs' / Path(run_id).name if run_id else CHECKPOINT_DIR / 'document_restorer'
    preview_path = preview_root / 'previews' / Path(filename).name
    if not preview_path.exists() or preview_path.suffix.lower() != '.png':
        raise HTTPException(status_code=404, detail='Preview not found')
    return FileResponse(preview_path, media_type='image/png')

@app.get("/api/training/runs")
async def training_runs(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    output = _safe_evaluation_output(output)
    preview_path = BASE_DIR / output / 'preview_grid.png'
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail='Evaluation preview not found')
    return FileResponse(preview_path, media_type='image/png')

@app.get("/api/training/evaluation/metrics")
async def evaluation_metrics(request: Request, output: str = 'evaluation/document_restorer'):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
        try:
            ds_path = Path(info.get('abs_path') or (BASE_DIR / path))
            info['updated_at'] = ds_path.stat().st_mtime
        except Exception:
            info['updated_at'] = 0
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
    all_datasets.sort(key=lambda item: item.get('updated_at', 0), reverse=True)
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
    allowed_roots = [(BASE_DIR / 'data' / 'datasets').resolve(), (BASE_DIR / 'datasets').resolve(), (BASE_DIR / 'datasets' / 'magang').resolve(), Path('/home/wahyu/DocRes/data/synthetic_appearance_v1').resolve()]
    if not any(root == allowed or allowed in root.parents for allowed in allowed_roots):
        raise HTTPException(status_code=400, detail='Dataset path outside allowed roots')
    if not root.is_dir():
        raise HTTPException(status_code=404, detail='Dataset folder not found')
    return root


@app.get('/api/datasets/explorer')
async def browse_dataset(request: Request, dataset: str, path: str = '', offset: int = 0, limit: int = 60):
    """Browse one registered dataset without allowing filesystem traversal."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
async def dataset_explorer_image(request: Request, dataset: str, path: str, size: int = 0):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    root = _dataset_root(dataset)
    image_path = (root / path).resolve()
    image_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    if root not in image_path.parents or not image_path.is_file() or image_path.suffix.lower() not in image_suffixes:
        raise HTTPException(status_code=404, detail='Dataset image not found')
    if size > 0:
        import cv2
        from io import BytesIO
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            scale = min(size / w, size / h)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            from fastapi.responses import Response
            return Response(content=buf.tobytes(), media_type='image/jpeg')
    return FileResponse(image_path)


@app.get("/api/datasets/validate")
async def validate_dataset(request: Request, path: str):
    """Validate a specific dataset path."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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

# =====================================================================
#  DATASET AUDIT & PREPARATION
# =====================================================================
from backend.utils.dataset_audit import audit_pair, prepare_pair, prepare_dataset, validate_dataset as validate_dataset_audit

@app.post("/api/datasets/audit-pair")
async def api_audit_pair(request: Request):
    """Audit a shadow/clean image pair for dataset feasibility."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import tempfile, os

    form = await request.form()
    shadow_file = form.get('shadow')
    clean_file = form.get('clean')

    if not shadow_file or not clean_file:
        raise HTTPException(status_code=400, detail='Both shadow and clean files required')

    with tempfile.TemporaryDirectory() as tmpdir:
        shadow_path = os.path.join(tmpdir, 'shadow.jpg')
        clean_path = os.path.join(tmpdir, 'clean.jpg')

        shadow_bytes = await shadow_file.read()
        clean_bytes = await clean_file.read()

        with open(shadow_path, 'wb') as f:
            f.write(shadow_bytes)
        with open(clean_path, 'wb') as f:
            f.write(clean_bytes)

        result = audit_pair(shadow_path, clean_path)

    return result


@app.post("/api/datasets/prepare")
async def api_prepare_pair(request: Request):
    """Prepare a shadow/clean pair: resize, align, save to dataset dir."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import tempfile, os

    form = await request.form()
    shadow_file = form.get('shadow')
    clean_file = form.get('clean')
    output_dir = form.get('output_dir', 'datasets/paired/custom_prepared')
    target_size_str = form.get('target_size', '768,768')
    do_align = form.get('align', 'true').lower() == 'true'

    if not shadow_file or not clean_file:
        raise HTTPException(status_code=400, detail='Both shadow and clean files required')

    target_size = tuple(map(int, target_size_str.split(',')))
    full_output = str(BASE_DIR / output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        shadow_path = os.path.join(tmpdir, 'shadow.jpg')
        clean_path = os.path.join(tmpdir, 'clean.jpg')

        shadow_bytes = await shadow_file.read()
        clean_bytes = await clean_file.read()

        with open(shadow_path, 'wb') as f:
            f.write(shadow_bytes)
        with open(clean_path, 'wb') as f:
            f.write(clean_bytes)

        result = prepare_pair(shadow_path, clean_path, full_output, target_size, do_align)

    return result


@app.get("/api/datasets/validate-pairs")
async def api_validate_pairs(request: Request, path: str):
    """Validate all pairs in a prepared dataset directory."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading

    target = str(BASE_DIR / path)
    result = validate_dataset_audit(target)
    return result


@app.get("/api/datasets/audit-local")
async def api_audit_local(request: Request, shadow: str, clean: str):
    """Audit local files on server by path."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import os

    shadow_path = str(BASE_DIR / shadow)
    clean_path = str(BASE_DIR / clean)

    if not os.path.exists(shadow_path):
        raise HTTPException(status_code=404, detail='Shadow file not found')
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail='Clean file not found')

    result = audit_pair(shadow_path, clean_path)
    return result


@app.post("/api/datasets/prepare-local")
async def api_prepare_local(request: Request):
    """Prepare local files on server."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import os

    form = await request.form()
    shadow_path = str(BASE_DIR / form.get('shadow', ''))
    clean_path = str(BASE_DIR / form.get('clean', ''))
    output_dir = str(BASE_DIR / form.get('output_dir', 'datasets/paired/custom_prepared'))
    target_size = tuple(map(int, form.get('target_size', '768,768').split(',')))
    do_align = form.get('align', 'true').lower() == 'true'

    if not os.path.exists(shadow_path):
        raise HTTPException(status_code=404, detail='Shadow file not found')
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail='Clean file not found')

    result = prepare_pair(shadow_path, clean_path, output_dir, target_size, do_align)
    return result


# =====================================================================
#  DATASET CRUD
# =====================================================================
from backend.utils.dataset_audit import (
    list_datasets, get_dataset, create_dataset, delete_dataset, update_dataset_meta,
    audit_pair, prepare_pair
)

@app.get("/api/datasets/custom")
async def api_list_datasets(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    return {"datasets": list_datasets()}

@app.post("/api/datasets/custom")
async def api_create_dataset(request: Request):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    form = await request.form()
    slug = form.get("slug", "")
    name = form.get("name", "")
    description = form.get("description", "")
    target_size = form.get("target_size", "768,768")
    if not slug:
        raise HTTPException(status_code=400, detail="slug required")
    result = create_dataset(slug, name, description, target_size)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/api/datasets/custom/{slug}")
async def api_get_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    ds = get_dataset(slug)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

@app.delete("/api/datasets/custom/{slug}")
async def api_delete_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    if not delete_dataset(slug):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True}

@app.put("/api/datasets/custom/{slug}")
async def api_update_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    form = await request.form()
    name = form.get("name")
    description = form.get("description")
    result = update_dataset_meta(slug, name, description)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/datasets/custom/{slug}/audit")
async def api_audit_in_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import tempfile
    ds_path = os.path.join("datasets/paired/custom", slug)
    if not os.path.exists(ds_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    form = await request.form()
    shadow_file = form.get("shadow")
    clean_file = form.get("clean")
    if not shadow_file or not clean_file:
        raise HTTPException(status_code=400, detail="Both shadow and clean files required")
    with tempfile.TemporaryDirectory() as tmpdir:
        shadow_path = os.path.join(tmpdir, "shadow.jpg")
        clean_path = os.path.join(tmpdir, "clean.jpg")
        with open(shadow_path, "wb") as f:
            f.write(await shadow_file.read())
        with open(clean_path, "wb") as f:
            f.write(await clean_file.read())
        result = audit_pair(shadow_path, clean_path)
    return result

@app.post("/api/datasets/custom/{slug}/add")
async def api_add_to_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    import tempfile
    ds_path = os.path.join("datasets/paired/custom", slug)
    if not os.path.exists(ds_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    form = await request.form()
    shadow_file = form.get("shadow")
    clean_file = form.get("clean")
    if not shadow_file or not clean_file:
        raise HTTPException(status_code=400, detail="Both shadow and clean files required")
    with tempfile.TemporaryDirectory() as tmpdir:
        shadow_path = os.path.join(tmpdir, "shadow.jpg")
        clean_path = os.path.join(tmpdir, "clean.jpg")
        with open(shadow_path, "wb") as f:
            f.write(await shadow_file.read())
        with open(clean_path, "wb") as f:
            f.write(await clean_file.read())
        result = prepare_pair(shadow_path, clean_path, ds_path, (768, 768), True)
    return result

@app.delete("/api/datasets/custom/{slug}/{filename}")
async def api_delete_pair(request: Request, slug: str, filename: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    ds_path = os.path.join("datasets/paired/custom", slug)
    input_path = os.path.join(ds_path, "input", filename)
    target_path = os.path.join(ds_path, "target", filename)
    deleted = []
    if os.path.exists(input_path):
        os.remove(input_path)
        deleted.append("input")
    if os.path.exists(target_path):
        os.remove(target_path)
        deleted.append("target")
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "deleted": deleted}


@app.post("/api/datasets/custom/{slug}/move/{filename}")
async def api_move_pair(request: Request, slug: str, filename: str):
    """Move pair from one dataset to another. Body: { target_slug: str }"""
    import shutil
    body = await request.json()
    target_slug = body.get("target_slug", "")
    if not target_slug:
        raise HTTPException(status_code=400, detail="target_slug required")
    
    src_path = os.path.join("datasets/paired/custom", slug)
    dst_path = os.path.join("datasets/paired/custom", target_slug)
    
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, "input"), exist_ok=True)
        os.makedirs(os.path.join(dst_path, "target"), exist_ok=True)
    
    moved = []
    for role in ("input", "target"):
        src = os.path.join(src_path, role, filename)
        dst = os.path.join(dst_path, role, filename)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved.append(role)
    
    if not moved:
        raise HTTPException(status_code=404, detail="File not found")
    return {"success": True, "moved": moved, "from": slug, "to": target_slug}


@app.get("/api/datasets/custom/{slug}/audit/{filename}")
async def api_audit_single_pair(request: Request, slug: str, filename: str):
    """Audit a single pair and return reasons."""
    import numpy as np
    from PIL import Image
    
    ds_path = os.path.join("datasets/paired/custom", slug)
    in_path = os.path.join(ds_path, "input", filename)
    gt_path = os.path.join(ds_path, "target", filename)
    
    if not os.path.exists(in_path) or not os.path.exists(gt_path):
        raise HTTPException(status_code=404, detail="Pair not found")
    
    in_img = np.array(Image.open(in_path).convert("RGB"))
    gt_img = np.array(Image.open(gt_path).convert("RGB"))
    
    reasons = []
    brightness_diff = float(gt_img.mean() - in_img.mean())
    gt_white_ratio = float((gt_img > 200).mean())
    diff = float(np.abs(in_img.astype(float) - gt_img.astype(float)).mean())
    
    if brightness_diff <= 10:
        reasons.append(f"Target kurang terang (delta={brightness_diff:.1f})")
    if gt_white_ratio <= 0.5:
        reasons.append(f"Target kurang bersih (white={gt_white_ratio:.2f})")
    if diff <= 5:
        reasons.append(f"Input/target terlalu mirip (diff={diff:.1f})")
    
    if in_img.size != gt_img.size:
        reasons.append(f"Resolusi beda: input={list(in_img.shape[:2][::-1])} target={list(gt_img.shape[:2][::-1])}")
    
    return {
        "filename": filename,
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "brightness_diff": round(brightness_diff, 1),
            "gt_white_ratio": round(gt_white_ratio, 3),
            "pixel_diff": round(diff, 1),
        }
    }


from backend.utils.dataset_audit import validate_dataset, get_recommendation

@app.get("/api/datasets/custom/{slug}/validate")
async def api_validate_dataset(request: Request, slug: str):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    ds_path = os.path.join("datasets/paired/custom", slug)
    if not os.path.exists(ds_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    result = validate_dataset(ds_path)
    result["recommendations"] = get_recommendation(result)
    return result

#  LIVE MODEL RELOAD (test during training)
# =====================================================================
@app.post("/api/model/reload")
async def reload_model(request: Request):
    """Reload best.pth from disk — allows testing model while training continues."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    try:
        import re as _re
        match = _re.search(r'(\d+)h\s+(\d+)m\s+(\d+)s', eta)
        if not match:
            return None
        hours, minutes, seconds = (int(value) for value in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
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

def _record_training_meta(kind: str, started_at: float, estimated_finish_at: float | None, cmd: list[str], **extra):
    TRAINING_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'kind': kind,
        'started_at': started_at,
        'started_at_wib': _format_wib(started_at),
        'estimated_finish_at': estimated_finish_at,
        'estimated_finish_wib': _format_wib(estimated_finish_at),
        'cmd': cmd,
    }
    payload.update(extra)
    TRAINING_META_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')

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

def _process_matches_training(pid: int | None) -> bool:
    if not pid or not _pid_running(pid):
        return False
    try:
        proc = psutil.Process(pid)
        cmdline = ' '.join(proc.cmdline())
        return 'train.py' in cmdline or 'evaluate_restorer.py' in cmdline
    except Exception:
        return False

def _find_training_pid() -> int | None:
    pid = _stored_training_pid()
    if _process_matches_training(pid):
        return pid
    meta = _stored_training_meta()
    active_run_id = meta.get('run_id') or ''
    try:
        result = subprocess.run(['pgrep', '-af', 'train.py|evaluate_restorer.py'], capture_output=True, text=True, timeout=2)
        for line in result.stdout.splitlines():
            parts = line.strip().split(' ', 1)
            if not parts or not parts[0].isdigit():
                continue
            cmdline = parts[1] if len(parts) > 1 else ''
            if active_run_id and active_run_id not in cmdline:
                continue
            if 'multiprocessing.forkserver' in cmdline:
                continue
            return int(parts[0])
    except Exception:
        pass
    return None

def _training_is_running() -> bool:
    if _training_process is not None and _training_process.poll() is None:
        return True
    pid = _find_training_pid()
    if pid:
        _record_training_pid(pid)
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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

@app.post("/api/docres576/process-json")
async def docres576_process_json(request: Request, file: UploadFile = File(...), adj_brightness: float = Form(1.0), adj_contrast: float = Form(1.0), task: str = Form("deshadowing")):
    """Test current DocRes 576 best checkpoint on CPU without touching training GPU."""
    unload_idle_models()
    image = _read_image(file)
    import glob as _g
    import pathlib as _pl; runs=sorted([d for d in (BASE_DIR.parent/"DocRes"/"finetune_logs").glob("base_multitask_*") if (d/"best.pth").exists()],key=lambda x:x.stat().st_mtime,reverse=True); ckpt=runs[0]/"best.pth" if runs else BASE_DIR.parent/"DocRes"/"finetune_logs"/"deshadow_v3_20260804_015921"/"best.pth"
    script = BASE_DIR.parent / "DocRes" / "test_deshadow.py"
    if not ckpt.exists():
        raise HTTPException(status_code=404, detail=f"DocRes 576 checkpoint not found: {ckpt}")
    start_total = time.time()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as inp:
        image.save(inp.name)
        in_path = inp.name
    out_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    script = BASE_DIR.parent / "DocRes" / "inference.py"
    try:
        cmd = [str(BASE_DIR.parent / "miniconda3" / "bin" / "python3"), str(script), "--model_path", str(ckpt), "--im_path", in_path, "--out_folder", os.path.dirname(out_path), "--task", task]
        _env = os.environ.copy(); _env["CUDA_VISIBLE_DEVICES"] = ""; proc = subprocess.run(cmd, cwd=str(BASE_DIR.parent / "DocRes"), capture_output=True, text=True, timeout=300, env=_env)
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=(proc.stderr or proc.stdout)[-1000:])
        final = Image.open(out_path).convert("RGB")
        if abs(adj_brightness - 1.0) > 0.001:
            final = ImageEnhance.Brightness(final).enhance(adj_brightness)
        if abs(adj_contrast - 1.0) > 0.001:
            final = ImageEnhance.Contrast(final).enhance(adj_contrast)
        total_ms = round((time.time() - start_total) * 1000)
        return {
            "success": True,
            "total_ms": total_ms,
            "mode": "docres576_best",
            "checkpoint": str(ckpt),
            "image": f"data:image/png;base64,{_pil_to_b64(final)}",
            "steps": [
                {"name":"input","label":"Input","status":"done","time_ms":0,"info":f"Original size: {image.size[0]}x{image.size[1]}"},
                {"name":"docres576","label":"DocRes 576 Best","status":"done","time_ms":total_ms,"info":"CPU inference, current best.pth"},
                {"name":"adjust","label":"Brightness/Contrast","status":"done","time_ms":0,"info":f"brightness={adj_brightness}, contrast={adj_contrast}"},
            ],
            "pipeline": ["input", "docres576_best", "brightness_contrast"],
        }
    finally:
        try: os.unlink(in_path)
        except Exception: pass
        try: os.unlink(out_path)
        except Exception: pass

@app.post("/api/pipeline/process")
async def pipeline_process(request: Request, file: UploadFile = File(...)):
    """Run the document restoration pipeline in the production order."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    if get_doc_restorer_model() is None:
        raise HTTPException(status_code=503, detail='DocumentRestorerNet checkpoint not loaded')
    image = _read_image(file)
    with torch.inference_mode():
        final, _, _ = run_document_restoration_pipeline(
            get_doc_restorer_model(), image, device,
            tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
        )
    return _pil_to_response(final, filename='pipeline_result.png')

@app.post("/api/pipeline/process-json")
async def pipeline_process_json(request: Request, file: UploadFile = File(...), mode: str = Form("full"), adj_brightness: float = Form(1.0), adj_contrast: float = Form(1.0)):
    """Run the production pipeline and return ordered step metadata."""
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    if get_doc_restorer_model() is None:
        raise HTTPException(status_code=503, detail='DocumentRestorerNet checkpoint not loaded')
    image = _read_image(file)
    start_total = time.time()
    with torch.inference_mode():
        final, _, info = run_document_restoration_pipeline(
            get_doc_restorer_model(), image, device,
            tile_size=RESTORATION_TILE_SIZE, overlap=RESTORATION_TILE_OVERLAP,
            mode=mode,
        )
    # Optional light post-adjust for web testing (works best with ai_only)
    if abs(adj_brightness - 1.0) > 0.001:
        final = ImageEnhance.Brightness(final).enhance(adj_brightness)
    if abs(adj_contrast - 1.0) > 0.001:
        final = ImageEnhance.Contrast(final).enhance(adj_contrast)

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
                         shadow_loss_weight: float = Form(1.5),
                         illumination_weight: float = Form(0.20),
                         mask_loss_weight: float = Form(0.25),
                         gradient_weight: float = Form(0.05),
                         color_weight: float = Form(0.15),
                         identity_weight: float = Form(1.0),
                         color_preservation_weight: float = Form(0.9),
                         text_weight: float = Form(0.2),
                         non_shadow_weight: float = Form(0.5),
                         warmup_epochs: int = Form(3),
                         max_train_samples: int = Form(0),
                         max_val_samples: int = Form(0)):
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    global _training_process, _training_log, _training_started_at, _training_estimated_finish_at, _training_kind

    if _training_is_running():
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
        color_preservation_weight=color_preservation_weight,
        text_weight=text_weight,
        non_shadow_weight=non_shadow_weight,
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
        '--color-preservation-weight', str(color_preservation_weight),
        '--text-weight', str(text_weight),
        '--non-shadow-weight', str(non_shadow_weight),
        '--warmup-epochs', str(warmup_epochs),
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
        training_mode = 'fine_tune' if (resume and resume_weights_only) else ('resume' if resume else 'train')
        run_config = {
            'epochs': epochs, 'batch_size': batch_size, 'size': size, 'lr': lr,
            'base_channels': base_channels, 'workers': workers, 'device': device,
            'early_stop_patience': early_stop_patience, 'min_delta': min_delta, 'grad_clip_norm': grad_clip_norm,
            'perceptual_weight': perceptual_weight, 'ssim_weight': ssim_weight,
            'shadow_loss_weight': shadow_loss_weight, 'illumination_weight': illumination_weight,
            'mask_loss_weight': mask_loss_weight, 'gradient_weight': gradient_weight,
            'color_weight': color_weight, 'identity_weight': identity_weight,
            'color_preservation_weight': color_preservation_weight, 'text_weight': text_weight,
            'non_shadow_weight': non_shadow_weight, 'warmup_epochs': warmup_epochs,
            'resume': resume, 'resume_weights_only': resume_weights_only,
            'paired_data': paired_data, 'clean_data': clean_data, 'identity_data': identity_data,
            'validation_paired_data': validation_paired_data,
        }
        _record_training_meta(
            _training_kind, _training_started_at, _training_estimated_finish_at, cmd,
            run_id=run_id, run_output=run_output, publish_output=publish_output,
            training_mode=training_mode, run_config=run_config, log_path=str(TRAINING_LOG_PATH.relative_to(BASE_DIR)),
        )

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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
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
    unload_idle_models()
    # require_api_auth(request)  # Public for image loading
    with _training_lock:
        running = _training_is_running()
        if TRAINING_LOG_PATH.exists():
            _load_training_log(TRAINING_LOG_PATH.read_text(encoding='utf-8').splitlines())
        lines = _training_log[offset:]
    return {'running': running, 'lines': lines, 'total': len(_training_log)}


# === Synthetic Shadow Generator ===
import io, base64, random, math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def _make_object_shadow_mask(w, h, obj_type, intensity, width_pct, margin_x=0.0, margin_y=0.0):
    """Generate realistic object shadow mask using PIL polygon drawing.
    obj_type: hand, phone, finger
    margin_x: 0=at left/right edge, 1.0=at center
    margin_y: 0=at bottom edge, 1.0=at center
    """
    from PIL import Image, ImageDraw, ImageFilter
    import math

    rng = np.random.RandomState()

    # Compute position based on margin (0=edge, 1.0=center)
    # X: margin_x=0 -> edge (random side), margin_x=1.0 -> center
    # Y: margin_y=0 -> bottom edge, margin_y=1.0 -> center
    if margin_x < 0.5:
        # Left edge side
        cx = w * (0.05 + margin_x * 0.45)
    else:
        # Right edge side
        cx = w * (0.5 + (margin_x - 0.5) * 0.9)
    cy = h * (0.95 - margin_y * 0.45)  # 0.95=bottom, 0.5=center

    if obj_type == "hand":
        palm_cx = cx
        palm_cy = cy
        palm_rx = w * 0.18
        palm_ry = h * 0.12

        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        # Draw palm (ellipse)
        draw.ellipse([palm_cx - palm_rx, palm_cy - palm_ry, palm_cx + palm_rx, palm_cy + palm_ry], fill=240)

        # Draw 4 fingers (index, middle, ring, pinky) - large and spread
        fingers = [
            {"dx": -0.14, "dy": -0.25, "fw": 0.05, "fh": 0.25, "angle": -15},
            {"dx": -0.05, "dy": -0.30, "fw": 0.05, "fh": 0.30, "angle": -3},
            {"dx": 0.05, "dy": -0.28, "fw": 0.05, "fh": 0.28, "angle": 7},
            {"dx": 0.12, "dy": -0.20, "fw": 0.045, "fh": 0.20, "angle": 15},
        ]

        for f in fingers:
            fx = palm_cx + w * f["dx"] + rng.randint(-10, 10)
            fy = palm_cy + h * f["dy"] + rng.randint(-5, 5)
            fw = w * f["fw"]
            fh = h * f["fh"]

            n_pts = 40
            points = []
            for i in range(n_pts):
                t = i / (n_pts - 1)
                width_factor = 1.0 - 0.4 * t
                tip_round = math.sin(t * math.pi)
                px = fx + fw * width_factor * tip_round
                py = fy + fh * t
                points.append((px, py))
            for i in range(n_pts - 1, -1, -1):
                t = i / (n_pts - 1)
                width_factor = 1.0 - 0.4 * t
                tip_round = math.sin(t * math.pi)
                px = fx - fw * width_factor * tip_round
                py = fy + fh * t
                points.append((px, py))

            if len(points) >= 3:
                draw.polygon(points, fill=220)

        # Draw thumb (from side)
        thumb_x = palm_cx - w * 0.20
        thumb_y = palm_cy + h * 0.02
        thumb_w = w * 0.04
        thumb_h = h * 0.18

        n_pts = 40
        points = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            width_factor = 1.0 - 0.5 * t
            tip_round = math.sin(t * math.pi)
            px = thumb_x + thumb_w * width_factor * tip_round
            py = thumb_y + thumb_h * t
            points.append((px, py))
        for i in range(n_pts - 1, -1, -1):
            t = i / (n_pts - 1)
            width_factor = 1.0 - 0.5 * t
            tip_round = math.sin(t * math.pi)
            px = thumb_x - thumb_w * width_factor * tip_round
            py = thumb_y + thumb_h * t
            points.append((px, py))
        if len(points) >= 3:
            draw.polygon(points, fill=210)

        blur_radius = max(int(min(w, h) * 0.003), 2)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    elif obj_type == "phone":
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        phone_cx = cx
        phone_cy = cy
        phone_w = w * rng.uniform(0.18, 0.30)
        phone_h = h * rng.uniform(0.30, 0.45)
        phone_angle = rng.uniform(-20, 20)

        x1 = int(phone_cx - phone_w / 2)
        y1 = int(phone_cy - phone_h / 2)
        x2 = int(phone_cx + phone_w / 2)
        y2 = int(phone_cy + phone_h / 2)
        radius = int(min(phone_w, phone_h) * 0.08)

        draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=240)

        notch_w = phone_w * 0.12
        notch_h = phone_h * 0.025
        notch_x = int(phone_cx - notch_w / 2)
        notch_y = int(phone_cy - phone_h / 2)
        draw.ellipse([notch_x, notch_y, notch_x + notch_w, notch_y + notch_h], fill=200)

        blur_radius = max(int(min(w, h) * 0.003), 2)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_img = mask_img.rotate(phone_angle, expand=False, center=(int(phone_cx), int(phone_cy)))

    elif obj_type == "finger":
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        finger_cx = cx
        finger_cy = cy
        finger_w = w * 0.06
        finger_h = h * 0.22
        finger_angle = rng.uniform(-35, 35)

        n_pts = 50
        points = []
        for i in range(n_pts):
            t = i / (n_pts - 1)
            width_factor = 1.0 - 0.5 * t
            tip_round = math.sin(t * math.pi)
            px = finger_cx + finger_w * width_factor * tip_round
            py = finger_cy - finger_h * t
            points.append((px, py))
        for i in range(n_pts - 1, -1, -1):
            t = i / (n_pts - 1)
            width_factor = 1.0 - 0.5 * t
            tip_round = math.sin(t * math.pi)
            px = finger_cx - finger_w * width_factor * tip_round
            py = finger_cy - finger_h * t
            points.append((px, py))

        if len(points) >= 3:
            draw.polygon(points, fill=240)

        nail_cx = finger_cx
        nail_cy = finger_cy - finger_h * 0.82
        nail_w = finger_w * 0.5
        nail_h = finger_h * 0.06
        draw.ellipse([nail_cx - nail_w, nail_cy - nail_h, nail_cx + nail_w, nail_cy + nail_h], fill=200)

        blur_radius = max(int(min(w, h) * 0.003), 2)
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_img = mask_img.rotate(finger_angle, expand=False, center=(int(finger_cx), int(finger_cy)))

    else:
        return np.zeros((h, w), dtype=np.float32)

    # Convert to numpy array    # Convert to numpy array
    mask = np.array(mask_img).astype(np.float32) / 255.0
    mask = mask * intensity
    mask = np.clip(mask, 0, 1)
    return mask
def generate_synthetic_shadow(img: Image.Image, position: str = "bottom", intensity: float = 0.7, width_pct: float = 0.4) -> Image.Image:
    """Generate synthetic shadow on image.
    position: left, right, top, bottom, all, bottom_left, bottom_right
    intensity: 0.0-1.0 (shadow darkness)
    width_pct: how much of the image the shadow covers (0.0-1.0)
    """
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    # Create gradient shadow mask based on position
    if position == "bottom":
        start = int(h * (1 - width_pct))
        for y in range(start, h):
            t = (y - start) / max(h - start, 1)
            fade = intensity * (t ** 0.6)
            mask[y, :] = fade
    elif position == "top":
        end = int(h * width_pct)
        for y in range(end):
            t = 1 - (y / max(end, 1))
            fade = intensity * (t ** 0.6)
            mask[y, :] = fade
    elif position == "left":
        end = int(w * width_pct)
        for x in range(end):
            t = 1 - (x / max(end, 1))
            fade = intensity * (t ** 0.6)
            mask[:, x] = fade
    elif position == "right":
        start = int(w * (1 - width_pct))
        for x in range(start, w):
            t = (x - start) / max(w - start, 1)
            fade = intensity * (t ** 0.6)
            mask[:, x] = fade
    elif position == "all":
        for y in range(h):
            for x in range(w):
                dy = min(y, h - y) / (h / 2)
                dx = min(x, w - x) / (w / 2)
                edge_factor = 1 - min(dy, dx)
                if edge_factor > 0:
                    mask[y, x] = intensity * (edge_factor ** 0.5)
    elif position == "bottom_left":
        for y in range(h):
            for x in range(w):
                dist = (x / w + y / h) / 2
                if dist > 0.5:
                    mask[y, x] = intensity * ((dist - 0.5) * 2) ** 0.6
    elif position == "bottom_right":
        for y in range(h):
            for x in range(w):
                dist = ((w - x) / w + y / h) / 2
                if dist > 0.5:
                    mask[y, x] = intensity * ((dist - 0.5) * 2) ** 0.6

    # Apply shadow mask with blur for soft edges
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=15))
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0

    # Apply shadow: darken the image where mask is dark
    shadow_color = np.array([0.0, 0.0, 0.0])
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - mask_arr) + shadow_color[c] * mask_arr * arr[:, :, c] * intensity

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_object_shadow(img: Image.Image, obj_type: str = "hand", intensity: float = 0.7, width_pct: float = 0.4, margin_x: float = 0.0, margin_y: float = 0.0) -> Image.Image:
    """Generate realistic object shadow on image.
    obj_type: hand, phone, finger
    intensity: 0.0-1.0 (shadow darkness)
    width_pct: how much of the image the shadow covers (0.0-1.0)
    """
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    mask = _make_object_shadow_mask(w, h, obj_type, intensity, width_pct, margin_x, margin_y)

    # Apply shadow mask with blur for soft edges
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.005))
    mask_arr = np.array(mask_img).astype(np.float32) / 255.0

    # Apply shadow: darken the image where mask is dark
    shadow_color = np.array([0.0, 0.0, 0.0])
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * (1 - mask_arr) + shadow_color[c] * mask_arr * arr[:, :, c] * intensity

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

@app.post("/api/synthetic-shadow/generate")
async def generate_shadow(
    file: UploadFile = File(...),
    position: str = Form("bottom"),
    intensity: float = Form(0.7),
    width_pct: float = Form(0.4),
    obj_type: str = Form("none"),
    margin_x: float = Form(0.0),
    margin_y: float = Form(0.0),
):
    """Generate synthetic shadow on uploaded image.
    obj_type: none (gradient), hand, phone, finger
    margin_x: horizontal offset from edge (0.0-1.0)
    margin_y: vertical offset from edge (0.0-1.0)
    """
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Resize if too large
        max_dim = 1024
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if obj_type in ("hand", "phone", "finger"):
            result = generate_object_shadow(img, obj_type, intensity, width_pct, margin_x, margin_y)
        else:
            result = generate_synthetic_shadow(img, position, intensity, width_pct)
        # Return as JPEG base64 (much smaller than PNG)
        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {"success": True, "image": f"data:image/jpeg;base64,{b64}", "position": position, "intensity": intensity, "obj_type": obj_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthetic-shadow/save")
async def save_shadow_pairs(
    files: List[UploadFile] = File(...),
    position: str = Form("bottom"),
    intensity: float = Form(0.7),
    width_pct: float = Form(0.4),
    obj_type: str = Form("none"),
    margin_x: float = Form(0.0),
    margin_y: float = Form(0.0),
    output_dir: str = Form("datasets/paired/custom"),
):
    """Save shadow pairs to dataset directory."""
    try:
        out_dir = BASE_DIR / output_dir
        input_dir = out_dir / "input"
        target_dir = out_dir / "target"
        input_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Find next index
        existing = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        idx = len(existing) + 1

        saved = []
        for f in files:
            data = await f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            # Resize if too large
            max_dim = 1024
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # Save target (clean)
            target_path = target_dir / f"pair_{idx:04d}.png"
            img.save(str(target_path))
            # Generate and save shadow
            shadow = generate_synthetic_shadow(img, position, intensity, width_pct)
            input_path = input_dir / f"pair_{idx:04d}.png"
            shadow.save(str(input_path))
            saved.append(f"pair_{idx:04d}.png")
            idx += 1

        return {"success": True, "saved": len(saved), "pairs": saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

