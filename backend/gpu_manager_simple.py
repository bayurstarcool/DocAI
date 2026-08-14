"""GPU memory manager: simple request-based unload.

Unloads idle models (>5min) to free BOTH VRAM and CPU RAM. Critical: after
Python drops model refs + gc.collect(), glibc still holds freed heap arenas.
malloc_trim(0) returns that memory to the OS, preventing the slow RSS creep
that previously grew the backend to 60+GB over many large-image requests.
"""
import ctypes
import ctypes.util
import gc
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_last_use = {}
_IDLE_TIMEOUT = 300

# Bind glibc malloc_trim once (Linux). No-op elsewhere.
_libc = None
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int
except Exception:
    _libc = None


def _release_ram():
    """Return freed heap arenas to the OS (glibc). Safe no-op if unavailable."""
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def mark_used(name):
    _last_use[name] = time.time()


def unload_idle_models():
    """Check and unload models idle >5min. Called on each request."""
    if not HAS_TORCH or not _last_use:
        return
    now = time.time()
    unloaded = False
    try:
        import backend.app as app
        for name in list(_last_use.keys()):
            age = now - _last_use[name]
            if age > _IDLE_TIMEOUT:
                if name == "task" and getattr(app, '_docres_task', None) is not None:
                    del app._docres_task; app._docres_task = None; unloaded = True
                elif name == "base" and getattr(app, '_docres_base', None) is not None:
                    del app._docres_base; app._docres_base = None; unloaded = True
                elif name == "docres" and getattr(app, '_docres', None) is not None:
                    del app._docres; app._docres = None; unloaded = True
                elif name == "shadow" and getattr(app, '_docshadow', None) is not None:
                    del app._docshadow; app._docshadow = None; unloaded = True
                _last_use.pop(name, None)
        if unloaded:
            gc.collect()
            try: torch.cuda.empty_cache()
            except: pass
            _release_ram()  # return CPU heap to OS (prevents RSS creep)
            print(f"[GPU] unloaded idle models: {torch.cuda.memory_allocated()//1024//1024}MB VRAM", flush=True)
    except Exception as e:
        print(f"[GPU] unload error: {e}", flush=True)


def reclaim_ram_now():
    """Force RAM reclaim without waiting for idle timeout.
    Call after heavy large-image requests to trim per-request numpy/PIL buffers."""
    gc.collect()
    if HAS_TORCH:
        try: torch.cuda.empty_cache()
        except: pass
    _release_ram()
