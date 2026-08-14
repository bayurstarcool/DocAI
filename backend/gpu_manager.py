"""GPU memory manager: auto-unload idle models."""
import time
import threading
import gc

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Track last use time for each model
_last_use = {}
_IDLE_TIMEOUT = 300  # 5 minutes


def mark_used(name):
    _last_use[name] = time.time()


def unload_idle_models():
    """Unload models that haven't been used for IDLE_TIMEOUT seconds."""
    if not HAS_TORCH:
        return
    now = time.time()
    to_unload = []
    for name, last in list(_last_use.items()):
        if now - last > _IDLE_TIMEOUT:
            to_unload.append(name)

    if not to_unload:
        return

    import backend.app as app
    for name in to_unload:
        if name == 'docres_task' and app._docres_task is not None:
            del app._docres_task
            app._docres_task = None
            _last_use.pop(name, None)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPU] Unloaded {name}")
        elif name == 'docres_base' and app._docres_base is not None:
            del app._docres_base
            app._docres_base = None
            _last_use.pop(name, None)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPU] Unloaded {name}")
        elif name == 'docres' and app._docres is not None:
            del app._docres
            app._docres = None
            _last_use.pop(name, None)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GPU] Unloaded {name}")


def get_gpu_usage():
    """Return GPU memory usage in MB."""
    if not HAS_TORCH:
        return 0, 0
    used = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return round(used), round(reserved)
