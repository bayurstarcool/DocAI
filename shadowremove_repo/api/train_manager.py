from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

EPOCH_RE = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.]+)\s+val_psnr=([0-9.]+)\s+best=([0-9.]+)")

class TrainJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.status = "idle"
        self.cmd: list[str] = []
        self.logs: deque[str] = deque(maxlen=1000)
        self.metrics: dict[str, Any] = {}
        self.worker: threading.Thread | None = None

    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("training already running")
            self.logs.clear()
            self.metrics = {}
            self.started_at = time.time()
            self.finished_at = None
            self.returncode = None
            self.status = "running"
            self.cmd = self._build_cmd(params)
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("OMP_NUM_THREADS", str(params.get("threads", 8)))
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=Path(__file__).resolve().parents[1],
                start_new_session=True,
            )
            self.worker = threading.Thread(target=self._read_logs, daemon=True)
            self.worker.start()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                self.status = "idle" if self.returncode is None else self.status
                return self.snapshot()
            self.status = "stopping"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        proc = self.proc
        if proc is not None:
            code = proc.poll()
            if code is not None and self.status in {"running", "stopping"}:
                self.returncode = code
                self.finished_at = self.finished_at or time.time()
                self.status = "completed" if code == 0 else "failed"
        return {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "cmd": self.cmd,
            "metrics": self.metrics,
            "logs": list(self.logs),
            "checkpoints": self._checkpoints(),
        }

    def _build_cmd(self, p: dict[str, Any]) -> list[str]:
        cmd = [sys.executable, "train.py"]
        paired_shadow = str(p.get("paired_shadow") or "").strip()
        paired_clean = str(p.get("paired_clean") or "").strip()
        paired_mask = str(p.get("paired_mask") or "").strip()
        if paired_shadow and paired_clean:
            cmd += ["--paired-shadow", paired_shadow, "--paired-clean", paired_clean]
            if paired_mask:
                cmd += ["--paired-mask", paired_mask]
        else:
            cmd += ["--data", str(p.get("data") or "data/clean_docs")]
        for key, flag in [
            ("epochs", "--epochs"), ("batch", "--batch"), ("size", "--size"),
            ("lr", "--lr"), ("base", "--base"), ("workers", "--workers"),
            ("out", "--out"), ("grad_accum", "--grad-accum"), ("device", "--device"),
        ]:
            value = p.get(key)
            if value not in (None, ""):
                cmd += [flag, str(value)]
        resume = str(p.get("resume") or "").strip()
        if resume:
            cmd += ["--resume", resume]
        init_weights = str(p.get("init_weights") or "").strip()
        if init_weights:
            cmd += ["--init-weights", init_weights]
        if bool(p.get("amp", True)):
            cmd += ["--amp"]
        return cmd

    def _read_logs(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            with self.lock:
                self.logs.append(line)
                match = EPOCH_RE.search(line)
                if match:
                    self.metrics = {
                        "epoch": int(match.group(1)),
                        "train_loss": float(match.group(2)),
                        "val_psnr": float(match.group(3)),
                        "best": float(match.group(4)),
                    }
        code = self.proc.wait()
        with self.lock:
            self.returncode = code
            self.finished_at = time.time()
            if self.status == "stopping":
                self.status = "stopped"
            else:
                self.status = "completed" if code == 0 else "failed"

    def _checkpoints(self) -> list[dict[str, Any]]:
        root = Path("runs")
        items: list[dict[str, Any]] = []
        for path in root.rglob("*.pth") if root.exists() else []:
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append({"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime})
        return sorted(items, key=lambda x: x["mtime"], reverse=True)[:20]

train_job = TrainJob()
