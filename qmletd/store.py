"""Result persistence: one JSON (metadata + metrics) and one NPZ (scores) per run."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config


def run_name(model: str, d, M, split, tag: str = "") -> str:
    m = "full" if M is None else str(M)
    base = f"{model}__d{d}__M{m}__s{split}"
    return f"{base}__{tag}" if tag else base


def exists(experiment: str, name: str) -> bool:
    return (config.RESULTS / experiment / f"{name}.json").exists()


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=config.ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "processor": platform.processor(), "git": _git_rev()}
    for mod in ("numpy", "scipy", "sklearn", "imblearn", "torch", "qiskit", "qiskit_aer",
                "lightgbm", "dimod", "dwave.samplers"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            env[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save(experiment: str, name: str, meta: dict, metrics: dict | None = None, **arrays) -> Path:
    out_dir = config.RESULTS / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "experiment": experiment, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               **meta, "metrics": metrics, "env": environment()}
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
    if arrays:
        np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    return out_dir / f"{name}.json"
