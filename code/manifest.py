"""Record the environment of a run so the experiment can be reproduced and reported."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ["openai", "anthropic", "numpy", "pandas", "scikit-learn", "sentence-transformers",
            "datasets", "torch", "transformers", "pyyaml", "vllm"]


def _gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_manifest(path, stage, cfg, config_path):
    """Write a JSON description of this run next to its outputs."""
    info = {
        "stage": stage,
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "declared_hardware": cfg.get("hardware", {}),
        "owned_hardware_assumptions": cfg.get("owned_hardware", {}),
        "detected": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "ram_gb": _ram_gb(),
            "gpus": _gpus(),
            "python": sys.version.split()[0],
        },
        "packages": {name: _version(name) for name in PACKAGES},
        "tiers": [{k: t.get(k) for k in ("key", "kind", "name", "revision", "price_in", "price_out")}
                  for t in cfg["tiers"]],
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
    }
    Path(path).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
