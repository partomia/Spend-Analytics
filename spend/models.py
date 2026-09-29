"""Where the two pretrained models come from and which device they run on.

model.device "" means CUDA when a GPU is visible, else CPU (both models are fast
enough on CPU, PLAN decision 7). model.local_dir points at a folder holding
both downloads for an air-gapped cluster: <local_dir>/all-MiniLM-L6-v2 and
<local_dir>/chronos-2.
"""

from __future__ import annotations

import os
from importlib import metadata

from spend.config import settings


def resolve_device(device: str | None = None) -> str:
    device = device if device is not None else settings()["model"]["device"]
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_path(model_id: str) -> str:
    local = settings()["model"]["local_dir"]
    if local:
        return os.path.join(local, model_id.split("/")[-1])
    return model_id


def package_versions(*packages: str) -> str:
    out = []
    for p in packages:
        try:
            out.append(f"{p}=={metadata.version(p)}")
        except metadata.PackageNotFoundError:
            continue
    return ",".join(out)
