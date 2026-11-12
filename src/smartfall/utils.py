"""Explicit seeds, JSON output, and metadata-bearing checkpoints."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(model, path):
    """Use restricted weights-only loading; accept archived bare state dicts."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("model_state", payload), strict=True)
    return payload
