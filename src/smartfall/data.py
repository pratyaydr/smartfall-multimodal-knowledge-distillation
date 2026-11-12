"""Validated five-array data schema and explicit subject partitions."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

KEYS = ("accelerometer", "skeleton", "labels", "subjects", "activities")


def load_arrays(path: str | Path) -> dict:
    """Load numeric arrays without enabling pickle deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        data = {key: archive[key] for key in KEYS}
    validate_arrays(data)
    return data


def validate_arrays(data: dict) -> None:
    """Enforce aligned fixed-shape tensors and binary labels."""
    n = len(data["labels"])
    if n == 0:
        raise ValueError("Dataset is empty")
    expected = {
        "accelerometer": (n, 64, 4),
        "skeleton": (n, 64, 32, 3),
        "labels": (n,),
        "subjects": (n,),
        "activities": (n,),
    }
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {data[key].shape}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if not np.isin(data["labels"], [0, 1]).all():
        raise ValueError("labels must be 0 (ADL) or 1 (fall)")
    for key in ("subjects", "activities"):
        if not np.equal(data[key], np.floor(data[key])).all():
            raise ValueError(f"{key} must be integer-valued")
    if not np.array_equal(data["labels"], (data["activities"] > 9).astype(int)):
        raise ValueError("labels disagree with the notebook activity > 9 mapping")


def split_arrays(data: dict, split: dict) -> tuple[dict, dict]:
    """Reject subject overlap and report samples excluded by the protocol."""
    groups = {name: set(split[name]) for name in ("train", "val", "test")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if groups[a] & groups[b]:
            raise ValueError(f"Subject overlap between {a} and {b}")
    result, summary = {}, {}
    for name, ids in groups.items():
        mask = np.isin(data["subjects"], list(ids))
        result[name] = {key: values[mask] for key, values in data.items()}
        if not mask.any():
            raise ValueError(f"{name} split contains no samples")
        summary[name] = {
            "samples": int(mask.sum()),
            "falls": int(data["labels"][mask].sum()),
            "subjects": sorted(map(int, np.unique(data["subjects"][mask]))),
            "missing_configured_subjects": sorted(ids - set(data["subjects"][mask])),
        }
    assigned = set.union(*groups.values())
    excluded = ~np.isin(data["subjects"], list(assigned))
    summary["excluded"] = {
        "samples": int(excluded.sum()),
        "subjects": sorted(map(int, np.unique(data["subjects"][excluded]))),
    }
    return result, summary


class SmartFallDataset(Dataset):
    """Notebook-compatible (modalities, binary target) batches."""

    def __init__(self, arrays: dict):
        self.acc = torch.as_tensor(arrays["accelerometer"], dtype=torch.float32)
        self.skl = torch.as_tensor(arrays["skeleton"], dtype=torch.float32)
        self.labels = torch.as_tensor(arrays["labels"], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"accelerometer": self.acc[index], "skeleton": self.skl[index]}, self.labels[
            index
        ].unsqueeze(0)


def make_loaders(path, split_path, config, device="cpu"):
    arrays = load_arrays(path)
    split = json.loads(Path(split_path).read_text())
    parts, summary = split_arrays(arrays, split)
    loaders = {}
    for name, part in parts.items():
        loaders[name] = DataLoader(
            SmartFallDataset(part),
            batch_size=config["batch_size"],
            shuffle=name == "train",
            num_workers=0,
            pin_memory=str(device).startswith("cuda"),
            drop_last=name == "train" and config["drop_last"],
        )
        if len(loaders[name]) == 0:
            raise ValueError(f"{name} has no batches; lower batch_size or disable drop_last")
    return loaders, summary
