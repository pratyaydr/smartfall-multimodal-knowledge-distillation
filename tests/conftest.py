import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def sample_arrays():
    rng = np.random.default_rng(7)
    n = 13
    labels = np.arange(n) % 2
    return {
        "accelerometer": rng.normal(size=(n, 64, 4)).astype("float32"),
        "skeleton": rng.normal(size=(n, 64, 32, 3)).astype("float32"),
        "labels": labels.astype("int32"),
        "subjects": np.array([29] * 4 + [38] * 4 + [56] * 4 + [99], dtype="int32"),
        "activities": np.where(labels == 1, 10, 1).astype("int32"),
    }
