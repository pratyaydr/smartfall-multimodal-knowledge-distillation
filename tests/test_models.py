import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smartfall.losses import (
    FocalLoss,
    it_feature_kd_loss,
    it_logit_kd_loss,
    light_feature_kd_loss,
    light_logit_kd_loss,
)
from smartfall.models import build_model

ROOT = Path(__file__).resolve().parents[1]


def source_model(cell, name):
    notebook = json.loads((ROOT / "notebooks/original_kaggle.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][cell]["source"]))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(compile(ast.Module(body=classes, type_ignores=[]), "<source models>", "exec"), ns)
    return ns[name]()


@pytest.mark.parametrize(
    "key,cell,name,outputs",
    [
        ("it_student", 8, "FinalITStudent", 3),
        ("teacher", 10, "FinalKDTeacher", 2),
        ("lighthart", 21, "LightHART", 3),
        ("shared_teacher", 22, "SharedCNNTeacher", 2),
    ],
)
def test_models_match_original_forward_with_identical_weights(key, cell, name, outputs):
    torch.manual_seed(42)
    original = source_model(cell, name).eval()
    rebuilt = build_model(key).eval()
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    data = {"accelerometer": torch.randn(2, 64, 4), "skeleton": torch.randn(2, 64, 32, 3)}
    with torch.no_grad():
        expected = original(data)
        actual = rebuilt(data)
    assert len(actual) == outputs and actual[0].shape == (2, 1) and actual[1].shape == (2, 128)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sum(p.numel() for p in original.parameters()) == sum(
        p.numel() for p in rebuilt.parameters()
    )


@pytest.mark.parametrize("key", ["it_student", "lighthart"])
def test_student_requires_only_accelerometer(key):
    model = build_model(key).eval()
    acc = torch.randn(2, 64, 4)
    with torch.no_grad():
        a = model({"accelerometer": acc})[0]
        b = model({"accelerometer": acc, "skeleton": torch.randn(2, 64, 32, 3)})[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "feature,logit",
    [(it_feature_kd_loss, it_logit_kd_loss), (light_feature_kd_loss, light_logit_kd_loss)],
)
def test_kd_objective_is_finite_differentiable_and_zero_for_equal_inputs(feature, logit):
    s = torch.randn(4, 128, requires_grad=True)
    t = torch.randn(4, 128)
    z = torch.randn(4, requires_grad=True)
    tz = torch.randn(4)
    loss = feature(s, t) + logit(z, tz, 2.0)
    loss.backward()
    assert torch.isfinite(loss) and s.grad.abs().sum() > 0 and z.grad.abs().sum() > 0
    torch.testing.assert_close(feature(t, t), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(logit(tz, tz, 2.0), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_focal_alpha_is_a_global_scale():
    logits = torch.tensor([-1.0, 1.0])
    labels = torch.tensor([0.0, 1.0])
    a = FocalLoss(alpha=0.8)(logits, labels)
    b = FocalLoss(alpha=0.4)(logits, labels)
    np.testing.assert_allclose(a.numpy(), 2 * b.numpy())


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("made_up")
