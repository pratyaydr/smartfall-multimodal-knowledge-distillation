import numpy as np
import pytest
import torch

from smartfall.cli import checkpoint_model
from smartfall.evaluate import best_threshold, binary_metrics
from smartfall.models import build_model


def test_fall_metrics_match_hand_computed_confusion_matrix():
    y = [0, 0, 1, 1]
    p = [0.1, 0.8, 0.6, 0.4]
    m = binary_metrics(y, p, 0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 0.5


def test_threshold_uses_validation_accuracy_and_first_tie():
    assert best_threshold([0, 1], [0.1, 0.9]) == pytest.approx(0.30)
    assert best_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.30)


def test_single_class_auc_is_explicitly_unavailable():
    assert binary_metrics([0, 0], [0.1, 0.2], 0.5)["auc"] is None


@pytest.mark.parametrize("p", [[np.nan], [1.1], [-0.1]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        binary_metrics([0], p, 0.5)


def test_archived_weights_cannot_silently_default_to_threshold(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save(build_model("it_student").state_dict(), path)
    with pytest.raises(ValueError, match="require --config"):
        checkpoint_model(path)
