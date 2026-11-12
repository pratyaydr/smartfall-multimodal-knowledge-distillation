"""Validation-only threshold tuning and explicit binary fall metrics."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold: float) -> dict:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be nonempty and aligned")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Expected binary labels and finite probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError("Probabilities and threshold must lie in [0,1]")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "samples": int(len(labels)),
        "positive_class": "fall",
    }


def best_threshold(labels, probabilities, grid=(0.30, 0.81, 0.02)) -> float:
    """Maximize validation accuracy; first improving threshold wins ties."""
    scores = np.arange(*grid)
    if len(scores) == 0:
        raise ValueError("Empty threshold grid")
    best_acc, threshold = 0.0, 0.5
    for value in scores:
        accuracy = binary_metrics(labels, probabilities, float(value))["accuracy"]
        if accuracy > best_acc:
            best_acc, threshold = accuracy, float(value)
    return threshold


@torch.no_grad()
def collect_predictions(model, loader, device="cpu", criterion=None):
    model.eval()
    labels, probabilities, losses = [], [], []
    for data, target in loader:
        data = {key: value.to(device) for key, value in data.items()}
        target = target.float().to(device).view(-1)
        logits = model(data)[0].view(-1)
        if criterion is not None:
            losses.append(float(criterion(logits, target)))
        labels.extend(target.cpu().numpy())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    if not labels:
        raise ValueError("Cannot evaluate an empty loader")
    # Historical loss: average of batch means, rather than sample-weighted mean.
    return np.asarray(labels), np.asarray(probabilities), float(np.mean(losses)) if losses else None
