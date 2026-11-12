"""Portable six-experiment runner preserving notebook-specific objectives."""

import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import make_loaders
from .evaluate import best_threshold, binary_metrics, collect_predictions
from .losses import (
    FocalLoss,
    it_feature_kd_loss,
    it_logit_kd_loss,
    light_feature_kd_loss,
    light_logit_kd_loss,
)
from .models import build_model
from .utils import file_hash, load_checkpoint, seed_everything, write_json


def make_criterion(config, device):
    if config["loss"] == "focal":
        return FocalLoss(config["focal_alpha"], config["focal_gamma"])
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config["pos_weight"]], device=device))


def train(
    config,
    dataset,
    split_path,
    output,
    device="cpu",
    teacher_checkpoint=None,
    student_checkpoint=None,
):
    """Train, select by validation, then evaluate the selected checkpoint once.

    A fresh output directory prevents silent checkpoint collisions. This runner
    adds metadata and explicit seeds; archived numbers are not guaranteed.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if config["epochs"] < 1:
        raise ValueError("epochs must be positive")
    kd = config.get("distillation")
    if kd and not teacher_checkpoint:
        raise ValueError("KD requires --teacher-checkpoint")
    if kd and kd["requires_student_initialization"] and not student_checkpoint:
        raise ValueError("IT KD requires the pretrained --student-checkpoint")
    if student_checkpoint and not (kd and kd["requires_student_initialization"]):
        raise ValueError("Student initialization is supported only for historical IT KD")
    seed_everything(config["seed"])
    loaders, summary = make_loaders(dataset, split_path, config, device)
    model = build_model(config["model"], config["model_kwargs"]).to(device)
    if student_checkpoint:
        load_checkpoint(model, student_checkpoint)
    teacher = None
    if kd:
        teacher = build_model(kd["teacher_model"]).to(device)
        load_checkpoint(teacher, teacher_checkpoint)
        teacher.eval()
        teacher.requires_grad_(False)
    criterion = make_criterion(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    if config["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["max_learning_rate"],
            epochs=config["epochs"],
            steps_per_epoch=len(loaders["train"]),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "split_summary.json", summary)
    write_json(
        output / "provenance.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "device": str(device),
            "dataset_sha256": file_hash(dataset),
            "subject_split_sha256": file_hash(split_path),
            "teacher_sha256": file_hash(teacher_checkpoint) if teacher_checkpoint else None,
            "student_sha256": file_hash(student_checkpoint) if student_checkpoint else None,
        },
    )
    best_score, stale, history = float("-inf"), 0, []
    for epoch in range(config["epochs"]):
        model.train()
        totals = []
        components = []
        for data, labels in loaders["train"]:
            data = {key: value.to(device) for key, value in data.items()}
            labels = labels.float().to(device).view(-1)
            smooth = config["label_smoothing"]
            targets = labels * (1 - smooth) + smooth / 2
            aug = config.get("external_augmentation")
            if aug:
                acc = data["accelerometer"]
                noise = torch.randn_like(acc) * aug["noise"]
                scale = 1 + torch.randn(acc.size(0), 1, 1, device=device) * aug["scale"]
                data["accelerometer"] = acc * scale + noise
            optimizer.zero_grad()
            # Source KD evaluates teacher before student's internal augmentation.
            teacher_logits = teacher_feat = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits, teacher_feat = teacher(data)
                    teacher_logits = teacher_logits.view(-1)
            result = model(data)
            logits = result[0].view(-1)
            hard = criterion(logits, targets)
            loss = hard
            if teacher is not None:
                if kd["logit_loss"] == "symmetric_kl":
                    feature_loss = (
                        it_feature_kd_loss(result[1], teacher_feat)
                        if epoch >= kd["feature_start_epoch"]
                        else torch.zeros((), device=device)
                    )
                    logit_loss = it_logit_kd_loss(logits, teacher_logits, kd["temperature"])
                else:
                    feature_loss = light_feature_kd_loss(result[1], teacher_feat)
                    logit_loss = light_logit_kd_loss(logits, teacher_logits, kd["temperature"])
                weights = kd["weights"]
                loss = weights[0] * hard + weights[1] * feature_loss + weights[2] * logit_loss
                components.append(
                    [float(hard.detach()), float(feature_loss.detach()), float(logit_loss.detach())]
                )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if config["scheduler"] == "onecycle":
                scheduler.step()
            totals.append(float(loss.detach()))
        if config["scheduler"] != "onecycle":
            scheduler.step()
        y, probabilities, val_loss = collect_predictions(model, loaders["val"], device, criterion)
        threshold = best_threshold(y, probabilities, config["threshold_grid"])
        metrics = binary_metrics(y, probabilities, threshold)
        metrics["loss"] = val_loss
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(totals)), "validation": metrics}
        if components:
            record["train_hard_feature_logit_mean"] = np.mean(components, axis=0).tolist()
        history.append(record)
        score = metrics[config["selection_metric"]]
        if score > best_score:
            best_score, stale = score, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "threshold": threshold,
                    "epoch": epoch + 1,
                    "validation": metrics,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch + 1}: train_loss={record['train_loss']:.4f}, val_accuracy={metrics['accuracy']:.4f}, val_f1={metrics['f1']:.4f}"
        )
        if stale >= config["patience"]:
            break
    checkpoint = load_checkpoint(model, output / "best.pt")
    y, probabilities, test_loss = collect_predictions(model, loaders["test"], device, criterion)
    test = binary_metrics(y, probabilities, checkpoint["threshold"])
    test["loss"] = test_loss
    write_json(output / "test_metrics.json", test)
    np.savez_compressed(output / "test_predictions.npz", labels=y, probabilities=probabilities)
    return test
