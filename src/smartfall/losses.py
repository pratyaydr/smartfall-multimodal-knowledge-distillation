"""Historical focal and KD objectives; no change to their mathematics.

Focal alpha is a global scalar, not a class-dependent alpha-balancing term.
The IT KL uses [-z,z], which differs from a Bernoulli [0,z] parameterization.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce
        return loss.mean()


def it_feature_kd_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    loss = 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1)
    return loss.mean()


def it_logit_kd_loss(student_logits, teacher_logits, temperature=2.0):
    student_logits = torch.stack([-student_logits, student_logits], dim=-1)
    teacher_logits = torch.stack([-teacher_logits, teacher_logits], dim=-1)
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(s, t, reduction="batchmean")
    return loss * temperature**2


def light_feature_kd_loss(student_feat, teacher_feat):
    if teacher_feat.dim() == 3:
        teacher_feat = teacher_feat.mean(dim=1)
    if student_feat.dim() == 3:
        student_feat = student_feat.mean(dim=1)
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    loss = 1 - F.cosine_similarity(student_feat, teacher_feat, dim=-1)
    return loss.mean()


def light_logit_kd_loss(student_logits, teacher_logits, temperature):
    student_probs = torch.sigmoid(student_logits / temperature)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    loss = F.mse_loss(student_probs, teacher_probs)
    return loss * temperature**2
