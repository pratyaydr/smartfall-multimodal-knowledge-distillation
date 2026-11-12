"""Notebook cell 8: preserved model architecture and forward behavior."""

import torch
from torch import nn
from torch.nn import functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class ITBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = residual + self.drop_path(x)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)
        return x


class FinalITStudent(nn.Module):
    def __init__(
        self, frames=64, channels=4, embed_dim=128, depth=5, heads=8, dropout=0.1, drop_path=0.15
    ):
        super().__init__()
        self.frames = frames
        self.embed_dim = embed_dim
        self.conv_stem = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.patch_embed = nn.Sequential(nn.Linear(4 * 128, embed_dim), nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_tokens = frames // 4 + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                ITBlock(dim=embed_dim, heads=heads, dropout=dropout, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = TemporalAttentionPooling(embed_dim)
        self.feature_head = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Dropout(0.2))
        self.kd_projector = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def create_patches(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T // 4, 4 * C)
        x = self.patch_embed(x)
        return x

    def forward(self, data):
        x = data["accelerometer"]
        B = x.shape[0]
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        x = x.permute(0, 2, 1)
        x = self.conv_stem(x)
        x = x.permute(0, 2, 1)
        x = self.create_patches(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        pooled = self.pool(x[:, 1:])
        raw_features = torch.cat([cls_token, pooled], dim=1)
        raw_features = self.feature_head(raw_features)
        kd_features = self.kd_projector(raw_features)
        kd_features = F.normalize(kd_features, dim=-1)
        logits = self.classifier(raw_features)
        return (logits, kd_features, raw_features)
