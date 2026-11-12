"""Notebook cell 21: preserved model architecture and forward behavior."""

import torch
from torch import nn
from torch.nn import functional as F

from .it import DropPath


class ResidualDWBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.08, mlp_ratio=2, drop_path=0.08):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = embed_dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.drop_path(attn_out)
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.drop_path(ffn_out)
        return x


class LightHART(nn.Module):
    def __init__(
        self, acc_frames=64, acc_coords=4, embed_dim=128, num_heads=8, num_layers=5, dropout=0.08
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv_stem = nn.Sequential(
            nn.Conv1d(acc_coords, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualDWBlock(64),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            ResidualDWBlock(embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, acc_frames, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, 0.08, num_layers)]
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, drop_path=dpr[i]
                )
                for i in range(num_layers)
            ]
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        self.pool_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.kd_projector = nn.Sequential(nn.Linear(embed_dim, 128), nn.LayerNorm(128))
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, data):
        x = data["accelerometer"]
        if self.training:
            noise = torch.randn_like(x) * 0.015
            scale = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * 0.05
            x = x * scale + noise
        x = x.permute(0, 2, 1)
        x = self.conv_stem(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.temporal_norm(x)
        temp = x.permute(0, 2, 1)
        temp = self.temporal_refine(temp)
        temp = temp.permute(0, 2, 1)
        x = x + temp
        features = x
        weights = self.pool_attn(x)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)
        kd_features = self.kd_projector(pooled)
        kd_features = F.normalize(kd_features, dim=-1)
        logits = self.classifier(pooled)
        return (logits, kd_features, features)
