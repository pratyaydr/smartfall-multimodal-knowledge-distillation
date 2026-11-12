"""Notebook cell 22: preserved model architecture and forward behavior."""

import torch
from torch import nn
from torch.nn import functional as F

from .it import DropPath
from .lighthart import ResidualDWBlock
from .teacher import MultiScaleSpatialConv


class ITBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
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


class CrossModalFusion(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.attn_si = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_is = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, skel, imu):
        s = self.norm_s(skel)
        i = self.norm_i(imu)
        s_attn, _ = self.attn_si(s, i, i)
        i_attn, _ = self.attn_is(i, s, s)
        s_out = skel + s_attn
        i_out = imu + i_attn
        fused = s_out + i_out
        return self.out_proj(fused)


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
        return (x * weights).sum(dim=1)


class SharedCNNTeacher(nn.Module):
    def __init__(
        self,
        frames=64,
        joints=32,
        acc_channels=4,
        embed_dim=128,
        heads=8,
        skel_depth=3,
        imu_depth=5,
        temp_depth=2,
        dropout=0.1,
        drop_path=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.imu_conv_stem = nn.Sequential(
            nn.Conv1d(acc_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualDWBlock(64),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            ResidualDWBlock(embed_dim),
        )
        self.spatial = nn.Sequential(
            MultiScaleSpatialConv(3, embed_dim // 2),
            MultiScaleSpatialConv(embed_dim // 2, embed_dim),
        )
        self.skel_embed = nn.Sequential(
            nn.Linear(joints * embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.pos_skel = nn.Parameter(torch.zeros(1, frames, embed_dim))
        self.pos_imu = nn.Parameter(torch.zeros(1, frames, embed_dim))
        self.pos_temp = nn.Parameter(torch.zeros(1, frames, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.skel_blocks = nn.ModuleList(
            [
                ITBlock(embed_dim, heads, dropout=dropout, drop_path=drop_path * 0.5)
                for _ in range(skel_depth)
            ]
        )
        self.skel_norm = nn.LayerNorm(embed_dim)
        self.imu_blocks = nn.ModuleList(
            [
                ITBlock(embed_dim, heads, dropout=dropout, drop_path=drop_path * 0.5)
                for _ in range(imu_depth)
            ]
        )
        self.imu_norm = nn.LayerNorm(embed_dim)
        self.fusion = CrossModalFusion(embed_dim, heads, dropout)
        self.temp_blocks = nn.ModuleList(
            [
                ITBlock(embed_dim, heads, dropout=dropout, drop_path=drop_path)
                for _ in range(temp_depth)
            ]
        )
        self.temp_norm = nn.LayerNorm(embed_dim)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        self.pool = TemporalAttentionPooling(embed_dim)
        self.kd_projector = nn.Sequential(nn.Linear(embed_dim, 128), nn.LayerNorm(128))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, data):
        acc = data["accelerometer"]
        skl = data["skeleton"]
        if skl.ndim == 4 and skl.shape[-1] == 3:
            skl = skl.permute(0, 3, 1, 2)
        B, _, T, J = skl.shape
        sf = self.spatial(skl)
        sf = sf.permute(0, 2, 3, 1)
        sf = sf.reshape(B, T, J * self.embed_dim)
        sf = self.skel_embed(sf)
        sf = sf + self.pos_skel[:, :T]
        sf = self.pos_drop(sf)
        for blk in self.skel_blocks:
            sf = blk(sf)
        sf = self.skel_norm(sf)
        imf = acc.permute(0, 2, 1)
        imf = self.imu_conv_stem(imf)
        imf = imf.permute(0, 2, 1)
        imf = imf + self.pos_imu[:, :T]
        imf = self.pos_drop(imf)
        for blk in self.imu_blocks:
            imf = blk(imf)
        imf = self.imu_norm(imf)
        fused = self.fusion(sf, imf)
        x = fused + self.pos_temp[:, :T]
        x = self.pos_drop(x)
        for blk in self.temp_blocks:
            x = blk(x)
        x = self.temp_norm(x)
        temp = x.permute(0, 2, 1)
        temp = self.temporal_refine(temp)
        temp = temp.permute(0, 2, 1)
        x = x + temp
        pooled = self.pool(x)
        kd_features = self.kd_projector(pooled)
        kd_features = F.normalize(kd_features, dim=-1)
        logits = self.head(pooled)
        return (logits, kd_features)
