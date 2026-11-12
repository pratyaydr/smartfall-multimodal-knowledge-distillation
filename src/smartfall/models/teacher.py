"""Notebook cell 10: preserved model architecture and forward behavior."""

import torch
from torch import nn
from torch.nn import functional as F

from .it import ITBlock, TemporalAttentionPooling


class SE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x):
        w = x.mean(dim=(-2, -1))
        w = self.fc(w)
        return x * w.unsqueeze(-1).unsqueeze(-1)


class MultiScaleSpatialConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 3
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.GELU())
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, (1, 3), padding=(0, 1)), nn.BatchNorm2d(mid), nn.GELU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch - 2 * mid, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_ch - 2 * mid),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.GELU())
        self.se = SE(out_ch)
        self.residual = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        x = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        x = self.fuse(x)
        x = self.se(x)
        return F.gelu(x + residual)


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
        if i_out.size(1) != s_out.size(1):
            i_out = F.interpolate(
                i_out.transpose(1, 2), size=s_out.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        fused = s_out + i_out
        return self.out_proj(fused)


class FinalKDTeacher(nn.Module):
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
        drop_path=0.15,
    ):
        super().__init__()
        self.frames = frames
        self.embed_dim = embed_dim
        self.imu_tokens = frames // 4
        self.spatial = nn.Sequential(
            MultiScaleSpatialConv(3, embed_dim // 2),
            MultiScaleSpatialConv(embed_dim // 2, embed_dim),
        )
        self.skel_embed = nn.Sequential(
            nn.Linear(joints * embed_dim, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.imu_conv_stem = nn.Sequential(
            nn.Conv1d(acc_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.imu_patch_embed = nn.Sequential(nn.Linear(4 * 128, embed_dim), nn.LayerNorm(embed_dim))
        self.pos_skel = nn.Parameter(torch.zeros(1, frames, embed_dim))
        self.pos_imu = nn.Parameter(torch.zeros(1, self.imu_tokens, embed_dim))
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
        self.pool = TemporalAttentionPooling(embed_dim)
        self.feature_norm = nn.LayerNorm(embed_dim)
        self.feature_dropout = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.pos_skel, std=0.02)
        nn.init.trunc_normal_(self.pos_imu, std=0.02)
        nn.init.trunc_normal_(self.pos_temp, std=0.02)

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
        B, T2, C = imf.shape
        imf = imf.reshape(B, T2 // 4, 4 * C)
        imf = self.imu_patch_embed(imf)
        imf = imf + self.pos_imu
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
        features = self.pool(x)
        features = self.feature_norm(features)
        features = self.feature_dropout(features)
        features = F.normalize(features, dim=-1)
        logits = self.head(features)
        return (logits, features)
