"""
reg_model.py
============
All model definitions for plant measurement regression.

Available models:
  EfficientNetReg  — EfficientNet-B3 backbone (pretrained ImageNet)
  ResNet34Reg      — ResNet-34 backbone (pretrained ImageNet)
  CustomCNNReg     — Lightweight CNN trained from scratch

All share the same MLP head and output shape [B, 32]:
  [0]      stem_length_m
  [1..15]  internode_lengths_m  (zero-padded)
  [16..31] leaf_lengths_m       (zero-padded)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    resnet34,        ResNet34_Weights,
)

from reg_dataset import N_TARGETS   # 32


# =============================================================================
# Shared MLP head
# =============================================================================

def _make_head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(128, N_TARGETS),
        nn.ReLU(inplace=True),   # lengths are non-negative
    )


# =============================================================================
# Model 1: EfficientNet-B3 (baseline — pretrained)
# =============================================================================

class EfficientNetReg(nn.Module):
    def __init__(self, freeze_encoder: bool = False, dropout: float = 0.3):
        super().__init__()
        backbone     = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.encoder = backbone.features          # [B, 1536, 20, 20] @ 640px
        self.pool    = nn.AdaptiveAvgPool2d(1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = _make_head(1536, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(x)).flatten(1))


# =============================================================================
# Model 2: ResNet-34 (pretrained)
# =============================================================================

class ResNet34Reg(nn.Module):
    def __init__(self, freeze_encoder: bool = False, dropout: float = 0.3):
        super().__init__()
        backbone  = resnet34(weights=ResNet34_Weights.DEFAULT)
        # Drop avgpool + fc — keep conv layers only
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B, 512, 20, 20]
        self.pool    = nn.AdaptiveAvgPool2d(1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = _make_head(512, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(x)).flatten(1))


# =============================================================================
# Model 3: Custom CNN (trained from scratch)
# =============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.downsample(x)


class CustomCNNReg(nn.Module):
    """
    5-stage residual CNN trained from scratch.
    ~6M parameters — lighter than the pretrained models.
    Input: [B, 3, 640, 640] → feature map [B, 512] → head → [B, 32]
    """
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            # stem: 640 → 320
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),   # → 160
            # stage 1: 160 → 160
            _ConvBlock(32,  64),
            _ConvBlock(64,  64),
            # stage 2: 160 → 80
            _ConvBlock(64,  128, stride=2),
            _ConvBlock(128, 128),
            # stage 3: 80 → 40
            _ConvBlock(128, 256, stride=2),
            _ConvBlock(256, 256),
            # stage 4: 40 → 20
            _ConvBlock(256, 512, stride=2),
            _ConvBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = _make_head(512, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.encoder(x)).flatten(1))


# =============================================================================
# Masked MSE loss  (shared by all training scripts)
# =============================================================================

def masked_mse(pred: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    MSE over valid (non-padded) positions only.
    pred, target, mask: [B, 32]  — mask is bool.
    """
    diff = (pred - target) ** 2
    return (diff * mask.float()).sum() / mask.float().sum().clamp(min=1)


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    x = torch.randn(2, 3, 640, 640)
    for Model, name in [
        (EfficientNetReg, "EfficientNet-B3"),
        (ResNet34Reg,     "ResNet-34     "),
        (CustomCNNReg,    "Custom CNN    "),
    ]:
        m      = Model()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        out    = m(x)
        print(f"{name}  params={params:.1f}M  out={tuple(out.shape)}  "
              f"min={out.min():.3f}  max={out.max():.3f}")
