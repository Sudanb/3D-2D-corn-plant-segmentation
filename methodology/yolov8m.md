# YOLOv8m-seg — Model Methodology

## Overview

YOLOv8m-seg is the medium-scale variant of Ultralytics YOLOv8, extended for instance segmentation. It is a single-stage, anchor-free detector that jointly predicts bounding boxes, class labels, and instance masks in a single forward pass.

## Architecture

```
Input (640×640×3)
    │
    ▼
CSPDarknet Backbone (P3/P4/P5 feature maps)
    │
    ▼
PANet Neck (multi-scale feature fusion)
    │
    ▼
Segment Head
  ├── Detection branch  → boxes + class scores
  └── Mask branch       → 32 prototype masks × coefficients → binary masks
```

| Component | Detail |
|-----------|--------|
| Backbone | CSPDarknet (C2f blocks) |
| Neck | PANet (bi-directional FPN) |
| Head | Decoupled Segment head |
| Mask representation | Linear combination of 32 prototype masks |
| Anchor type | Anchor-free (DFL regression) |

## Parameters

| Property | Value |
|----------|-------|
| Total parameters | 27.24 M |
| Trainable parameters | 27.24 M |
| GFLOPs | 104.7 |
| Layers | 192 |

## Loss Functions

| Loss | Target | Weight |
|------|--------|--------|
| Box loss (CIoU) | Bounding box regression | 7.5 |
| Class loss (BCE) | Object classification | 0.5 |
| DFL loss | Distribution focal (box precision) | 1.5 |
| Mask loss (BCE) | Per-pixel mask prediction | 1.0 |

## Training Configuration

| Setting | Value |
|---------|-------|
| Pretrained weights | YOLOv8m-seg (COCO) |
| Epochs | 10 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | Auto (AdamW warm-up → SGD) |
| Learning rate (lr0 / lrf) | 0.01 / 0.01 |
| Weight decay | 0.0005 |
| Warmup epochs | 3 |
| AMP | Enabled |
| Augmentation | Mosaic, HSV jitter, horizontal flip, random erase |
| Workers | 0 (Windows) |
| Classes | 2 and 17 (separate runs) |

## Key Characteristics

- **Speed**: Fastest inference among evaluated models; optimised for real-time use
- **Mask quality**: Masks are coarse (prototype-based) compared to pixel-level predictors
- **Strength**: Robust to scale variation via multi-scale PANet features
- **Weakness**: Box-first design means mask quality is bounded by localisation accuracy
