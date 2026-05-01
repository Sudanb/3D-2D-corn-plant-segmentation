# YOLO26m-seg — Model Methodology

## Overview

YOLO26m-seg is a single-stage instance segmentation model from the Ultralytics YOLO11 family, with approximately 26 million parameters (medium scale). It follows the same anchor-free, single-pass paradigm as YOLOv8 but incorporates architectural improvements in the C3k2 backbone blocks and attention-augmented feature fusion, improving accuracy with comparable compute.

## Architecture

```
Input (640×640×3)
    │
    ▼
C3k2 Backbone (CSP variant with improved gradient flow)
    │
    ▼
C2PSA Neck (cross-stage partial + positional self-attention)
    │
    ▼
Segment Head
  ├── Detection branch  → boxes + class scores (DFL)
  └── Mask branch       → prototype masks × coefficients → binary masks
```

| Component | Detail |
|-----------|--------|
| Backbone | C3k2 (improved CSP Darknet) |
| Neck | C2PSA (attention-enhanced FPN) |
| Head | Decoupled Segment head |
| Mask representation | Linear combination of prototype masks |
| Anchor type | Anchor-free (DFL regression) |

## Parameters

| Property | Value |
|----------|-------|
| Total parameters | ~26.4 M |
| Trainable parameters | ~26.4 M |
| GFLOPs | ~110 |
| Model family | YOLO11-seg (medium) |

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
| Pretrained weights | YOLO26m-seg (COCO) |
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

- **vs YOLOv8m**: Similar parameter count but attention in the neck improves feature aggregation for overlapping instances
- **Strength**: Better handling of occluded instances than YOLOv8m due to attention mechanism
- **Weakness**: Same prototype-mask limitation as YOLOv8; mask boundaries remain approximate
- **Speed**: Comparable to YOLOv8m; marginally slower due to attention in neck
