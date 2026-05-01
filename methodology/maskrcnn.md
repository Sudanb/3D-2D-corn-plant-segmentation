# Mask R-CNN — Model Methodology

## Overview

Mask R-CNN (He et al., ICCV 2017) is a two-stage instance segmentation framework that extends Faster R-CNN by adding a parallel mask prediction branch on top of region-of-interest (RoI) features. It is a widely used baseline in instance segmentation benchmarks and provides a strong reference point between single-stage YOLO models and transformer-based EoMT.

## Architecture

```
Input (variable size, resized to fit FPN)
    │
    ▼
ResNet-50 Backbone (C2–C5 feature maps)
    │
    ▼
Feature Pyramid Network (FPN, P2–P6)
    │
    ▼
Region Proposal Network (RPN)
    │ → ~1000 region proposals
    ▼
RoIAlign (7×7 for box/class, 14×14 for mask)
    │
    ├── Box head (2-layer FC) → class scores + box deltas
    └── Mask head (4× Conv + deconv) → 28×28 binary mask per class
```

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-50 |
| Neck | FPN (P2–P6, 256 channels) |
| Region proposals | RPN (anchor-based) |
| RoI pooling | RoIAlign |
| Box head | 2-layer FC (1024-d) |
| Mask head | 4× Conv 3×3 (256-d) + deconv → sigmoid |
| Mask output | Per-class 28×28 binary mask |

## Parameters

| Property | Value |
|----------|-------|
| Backbone (ResNet-50) | ~23.5 M |
| FPN + heads | ~20.5 M |
| Total | ~44 M |
| Pretrained | COCO instance segmentation |

## Loss Functions

| Loss | Target |
|------|--------|
| RPN classification (BCE) | Foreground/background proposals |
| RPN regression (Smooth L1) | Anchor box refinement |
| RoI classification (CE) | Per-RoI class prediction |
| RoI box regression (Smooth L1) | Per-RoI box refinement |
| Mask loss (BCE per class) | 28×28 binary mask per predicted class |

Total loss is the sum of all five terms weighted equally.

## Key Differences from Other Models

| Property | Mask R-CNN | YOLO models | EoMT |
|----------|-----------|-------------|------|
| Stages | 2 (propose → refine) | 1 | 1 (query-based) |
| Mask source | RoI-cropped features | Prototype blend | Direct query output |
| Mask resolution | 28×28 (upsampled) | ~160×160 prototype | Full resolution |
| Assignment | IoU-threshold matching | IoU-threshold NMS | Hungarian matching |
| Backbone | CNN (ResNet) | CNN (CSP) | ViT (DINOv2) |

## Key Characteristics

- **Strength**: High-quality masks from dedicated RoI-aligned mask branch
- **Strength**: Two-stage refinement produces accurate box localisation before masking
- **Strength**: Well-established baseline — widely cited in literature
- **Weakness**: Slower than single-stage models; RPN adds overhead
- **Weakness**: Fixed 28×28 mask resolution requires upsampling for fine boundaries
- **Weakness**: Anchor design requires tuning for non-standard object shapes (elongated leaves)

## References

He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV 2017*.
