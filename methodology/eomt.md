# EoMT (Encoder-only Mask Transformer) — Model Methodology

## Overview

EoMT (CVPR 2025) is a transformer-based instance segmentation model that eliminates the conventional decoder by injecting learnable query tokens directly into the last four blocks of a frozen Vision Transformer (ViT) backbone. Queries interact with patch tokens through standard self-attention, producing mask and class predictions without a separate cross-attention decoder.

## Architecture

```
Input (640×640×3)
    │
    ▼
Patch Embedding (14×14 patches → 46×46 tokens at 640px)
    │
    ▼
DINOv2 ViT-S Backbone (frozen, 12 blocks)
    │         │
    │    Blocks 9–12: learnable query tokens (Q) injected
    │         │         alongside patch tokens
    │         ▼
    │    Self-attention (Q ↔ patch tokens)
    │
    ▼
Query outputs → Class head (linear, nc+1)
Patch outputs → Upscale (2× bilinear ×2) → Mask head (3-layer MLP)
    │
    ▼
Mask logits (Q × H × W) → Hungarian matching → Binary masks
```

| Component | Detail |
|-----------|--------|
| Backbone | DINOv2 ViT-S/14 with 4 register tokens (frozen) |
| Query injection | Last 4 ViT blocks (blocks 9–12) |
| Number of queries | 100 |
| Mask head | 3-layer MLP on query embeddings |
| Class head | Linear layer (num_classes + 1 background) |
| Upscaling | 2× bilinear upsampling × 2 stages |
| Assignment | Hungarian matching (Mask2Former matcher) |

## Parameters

| Property | Value |
|----------|-------|
| Backbone (ViT-S, frozen) | ~22 M |
| Trainable (queries + heads) | ~4 M |
| Total | ~26 M |
| Input patch size | 14 × 14 |
| Feature resolution | 40 × 40 (at 640px input) |

## Loss Functions

| Loss | Target | Coefficient |
|------|--------|-------------|
| Binary cross-entropy | Per-pixel mask | mask_coefficient |
| Dice loss | Mask shape / overlap | dice_coefficient |
| Cross-entropy | Class label (per query) | class_coefficient |

Losses are computed after Hungarian matching between predicted queries and ground-truth instances. Applied at each of the last 4 transformer blocks (auxiliary losses) and summed.

## Attention Mask Annealing

During training, a binary attention mask restricts query-to-patch attention to predicted foreground regions, progressively relaxed across 4 annealing windows:

| Epoch range | Annealing start step | Annealing end step |
|-------------|---------------------|-------------------|
| 1 → 2 | 1706 | 3412 |
| 2 → 3 | 3412 | 5118 |
| 3 → 4 | 5118 | 6824 |
| 4 → 5 | 6824 | 8530 |

(Based on 27,300 images / batch 16 ≈ 1,706 steps/epoch)

## Training Configuration

| Setting | Value |
|---------|-------|
| Pretrained backbone | DINOv2 ViT-S (ImageNet-22k) |
| Backbone frozen | Yes |
| Epochs | 10 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | AdamW |
| Augmentation | Color jitter, random scale [0.5–2.0], random crop/pad |
| Validation batches (during training) | 100 (limit to avoid OOM) |
| Framework | PyTorch Lightning + torch.compile |
| Logger | CSVLogger |
| Classes | 2 and 17 (separate runs) |

## Key Characteristics

- **Strength**: Directly optimised for mask quality via Dice loss; no box prediction step
- **Strength**: Frozen DINOv2 backbone provides strong semantic features with minimal fine-tuning
- **Weakness**: Slowest inference due to ViT quadratic attention; not real-time
- **Weakness**: torch.compile warmup adds ~30–50 batch delay at start of training
- **Occluded instances**: Hungarian matching with per-query masks is better suited for dense, overlapping instances than prototype-based approaches
