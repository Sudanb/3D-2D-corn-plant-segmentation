# Experimental Setup — General Methodology

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4500 Ada Generation (24 GB VRAM) |
| Platform | Windows 11 Pro (WSL2 for EOMT training) |
| CUDA | 12.x via WDDM (Windows) / WSL2 (EOMT) |

## Dataset — MaizeField3D

| Property | Value |
|----------|-------|
| Domain | Precision agriculture — field-grown maize |
| Task | Instance segmentation |
| Total images | 39,000 |
| Training images | 27,300 |
| Annotation format | COCO JSON (polygon) |
| Image format | PNG, RGB |
| Input resolution | 640 × 640 (resized/padded) |
| Splits | Train / Val / Test |

### Class Configurations

Two experimental class configurations were evaluated across all models:

| Config | Classes | Labels |
|--------|---------|--------|
| 2-class | 2 | stem, leaf |
| 17-class | 17 | stem, leaf1–leaf16 |

In the 17-class setting, each individual leaf (ordered by emergence) is treated as a distinct category. In the 2-class setting, all leaf instances are merged into a single *leaf* category.

## Annotation Details

- **Format**: COCO JSON with polygon segmentation per instance
- **Masks**: Binary per-instance masks decoded from polygons at runtime using `pycocotools`
- **Category IDs**: 1-indexed (1 = stem, 2–17 = leaf1–leaf16)
- **Crowd flags**: All `iscrowd = 0`
- **Instances per image**: Variable; higher-numbered leaves are rarer and more occluded

## Models Evaluated

| Model | Type | Framework | Classes |
|-------|------|-----------|---------|
| YOLOv8m-seg | Single-stage | Ultralytics | 2, 17 |
| YOLO26m-seg | Single-stage | Ultralytics | 2, 17 |
| EoMT (DINOv2-S) | Transformer query-based | PyTorch Lightning | 2, 17 |
| Mask R-CNN | Two-stage | Detectron2 / torchvision | 2, 17 |

## Training Configuration

| Setting | YOLO models | EoMT | Mask R-CNN |
|---------|-------------|------|------------|
| Epochs | 10 | 10 | 10 |
| Batch size | 16 | 16 | 16 |
| Image size | 640 × 640 | 640 × 640 | 640 × 640 |
| Optimizer | Auto (SGD/Adam) | AdamW | SGD |
| Pretrained | Yes (COCO) | Yes (DINOv2) | Yes (COCO) |
| Data augmentation | Mosaic, HSV, flip | Color jitter, scale [0.5–2.0] | Flip, scale |
| Early stopping patience | 20 (disabled at 10ep) | — | — |

## Evaluation Protocol

All models are evaluated on the held-out **test split** using COCO instance segmentation metrics:

| Metric | Description |
|--------|-------------|
| mAP50 | Mask IoU @ 0.50 |
| mAP50-95 | Mask IoU @ 0.50:0.05:0.95 |
| Recall | Max recall @ 100 detections |

Per-class mAP50, mAP50-95, and Recall are reported for all models to enable fine-grained comparison across leaf emergence order.

> **Note on comparability**: All models are evaluated with the same COCO mask mAP metric on the same test set. Training loss formulations differ by architecture design (see individual model files) but do not affect the evaluation metric.
