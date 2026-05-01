# Evaluation Protocol

## Overview

All models are evaluated on the held-out **test split** (59 plants × 75 views = 4,425 images) using a unified evaluation pipeline. No test images are seen during training or hyperparameter selection. Results are reported for two class configurations independently: 2-class (stem, leaf) and 17-class (stem, leaf1–leaf16).

## Test Split

| Property | Value |
|----------|-------|
| Plants | 59 (held out at plant level — no image-level leakage) |
| Images | 4,425 (59 × 75 views) |
| Split seed | 3 |
| Annotation format | COCO JSON polygon segmentation |

## Accuracy Metrics

All accuracy metrics use **COCO mask IoU** (instance segmentation, not bounding box). Evaluated using `pycocotools.COCOeval` for Mask R-CNN and YOLO; `torchmetrics.MeanAveragePrecision` for EoMT.

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **mAP50** | Mean Average Precision at mask IoU = 0.50 | IoU ≥ 0.50 |
| **mAP50-95** | Mean AP averaged over IoU thresholds 0.50:0.05:0.95 (10 thresholds) | IoU 0.50–0.95 |
| **Recall** | Maximum recall at up to 100 detections per image | — |

Reported at two granularities:
- **Overall** — averaged across all classes
- **Per-class** — separately for stem and each leaf class (leaf1–leaf16 in 17-class config)

Per-class recall for Mask R-CNN and YOLO is extracted from `COCOeval.eval["recall"]` (shape `[T, K, A, M]`, area=all, maxDet=100, averaged over IoU thresholds). For EoMT it is `mar_100_per_class` from `torchmetrics`.

## Efficiency Metrics

| Metric | Description | How Measured |
|--------|-------------|--------------|
| **FPS** | Inference throughput (images per second) | Wall-clock time over full test set ÷ num_images (EoMT, Mask R-CNN); `metrics.speed["inference"]` (YOLO) |
| **ms/image** | Mean per-image inference latency | 1000 / FPS |
| **Parameters (M)** | Total trainable + frozen model parameters | Reported from architecture specs |
| **GFLOPs** | Floating point operations per forward pass at 640×640 | Reported from architecture specs |

> **Note on FPS measurement**: YOLO FPS is measured by Ultralytics' internal timer (inference step only, excluding pre/postprocessing). EoMT and Mask R-CNN FPS is measured as wall-clock time across the full test set including pre/postprocessing on GPU, so values are not directly comparable across frameworks. They are indicative of relative throughput only.

## Model Parameter and Complexity Summary

| Model | Parameters (M) | GFLOPs |
|-------|---------------|--------|
| YOLOv8m-seg | 27.2 | 104.7 |
| YOLO11m-seg (26m) | 26.4 | ~100 |
| EoMT (DINOv2 ViT-S) | 26 (22 frozen + 4 trainable) | — |
| Mask R-CNN (ResNet-50) | 44 | — |

## Evaluation Scripts

| Script | Models | Output |
|--------|--------|--------|
| `eval_yolo.py` | YOLOv8m-seg, YOLO11m-seg | CSV + JSON per run |
| `MaskRCNN/eval_maskrcnn.py` | Mask R-CNN 2-cls, 17-cls | CSV + JSON per run |
| `EOMT/eomt/eval_eomt.py` | EoMT ViT-S 2-cls, 17-cls | CSV + JSON per run |

All scripts export the same column schema for direct comparison:

```
class, mAP50, mAP50-95, Recall
```

JSON outputs additionally include `inference_fps`, `inference_ms_per_image`, and `inference_total_s`.

## Hardware

All evaluations run on:

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4500 Ada Generation (24 GB VRAM) |
| Platform | Windows 11 Pro |
| CUDA | 12.x |
| Batch size (eval) | 4 (Mask R-CNN), 8 (EoMT), 16 (YOLO default) |

## Comparability Notes

- All models evaluated on the **same test set** with the **same COCO mask IoU metric**
- Training loss formulations differ by architecture (BCE/Dice/CE for EoMT; box+cls+mask+DFL for YOLO; 5-term RPN+RoI+mask for Mask R-CNN) but do not affect the evaluation metric
- All models trained for **10 epochs** with **batch size 16** for a fair training budget comparison
- Pretrained weights used for all models (COCO for YOLO and Mask R-CNN; ImageNet-22k DINOv2 for EoMT)
- Data augmentation varies by framework (see individual model methodology files); this is an inherent architectural difference, not a controlled variable
