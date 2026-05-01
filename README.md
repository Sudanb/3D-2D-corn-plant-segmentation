# MaizeField3D — Instance Segmentation Benchmark

Comparative study of four instance segmentation architectures on synthetically projected TLS point clouds of field-grown maize plants.

## Models

| Model | Type | Classes | Script |
|-------|------|---------|--------|
| YOLOv8m-seg | Single-stage CNN | 2, 17 | `train_yolo.py` |
| YOLO11m-seg (26m) | Single-stage CNN + attention | 2, 17 | `train_yolo.py` |
| Mask R-CNN | Two-stage CNN | 2, 17 | `MaskRCNN/train_maskrcnn.py` |
| EoMT (DINOv2-S) | Encoder-only Transformer | 2, 17 | `EOMT/eomt/main.py` |

## Repository Structure

```
CV_datasets/
├── train_yolo.py              # YOLO training (all 4 runs)
├── eval_yolo.py               # YOLO evaluation → CSV + JSON
├── dataset_build.py           # PLY → projected image pipeline
├── training_format.py         # Convert images to COCO/YOLO format
├── MaskRCNN/
│   ├── train_maskrcnn.py
│   └── eval_maskrcnn.py
├── EOMT/eomt/
│   ├── main.py                # EoMT training entry point
│   ├── eval_eomt.py           # EoMT evaluation
│   └── configs/dinov2/maize/instance/
│       ├── eomt_small_640.yaml
│       └── eomt_base_640.yaml
└── methodology/               # Methodology documentation
    ├── general.md
    ├── data_preparation.md
    ├── decisions.md
    ├── evaluation_protocol.md
    ├── yolov8m.md
    ├── yolo26m.md
    ├── maskrcnn.md
    └── eomt.md
```

## Environment Setup

| Model | Platform | Environment |
|-------|----------|-------------|
| YOLO | Windows (PowerShell) | pip — `ultralytics` |
| Mask R-CNN | Windows (PowerShell) | pip — `torchvision`, `pycocotools` |
| EoMT | WSL2 | conda `eomt` (see `EOMT/eomt/README.md`) |

## Hardware

- GPU: NVIDIA RTX 4500 Ada Generation (24 GB VRAM)
- Platform: Windows 11 Pro + WSL2
- CUDA: 12.x

## Dataset

Not included in this repository (large binary files). The pipeline to regenerate projected images from TLS point clouds is in `dataset_build.py`. See `methodology/data_preparation.md` for full details.

- 393 field-grown maize plants
- 75 projected views per plant → 29,475 total images
- Split: 275 train / 59 val / 59 test (plant-level, seed=3)
