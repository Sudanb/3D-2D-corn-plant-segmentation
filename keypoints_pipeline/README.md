# Keypoints Pipeline — Plant Morphology Regression

Predicts maize plant measurements (stem length, internode lengths, leaf lengths)
from projected RGB images using a CNN regression model.

**Input:** 512 synthetic Zea mays point clouds (PLY) + projected RGB images  
**Output:** Per-plant measurements in metres, predicted from single or multi-view images

---

## Directory Structure

```
keypoints_pipeline/
├── measure_plants.py       # Extract ground truth measurements from 3D PLY files
├── visualize_suspicious.py # Visual QC — check keypoint placement on plant images
├── project_keypoints.py    # Project 3D keypoints → 2D pixel coordinates
├── reg_dataset.py          # PyTorch dataset (images + labels)
├── reg_model.py            # Model definitions (EfficientNet-B3, ResNet-34, Custom CNN)
├── train_engine.py         # Shared training loop (do not run directly)
├── train_efficientnet.py   # Train EfficientNet-B3
├── train_resnet34.py       # Train ResNet-34
├── train_custom.py         # Train custom CNN from scratch
├── eval_reg.py             # Multi-view evaluation + metrics
├── DECISIONS.txt           # Full decision log and design rationale
├── requirements.txt        # Python dependencies
├── labels.json             # Ground truth measurements (output of measure_plants.py)
├── keypoints.json          # 3D attachment point coordinates
├── training_data_fmt_17/   # Projected RGB images (copy from CV_datasets)
│   └── images/
│       ├── train/          # plant_XXXX_rgb_YYY.png
│       ├── val/
│       └── test/
└── checkpoints/            # Saved weights (created during training)
    ├── efficientnet_b3/
    ├── resnet34/
    └── custom_cnn/
```

---

## Requirements

- Python 3.10+
- CUDA 11.8 compatible GPU (tested on Windows 11)
- ~5 GB VRAM minimum for batch_size=16 at 640×640

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
```

### 2. Install PyTorch with CUDA 11.8

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> If you have a different CUDA version, find the matching wheel at
> https://pytorch.org/get-started/locally/

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` lists torch/torchvision for reference but they must be
> installed via the PyTorch index (step 2) — plain `pip install -r` will
> install CPU-only torch from PyPI if done without the index URL.

### 4. Verify GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Data Setup

The pipeline expects the following files to be present before training:

| File / Folder | How to obtain |
|---|---|
| `labels.json` | Run `python measure_plants.py` |
| `keypoints.json` | Run `python measure_plants.py` |
| `training_data_fmt_17/` | Copy from `CV_datasets/training_data_fmt_17/` |

The raw PLY directories are read from hardcoded paths in `measure_plants.py`
and `reg_dataset.py`. Update these if your dataset is in a different location:

```python
# reg_dataset.py
RAW_PCD_DIR = r"C:\...\FielGrwon_ZeaMays_RawPCD_100k\..."
SEG_PCD_DIR = r"C:\...\FielGrwon_ZeaMays_SegmentedPCD_100k\..."
```

---

## Running the Pipeline

### Step 1 — Extract measurements from 3D PLY files

```bash
python measure_plants.py
```

Outputs `labels.json` and `keypoints.json` into this directory.

### Step 2 — Sanity check dataset

```bash
python reg_dataset.py
```

Expected output:
```
train:  27300 samples  img=(3, 640, 640)  valid_targets=24/32  stem=X.XXXm
val  :   5850 samples  ...
test :   5850 samples  ...
```

### Step 3 — Verify models load correctly

```bash
python reg_model.py
```

Expected output:
```
EfficientNet-B3  params=11.6M  out=(2, 32)  min=0.000  max=X.XXX
ResNet-34        params=21.6M  out=(2, 32)  min=0.000  max=X.XXX
Custom CNN       params=11.5M  out=(2, 32)  min=0.000  max=X.XXX
```

### Step 4 — Train

Run one or all models (each is independent):

```bash
python train_efficientnet.py   # EfficientNet-B3 baseline
python train_resnet34.py       # ResNet-34
python train_custom.py         # Custom CNN from scratch
```

Checkpoints saved to `checkpoints/{run_name}/best.pt` and `last.pt`.  
Training history saved to `checkpoints/{run_name}/history.json`.

> **Note:** The first epoch takes longer to start on Windows due to the
> DataLoader globbing 27,300 filenames at initialisation. This is normal.

### Step 5 — Evaluate

```bash
python eval_reg.py --model efficientnet --ckpt checkpoints/efficientnet_b3/best.pt
python eval_reg.py --model resnet34     --ckpt checkpoints/resnet34/best.pt
python eval_reg.py --model custom       --ckpt checkpoints/custom_cnn/best.pt
```

Reports both **single-view** and **multi-view (75-view average)** MAE, RMSE,
and R² per measurement slot (stem, each internode, each leaf).  
Results saved to `checkpoints/{run_name}/eval_results.json`.

---

## Target Layout

Each sample produces a 32-value prediction vector:

| Index | Meaning |
|---|---|
| 0 | `stem_length_m` |
| 1 – 15 | `internode_lengths_m[0]` … `[14]` (zero-padded if fewer) |
| 16 – 31 | `leaf_lengths_m[0]` … `[15]` (zero-padded if fewer) |

A boolean mask accompanies each sample to exclude padded slots from the loss.

---

## Training Configuration

Key hyperparameters are at the top of each `train_*.py` file:

| Parameter | EfficientNet | ResNet-34 | Custom CNN |
|---|---|---|---|
| Epochs | 60 | 60 | 80 |
| Batch size | 16 | 16 | 16 |
| Encoder LR | 1e-4 | 1e-4 | 3e-4 |
| Head LR | 1e-3 | 1e-3 | 3e-4 |
| Patience | 15 | 15 | 20 |

Pretrained models (EfficientNet, ResNet) use a lower encoder LR to preserve
ImageNet weights during fine-tuning. The custom CNN uses a single LR for all
layers since there are no pretrained weights.

---

## Design Notes

See `DECISIONS.txt` for the full rationale behind every design choice,
including the three failed stem measurement approaches and why the final
leaf-attachment waypath method was adopted.
