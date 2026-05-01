# EoMT — Training & Evaluation

Encoder-only Mask Transformer (CVPR 2025) with frozen DINOv2 ViT-S backbone, fine-tuned on MaizeField3D for instance segmentation.

## Environment

**Platform**: WSL2 (Linux) — EoMT requires Linux due to PyTorch Lightning / torch.compile dependencies.  
**Conda environment**: `eomt`

### Setup (first time only)

```bash
# Inside WSL2
conda create -n eomt python==3.13.2
conda activate eomt
cd /mnt/c/Users/sudanb/Desktop/CV_datasets/EOMT/eomt
pip install -r requirements.txt
```

### Activate environment

```bash
wsl
conda activate eomt
cd /mnt/c/Users/sudanb/Desktop/CV_datasets/EOMT/eomt
```

## Training

### 2-class (stem + leaf)

```bash
python main.py fit \
  -c configs/dinov2/maize/instance/eomt_small_640.yaml \
  --data.path /mnt/c/Users/sudanb/Desktop/CV_datasets/training_data_fmt
```

### 17-class (stem + leaf1–leaf16)

```bash
python main.py fit \
  -c configs/dinov2/maize/instance/eomt_small_640.yaml \
  --data.path /mnt/c/Users/sudanb/Desktop/CV_datasets/training_data_fmt_17
```

> Change `num_classes` in the yaml to 17 for the 17-class run and update `CLASS_MAPPING` in `datasets/coco_instance.py`.

### Key training config

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | AdamW |
| Backbone | DINOv2 ViT-S/14 (frozen) |
| Trainable params | ~4 M (queries + heads) |
| Logger | CSVLogger → `logs/maize_instance_eomt_small_640/` |
| Framework | PyTorch Lightning + torch.compile |

### Architecture

- Backbone: DINOv2 ViT-S/14 with 4 register tokens (frozen, ~22M params)
- 100 learnable query tokens injected into last 4 ViT blocks
- Hungarian matching (Mask2Former matcher)
- Loss: BCE mask + Dice + cross-entropy class (auxiliary at each of 4 blocks)
- Attention mask annealing over epochs 1–5

Checkpoints saved to:
```
EOMT/eomt/logs/maize_instance_eomt_small_640/version_0/checkpoints/epoch=9-step=17060.ckpt
```

## Evaluation

### 2-class eval

```bash
# Inside WSL2 with eomt env active
cd /mnt/c/Users/sudanb/Desktop/CV_datasets/EOMT/eomt

python eval_eomt.py \
  --ckpt "logs/maize_instance_eomt_small_640/version_0/checkpoints/epoch=9-step=17060.ckpt" \
  --data_path "/mnt/c/Users/sudanb/Desktop/CV_datasets/training_data_fmt" \
  --model small \
  --out "/mnt/c/Users/sudanb/Desktop/CV_datasets/eval_eomt_2cls.json"
```

### 17-class eval

```bash
python eval_eomt.py \
  --ckpt "eomt/efub7kso/checkpoints/epoch=9-step=17060.ckpt" \
  --data_path "/mnt/c/Users/sudanb/Desktop/CV_datasets/training_data_fmt_17" \
  --model small \
  --out "/mnt/c/Users/sudanb/Desktop/CV_datasets/eval_eomt_17cls.json"
```

> For 17-class eval, update `num_classes=17` and `CLASS_NAMES` in `eval_eomt.py` before running.

### Output files

| File | Content |
|------|---------|
| `eval_eomt_2cls.csv` | class, mAP50, mAP50-95, Recall |
| `eval_eomt_2cls.json` | Same + FPS, ms/image, inference time |
| `eval_eomt_17cls.csv` | 17-class results |
| `eval_eomt_17cls.json` | 17-class results + timing |
