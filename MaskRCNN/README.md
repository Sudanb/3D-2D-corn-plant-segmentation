# Mask R-CNN — Training & Evaluation

Two-stage instance segmentation using torchvision's Mask R-CNN (ResNet-50 + FPN), fine-tuned on MaizeField3D.

## Environment

**Platform**: Windows (PowerShell)  
**No conda required** — standard pip environment.

```powershell
pip install torch torchvision pycocotools pillow numpy
```

## Training

Edit the top of `train_maskrcnn.py` to switch between 2-class and 17-class:

```python
NUM_CLASS_MODE = 2   # change to 17 for 17-class run
```

Then run:

```powershell
cd C:\Users\sudanb\Desktop\CV_datasets
python MaskRCNN\train_maskrcnn.py
```

### Key training config

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | SGD (lr=0.005, momentum=0.9) |
| AMP | BF16 (Ada tensor cores) |
| Workers | 4 |
| Pretrained | COCO instance segmentation |

Checkpoints saved to:
```
MaskRCNN/maskrcnn_2class/maskrcnn_2class_best.pth
MaskRCNN/maskrcnn_17class/maskrcnn_17class_best.pth
```

### Architecture

- Backbone: ResNet-50 + FPN (P2–P6)
- Heads: FastRCNNPredictor (box) + MaskRCNNPredictor (28×28 mask)
- Loss: RPN BCE + RPN Smooth L1 + RoI CE + RoI Smooth L1 + mask BCE (equal weights)
- ~44 M parameters

## Evaluation

Runs fresh inference on the test split for both class configs. No retraining needed.

```powershell
cd C:\Users\sudanb\Desktop\CV_datasets
python MaskRCNN\eval_maskrcnn.py
```

### Output files

| File | Content |
|------|---------|
| `eval_maskrcnn_2cls.csv` | class, mAP50, mAP50-95, Recall |
| `eval_maskrcnn_2cls.json` | Same + FPS, ms/image, inference time |
| `eval_maskrcnn_17cls.csv` | 17-class results |
| `eval_maskrcnn_17cls.json` | 17-class results + timing |
