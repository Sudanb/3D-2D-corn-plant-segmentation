# YOLO — Training & Evaluation

YOLOv8m-seg and YOLO11m-seg (yolo26m) instance segmentation on MaizeField3D.

## Environment

**Platform**: Windows (PowerShell)  
**No conda required** — standard pip environment.

```powershell
pip install ultralytics
```

## Training

Runs all 4 configurations sequentially (2-class and 17-class for both models):

```powershell
cd C:\Users\sudanb\Desktop\CV_datasets
python train_yolo.py
```

### What it runs

| Run name | Model | Classes | Data |
|----------|-------|---------|------|
| `yolov8m_2cls_10ep` | YOLOv8m-seg | 2 (stem, leaf) | `training_data_fmt/` |
| `yolov8m_17cls_10ep` | YOLOv8m-seg | 17 (stem, leaf1–leaf16) | `training_data_fmt_17/` |
| `yolo26m_2cls_10ep` | YOLO11m-seg | 2 | `training_data_fmt/` |
| `yolo26m_17cls_10ep` | YOLO11m-seg | 17 | `training_data_fmt_17/` |

### Key training config

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Optimizer | Auto (AdamW → SGD) |
| AMP | Enabled |
| Workers | 0 (Windows requirement) |
| Pretrained | COCO |

Weights saved to:
```
runs/segment/runs/segment/maize_runs/<run_name>/weights/best.pt
```

## Evaluation

```powershell
cd C:\Users\sudanb\Desktop\CV_datasets
python eval_yolo.py
```

Evaluates all 4 runs on the test split. Skips any run whose `best.pt` is missing.

### Output files (per run)

| File | Content |
|------|---------|
| `eval_yolo_<run>.csv` | class, mAP50, mAP50-95, Recall |
| `eval_yolo_<run>.json` | Same + FPS, ms/image, speed breakdown |
