# YOLO Instance Segmentation Pipeline

Trains a YOLOv2-6m-seg model to segment maize plant parts (stem + 16 leaves)
from projected RGB images.

---

## Directory Structure

```
CV_datasets/
├── dataset_build.py          # Render PLY point clouds → projected RGB images
├── training_format.py        # Convert rendered images → YOLO/COCO dataset format
├── yolo_seg.py               # Train + evaluate YOLO segmentation model
├── training_data/            # Raw renders (input to training_format.py)
│   ├── train/plant_XXXX/rgb_YYY.png + mask_YYY.png
│   ├── val/
│   └── test/
├── training_data_fmt_17/     # Formatted dataset (output of training_format.py)
│   ├── images/train|val|test/
│   ├── labels/train|val|test/   ← YOLO polygon format .txt
│   ├── annotations/             ← COCO JSON (for MMDetection if needed)
│   └── maize.yaml
└── maize_runs/               # YOLO training outputs
    └── yolo26m_pipeline_test/
        └── weights/best.pt
```

---

## Classes (17 total)

| ID | Name  | Description              |
|----|-------|--------------------------|
| 0  | stem  | Main plant stem          |
| 1  | leaf1 | Lowest leaf (bottom)     |
| 2  | leaf2 |                          |
| …  | …     |                          |
| 16 | leaf16| Topmost leaf             |

Leaves are ordered **bottom → top** by mean Z height in the 3D point cloud.

---

## Setup

Same `.venv` as the keypoints pipeline. Additional dependency:

```bash
pip install ultralytics
```

---

## Running the Pipeline

### Step 1 — Render point clouds to images

```bash
python dataset_build.py
```

Renders each PLY plant from 75 camera views (5 azimuths × 5 elevations × 3 distances)
into `training_data/`. Split: 70% train / 15% val / 15% test.

### Step 2 — Format for YOLO

```bash
python training_format.py
```

Converts `training_data/` into YOLO polygon label format and COCO JSON.
Output: `training_data_fmt_17/` with `maize.yaml`.

Key config at top of file:
```python
NUM_CLASSES = 17          # stem + leaf1..leaf16
OUTPUT_DIR  = "training_data_fmt_17"
```

### Step 3 — Train

```bash
python yolo_seg.py
```

Config inside `yolo_seg.py`:

| Parameter | Value |
|-----------|-------|
| Model     | yolo26m-seg.pt |
| Epochs    | 100 |
| Patience  | 20 |
| Image size | 640×640 |
| Batch size | 16 |
| Cache     | False (avoids RAM overflow) |
| Workers   | 0 (Windows stability) |

Outputs saved to `maize_runs/yolo26m_pipeline_test/`.

### Step 4 — Evaluate

Evaluation runs automatically after training on the test split:

```
mAP50      : X.XXXX
mAP50-95   : X.XXXX
Precision  : X.XXXX
Recall     : X.XXXX
```

Best weights: `maize_runs/yolo26m_pipeline_test/weights/best.pt`

---

## Notes

- `cache=False` — set explicitly to prevent RAM overflow (full dataset ~38GB cached)
- `workers=0` — required on Windows to avoid DataLoader multiprocessing errors
- COCO JSON is also exported alongside YOLO labels for MMDetection compatibility
- Minimum contour area of 20px filters out projection noise in masks
