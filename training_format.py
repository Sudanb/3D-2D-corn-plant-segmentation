"""
convert_to_training_format.py
==============================
Converts the per-plant output of generate_dataset.py into two formats
consumed by downstream training frameworks:

  1. COCO JSON  — for MMDetection (Mask2Former, SOLOv2, RTMDet, QueryInst)
  2. YOLO seg   — for YOLOv8-seg

Input directory layout (produced by generate_dataset.py):
  training_data/
  ├── train/
  │   ├── plant_0001/
  │   │   ├── rgb_001.png
  │   │   ├── seg_001.png
  │   │   └── mask_001.png   ← integer instance IDs (uint8)
  │   └── ...
  ├── val/
  └── test/

Output directory layout:
  training_data_fmt/
  ├── images/
  │   ├── train/   plant0001_v001.png  ...
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/   plant0001_v001.txt  ...  (YOLO polygon format)
  │   ├── val/
  │   └── test/
  └── annotations/
      ├── instances_train.json   (COCO format)
      ├── instances_val.json
      └── instances_test.json
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to native Python types for json.dump."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# =============================================================================
# Configuration — must match generate_dataset.py
# =============================================================================
INPUT_DIR    = r"C:\Users\sudanb\Desktop\CV_datasets\training_data"
OUTPUT_DIR   = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt_17" #for 17 classes
#OUTPUT_DIR   = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt"
IMAGE_HEIGHT = 640
IMAGE_WIDTH  = 640

# Class definitions
# ID 1 = stem, IDs 2..N = leaves (per build_color_to_id ordering)
# We map these to two semantic classes for training:
#   class 0 → stem
#   class 1 → leaf
# Change to 17 classes if you want per-leaf-position classification.
NUM_CLASSES  = 17   # 2 for stem/leaf; 17 for per-leaf (leaf1=bottom, leaf16=top)
#CLASS_NAMES  = ["stem", "leaf"]
CLASS_NAMES = ["stem"] + [f"leaf{i}" for i in range(1, 17)]  # stem, leaf1..leaf16

SPLITS = ["train", "val", "test"]

# Minimum contour area to keep (filters out projection noise)
MIN_CONTOUR_AREA = 20   # pixels


# =============================================================================
# Helpers
# =============================================================================

def instance_id_to_class(instance_id: int) -> int | None:
    """
    Map integer instance ID (from mask PNG) to a 0-indexed class index.

    With NUM_CLASSES=2:
      instance_id == 1 → class 0 (stem)
      instance_id >= 2 → class 1 (leaf)

    With NUM_CLASSES=17 (per-leaf):
      instance_id == 1  → class 0  (stem)
      instance_id == 2  → class 1  (leaf1, bottom)
      ...
      instance_id == 17 → class 16 (leaf16, top)

    Returns None if instance_id falls outside the expected range so callers
    can skip rather than write a corrupt label.
    """
    if NUM_CLASSES == 2:
        return 0 if instance_id == 1 else 1
    else:
        class_idx = instance_id - 1   # 0-indexed
        if class_idx >= NUM_CLASSES:
            print(f"  WARNING: instance_id={instance_id} exceeds NUM_CLASSES={NUM_CLASSES}, skipping.")
            return None
        return class_idx


def mask_to_polygons(binary: np.ndarray, max_points: int = 100) -> list[list[float]]:
    """
    Convert a binary instance mask to a list of polygon contours.
    Each polygon is a flat list [x1, y1, x2, y2, ...] in pixel coords.

    - Filters contours smaller than MIN_CONTOUR_AREA
    - Simplifies contours with more than max_points using RDP algorithm
    - Handles disconnected regions (returns multiple polygons per instance)
    - Safe with uint8 and uint16 input masks
    """
    binary = (binary > 0).astype(np.uint8)   # safe dtype conversion

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        if c.shape[0] < 3:
            continue

        # simplify dense contours — preserves shape, reduces point count
        if c.shape[0] > max_points:
            epsilon = 0.005 * cv2.arcLength(c, True)
            c       = cv2.approxPolyDP(c, epsilon, True)

        if c.shape[0] < 3:   # re-check after simplification
            continue

        polygons.append(c.reshape(-1).tolist())

    return polygons


def collect_image_paths(input_dir: str, split: str) -> list[tuple[Path, Path, Path]]:
    """
    Walk the per-plant subdirectory structure and collect (rgb, mask) path tuples.
    Returns list of (rgb_path, mask_path, plant_stem) sorted stably.
    """
    split_dir = Path(input_dir) / split
    if not split_dir.exists():
        return []

    triplets = []
    for plant_dir in sorted(split_dir.iterdir()):
        if not plant_dir.is_dir():
            continue
        for rgb_path in sorted(plant_dir.glob("rgb_*.png")):
            view_id      = rgb_path.stem.replace("rgb_", "")    # e.g. "001"
            mask_path    = plant_dir / f"mask_{view_id}.png"
            if not mask_path.exists():
                print(f"  WARNING: mask missing for {rgb_path}, skipping.")
                continue
            triplets.append((rgb_path, mask_path, plant_dir.name))
    return triplets


# =============================================================================
# COCO conversion
# =============================================================================

def build_coco_json(triplets: list, img_id_offset: int = 0) -> tuple[list, list]:
    """
    Build COCO `images` and `annotations` lists from a list of (rgb, mask, stem) tuples.
    Returns (images, annotations).
    """
    images      = []
    annotations = []
    ann_id      = 1

    iterable = tqdm(enumerate(triplets, 1), total=len(triplets)) if TQDM else enumerate(triplets, 1)

    for local_img_id, (rgb_path, mask_path, plant_stem) in iterable:
        img_id    = local_img_id + img_id_offset
        flat_name = f"{plant_stem}_{rgb_path.stem}.png"   # e.g. plant_0001_rgb_001.png

        probe = cv2.imread(str(rgb_path))
        if probe is None:
            print(f"  WARNING: could not read {rgb_path}, skipping.")
            continue
        h, w = probe.shape[:2]

        images.append({
            "id"        : img_id,
            "file_name" : flat_name,
            "height"    : h,
            "width"     : w,
        })

        mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
        if mask is None:
            continue

        for inst_id in np.unique(mask):
            if inst_id == 0:
                continue   # background

            class_idx = instance_id_to_class(inst_id)
            if class_idx is None:
                continue
            category_id = class_idx + 1              # COCO is 1-indexed

            binary   = (mask == inst_id).astype(np.uint8)
            polygons = mask_to_polygons(binary)
            if not polygons:
                continue

            x, y, bw, bh = cv2.boundingRect(binary)
            area         = int(binary.sum())

            annotations.append({
                "id"          : ann_id,
                "image_id"    : img_id,
                "category_id" : category_id,
                "segmentation": polygons,
                "area"        : area,
                "bbox"        : [x, y, bw, bh],
                "iscrowd"     : 0,
            })
            ann_id += 1

    return images, annotations


def save_coco_json(images: list, annotations: list, out_path: Path):
    categories = [
        {"id": i + 1, "name": name, "supercategory": "plant"}
        for i, name in enumerate(CLASS_NAMES)
    ]
    coco = {
        "info"       : {"description": "MaizeField3D projected dataset"},
        "images"     : images,
        "annotations": annotations,
        "categories" : categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, cls=_NumpyEncoder)
    print(f"  Saved {out_path}  ({len(images)} images, {len(annotations)} annotations)")


# =============================================================================
# YOLO conversion
# =============================================================================

def mask_to_yolo_lines(mask: np.ndarray, img_h: int, img_w: int) -> list[str]:
    """
    Convert integer instance mask → YOLO segmentation label lines.
    Each line: class_id x1 y1 x2 y2 ... (normalised polygon coords, no trailing newline)
    """
    lines = []
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue

        class_idx = instance_id_to_class(inst_id)
        if class_idx is None:
            continue
        binary    = (mask == inst_id).astype(np.uint8)
        polygons  = mask_to_polygons(binary)

        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] /= img_w
            pts[:, 1] /= img_h
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
            lines.append(f"{class_idx} {coords}")

    return lines


# =============================================================================
# Main
# =============================================================================

def convert():
    out_root = Path(OUTPUT_DIR)

    img_dirs  = {s: out_root / "images"      / s for s in SPLITS}
    lbl_dirs  = {s: out_root / "labels" / s for s in SPLITS}
    ann_dir   =              out_root / "annotations"

    for d in list(img_dirs.values()) + list(lbl_dirs.values()) + [ann_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n{'='*60}")
        print(f"  {split.upper()}")
        print(f"{'='*60}")

        triplets = collect_image_paths(INPUT_DIR, split)
        if not triplets:
            print(f"  No data found for split '{split}', skipping.")
            continue

        print(f"  Found {len(triplets)} images")

        # --- copy images (flat structure) and write YOLO labels ---
        iterable = (tqdm(triplets, desc="  copying + YOLO labels")
                    if TQDM else triplets)

        for rgb_path, mask_path, plant_stem in iterable:
            view_id   = rgb_path.stem.replace("rgb_", "")
            flat_name = f"{plant_stem}_{rgb_path.stem}.png"
            flat_stem = f"{plant_stem}_{rgb_path.stem}"

            # copy RGB image
            shutil.copy2(rgb_path, img_dirs[split] / flat_name)

            # YOLO label — derive h/w from mask so it works at any resolution
            mask  = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
            h, w  = mask.shape
            lines = mask_to_yolo_lines(mask, h, w)

            label_path = lbl_dirs[split] / f"{flat_stem}.txt"
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

        # --- COCO JSON ---
        print(f"  Building COCO JSON...")
        images, annotations = build_coco_json(triplets)
        save_coco_json(images, annotations, ann_dir / f"instances_{split}.json")

    # --- YOLO dataset YAML ---
    yaml_path = out_root / "maize.yaml"
    yaml_content = f"""# MaizeField3D instance segmentation dataset
path: {out_root.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {NUM_CLASSES}
names: [{", ".join(f'"{n}"' for n in CLASS_NAMES)}]
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nSaved YOLO dataset YAML: {yaml_path}")

    # --- MMDetection dataset config snippet ---
    mmdet_snippet = out_root / "mmdet_dataset_config.py"
    mmdet_content = f"""# Paste this into your MMDetection config
data_root = r'{out_root.as_posix()}'

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=dict(classes={tuple(CLASS_NAMES)}, palette=[(255,255,255),(0,200,0)]),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=dict(classes={tuple(CLASS_NAMES)}, palette=[(255,255,255),(0,200,0)]),
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/instances_val.json',
    metric=['segm'],
    classwise=True,
)
"""
    with open(mmdet_snippet, "w") as f:
        f.write(mmdet_content)
    print(f"Saved MMDetection config snippet: {mmdet_snippet}")
    print("\nDone.")


if __name__ == "__main__":
    convert()