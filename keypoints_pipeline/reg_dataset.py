"""
reg_dataset.py
==============
PyTorch Dataset for plant measurement regression.

Each sample: (image [3×640×640 float32], targets [32 float32], mask [32 bool])

Target layout (32 values):
  [0]       stem_length_m
  [1..15]   internode_lengths_m  (up to 15, zero-padded if fewer)
  [16..31]  leaf_lengths_m       (up to 16, zero-padded if fewer)

Mask is True where the value is valid (not padding).
Plants with stem_length_m < MIN_STEM_M are excluded from all splits.
"""

import json
import random
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ── Paths — relative to this file so the pipeline is self-contained ───────────
_HERE        = Path(__file__).parent
IMAGES_DIR   = _HERE / "training_data_fmt_17" / "images"
LABELS_JSON  = _HERE / "labels.json"

# ── Must mirror dataset_build.py exactly ─────────────────────────────────────
RAW_PCD_DIR  = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_RawPCD_100k\FielGrwon_ZeaMays_RawPCD_100k"
SEG_PCD_DIR  = r"C:\Users\sudanb\Desktop\CV_datasets\FielGrwon_ZeaMays_SegmentedPCD_100k\FielGrwon_ZeaMays_SegmentedPCD_100k"
RANDOM_SEED  = 3

# ── Label layout ─────────────────────────────────────────────────────────────
MAX_INTERNODES = 15
MAX_LEAVES     = 16
N_TARGETS      = 1 + MAX_INTERNODES + MAX_LEAVES   # 32

MIN_STEM_M     = 1.0   # plants below this are excluded (incomplete stem data)


# =============================================================================
# Plant index → PLY stem mapping
# =============================================================================

def build_plant_map() -> dict[str, str]:
    """
    Reproduces dataset_build.py's shuffled plant ordering.
    Returns {"plant_0001": "0042", "plant_0002": "0007", ...}
    """
    raw_stems = {Path(f).stem for f in os.listdir(RAW_PCD_DIR) if f.endswith(".ply")}
    seg_stems = {Path(f).stem for f in os.listdir(SEG_PCD_DIR) if f.endswith(".ply")}
    common    = sorted(raw_stems & seg_stems)

    random.seed(RANDOM_SEED)
    random.shuffle(common)

    return {f"plant_{i:04d}": stem for i, stem in enumerate(common, start=1)}


# =============================================================================
# Label encoding
# =============================================================================

def make_target(label: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode one plant's label dict into a fixed-length vector + validity mask.
    Returns (targets float32 [32], mask bool [32]).
    """
    targets = np.zeros(N_TARGETS, dtype=np.float32)
    mask    = np.zeros(N_TARGETS, dtype=bool)

    targets[0] = label["stem_length_m"]
    mask[0]    = True

    for i, v in enumerate(label.get("internode_lengths_m", [])[:MAX_INTERNODES]):
        targets[1 + i] = v
        mask[1 + i]    = True

    for i, v in enumerate(label.get("leaf_lengths_m", [])[:MAX_LEAVES]):
        targets[1 + MAX_INTERNODES + i] = v
        mask[1 + MAX_INTERNODES + i]    = True

    return targets, mask


# =============================================================================
# Dataset
# =============================================================================

class PlantRegDataset(Dataset):
    """
    Maps projected plant images to regression targets.

    Args:
        split   : "train" | "val" | "test"
        augment : apply random flip + colour jitter (use only for train)
    """

    def __init__(self, split: str, augment: bool = False):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        with open(LABELS_JSON) as f:
            labels_raw = json.load(f)

        plant_map = build_plant_map()

        img_dir = IMAGES_DIR / split
        self.samples: list[tuple[Path, np.ndarray, np.ndarray]] = []

        for img_path in sorted(img_dir.glob("*.png")):
            plant_id = img_path.stem.split("_rgb_")[0]   # "plant_0001"
            ply_stem = plant_map.get(plant_id)
            if ply_stem is None:
                continue
            label = labels_raw.get(ply_stem)
            if label is None:
                continue
            if label["stem_length_m"] < MIN_STEM_M:
                continue

            targets, mask = make_target(label)
            self.samples.append((img_path, targets, mask))

        self.transform = _build_transform(augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, targets, mask = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return (
            self.transform(img),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
        )


def _build_transform(augment: bool) -> T.Compose:
    ops = []
    if augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = PlantRegDataset(split)
        img, tgt, mask = ds[0]
        print(f"{split:5s}: {len(ds):5d} samples  "
              f"img={tuple(img.shape)}  "
              f"valid_targets={int(mask.sum())}/32  "
              f"stem={tgt[0]:.3f}m")
