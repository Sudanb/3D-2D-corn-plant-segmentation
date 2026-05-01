import cv2
import numpy as np
from pathlib import Path

img_dir  = Path(r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt\images\train")
mask_dir = Path(r"C:\Users\sudanb\Desktop\CV_datasets\training_data\train")

stats = {
    "total"        : 0,
    "cropped"      : 0,
    "empty"        : 0,
    "small_plant"  : 0,  # plant covers <10% of image
}

for img_path in sorted(img_dir.glob("*.png"))[:500]:
    # find matching mask
    parts      = img_path.stem.split("_rgb_")
    plant_stem = parts[0]
    view_id    = parts[1]
    mask_path  = mask_dir / plant_stem / f"mask_{view_id}.png"

    if not mask_path.exists():
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_ANYDEPTH)
    stats["total"] += 1

    plant_pixels = (mask > 0).sum()
    total_pixels = mask.size

    # empty — no plant visible at all
    if plant_pixels == 0:
        stats["empty"] += 1
        continue

    # small — plant covers less than 10% of frame
    if plant_pixels / total_pixels < 0.10:
        stats["small_plant"] += 1

    # cropped — plant pixels touch boundary
    plant = mask > 0
    if (plant[:3, :].any() or plant[-3:, :].any() or
        plant[:, :3].any() or plant[:, -3:].any()):
        stats["cropped"] += 1

for k, v in stats.items():
    pct = f"({v/stats['total']:.1%})" if stats["total"] > 0 and k != "total" else ""
    print(f"{k:<15}: {v} {pct}")