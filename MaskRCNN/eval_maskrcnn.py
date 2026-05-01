"""
eval_maskrcnn.py
================
Run Mask R-CNN inference on the test split and export
per-class mAP50, mAP50-95, and Recall to CSV — same format
as eval_eomt.py and eval_yolo.py for direct comparison.

Usage:
  python eval_maskrcnn.py
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

BASE = Path(r"C:\Users\sudanb\Desktop\CV_datasets")

RUNS = [
    {
        "name":        "maskrcnn_2cls",
        "checkpoint":  BASE / "MaskRCNN/maskrcnn_2class/maskrcnn_2class_best.pth",
        "data_root":   BASE / "training_data_fmt",
        "num_classes": 2 + 1,
        "class_names": ["stem", "leaf"],
        "out":         BASE / "eval_maskrcnn_2cls.csv",
    },
    {
        "name":        "maskrcnn_17cls",
        "checkpoint":  BASE / "MaskRCNN/maskrcnn_17class/maskrcnn_17class_best.pth",
        "data_root":   BASE / "training_data_fmt_17",
        "num_classes": 17 + 1,
        "class_names": ["stem", "leaf1", "leaf2", "leaf3", "leaf4", "leaf5", "leaf6",
                        "leaf7", "leaf8", "leaf9", "leaf10", "leaf11", "leaf12",
                        "leaf13", "leaf14", "leaf15", "leaf16"],
        "out":         BASE / "eval_maskrcnn_17cls.csv",
    },
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaizeDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.img_dir = os.path.join(root, "images", split)
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = F.to_tensor(Image.open(
            os.path.join(self.img_dir, img_info["file_name"])
        ).convert("RGB"))
        return img, img_id


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes, checkpoint):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def per_class_ap(coco_gt, results_file, iou_type="segm"):
    coco_dt = coco_gt.loadRes(str(results_file))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # precision shape: (T=10, R=101, K=num_cats, A=4, M=2)
    # T: IoU thresholds 0.5:0.05:0.95
    # R: recall thresholds
    # K: categories
    # A: area ranges (all, small, medium, large)
    # M: max detections
    precision = ev.eval["precision"]   # (10, 101, K, 4, 2)
    recall_arr = ev.eval["recall"]     # (10, K, 4, 2)
    cat_ids = ev.params.catIds

    per_class = {}
    for ki, cat_id in enumerate(cat_ids):
        # mAP50-95: mean over all T, recall thresholds where precision != -1
        pr_all = precision[:, :, ki, 0, 2]   # (10, 101) — area=all, maxDet=100
        pr_all = pr_all[pr_all > -1]
        ap5095 = float(np.mean(pr_all)) if len(pr_all) else 0.0

        # mAP50: T index 0 = IoU 0.50
        pr50 = precision[0, :, ki, 0, 2]
        pr50 = pr50[pr50 > -1]
        ap50 = float(np.mean(pr50)) if len(pr50) else 0.0

        # Recall: mean over IoU thresholds at maxDet=100
        rec = recall_arr[:, ki, 0, 2]
        rec = rec[rec > -1]
        rec_val = float(np.mean(rec)) if len(rec) else 0.0

        per_class[cat_id] = {"mAP50": round(ap50, 4),
                              "mAP50-95": round(ap5095, 4),
                              "Recall": round(rec_val, 4)}

    # overall from COCOeval stats
    overall = {
        "mAP50":    round(float(ev.stats[1]), 4),
        "mAP50-95": round(float(ev.stats[0]), 4),
        "Recall":   round(float(ev.stats[8]), 4),   # AR@100
    }
    return overall, per_class, cat_ids


def eval_run(cfg):
    print(f"\n{'='*60}\n  {cfg['name']}\n{'='*60}")

    dataset = MaizeDataset(str(cfg["data_root"]), "test")
    loader = DataLoader(dataset, batch_size=4, num_workers=0,
                        collate_fn=collate_fn)

    model = get_model(cfg["num_classes"], cfg["checkpoint"])

    results = []
    num_images = len(dataset)
    t0 = time.time()
    with torch.no_grad():
        for imgs, img_ids in loader:
            imgs = [img.to(DEVICE) for img in imgs]
            outputs = model(imgs)
            for img_id, out in zip(img_ids, outputs):
                masks = out["masks"].squeeze(1).cpu().numpy() > 0.5
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                for mask, score, label in zip(masks, scores, labels):
                    rle = coco_mask.encode(
                        np.asfortranarray(mask.astype(np.uint8))
                    )
                    rle["counts"] = rle["counts"].decode("utf-8")
                    results.append({
                        "image_id":    int(img_id),
                        "category_id": int(label),
                        "segmentation": rle,
                        "score":        float(score),
                    })
    elapsed = time.time() - t0
    fps = num_images / elapsed
    print(f"\n  Inference: {elapsed:.1f}s total | {fps:.2f} FPS | {1000/fps:.1f} ms/image")

    tmp_file = cfg["out"].with_name(cfg["name"] + "_tmp_results.json")
    with open(tmp_file, "w") as f:
        json.dump(results, f)

    print(f"\nRunning COCOeval on {len(dataset)} test images...")
    overall, per_class, cat_ids = per_class_ap(dataset.coco, tmp_file)
    tmp_file.unlink()

    class_names = cfg["class_names"]

    rows = [{"class": "all", **overall}]
    for cat_id in cat_ids:
        name = class_names[cat_id - 1] if (cat_id - 1) < len(class_names) else f"class_{cat_id}"
        rows.append({"class": name, **per_class[cat_id]})

    # Print table
    print(f"\n  {'Class':<12} {'mAP50':>8} {'mAP50-95':>10} {'Recall':>8}")
    print("  " + "-" * 42)
    for r in rows:
        print(f"  {r['class']:<12} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['Recall']:>8.4f}")

    # Save CSV
    with open(cfg["out"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "mAP50", "mAP50-95", "Recall"])
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON
    json_path = cfg["out"].with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "model":                  cfg["name"],
            "mAP50":                  overall["mAP50"],
            "mAP50-95":               overall["mAP50-95"],
            "Recall":                 overall["Recall"],
            "inference_fps":          round(fps, 2),
            "inference_ms_per_image": round(1000 / fps, 2),
            "inference_total_s":      round(elapsed, 2),
            "per_class": {r["class"]: {"mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"],
                                       "Recall": r["Recall"]} for r in rows},
        }, f, indent=2)

    print(f"\n  CSV  → {cfg['out']}")
    print(f"  JSON → {json_path}")


if __name__ == "__main__":
    for cfg in RUNS:
        eval_run(cfg)
    print("\nDone.")
