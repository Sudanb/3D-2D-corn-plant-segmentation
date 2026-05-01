"""
eval_yolo.py
============
Run YOLO segmentation validation on test split and export
per-class mAP50, mAP50-95, and Recall to CSV — same format
as eval_eomt.py output for direct comparison.

Usage:
  python eval_yolo.py
"""

import csv
import json
from pathlib import Path
from ultralytics import YOLO

BASE = Path(r"C:\Users\sudanb\Desktop\CV_datasets")

RUNS = [
    {
        "name":       "yolov8m_2cls_10ep",
        "weights":    BASE / "runs/segment/runs/segment/maize_runs/yolov8m_2cls_10ep-4/weights/best.pt",
        "data":       BASE / "training_data_fmt/maize.yaml",
        "classes":    ["stem", "leaf"],
        "out":        BASE / "eval_yolo_yolov8m_2cls.csv",
    },
    {
        "name":       "yolov8m_17cls_10ep",
        "weights":    BASE / "runs/segment/runs/segment/maize_runs/yolov8m_17cls_10ep/weights/best.pt",
        "data":       BASE / "training_data_fmt_17/maize.yaml",
        "classes":    ["stem", "leaf1", "leaf2", "leaf3", "leaf4", "leaf5", "leaf6",
                       "leaf7", "leaf8", "leaf9", "leaf10", "leaf11", "leaf12",
                       "leaf13", "leaf14", "leaf15", "leaf16"],
        "out":        BASE / "eval_yolo_yolov8m_17cls.csv",
    },
    {
        "name":       "yolo26m_2cls_10ep",
        "weights":    BASE / "runs/segment/runs/segment/maize_runs/yolo26m_2cls_10ep/weights/best.pt",
        "data":       BASE / "training_data_fmt/maize.yaml",
        "classes":    ["stem", "leaf"],
        "out":        BASE / "eval_yolo_yolo26m_2cls.csv",
    },
    {
        "name":       "yolo26m_17cls_10ep",
        "weights":    BASE / "runs/segment/runs/segment/maize_runs/yolo26m_17cls_10ep/weights/best.pt",
        "data":       BASE / "training_data_fmt_17/maize.yaml",
        "classes":    ["stem", "leaf1", "leaf2", "leaf3", "leaf4", "leaf5", "leaf6",
                       "leaf7", "leaf8", "leaf9", "leaf10", "leaf11", "leaf12",
                       "leaf13", "leaf14", "leaf15", "leaf16"],
        "out":        BASE / "eval_yolo_yolo26m_17cls.csv",
    },
]


def eval_run(cfg):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"{'='*60}")

    model = YOLO(cfg["weights"])
    metrics = model.val(
        data=str(cfg["data"]),
        split="test",
        verbose=True,
    )

    seg = metrics.seg
    class_indices = metrics.ap_class_index   # classes that appeared in val
    class_names   = cfg["classes"]

    # YOLO reports speed as dict: {preprocess, inference, postprocess} in ms/image
    speed = metrics.speed  # ms per image
    inference_ms = speed.get("inference", 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    # per-class arrays aligned to class_indices
    ap50_per   = seg.ap50          # shape: (num_detected_classes,)
    ap5095_per = seg.ap            # shape: (num_detected_classes,)
    recall_per = seg.r             # shape: (num_detected_classes,)

    rows = []

    # Overall row
    rows.append({
        "class":     "all",
        "mAP50":     round(float(seg.map50), 4),
        "mAP50-95":  round(float(seg.map),   4),
        "Recall":    round(float(seg.r.mean()), 4) if len(seg.r) else 0.0,
    })

    # Per-class rows
    for rank, cls_idx in enumerate(class_indices):
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        rows.append({
            "class":    name,
            "mAP50":    round(float(ap50_per[rank]),   4),
            "mAP50-95": round(float(ap5095_per[rank]), 4),
            "Recall":   round(float(recall_per[rank]),  4),
        })

    # Print table
    print(f"\n  Inference speed: {fps:.2f} FPS | {inference_ms:.2f} ms/image")
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
            "mAP50":                  rows[0]["mAP50"],
            "mAP50-95":               rows[0]["mAP50-95"],
            "Recall":                 rows[0]["Recall"],
            "inference_fps":          round(fps, 2),
            "inference_ms_per_image": round(inference_ms, 2),
            "speed_breakdown_ms":     {k: round(v, 2) for k, v in speed.items()},
            "per_class": {r["class"]: {"mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"],
                                       "Recall": r["Recall"]} for r in rows},
        }, f, indent=2)

    print(f"\n  CSV  → {cfg['out']}")
    print(f"  JSON → {json_path}")
    return rows


if __name__ == "__main__":
    for cfg in RUNS:  # noqa
        if not Path(cfg["weights"]).exists():
            print(f"\n[SKIP] {cfg['name']} — weights not found: {cfg['weights']}")
            continue
        eval_run(cfg)

    print("\nDone. All results saved to CV_datasets root.")
