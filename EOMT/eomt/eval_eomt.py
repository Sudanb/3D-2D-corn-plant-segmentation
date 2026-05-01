"""
eval_eomt.py
============
Evaluate a trained EoMT checkpoint on the maize test/val set.
Reports per-class mAP alongside overall mAP50, mAP50-95.

Usage:
  python eval_eomt.py \
    --ckpt path/to/checkpoint.ckpt \
    --data_path ~/maize_data \
    --model small          # small | base | large
    --split test           # test | val (default: test)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from datasets.coco_instance import MaizeInstance
from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_instance import MaskClassificationInstance

BACKBONES = {
    "small": "vit_small_patch14_reg4_dinov2",
    "base":  "vit_base_patch14_reg4_dinov2",
    "large": "vit_large_patch14_reg4_dinov2",
}

CLASS_NAMES = ["stem", "leaf"]


def build_model(backbone_name: str, ckpt_path: str, device: torch.device):
    encoder = ViT(img_size=(640, 640), backbone_name=backbone_name)
    network = EoMT(encoder=encoder, num_classes=2, num_q=100)
    model = MaskClassificationInstance.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        network=network,
        img_size=(640, 640),
        num_classes=2,
        attn_mask_annealing_enabled=False,
        strict=False,
    )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_eval(model, dataloader, device, limit_batches=None):
    # Two metrics: full IoU range (mAP50-95) and IoU=0.5 only (mAP50) for per-class
    metric     = MeanAveragePrecision(iou_type="segm", class_metrics=True).to(device)
    metric_50  = MeanAveragePrecision(iou_type="segm", class_metrics=True,
                                      iou_thresholds=[0.5]).to(device)

    for batch_idx, (imgs, targets) in enumerate(dataloader):
        if limit_batches and batch_idx >= limit_batches:
            break

        if batch_idx % 10 == 0:
            total = limit_batches or len(dataloader)
            print(f"  [{batch_idx}/{total}]", end="\r")

        imgs = [img.to(device) for img in imgs]
        img_sizes = [img.shape[-2:] for img in imgs]
        transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)

        mask_logits_per_layer, class_logits_per_layer = model(transformed)

        # Use final layer predictions only
        mask_logits = mask_logits_per_layer[-1]
        class_logits = class_logits_per_layer[-1]

        mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes
        )

        preds, gts = [], []
        for j in range(len(mask_logits)):
            scores = class_logits[j].softmax(dim=-1)[:, :-1]
            labels = (
                torch.arange(scores.shape[-1], device=device)
                .unsqueeze(0)
                .repeat(scores.shape[0], 1)
                .flatten(0, 1)
            )
            topk_scores, topk_indices = scores.flatten(0, 1).topk(
                model.eval_top_k_instances, sorted=False
            )
            labels = labels[topk_indices]
            topk_indices = topk_indices // scores.shape[-1]
            mask_logits[j] = mask_logits[j][topk_indices]

            masks = mask_logits[j] > 0
            mask_scores = (
                mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
            ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
            final_scores = topk_scores * mask_scores

            preds.append(dict(masks=masks, labels=labels, scores=final_scores))
            gts.append(dict(
                masks=targets[j]["masks"].to(device),
                labels=targets[j]["labels"].to(device),
                iscrowd=targets[j]["is_crowd"].to(device),
            ))

        metric.update(preds, gts)
        metric_50.update(preds, gts)

    print()
    results = metric.compute()
    results_50 = metric_50.compute()
    results["map_per_class_50"] = results_50.get("map_per_class", None)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--data_path",   required=True)
    parser.add_argument("--model",       default="small", choices=["small", "base", "large"])
    parser.add_argument("--split",       default="test",  choices=["val", "test"])
    parser.add_argument("--batch_size",  default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--limit_batches", default=None, type=int)
    parser.add_argument("--out",         default="eval_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEoMT Evaluation — DINOv2-{args.model.upper()} | split={args.split}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Data       : {args.data_path}")
    print(f"Device     : {device}\n")

    model = build_model(BACKBONES[args.model], args.ckpt, device)

    datamodule = MaizeInstance(
        path=args.data_path,
        num_classes=2,
        img_size=(640, 640),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        check_empty_targets=True,
    )
    datamodule.setup()
    dataloader = (
        datamodule.test_dataloader()
        if args.split == "test"
        else datamodule.val_dataloader()
    )

    num_images = len(dataloader.dataset)
    print(f"Running inference on {args.split} set ({len(dataloader)} batches, {num_images} images)...")
    t0 = time.perf_counter()
    results = run_eval(model, dataloader, device, args.limit_batches)
    elapsed = time.perf_counter() - t0
    fps = num_images / elapsed

    map5095    = float(results["map"])
    map50      = float(results["map_50"])
    map75      = float(results["map_75"])
    map_small  = float(results["map_small"])
    map_medium = float(results["map_medium"])
    map_large  = float(results["map_large"])
    per_class       = results.get("map_per_class", None)
    per_class_50    = results.get("map_per_class_50", None)
    mar_per_class   = results.get("mar_100_per_class", None)

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EoMT DINOv2-{args.model.upper()} — {args.split.upper()} results")
    print("=" * 60)
    print(f"  {'Class':<12} {'mAP50':>8} {'mAP50-95':>10} {'Recall':>8}")
    print("  " + "-" * 42)

    rows = []

    # Overall row
    mar_all = float(results.get("mar_100", 0.0))
    print(f"  {'all':<12} {map50:>8.4f} {map5095:>10.4f} {mar_all:>8.4f}")
    rows.append({"class": "all", "mAP50": round(map50, 4),
                 "mAP50-95": round(map5095, 4), "Recall": round(mar_all, 4)})

    # Per-class rows
    if per_class is not None:
        for i, ap5095 in enumerate(per_class.tolist()):
            name   = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            ap50   = float(per_class_50[i])  if per_class_50  is not None else 0.0
            recall = float(mar_per_class[i]) if mar_per_class is not None else 0.0
            print(f"  {name:<12} {ap50:>8.4f} {ap5095:>10.4f} {recall:>8.4f}")
            rows.append({"class": name, "mAP50": round(ap50, 4),
                         "mAP50-95": round(ap5095, 4), "Recall": round(recall, 4)})

    print("=" * 60)
    print(f"\n  Inference  : {elapsed:.1f}s total | {fps:.2f} FPS | {1000/fps:.1f} ms/image")
    print(f"\n  mAP75     : {map75:.4f}")
    print(f"  mAP Small : {map_small:.4f}")
    print(f"  mAP Medium: {map_medium:.4f}")
    print(f"  mAP Large : {map_large:.4f}")

    # ── Save CSV (YOLO-style) ────────────────────────────────────────
    stem = Path(args.out).stem
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "mAP50", "mAP50-95", "Recall"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV  saved to {csv_path}")

    # ── Save JSON ────────────────────────────────────────────────────
    output = {
        "model":         f"eomt_dinov2_{args.model}",
        "split":         args.split,
        "checkpoint":    str(args.ckpt),
        "mAP50-95":      round(map5095,    4),
        "mAP50":         round(map50,      4),
        "mAP75":         round(map75,      4),
        "mAP_small":     round(map_small,  4),
        "mAP_medium":    round(map_medium, 4),
        "mAP_large":     round(map_large,  4),
        "inference_fps": round(fps, 2),
        "inference_ms_per_image": round(1000 / fps, 2),
        "inference_total_s": round(elapsed, 2),
        "per_class":     {r["class"]: {"mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"],
                                       "Recall": r["Recall"]} for r in rows},
    }
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to {out_path}")


if __name__ == "__main__":
    main()
