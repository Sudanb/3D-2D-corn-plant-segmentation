# ---------------------------------------------------------------
# Mask R-CNN fine-tuning on Maize dataset
# Supports 2-class and 17-class configurations
# Logs to CSV + wandb: mAP, loss, timing, GPU memory, params, GFLOPs
#
# Changes from original:
#   - AMP (BF16 autocast + GradScaler) for ~2x throughput on Ada GPU
#   - Fixed WandB step conflict (all logs use monotonic global_step)
#   - num_workers=4 + persistent_workers + pin_memory for faster data loading
#   - torch.backends.cudnn.benchmark = True
#   - optimizer.zero_grad(set_to_none=True) for slightly faster zeroing
#   - LR note: 0.005 is reasonable for SGD at batch=16; scale if you change batch
# ---------------------------------------------------------------

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import os
import time
import json
import csv

# ── wandb (optional — gracefully disabled if not installed) ───────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging to CSV only. pip install wandb to enable.")

# ── Config ────────────────────────────────────────────────────────────────────
# Switch between runs by changing these 3 lines only
NUM_CLASS_MODE = 2                           # 2 or 17
DATA_ROOT = (
    r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt_17"
    if NUM_CLASS_MODE == 17 else
    r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt"
)
RUN_NAME       = f"maskrcnn_{NUM_CLASS_MODE}class"

NUM_CLASSES    = NUM_CLASS_MODE + 1          # +1 for background
BATCH_SIZE     = 16                          # 4500 Ada has 24GB — push to 32 if VRAM headroom remains
NUM_EPOCHS     = 10
LR             = 0.005                       # Good for SGD + batch=16; scale linearly if batch changes
NUM_WORKERS    = 4                           # Increase if CPU is not bottleneck; 0 on some Windows setups
SAVE_DIR       = os.path.join(
    r"C:\Users\sudanb\Desktop\CV_datasets\MaskRCNN", RUN_NAME
)
WANDB_PROJECT  = "maize_seg"
DEVICE         = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# Ada tensor cores love this
torch.backends.cudnn.benchmark = True


# ── Dataset ───────────────────────────────────────────────────────────────────
class MaizeDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.root    = root
        self.split   = split
        self.img_dir = os.path.join(root, "images", split)
        ann_file     = os.path.join(root, "annotations", f"instances_{split}.json")
        self.coco    = COCO(ann_file)
        self.ids     = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img      = F.to_tensor(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        masks, boxes, labels = [], [], []
        for ann in anns:
            rle  = coco_mask.frPyObjects(
                ann["segmentation"], img_info["height"], img_info["width"]
            )
            mask = coco_mask.decode(coco_mask.merge(rle))
            if mask.max() == 0:
                continue
            pos = np.where(mask)
            x1, y1 = np.min(pos[1]), np.min(pos[0])
            x2, y2 = np.max(pos[1]), np.max(pos[0])
            if x2 <= x1 or y2 <= y1:
                continue
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))
            boxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])

        if len(masks) == 0:
            return img, {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros(0, dtype=torch.int64),
                "masks":    torch.zeros(
                    (0, img_info["height"], img_info["width"]), dtype=torch.uint8
                ),
                "image_id": torch.tensor([img_id]),
                "area":     torch.zeros(0, dtype=torch.float32),
                "iscrowd":  torch.zeros(0, dtype=torch.int64),
            }

        boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks  = torch.stack(masks)
        area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return img, {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([img_id]),
            "area":     area,
            "iscrowd":  torch.zeros(len(labels), dtype=torch.int64),
        }


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ── Model info (params + GFLOPs) ──────────────────────────────────────────────
def get_model_info(model):
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        dummy = [torch.rand(3, 640, 640).to(DEVICE)]
        with torch.no_grad():
            with FlopCounterMode(model, display=False) as flop_counter:
                model(dummy)
        gflops = flop_counter.get_total_flops() / 1e9
        model.train()
    except Exception:
        gflops = -1.0   # FlopCounterMode not available in older torch
    return round(params_m, 2), round(gflops, 2)


# ── Collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, scaler, device, epoch, global_step, use_wandb):
    """
    Returns (avg_loss, global_step).
    global_step is monotonically incremented across all epochs — fixes the
    WandB 'step must be monotonically increasing' warning that was caused by
    resetting the step counter at every epoch boundary.
    """
    model.train()
    total_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # AMP forward pass — BF16 preferred on Ada (numerically stabler than FP16)
        with autocast(dtype=torch.bfloat16):
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

        # AMP backward + optimizer step
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss  += losses.item()
        global_step += 1

        if i % 100 == 0:
            print(f"  Epoch {epoch} [{i}/{len(loader)}]  loss={losses.item():.4f}")
            if use_wandb:
                wandb.log({
                    "train/step_loss": losses.item(),
                    **{f"train/{k}": v.item() for k, v in loss_dict.items()},
                }, step=global_step)

    return total_loss / len(loader), global_step


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, dataset):
    model.eval()
    results = []
    t_inf   = time.time()

    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                img_id = target["image_id"].item()
                boxes  = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                masks  = output["masks"].cpu().numpy()

                for i in range(len(scores)):
                    mask = (masks[i, 0] > 0.5).astype(np.uint8)
                    rle  = coco_mask.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    results.append({
                        "image_id":     img_id,
                        "category_id":  int(labels[i]),
                        "segmentation": rle,
                        "bbox": [
                            float(boxes[i][0]), float(boxes[i][1]),
                            float(boxes[i][2] - boxes[i][0]),
                            float(boxes[i][3] - boxes[i][1]),
                        ],
                        "score": float(scores[i]),
                    })

    inf_time   = time.time() - t_inf
    ms_per_img = inf_time / len(dataset) * 1000
    fps        = len(dataset) / inf_time

    if len(results) == 0:
        print("  No predictions — skipping eval")
        return 0.0, 0.0, inf_time, ms_per_img, fps

    results_file = os.path.join(SAVE_DIR, "val_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f)

    coco_dt   = dataset.coco.loadRes(results_file)
    coco_eval = COCOeval(dataset.coco, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return (
        float(coco_eval.stats[0]),   # mAP50-95
        float(coco_eval.stats[1]),   # mAP50
        inf_time,
        ms_per_img,
        fps,
    )


# ── CSV Logger ────────────────────────────────────────────────────────────────
def init_log(log_file):
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "avg_loss", "epoch_time_s", "cumulative_time_s",
            "mAP50_95", "mAP50", "val_inf_time_s", "ms_per_img", "fps",
            "gpu_mem_gb",
        ])

def append_log(log_file, row):
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device:    {DEVICE}")
    print(f"Torch:     {torch.__version__}")
    print(f"Run:       {RUN_NAME}")
    print(f"Classes:   {NUM_CLASS_MODE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"AMP:       BF16 (autocast + GradScaler)")

    # Datasets
    train_ds = MaizeDataset(DATA_ROOT, "train")
    val_ds   = MaizeDataset(DATA_ROOT, "val")
    test_ds  = MaizeDataset(DATA_ROOT, "test")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # num_workers > 0 for parallel data loading; persistent_workers avoids
    # re-spawning worker processes every epoch; pin_memory speeds up GPU transfer.
    # NOTE: if you hit multiprocessing errors on Windows, set NUM_WORKERS=0.
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0), pin_memory=True,
    )
    val_dl   = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0), pin_memory=True,
    )
    test_dl  = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0), pin_memory=True,
    )

    # Model
    model = get_model(NUM_CLASSES).to(DEVICE)
    torch.cuda.reset_peak_memory_stats()

    # Model info
    params_m, gflops = get_model_info(model)
    print(f"Params: {params_m}M  |  GFLOPs: {gflops}")

    # Save model info JSON
    model_info = {
        "model":       "maskrcnn_resnet50_fpn",
        "num_classes": NUM_CLASS_MODE,
        "params_m":    params_m,
        "gflops":      gflops,
    }
    with open(os.path.join(SAVE_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    # Optimizer + scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # AMP scaler — handles BF16 gradient scaling
    scaler = GradScaler()

    # wandb init
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT,
            name=RUN_NAME,
            config={
                "model":        "maskrcnn_resnet50_fpn",
                "num_classes":  NUM_CLASS_MODE,
                "batch_size":   BATCH_SIZE,
                "epochs":       NUM_EPOCHS,
                "lr":           LR,
                "optimizer":    "SGD",
                "scheduler":    "CosineAnnealingLR",
                "amp":          "BF16",
                "params_m":     params_m,
                "gflops":       gflops,
                "dataset":      DATA_ROOT,
                "train_images": len(train_ds),
                "val_images":   len(val_ds),
            }
        )

    # CSV log
    log_file        = os.path.join(SAVE_DIR, "training_log.csv")
    init_log(log_file)

    best_ap         = 0.0
    cumulative_time = 0.0
    training_start  = time.time()
    global_step     = 0   # monotonically increases across ALL epochs → fixes WandB warning

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        loss, global_step = train_one_epoch(
            model, optimizer, train_dl, scaler, DEVICE, epoch, global_step, use_wandb
        )

        scheduler.step()

        epoch_time      = time.time() - t0
        cumulative_time += epoch_time
        gpu_mem_gb      = torch.cuda.max_memory_allocated() / 1e9

        print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={loss:.4f}  "
              f"time={epoch_time:.1f}s  total={cumulative_time/3600:.2f}h  "
              f"gpu={gpu_mem_gb:.2f}GB")

        ap5095 = ap50 = inf_time = ms_per_img = fps = 0.0

        # Evaluate every 2 epochs
        if epoch % 2 == 0:
            print("  Running validation...")
            ap5095, ap50, inf_time, ms_per_img, fps = evaluate(
                model, val_dl, DEVICE, val_ds
            )
            print(f"  mAP50-95={ap5095:.4f}  mAP50={ap50:.4f}  "
                  f"inf={inf_time:.1f}s  ({ms_per_img:.1f}ms/img  {fps:.1f}FPS)")

            if ap5095 > best_ap:
                best_ap = ap5095
                ckpt = os.path.join(SAVE_DIR, f"{RUN_NAME}_best.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"  New best! Saved: {ckpt}")
                if use_wandb:
                    wandb.run.summary["best_mAP50_95"] = best_ap
                    wandb.run.summary["best_epoch"]    = epoch

        # All epoch-level metrics logged at global_step → no more WandB conflict
        if use_wandb:
            wandb.log({
                "epoch":                    epoch,
                "train/avg_loss":           loss,
                "train/epoch_time_s":       epoch_time,
                "train/cumulative_time_h":  cumulative_time / 3600,
                "train/lr":                 scheduler.get_last_lr()[0],
                "train/gpu_mem_gb":         gpu_mem_gb,
                "val/mAP50_95":             ap5095,
                "val/mAP50":                ap50,
                "val/inference_time_s":     inf_time,
                "val/ms_per_img":           ms_per_img,
                "val/fps":                  fps,
            }, step=global_step)

        # Log to CSV
        append_log(log_file, [
            epoch,
            round(loss, 6),
            round(epoch_time, 1),
            round(cumulative_time, 1),
            round(ap5095, 6),
            round(ap50, 6),
            round(inf_time, 1),
            round(ms_per_img, 2),
            round(fps, 2),
            round(gpu_mem_gb, 3),
        ])

        # Checkpoint every 2 epochs
        if epoch % 2 == 0:
            ckpt = os.path.join(SAVE_DIR, f"{RUN_NAME}_ep{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved: {ckpt}")

    # ── Final test set evaluation ─────────────────────────────────────────────
    print("\nRunning final test set evaluation...")
    best_ckpt = os.path.join(SAVE_DIR, f"{RUN_NAME}_best.pth")
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    test_ap5095, test_ap50, test_inf, test_ms, test_fps = evaluate(
        model, test_dl, DEVICE, test_ds
    )
    print(f"Test  mAP50-95={test_ap5095:.4f}  mAP50={test_ap50:.4f}  "
          f"inf={test_inf:.1f}s  ({test_ms:.1f}ms/img  {test_fps:.1f}FPS)")

    total_time = time.time() - training_start

    # Save final summary JSON
    summary = {
        "run_name":              RUN_NAME,
        "num_classes":           NUM_CLASS_MODE,
        "params_m":              params_m,
        "gflops":                gflops,
        "amp":                   "BF16",
        "best_val_mAP50_95":     best_ap,
        "test_mAP50_95":         test_ap5095,
        "test_mAP50":            test_ap50,
        "test_inference_time_s": test_inf,
        "test_ms_per_img":       test_ms,
        "test_fps":              test_fps,
        "total_training_time_s": round(total_time, 1),
        "total_training_time_h": round(total_time / 3600, 2),
        "peak_gpu_mem_gb":       round(gpu_mem_gb, 3),
    }
    summary_file = os.path.join(SAVE_DIR, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    if use_wandb:
        wandb.run.summary.update({
            "test/mAP50_95":         test_ap5095,
            "test/mAP50":            test_ap50,
            "test/fps":              test_fps,
            "test/ms_per_img":       test_ms,
            "total_training_time_h": round(total_time / 3600, 2),
            "peak_gpu_mem_gb":       round(gpu_mem_gb, 3),
        })
        wandb.finish()

    print(f"\nTraining complete.")
    print(f"Best val mAP50-95 : {best_ap:.4f}")
    print(f"Test mAP50-95     : {test_ap5095:.4f}")
    print(f"Total time        : {total_time/3600:.2f}h")
    print(f"Log               : {log_file}")