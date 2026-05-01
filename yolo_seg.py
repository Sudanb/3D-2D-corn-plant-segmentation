import torch
from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"GPU            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

    # Train
    model = YOLO("yolo26m-seg.pt")

    results = model.train(
        data    = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt\maize.yaml",
        epochs  = 100,
        patience= 20,
        imgsz   = 640,
        batch   = 16,
        device  = 0,
        workers = 0,
        project = "maize_runs",
        name    = "yolo26m_pipeline_test",
        cache   = False,
    )

    # Evaluate on test split using best weights
    save_dir  = Path(results.save_dir)
    best_ckpt = save_dir / "weights" / "best.pt"

    print(f"Loading weights from: {best_ckpt}")
    best_model = YOLO(str(best_ckpt))

    metrics = best_model.val(
        data    = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt\maize.yaml",
        split   = "test",
        imgsz   = 640,
        workers = 0,
    )

    print("\n========== TEST RESULTS ==========")
    print(f"mAP50      : {metrics.seg.map50:.4f}")
    print(f"mAP50-95   : {metrics.seg.map:.4f}")
    print(f"Precision  : {metrics.seg.mp:.4f}")
    print(f"Recall     : {metrics.seg.mr:.4f}")