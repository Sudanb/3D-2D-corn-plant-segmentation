from ultralytics import YOLO

BASE_DATA_2CLS  = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt\maize.yaml"
BASE_DATA_17CLS = r"C:\Users\sudanb\Desktop\CV_datasets\training_data_fmt_17\maize.yaml"

RUNS = [
    ("yolov8m-seg.pt",  BASE_DATA_2CLS,  "yolov8m_2cls_10ep"),
    ("yolov8m-seg.pt",  BASE_DATA_17CLS, "yolov8m_17cls_10ep"),
    ("yolo26m-seg.pt",  BASE_DATA_2CLS,  "yolo26m_2cls_10ep"),
    ("yolo26m-seg.pt",  BASE_DATA_17CLS, "yolo26m_17cls_10ep"),
]

if __name__ == "__main__":
    for model_weights, data_yaml, run_name in RUNS:
        print(f"\n{'='*60}\nStarting: {run_name}\n{'='*60}")
        YOLO(model_weights).train(
            data=data_yaml,
            epochs=10,
            batch=16,
            imgsz=640,
            device=0,
            workers=0,
            project="runs/segment/maize_runs",
            name=run_name,
        )
