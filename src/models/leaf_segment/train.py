import argparse
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR
    from src.data.download_dataset import download_roboflow_dataset
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt", help="yolov8n-seg.pt or yolov8s-seg.pt")
    ap.add_argument("--name", default="v1", help="run folder name under models/leaf_segment/")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    data_dir = DATA_PROCESSED_DIR / "leaf_segment"
    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        download_roboflow_dataset("aris-corn-leaf-segmentation", 1, "yolov8", data_dir)

    YOLO(args.model).train(
        data=str(data_yaml),
        imgsz=640,
        epochs=150,
        patience=30,
        batch=args.batch,
        device=0,
        workers=4,
        seed=42,
        project=str(MODELS_DIR / "leaf_segment"),
        name=args.name,
        val=True,
        pretrained=True,
        plots=False,
    )


if __name__ == "__main__":
    main()