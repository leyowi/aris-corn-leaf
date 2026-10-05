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

    DATA_DIR = DATA_PROCESSED_DIR / "leaf_segment"   
    MODEL_DIR = MODELS_DIR / "leaf_detect"
    data_yaml = DATA_DIR / "data.yaml"

    if not data_yaml.exists():
        download_roboflow_dataset("aris-corn-leaf-segmentation", 1, "yolov8", DATA_DIR)

    model = YOLO("yolov8n.pt")  # detection model, not -seg

    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=150,
        patience=30,
        batch=16,
        device=0,
        workers=4,
        seed=42,
        project=str(MODEL_DIR),
        name="v1",
        val=True,
        pretrained=True,
    )


if __name__ == "__main__":
    main()