import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR
    from ultralytics import YOLO

    data_yaml = DATA_PROCESSED_DIR / "baseline_seg" / "data.yaml"
    if not data_yaml.exists():
        sys.exit("Run src/data/build_baseline_dataset.py first.")

    model = YOLO("yolov8n-seg.pt")
    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=150,
        patience=30,
        batch=16,
        device=0,
        workers=4,
        seed=42,
        project=str(MODELS_DIR / "baseline_seg"),
        name="v1",
        val=True,
        pretrained=True,
        plots=False,   # avoids the RAM crash seen at the end of Stage 2 training
    )


if __name__ == "__main__":
    main()