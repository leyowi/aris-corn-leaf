import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT as ROOT
    from ultralytics import YOLO

    RUN_NAME = "v1-6"
    WEIGHTS_PATH = MODELS_DIR / "leaf_segment" / RUN_NAME / "weights" / "best.pt"
    TEST_IMAGES_DIR = DATA_PROCESSED_DIR / "leaf_segment" / "test" / "images"

    candidates = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No images found in {TEST_IMAGES_DIR}")

    image_path = random.choice(candidates)
    print(f"Testing on: {image_path.name}")

    out_dir = ROOT / "results" / "predictions"

    model = YOLO(str(WEIGHTS_PATH))
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        save=True,
        project=str(out_dir),
        name="run",
        exist_ok=True,  # overwrite the same folder every run
    )

    for r in results:
        n = len(r.boxes) if r.boxes is not None else 0
        print(f"Detected {n} corn-leaf instance(s)")
        if n:
            for i, c in enumerate(r.boxes.conf.tolist(), 1):
                print(f"  Leaf {i}: confidence {c:.3f}")

    print(f"Annotated image saved to: {out_dir / 'run'}")


if __name__ == "__main__":
    main()