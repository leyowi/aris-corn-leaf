import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR

SEED = 42
VALID_FRACTION = 0.20
KAGGLE = DATA_PROCESSED_DIR / "disease_classify" / "kaggle_crops"
CLASSES = ["corn-rust", "gray-leaf-spot", "northern-corn-leaf-blight", "healthy"]


def main():
    random.seed(SEED)
    print(f"{'Class':<28}{'Train':>7}{'Valid':>7}")
    for cls in CLASSES:
        train_dir, valid_dir = KAGGLE / "train" / cls, KAGGLE / "valid" / cls
        valid = sorted(valid_dir.glob("*.jpg"))
        total = len(valid) + len(list(train_dir.glob("*.jpg")))
        target_valid = round(total * VALID_FRACTION)

        random.shuffle(valid)
        for img in valid[target_valid:]:
            shutil.move(str(img), str(train_dir / img.name))

        print(f"{cls:<28}{len(list(train_dir.glob('*.jpg'))):>7}{len(list(valid_dir.glob('*.jpg'))):>7}")


if __name__ == "__main__":
    main()