import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR

SEED = 42
BASE = DATA_PROCESSED_DIR / "disease_classify"
KAGGLE = BASE / "kaggle_crops"
UNUSED = BASE / "kaggle_crops_unused"
LOGS = BASE / "kaggle_crops_logs"
CLASSES = ["corn-rust", "gray-leaf-spot", "northern-corn-leaf-blight", "healthy"]


def main():
    random.seed(SEED)

    # Move CSV logs out of the upload folder
    LOGS.mkdir(parents=True, exist_ok=True)
    for csv_file in KAGGLE.glob("*.csv"):
        shutil.move(str(csv_file), str(LOGS / csv_file.name))

    print(f"{'Split':<8}{'Class':<28}{'Before':>8}{'After':>8}")
    print("-" * 52)

    for split in ("train", "valid"):
        files = {c: sorted((KAGGLE / split / c).glob("*.jpg")) for c in CLASSES}
        cap = min(len(v) for v in files.values())

        for cls, imgs in files.items():
            random.shuffle(imgs)
            extra = imgs[cap:]
            dest = UNUSED / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img in extra:
                shutil.move(str(img), str(dest / img.name))
            print(f"{split:<8}{cls:<28}{len(imgs):>8}{cap:>8}")
        print("-" * 52)

    print(f"\nKept crops:   {KAGGLE}")
    print(f"Moved extras: {UNUSED}")
    print(f"CSV logs:     {LOGS}")


if __name__ == "__main__":
    main()