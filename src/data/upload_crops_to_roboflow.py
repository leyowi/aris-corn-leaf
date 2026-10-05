import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from roboflow import Roboflow
from src.utils.config import DATA_PROCESSED_DIR, ENV_PATH

CROPS_DIR = DATA_PROCESSED_DIR / "disease_classify" / "unlabeled"
PROJECT_NAME = "aris-disease-classify" 


def main():
    load_dotenv(ENV_PATH)
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("leouie-s-workspace").project(PROJECT_NAME)

    for split in ("train", "valid", "test"):
        crops = sorted((CROPS_DIR / split).glob("*.jpg"))
        print(f"Uploading {len(crops)} crops to '{split}'...")
        for crop in crops:
            project.upload(
                image_path=str(crop),
                split=split,
                batch_name=f"leaf-crops-{split}",
                num_retry_uploads=2,
            )

    print("Done.")


if __name__ == "__main__":
    main()