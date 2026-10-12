import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from roboflow import Roboflow
from src.utils.config import DATA_PROCESSED_DIR, ENV_PATH

PROJECT_NAME = "aris-disease-classify"
KAGGLE_DIR = DATA_PROCESSED_DIR / "disease_classify" / "kaggle_crops"


def main():
    load_dotenv(ENV_PATH)
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    ws = rf.workspace("leouie-s-workspace")

    ws.upload_dataset(
        str(KAGGLE_DIR),
        PROJECT_NAME,
        num_workers=4,
        project_type="single-label-classification",
        batch_name="kaggle-crops",
    )
    print("Upload finished.")


if __name__ == "__main__":
    main()