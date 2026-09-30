import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from roboflow import Roboflow
from src.utils.config import ENV_PATH


def download_roboflow_dataset(project_name, version_num, export_format, dest_dir):
    load_dotenv(ENV_PATH)
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("leouie-s-workspace").project(project_name)
    version = project.version(version_num)
    dataset = version.download(export_format, location=str(dest_dir), overwrite=True)

    contents = os.listdir(dataset.location)
    print(f"[download] Saved at: {dataset.location}")
    print(f"[download] Contents: {contents}")

    return dataset


if __name__ == "__main__":
    from src.utils.config import DATA_PROCESSED_DIR
    download_roboflow_dataset(
        "aris-corn-leaf-segmentation", 1, "yolov8",
        DATA_PROCESSED_DIR / "leaf_segment"
    )