import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT as ROOT
    from ultralytics import YOLO

    # Update these to whichever run/weights you're evaluating
    RUN_NAME = "v1-6"
    WEIGHTS_PATH = MODELS_DIR / "leaf_segment" / RUN_NAME / "weights" / "best.pt"
    DATA_YAML = DATA_PROCESSED_DIR / "leaf_segment" / "data.yaml"

    model = YOLO(str(WEIGHTS_PATH))
    metrics = model.val(data=str(DATA_YAML), split="test", plots=False)

    r = metrics.results_dict
    n_images = metrics.nt_per_image.sum() if hasattr(metrics, "nt_per_image") else "N/A"
    n_instances = int(metrics.nt_per_class.sum())

    report = f"""# ARIS - Leaf Segmentation (Stage 2) - Test Set Evaluation

**Model:** YOLOv8n-seg
**Run:** {RUN_NAME}
**Weights:** {WEIGHTS_PATH.relative_to(ROOT)}
**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test set:** {n_images} images, {n_instances} `corn-leaf` instances

## Detection (Bounding Box) Metrics

| Metric | Value |
|---|---|
| Precision | {r['metrics/precision(B)']:.4f} |
| Recall | {r['metrics/recall(B)']:.4f} |
| mAP@50 | {r['metrics/mAP50(B)']:.4f} |
| mAP@50-95 | {r['metrics/mAP50-95(B)']:.4f} |

## Segmentation (Mask) Metrics

| Metric | Value |
|---|---|
| Precision | {r['metrics/precision(M)']:.4f} |
| Recall | {r['metrics/recall(M)']:.4f} |
| mAP@50 | {r['metrics/mAP50(M)']:.4f} |
| mAP@50-95 | {r['metrics/mAP50-95(M)']:.4f} |
"""

    print(report)

    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leaf_segment_{RUN_NAME}_test_metrics.md"
    out_path.write_text(report)
    print(f"Saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()