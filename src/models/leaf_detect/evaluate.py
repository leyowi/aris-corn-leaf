import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT as ROOT
    from ultralytics import YOLO

    RUN_NAME = "v1"
    WEIGHTS_PATH = MODELS_DIR / "leaf_detect" / RUN_NAME / "weights" / "best.pt"
    DATA_YAML = DATA_PROCESSED_DIR / "leaf_segment" / "data.yaml"

    model = YOLO(str(WEIGHTS_PATH))
    metrics = model.val(data=str(DATA_YAML), split="test", plots=False)

    r = metrics.results_dict
    p, rc = r["metrics/precision(B)"], r["metrics/recall(B)"]
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    n_instances = int(metrics.nt_per_class.sum())
    speed = metrics.speed

    report = f"""# ARIS - Leaf Detection (Stage 1) - Test Set Evaluation

**Model:** YOLOv8n (detection)
**Run:** {RUN_NAME}
**Weights:** {WEIGHTS_PATH.relative_to(ROOT)}
**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test set:** {n_instances} `corn-leaf` instances (same split as Stage 2)

## Detection (Bounding Box) Metrics

| Metric | Value |
|---|---|
| Precision | {p:.4f} |
| Recall | {rc:.4f} |
| F1 score | {f1:.4f} |
| mAP@50 | {r['metrics/mAP50(B)']:.4f} |
| mAP@50-95 | {r['metrics/mAP50-95(B)']:.4f} |

## Inference Speed (dev GPU, per image)

| Stage | ms |
|---|---|
| Preprocess | {speed['preprocess']:.2f} |
| Inference | {speed['inference']:.2f} |
| Postprocess | {speed['postprocess']:.2f} |
"""

    print(report)

    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"leaf_detect_{RUN_NAME}_test_metrics.md"
    out_path.write_text(report)
    print(f"Saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()