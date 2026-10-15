import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR
    from ultralytics import YOLO

    RUN_NAME = "v1"   # update if training created v1-2, v1-3, ...
    weights = MODELS_DIR / "baseline_seg" / RUN_NAME / "weights" / "best.pt"
    data_yaml = DATA_PROCESSED_DIR / "baseline_seg" / "data.yaml"

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split="test", plots=False,
                  project=str(MODELS_DIR / "baseline_seg"), name=f"{RUN_NAME}_test", exist_ok=True)

    names = m.names
    support = m.nt_per_class

    def overall(res):
        p, r, m50, m5095 = res.mp, res.mr, res.map50, res.map
        return f"| {p:.4f} | {r:.4f} | {f1(p, r):.4f} | {m50:.4f} | {m5095:.4f} |"

    def per_class(res):
        rows = []
        for i, c in enumerate(m.ap_class_index):
            p, r, m50, m5095 = res.class_result(i)
            rows.append(f"| {names[c]} | {p:.4f} | {r:.4f} | {f1(p, r):.4f} | {m50:.4f} | {m5095:.4f} | {support[c]} |")
        return "\n".join(rows)

    head = "| Precision | Recall | F1 | mAP@50 | mAP@50-95 |\n|---|---|---|---|---|"
    head_c = "| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Instances |\n|---|---|---|---|---|---|---|"

    report = f"""# ARIS - Baseline (single-stage YOLOv8n-seg, 4 disease classes) - Test Set Evaluation

**Run:** {RUN_NAME}
**Weights:** {weights.relative_to(PROJECT_ROOT)}
**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test set:** same field test images as Stages 1-3, {int(support.sum())} leaf instances

## Overall - Mask (segmentation)

{head}
{overall(m.seg)}

## Overall - Box (detection)

{head}
{overall(m.box)}

## Per class - Mask

{head_c}
{per_class(m.seg)}

## Per class - Box

{head_c}
{per_class(m.box)}

## Speed (dev GPU, per image)

| Preprocess | Inference | Postprocess |
|---|---|---|
| {m.speed['preprocess']:.2f} ms | {m.speed['inference']:.2f} ms | {m.speed['postprocess']:.2f} ms |
"""
    print(report)
    out = PROJECT_ROOT / "results" / "metrics" / f"baseline_seg_{RUN_NAME}_test_metrics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved to: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()