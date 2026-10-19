import argparse
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1-6")
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    args = ap.parse_args()

    weights = MODELS_DIR / "leaf_segment" / args.run / "weights" / "best.pt"
    data_yaml = DATA_PROCESSED_DIR / "leaf_segment" / "data.yaml"
    split = "val" if args.split == "valid" else "test"

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, plots=False,
                               project=str(MODELS_DIR / "leaf_segment"),
                               name=f"{args.run}_{args.split}", exist_ok=True)

    def row(label, res):
        return (f"| {label} | {res.mp:.4f} | {res.mr:.4f} | {f1(res.mp, res.mr):.4f} | "
                f"{res.map50:.4f} | {res.map:.4f} |")

    params = sum(p.numel() for p in YOLO(str(weights)).model.parameters()) / 1e6
    report = f"""# ARIS - Leaf Segmentation (Stage 2) - {args.split.upper()} evaluation

**Run:** {args.run}  ({params:.2f} M parameters)
**Split:** {args.split}
**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Leaves:** {int(m.nt_per_class.sum())}

| Head | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
{row("Box", m.box)}
{row("Mask", m.seg)}

Speed (dev GPU): inference {m.speed['inference']:.2f} ms/image
"""
    print(report)
    out = PROJECT_ROOT / "results" / "metrics" / f"leaf_segment_{args.run}_{args.split}_metrics.md"
    out.write_text(report, encoding="utf-8")
    print(f"Saved to: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()