import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR  # noqa: E402
from src.pipeline.inference_pipeline import ARISPipeline, pick_best_classifier  # noqa: E402
from src.pipeline.evaluate_pipeline import load_ground_truth, mask_iou, to_mask  # noqa: E402

VAL_CROPS = DATA_PROCESSED_DIR / "disease_classify" / "labeled_v1" / "valid"
VAL_PHOTOS = DATA_PROCESSED_DIR / "baseline_seg" / "valid"
OUT = PROJECT_ROOT / "results" / "error_analysis"
OP_CONF = 0.25
SHORT = {"corn-rust": "RUST", "gray-leaf-spot": "GLS", "healthy": "HEALTHY",
         "northern-corn-leaf-blight": "NCLB"}


def contact_sheet(paths, out_path, thumb=180, cols=5):
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb, thumb))
        x, y = (n % cols) * thumb, (n // cols) * (thumb + 18)
        sheet.paste(im, (x + (thumb - im.width) // 2, y + (thumb - im.height) // 2))
        draw.text((x + 4, y + thumb + 2), p.name[:28], fill="black")
    sheet.save(out_path, quality=90)


# ------------------------------------------------------------------ Part A
@torch.no_grad()
def classifier_errors(pipe):
    out_dir = OUT / "classifier_val"
    classes = pipe.classes
    samples = [(p, classes.index(c.name)) for c in sorted(VAL_CROPS.iterdir()) if c.is_dir()
               for p in sorted(c.glob("*.jpg"))]

    preds, probs = [], []
    for i in range(0, len(samples), 32):
        batch = torch.stack([pipe.transform(Image.open(p).convert("RGB")) for p, _ in samples[i:i + 32]])
        pr = torch.softmax(pipe.classifier(batch.to(pipe.device)), 1).cpu()
        conf, k = pr.max(1)
        preds += k.tolist()
        probs += conf.tolist()

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    groups, review = {}, []
    for (path, t), k, c in zip(samples, preds, probs):
        cm[t, k] += 1
        if t != k:
            name = f"true_{classes[t]}__pred_{classes[k]}"
            dest = out_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            new = dest / f"{c:.2f}_{path.name}"
            shutil.copy2(path, new)
            groups.setdefault(name, []).append(new)
            review.append([new.relative_to(OUT).as_posix(), classes[t], classes[k], f"{c:.2f}", "", ""])

    for name, paths in groups.items():
        contact_sheet(sorted(paths), out_dir / f"{name}.jpg")
    return classes, cm, groups, review


# ------------------------------------------------------------------ Part B
def pipeline_errors(pipe, classes):
    out_dir = OUT / "pipeline_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_count = Counter()
    colors = {"correct": (0, 200, 0), "wrong": (0, 0, 255), "missed": (0, 220, 255)}  # BGR

    for item in load_ground_truth(VAL_PHOTOS):
        img = item["img"]
        h, w = img.shape[:2]
        leaves = pipe.predict(img, seg_conf=OP_CONF)
        pred_masks = [to_mask(l["polygon"], h, w) for l in leaves]

        canvas, has_error = img.copy(), False
        for g in item["gts"]:
            hits = [(l["conf"], l["class_name"]) for l, m in zip(leaves, pred_masks) if mask_iou(m, g["mask"]) >= 0.5]
            true_name = classes[g["cls"]]
            if not hits:
                status, text = "missed", f"MISSED {SHORT[true_name]}"
            else:
                pred_name = max(hits)[1]
                status = "correct" if pred_name == true_name else "wrong"
                text = SHORT[true_name] if status == "correct" else f"{SHORT[true_name]}->{SHORT[pred_name]}"
            status_count[(status, true_name)] += 1
            has_error |= status != "correct"

            contours, _ = cv2.findContours(g["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colors[status], 3)
            x, y, _, _ = cv2.boundingRect(contours[0]) if contours else (0, 0, 0, 0)
            cv2.putText(canvas, text, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(canvas, text, (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[status], 2)

        if has_error:
            cv2.imwrite(str(out_dir / item["path"].name), canvas)
    return status_count


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    clf_path, clf_val = pick_best_classifier()
    pipe = ARISPipeline(classifier_path=clf_path)
    print(f"Classifier: {clf_path.parent.name} (val macro F1 {clf_val:.4f})")

    print("Part A: classifier on validation crops...")
    classes, cm, groups, review = classifier_errors(pipe)
    print("Part B: pipeline on validation photos...")
    status = pipeline_errors(pipe, classes)

    with open(OUT / "review_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "labeled_as", "model_says", "model_confidence", "farmer_true_label", "notes"])
        w.writerows(review)

    hdr = "| True vs Predicted | " + " | ".join(SHORT[c] for c in classes) + " | Recall |"
    sep = "|---" * (len(classes) + 2) + "|"
    cm_rows = "\n".join(
        f"| {SHORT[c]} | " + " | ".join(str(v) for v in cm[i]) + f" | {cm[i, i] / max(cm[i].sum(), 1):.2f} |"
        for i, c in enumerate(classes))
    err_rows = "\n".join(f"| {n.replace('__', ' -> ').replace('true_', '').replace('pred_', '')} | {len(v)} |"
                         for n, v in sorted(groups.items(), key=lambda kv: -len(kv[1])))
    pipe_rows = "\n".join(
        f"| {SHORT[c]} | {status[('correct', c)]} | {status[('wrong', c)]} | {status[('missed', c)]} |"
        for c in classes)
    tot = {s: sum(v for (k, _), v in status.items() if k == s) for s in ("correct", "wrong", "missed")}

    report = f"""# ARIS - Error analysis (VALIDATION set)

Classifier: {clf_path.parent.name} (val macro F1 {clf_val:.4f})

## Part A - Stage 3 on validation leaf crops ({cm.sum()} crops)

{hdr}
{sep}
{cm_rows}

### Most common mistakes

| Mistake (true -> predicted) | Crops |
|---|---|
{err_rows}

Wrong crops and contact sheets: `results/error_analysis/classifier_val/`

## Part B - full pipeline on validation photos (confidence {OP_CONF})

| Class | Correct | Wrong disease | Missed |
|---|---|---|---|
{pipe_rows}
| **Total** | **{tot['correct']}** | **{tot['wrong']}** | **{tot['missed']}** |

Photos with mistakes (green correct, red wrong, yellow missed): `results/error_analysis/pipeline_val/`
"""
    (OUT / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()