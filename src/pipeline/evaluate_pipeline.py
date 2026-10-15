import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR  # noqa: E402
from src.pipeline.inference_pipeline import ARISPipeline, pick_best_classifier  # noqa: E402

TEST_DIR = DATA_PROCESSED_DIR / "baseline_seg" / "test"
BASELINE_WEIGHTS = MODELS_DIR / "baseline_seg" / "v1" / "weights" / "best.pt"
CLASSES = ["corn-rust", "gray-leaf-spot", "healthy", "northern-corn-leaf-blight"]
EVAL_CONF = 0.001       # low threshold for mAP (standard practice)
OP_CONF = 0.25          # operating threshold for P / R / F1 and speed
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def load_ground_truth():
    images = []
    for img_path in sorted((TEST_DIR / "images").glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = []
        lbl = TEST_DIR / "labels" / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().split("\n"):
                parts = line.split()
                if len(parts) < 7:
                    continue
                pts = np.array(list(map(float, parts[1:]))).reshape(-1, 2) * [w, h]
                gts.append({"cls": int(parts[0]), "mask": to_mask(pts, h, w)})
        images.append({"path": img_path, "img": img, "gts": gts})
    return images


def to_mask(polygon, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(polygon, dtype=np.int32)], 1)
    return m.astype(bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def predict_pipeline(pipe, images, conf):
    out = []
    for item in images:
        h, w = item["img"].shape[:2]
        leaves = pipe.predict(item["img"], seg_conf=conf, max_det=100)
        out.append([{"cls": CLASSES.index(l["class_name"]), "conf": l["conf"],
                     "mask": to_mask(l["polygon"], h, w)} for l in leaves])
    return out


def predict_baseline(model, images, conf):
    out = []
    for item in images:
        h, w = item["img"].shape[:2]
        r = model.predict(item["img"], conf=conf, max_det=100, verbose=False)[0]
        preds = []
        if r.masks is not None:
            for poly, c, k in zip(r.masks.xy, r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                if len(poly) >= 3:
                    preds.append({"cls": int(k), "conf": float(c), "mask": to_mask(poly, h, w)})
        out.append(preds)
    return out


def iou_tables(images, preds):
    """IoU between every prediction and every ground truth, per image."""
    return [np.array([[mask_iou(p["mask"], g["mask"]) for g in item["gts"]] for p in ps])
            .reshape(len(ps), len(item["gts"]))
            for item, ps in zip(images, preds)]


def match_class(images, preds, ious, cls, thr, min_conf=0.0):
    """Greedy matching by confidence (COCO style). Returns (sorted tp flags, n_gt)."""
    dets = [(p["conf"], i, j) for i, ps in enumerate(preds) for j, p in enumerate(ps)
            if p["cls"] == cls and p["conf"] >= min_conf]
    dets.sort(key=lambda d: -d[0])
    n_gt = sum(g["cls"] == cls for item in images for g in item["gts"])
    used = [set() for _ in images]
    tp = []
    for _, i, j in dets:
        best, best_k = thr, -1
        for k, g in enumerate(images[i]["gts"]):
            if g["cls"] != cls or k in used[i]:
                continue
            if ious[i][j, k] >= best:
                best, best_k = ious[i][j, k], k
        if best_k >= 0:
            used[i].add(best_k)
            tp.append(1)
        else:
            tp.append(0)
    return np.array(tp), n_gt


def average_precision(tp, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    ap = 0.0
    for r in np.linspace(0, 1, 101):                      # 101-point interpolation
        p = precision[recall >= r]
        ap += (p.max() if p.size else 0.0) / 101
    return ap


def score(images, preds):
    ious = iou_tables(images, preds)
    rows = {}
    for c, name in enumerate(CLASSES):
        ap50 = average_precision(*match_class(images, preds, ious, c, 0.5))
        ap5095 = np.nanmean([average_precision(*match_class(images, preds, ious, c, t)) for t in IOU_THRESHOLDS])
        tp, n_gt = match_class(images, preds, ious, c, 0.5, min_conf=OP_CONF)
        if n_gt == 0:                                     # class absent: leave out of the means
            rows[name] = {"P": np.nan, "R": np.nan, "F1": np.nan, "AP50": np.nan, "AP5095": np.nan, "n": 0}
            continue
        p = tp.sum() / len(tp) if len(tp) else 0.0
        r = tp.sum() / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows[name] = {"P": p, "R": r, "F1": f1, "AP50": ap50, "AP5095": ap5095, "n": n_gt}

    found = correct = total = 0
    for item, ps, iou in zip(images, preds, ious):
        for k, g in enumerate(item["gts"]):
            total += 1
            cands = [(p["conf"], p["cls"]) for j, p in enumerate(ps)
                     if p["conf"] >= OP_CONF and iou[j, k] >= 0.5]
            if cands:
                found += 1
                correct += int(max(cands)[1] == g["cls"])
    leaf = {"found": found / total, "acc_found": correct / found if found else 0.0,
            "found_correct": correct / total, "total": total}
    return rows, leaf


def mean_of(rows, key):
    return float(np.nanmean([r[key] for r in rows.values()]))


def time_per_image(fn, images, repeats=2):
    for item in images[:3]:
        fn(item["img"])                                  # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        for item in images:
            fn(item["img"])
    return (time.perf_counter() - t0) / (repeats * len(images)) * 1000


def main():
    from ultralytics import YOLO

    images = load_ground_truth()
    print(f"Test images: {len(images)}, leaves: {sum(len(i['gts']) for i in images)}")

    clf_path, clf_val = pick_best_classifier()
    print(f"Stage 3 classifier: {clf_path.parent.name} (val macro F1 {clf_val:.4f})")
    pipe = ARISPipeline(classifier_path=clf_path)
    baseline = YOLO(str(BASELINE_WEIGHTS))

    print("Running pipeline...")
    pipe_rows, pipe_leaf = score(images, predict_pipeline(pipe, images, EVAL_CONF))
    print("Running baseline...")
    base_rows, base_leaf = score(images, predict_baseline(baseline, images, EVAL_CONF))

    print("Timing...")
    pipe_ms = time_per_image(lambda im: pipe.predict(im, seg_conf=OP_CONF), images)
    base_ms = time_per_image(lambda im: baseline.predict(im, conf=OP_CONF, verbose=False), images)

    def overall(rows, leaf, ms):
        return (f"{mean_of(rows, 'P'):.4f} | {mean_of(rows, 'R'):.4f} | {mean_of(rows, 'F1'):.4f} | "
                f"{mean_of(rows, 'AP50'):.4f} | {mean_of(rows, 'AP5095'):.4f} | "
                f"{leaf['found']:.4f} | {leaf['acc_found']:.4f} | {leaf['found_correct']:.4f} | {ms:.1f}")

    def per_class(rows):
        return "\n".join(f"| {n} | {r['n']} | {r['P']:.4f} | {r['R']:.4f} | {r['F1']:.4f} | "
                         f"{r['AP50']:.4f} | {r['AP5095']:.4f} |" for n, r in rows.items())

    head_c = "| Class | Leaves | Precision | Recall | F1 | mAP@50 | mAP@50-95 |\n|---|---|---|---|---|---|---|"
    report = f"""# ARIS - 3-stage pipeline vs single-stage baseline (end to end)

**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test set:** {len(images)} raw field photos, {pipe_leaf['total']} labeled leaves
**Pipeline:** leaf_detect v1 (gate) -> leaf_segment v1-6 -> {clf_path.parent.name}
**Baseline:** baseline_seg v1 (YOLOv8n-seg, 4 disease classes)
**Scoring:** identical code for both systems, mask IoU. P/R/F1 and leaf-level at confidence {OP_CONF}.

## Overall

| System | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Leaves found | Correct (of found) | Found AND correct | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| 3-stage pipeline | {overall(pipe_rows, pipe_leaf, pipe_ms)} |
| Single-stage baseline | {overall(base_rows, base_leaf, base_ms)} |

## Per class - 3-stage pipeline

{head_c}
{per_class(pipe_rows)}

## Per class - single-stage baseline

{head_c}
{per_class(base_rows)}

Notes: precision/recall/F1/mAP are means over the 4 classes. "Leaves found" = a
prediction overlaps the leaf with mask IoU >= 0.5. Speed is measured on the dev GPU
at the operating threshold; Raspberry Pi 5 timing is measured separately.
"""
    print(report)
    out = PROJECT_ROOT / "results" / "comparison_report" / "pipeline_vs_baseline.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved to: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()