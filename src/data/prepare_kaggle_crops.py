import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- settings ---------------------------------------------------------------
TARGET_PER_CLASS = 600        # field + Kaggle, train + valid combined, per class
VALID_FRACTION = 0.20         # matches the field data's train/valid ratio
CONF = 0.40                   # min confidence for the leaf mask
MIN_AREA_RATIO = 0.05         # reject masks smaller than 5% of the image
DUP_HAMMING = 4               # perceptual hash distance treated as duplicate
PAD = 10
SEED = 42
KAGGLE_ID = "smaranjitghose/corn-or-maize-leaf-disease-dataset"
SEG_RUN = "v1-6"

CLASS_KEYWORDS = {
    "corn-rust": ["rust"],
    "gray-leaf-spot": ["gray", "grey"],
    "northern-corn-leaf-blight": ["blight"],
    "healthy": ["healthy"],
}
EXTS = {".jpg", ".jpeg", ".png"}


def find_kaggle_images(root):
    found = {c: set() for c in CLASS_KEYWORDS}
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        name = d.name.lower()
        imgs = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
        if not imgs:
            continue
        for cls, kws in CLASS_KEYWORDS.items():
            if any(k in name for k in kws):
                found[cls].update(imgs)
    return {c: sorted(v) for c, v in found.items()}


def dedupe(paths):
    import imagehash
    from PIL import Image

    kept, hashes = [], []
    for p in paths:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            continue
        if any(h - k <= DUP_HAMMING for k in hashes):
            continue
        hashes.append(h)
        kept.append(p)
    return kept


def count_field(labeled_dir):
    counts = {}
    for cls in CLASS_KEYWORDS:
        counts[cls] = sum(
            len(list((labeled_dir / s / cls).glob("*.jpg"))) for s in ("train", "valid")
        )
    return counts


def crop_with_model(model, img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "unreadable"
    r = model.predict(str(img_path), conf=CONF, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return None, "no_leaf_detected"

    best = int(r.boxes.conf.argmax())
    pts = r.masks.xy[best].astype(np.int32)
    h, w = img.shape[:2]
    if cv2.contourArea(pts) < MIN_AREA_RATIO * h * w:
        return None, "mask_too_small"

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    x, y, bw, bh = cv2.boundingRect(pts)
    x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
    x1, y1 = min(x + bw + PAD, w), min(y + bh + PAD, h)
    return masked[y0:y1, x0:x1], "ok"


def main():
    import kagglehub
    from ultralytics import YOLO
    from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR

    random.seed(SEED)
    out_dir = DATA_PROCESSED_DIR / "disease_classify" / "kaggle_crops"
    labeled_dir = DATA_PROCESSED_DIR / "disease_classify" / "labeled"

    print("Downloading Kaggle dataset...")
    kaggle_root = Path(kagglehub.dataset_download(KAGGLE_ID))
    print(f"Kaggle files at: {kaggle_root}")

    images = find_kaggle_images(kaggle_root)
    field = count_field(labeled_dir)
    model = YOLO(str(MODELS_DIR / "leaf_segment" / SEG_RUN / "weights" / "best.pt"))

    manifest, rejected = [], []
    summary = []

    for cls, paths in images.items():
        raw = len(paths)
        paths = dedupe(paths)
        random.shuffle(paths)
        need = max(0, TARGET_PER_CLASS - field[cls])
        n_valid_target = round(need * VALID_FRACTION)
        print(f"\n{cls}: {raw} raw, {len(paths)} after dedupe, field={field[cls]}, need={need}")

        accepted = 0
        for p in paths:
            if accepted >= need:
                break
            crop, status = crop_with_model(model, p)
            if crop is None:
                rejected.append([cls, str(p), status])
                continue
            split = "valid" if accepted < n_valid_target else "train"
            dest = out_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            name = f"kaggle_{cls}_{accepted:04d}.jpg"
            cv2.imwrite(str(dest / name), crop)
            manifest.append([name, cls, split, str(p)])
            accepted += 1

        summary.append((cls, raw, len(paths), field[cls], need, accepted))
        if accepted < need:
            print(f"  WARNING: only {accepted}/{need} usable crops for {cls}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop", "class", "split", "kaggle_source"])
        w.writerows(manifest)
    with open(out_dir / "rejected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "kaggle_source", "reason"])
        w.writerows(rejected)

    print(f"\n{'Class':<28}{'Raw':>6}{'Dedup':>7}{'Field':>7}{'Need':>6}{'Added':>7}")
    print("-" * 61)
    for cls, raw, dd, fld, need, acc in summary:
        print(f"{cls:<28}{raw:>6}{dd:>7}{fld:>7}{need:>6}{acc:>7}")
    print(f"\nRejected by the segment model: {len(rejected)} (see rejected.csv)")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()