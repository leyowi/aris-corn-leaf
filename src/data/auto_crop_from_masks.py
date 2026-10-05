import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR

SRC_DIR = DATA_PROCESSED_DIR / "leaf_segment"
OUT_DIR = DATA_PROCESSED_DIR / "disease_classify" / "unlabeled"
PAD = 10  

def crop_leaf(img, poly_norm):
    h, w = img.shape[:2]
    pts = (np.array(poly_norm).reshape(-1, 2) * [w, h]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)  # background becomes black

    x, y, bw, bh = cv2.boundingRect(pts)
    x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
    x1, y1 = min(x + bw + PAD, w), min(y + bh + PAD, h)
    return masked[y0:y1, x0:x1]


def main():
    rows = []

    for split in ("train", "valid", "test"):
        img_dir = SRC_DIR / split / "images"
        lbl_dir = SRC_DIR / split / "labels"
        out_split = OUT_DIR / split
        out_split.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            lines = lbl_path.read_text().strip().splitlines()

            for i, line in enumerate(lines):
                coords = list(map(float, line.split()))[1:]
                if len(coords) < 6:  # not a polygon
                    continue

                crop = crop_leaf(img, coords)
                crop_name = f"{img_path.stem}_leaf{i}.jpg"
                cv2.imwrite(str(out_split / crop_name), crop)

                rows.append([crop_name, img_path.name, split, i, crop.shape[1], crop.shape[0]])
                count += 1

        print(f"{split}: {count} leaf crops")

    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop", "source_image", "split", "leaf_index", "width", "height"])
        writer.writerows(rows)

    print(f"\nTotal: {len(rows)} crops")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()