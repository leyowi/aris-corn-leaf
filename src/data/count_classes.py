import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR
from src.data.download_dataset import download_roboflow_dataset

PROJECT_NAME = "aris-disease-classify"
VERSION = 1
DEST = DATA_PROCESSED_DIR / "disease_classify" / "labeled"


def original_name(stem):
    return stem.rsplit("_jpg.rf.", 1)[0]


def main():
    dataset = download_roboflow_dataset(PROJECT_NAME, VERSION, "folder", DEST)
    root = Path(dataset.location)

    counts = defaultdict(lambda: defaultdict(int))
    seen = defaultdict(list)  # original crop name -> [(split, class), ...]

    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for img in class_dir.glob("*.jpg"):
                counts[class_dir.name][split] += 1
                seen[original_name(img.stem)].append((split, class_dir.name))

    print(f"\n{'Class':<28}{'Train':>8}{'Valid':>8}{'Test':>8}{'Total':>8}")
    print("-" * 60)
    totals = defaultdict(int)
    for cls in sorted(counts):
        row = counts[cls]
        t = sum(row.values())
        for s in ("train", "valid", "test"):
            totals[s] += row[s]
        print(f"{cls:<28}{row['train']:>8}{row['valid']:>8}{row['test']:>8}{t:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<28}{totals['train']:>8}{totals['valid']:>8}{totals['test']:>8}{sum(totals.values()):>8}")

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    cross_split = {k: v for k, v in dupes.items() if len({s for s, _ in v}) > 1}

    print(f"\nUnique source crops: {len(seen)}")
    print(f"Crops appearing more than once: {len(dupes)}")
    print(f"Duplicates across DIFFERENT splits (leakage): {len(cross_split)}")
    for name, where in list(cross_split.items())[:10]:
        print(f"  {name}: {where}")


if __name__ == "__main__":
    main()