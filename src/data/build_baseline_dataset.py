import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR  # noqa: E402

SEG_DIR = DATA_PROCESSED_DIR / "leaf_segment"
LABELED_DIR = DATA_PROCESSED_DIR / "disease_classify" / "labeled_v1"
OUT_DIR = DATA_PROCESSED_DIR / "baseline_seg"
SPLITS = ("train", "valid", "test")


def original_crop_name(stem):
    # Roboflow adds "_jpg.rf.<hash>" on upload; strip only the last one
    return stem.rsplit("_jpg.rf.", 1)[0]


def norm(name):
    """Reduces a name to the original camera filename, e.g.
    '20260831_100457_jpg.rf.d3e3...' -> '20260831_100457'.
    Roboflow adds or rewrites the '_jpg.rf.<code>' part, so it is ignored."""
    name = re.sub(r"[^A-Za-z0-9]", "_", name)
    return re.split(r"_(?:jpg|jpeg|png)_rf_", name, flags=re.IGNORECASE)[0]


def load_crop_labels(classes):
    """Maps (source image stem, polygon index) -> (class id, split)."""
    lookup = {}
    for split in SPLITS:
        for cls_id, cls in enumerate(classes):
            for img in (LABELED_DIR / split / cls).glob("*.jpg"):
                name = original_crop_name(img.stem)          # <image stem>_leaf<i>
                src_stem, idx = name.rsplit("_leaf", 1)
                lookup[(norm(src_stem), int(idx))] = (cls_id, split)
    return lookup


def main():
    classes = sorted(p.name for p in (LABELED_DIR / "train").iterdir() if p.is_dir())
    lookup = load_crop_labels(classes)
    print(f"Classes: {classes}")
    print(f"Labeled leaf crops found: {len(lookup)}")

    if OUT_DIR.exists():
        for split in SPLITS:
            shutil.rmtree(OUT_DIR / split, ignore_errors=True)

    seen_bases = {}
    for split in SPLITS:
        for img in (SEG_DIR / split / "images").glob("*.jpg"):
            seen_bases.setdefault(norm(img.stem), []).append(img.name)
    clashes = {k: v for k, v in seen_bases.items() if len(v) > 1}
    if clashes:
        sys.exit(f"STOP: {len(clashes)} camera filenames appear more than once, "
                 f"matching would be ambiguous. Example: {next(iter(clashes.values()))}")

    used = set()
    counts = {s: [0] * len(classes) for s in SPLITS}
    dropped, split_mismatch, n_images = 0, 0, {s: 0 for s in SPLITS}

    for split in SPLITS:
        img_out = OUT_DIR / split / "images"
        lbl_out = OUT_DIR / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in sorted((SEG_DIR / split / "images").glob("*.jpg")):
            lbl = SEG_DIR / split / "labels" / f"{img.stem}.txt"
            lines = lbl.read_text().strip().splitlines() if lbl.exists() else []

            new_lines = []
            for i, line in enumerate(lines):
                parts = line.split()
                hit = lookup.get((norm(img.stem), i))
                if hit is None:
                    dropped += 1
                    continue
                cls_id, crop_split = hit
                if crop_split != split:
                    split_mismatch += 1
                used.add((norm(img.stem), i))
                counts[split][cls_id] += 1
                new_lines.append(" ".join([str(cls_id)] + parts[1:]))

            shutil.copy2(img, img_out / img.name)
            (lbl_out / f"{img.stem}.txt").write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            n_images[split] += 1

    yaml = [f"path: {OUT_DIR.as_posix()}", "train: train/images", "val: valid/images",
            "test: test/images", f"nc: {len(classes)}", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml) + "\n")

    print(f"\n{'Split':<8}{'Images':>8}" + "".join(f"{c[:14]:>16}" for c in classes))
    for s in SPLITS:
        print(f"{s:<8}{n_images[s]:>8}" + "".join(f"{n:>16}" for n in counts[s]))

    unused = len(lookup) - len(used)
    match_rate = len(used) / max(len(lookup), 1)
    print(f"\nPolygons dropped (no disease label): {dropped}")
    print(f"Labeled crops not matched to a polygon: {unused}")
    print(f"Crops whose split differs from their image: {split_mismatch} (must be 0)")
    print(f"Saved to: {OUT_DIR}")

    if match_rate < 0.95 or split_mismatch:
        example_crop = next(iter(lookup))[0] if lookup else "-"
        example_img = next((SEG_DIR / "train" / "images").glob("*.jpg"), Path("-")).stem
        sys.exit(
            f"\nSTOP: only {match_rate:.0%} of crops matched a polygon "
            f"(or splits differ). Do NOT train on this.\n"
            f"  example crop source name: {example_crop}\n"
            f"  example image name:       {norm(example_img)}\n"
            f"Send these two lines to fix the matching."
        )
    print(f"Match rate: {match_rate:.1%} - OK to train.")


if __name__ == "__main__":
    main()