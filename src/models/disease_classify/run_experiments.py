import argparse
import statistics as st
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from common import MODELS_DIR, run_name  # noqa: E402

SEEDS = [42, 7, 123]
SCRIPTS = {"mobilenet_v2": "train_mobilenet.py", "resnet50": "train_resnet.py"}

# (model, data, input). Already-trained runs are skipped, only evaluated again.
EXPERIMENTS = [
    ("mobilenet_v2", "v1", "square"),
    ("mobilenet_v2", "v2", "square"),
    ("resnet50", "v1", "square"),
    ("resnet50", "v2", "square"),
    ("mobilenet_v2", "v1", "wide"),     # 128 x 512 input
    ("mobilenet_v2", "v1", "wide_hr"),  # new: 192 x 768 input
]


def run(cmd):
    print("\n>>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def validation_table():
    """Mean +/- std of VALIDATION macro F1 per setup, read from the saved checkpoints."""
    print("\n## Validation comparison (decide with this table only)\n")
    print("| Model | Data | Input | Seeds | Val macro F1 | Best epochs |")
    print("|---|---|---|---|---|---|")
    for arch, data, inp in EXPERIMENTS:
        vals, epochs = [], []
        for seed in SEEDS:
            ckpt = MODELS_DIR / "disease_classify" / run_name(arch, data, seed, inp) / "best.pt"
            if ckpt.exists():
                c = torch.load(ckpt, map_location="cpu")
                vals.append(c["val_macro_f1"])
                epochs.append(c["epoch"])
        if vals:
            spread = f" ± {st.stdev(vals):.3f}" if len(vals) > 1 else ""
            print(f"| {arch} | {data} | {inp} | {len(vals)} | {st.mean(vals):.3f}{spread} | {epochs} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-only", action="store_true",
                    help="train missing runs and compare on validation; do not touch the test set")
    args = ap.parse_args()

    py = sys.executable
    for arch, data, inp in EXPERIMENTS:
        for seed in SEEDS:
            ckpt = MODELS_DIR / "disease_classify" / run_name(arch, data, seed, inp) / "best.pt"
            if ckpt.exists():
                print(f"\n[skip training] {ckpt.parent.name} already trained")
            else:
                run([py, str(HERE / SCRIPTS[arch]), "--data", data, "--seed", str(seed), "--input", inp])
            if not args.val_only:
                run([py, str(HERE / "evaluate.py"), "--arch", arch, "--data", data,
                     "--seed", str(seed), "--input", inp])

    if args.val_only:
        validation_table()
    else:
        run([py, str(HERE / "summarize_runs.py")])


if __name__ == "__main__":
    main()