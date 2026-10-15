import subprocess
import sys
from pathlib import Path

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
    ("mobilenet_v2", "v1", "wide"),     # new: 128 x 512 input
]


def run(cmd):
    print("\n>>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    py = sys.executable
    for arch, data, inp in EXPERIMENTS:
        for seed in SEEDS:
            ckpt = MODELS_DIR / "disease_classify" / run_name(arch, data, seed, inp) / "best.pt"
            if ckpt.exists():
                print(f"\n[skip training] {ckpt.parent.name} already trained")
            else:
                run([py, str(HERE / SCRIPTS[arch]), "--data", data, "--seed", str(seed), "--input", inp])
            run([py, str(HERE / "evaluate.py"), "--arch", arch, "--data", data,
                 "--seed", str(seed), "--input", inp])

    run([py, str(HERE / "summarize_runs.py")])


if __name__ == "__main__":
    main()