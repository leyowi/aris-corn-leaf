import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from common import MODELS_DIR, run_name  # noqa: E402

SEEDS = [42, 7, 123]
SETUPS = [("mobilenet_v2", "train_mobilenet.py"), ("resnet50", "train_resnet.py")]
DATAS = ["v1", "v2"]


def run(cmd):
    print("\n>>> " + " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    py = sys.executable
    for arch, train_script in SETUPS:
        for data in DATAS:
            for seed in SEEDS:
                ckpt = MODELS_DIR / "disease_classify" / run_name(arch, data, seed) / "best.pt"
                if ckpt.exists():
                    print(f"\n[skip training] {ckpt.parent.name} already trained")
                else:
                    run([py, str(HERE / train_script), "--data", data, "--seed", str(seed)])
                run([py, str(HERE / "evaluate.py"), "--arch", arch, "--data", data, "--seed", str(seed)])

    run([py, str(HERE / "summarize_runs.py")])


if __name__ == "__main__":
    main()