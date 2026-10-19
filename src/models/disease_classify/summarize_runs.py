import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]

RUNS_CSV = PROJECT_ROOT / "results" / "metrics" / "disease_classify_runs.csv"
OUT = PROJECT_ROOT / "results" / "comparison_report" / "disease_classify_summary.md"


def fmt(values):
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{st.mean(values):.3f} ± {st.stdev(values):.3f}"


def main():
    if not RUNS_CSV.exists():
        sys.exit(f"No runs found at {RUNS_CSV}. Run evaluate.py first.")

    with open(RUNS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        groups[(r["arch"], r["data"], r.get("input") or "square")].append(r)

    f1_cols = [c for c in rows[0] if c.startswith("f1_")]
    lines = [
        "# ARIS - Disease Classification (Stage 3) - Comparison",
        "",
        "Field-only test set (55 crops). Values are mean ± standard deviation across seeds.",
        "Input: square = pad to 224 x 224; wide = leaf horizontal, fit to 128 x 512; wide_hr = 192 x 768.",
        "v1 = field only, v2 = field + Kaggle (train/valid only).",
        "",
        "## Overall",
        "",
        "| Model | Data | Input | Seeds | Val macro F1 | Test accuracy | Test macro F1 | Params (M) | GPU ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (arch, data, inp), rs in sorted(groups.items()):
        g = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        lines.append(
            f"| {arch} | {data} | {inp} | {len(rs)} | {fmt(g('val_macro_f1'))} | {fmt(g('test_acc'))} | "
            f"{fmt(g('test_macro_f1'))} | {rs[0]['params_m']} | {fmt(g('gpu_ms'))} |"
        )

    lines += ["", "## Per class test F1", "",
              "| Model | Data | Input | " + " | ".join(c[3:] for c in f1_cols) + " |",
              "|---|---|---|" + "---|" * len(f1_cols)]
    for (arch, data, inp), rs in sorted(groups.items()):
        cells = [fmt([float(r[c]) for r in rs]) for c in f1_cols]
        lines.append(f"| {arch} | {data} | {inp} | " + " | ".join(cells) + " |")

    report = "\n".join(lines) + "\n"
    print(report)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"Saved to {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()