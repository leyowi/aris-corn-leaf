import argparse
import os
import time
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from torch.utils.data import DataLoader
    from torchvision import datasets

    from common import PROJECT_ROOT, MODELS_DIR, build_model, get_data_root, get_transforms, run_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["mobilenet_v2", "resnet50"], required=True)
    parser.add_argument("--data", choices=["v1", "v2"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", choices=["square", "wide", "wide_hr"], default="square")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = run_name(args.arch, args.data, args.seed, args.input)
    ckpt = torch.load(MODELS_DIR / "disease_classify" / run / "best.pt", map_location=device)
    classes = ckpt["classes"]

    model = build_model(args.arch, len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    test_ds = datasets.ImageFolder(get_data_root(args.data) / "test", get_transforms(train=False, input_mode=ckpt.get("input_mode", "square")))
    assert test_ds.classes == classes, "Test class folders differ from training"
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            labels += y.tolist()

    # Inference speed, batch of 1 (closest to one leaf at a time on the rover)
    x1 = test_ds[0][0].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 100 * 1000

    acc = accuracy_score(labels, preds)
    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=range(len(classes)), zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    params = sum(t.numel() for t in model.parameters()) / 1e6

    rows = "\n".join(
        f"| {c} | {p[i]:.4f} | {r[i]:.4f} | {f1[i]:.4f} | {support[i]} |" for i, c in enumerate(classes)
    )
    data_desc = "field only" if args.data == "v1" else "field + Kaggle (train/valid only)"

    report = f"""# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** {args.arch}
**Training data:** {args.data} ({data_desc})
**Seed:** {args.seed}
**Input:** {args.input} ({ {"square": "224 x 224", "wide": "128 x 512", "wide_hr": "192 x 768"}[args.input] })
**Best epoch:** {ckpt['epoch']} (val macro F1 {ckpt['val_macro_f1']:.4f})
**Evaluated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Test set:** {len(labels)} leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | {acc:.4f} |
| Macro precision | {mp:.4f} |
| Macro recall | {mr:.4f} |
| Macro F1 | {mf1:.4f} |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{rows}

## Model cost

| Item | Value |
|---|---|
| Parameters | {params:.2f} M |
| Inference (dev GPU, batch 1) | {ms:.2f} ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
"""
    print(report)

    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"disease_classify_{run}_test_metrics.md").write_text(report)

    cm = confusion_matrix(labels, preds, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{args.arch} ({args.data}, {args.input}) - field test set")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"disease_classify_{run}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved report and confusion matrix to {out_dir.relative_to(PROJECT_ROOT)}")

    # One row per run, used by summarize_runs.py (re-running replaces the old row)
    import csv
    runs_csv = out_dir / "disease_classify_runs.csv"
    header = ["arch", "data", "seed", "input", "best_epoch", "val_macro_f1", "test_acc",
              "test_macro_precision", "test_macro_recall", "test_macro_f1"] + \
             [f"f1_{c}" for c in classes] + ["params_m", "gpu_ms"]
    row = [args.arch, args.data, args.seed, args.input, ckpt["epoch"], round(ckpt["val_macro_f1"], 4),
           round(acc, 4), round(mp, 4), round(mr, 4), round(mf1, 4)] + \
          [round(v, 4) for v in f1] + [round(params, 2), round(ms, 2)]
    rows = []
    if runs_csv.exists():
        with open(runs_csv, newline="") as fh:
            old = list(csv.reader(fh))
        rows = old[1:]
        if "input" not in old[0]:  # older file without the input column
            rows = [r[:3] + ["square"] + r[3:] for r in rows]
        key = [args.arch, args.data, str(args.seed), args.input]
        rows = [r for r in rows if r[:4] != key]
    rows.append([str(v) for v in row])
    with open(runs_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()