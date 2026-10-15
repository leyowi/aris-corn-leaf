import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # limits OpenBLAS memory use

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR  # noqa: E402
from src.data.download_dataset import download_roboflow_dataset  # noqa: E402

ROBOFLOW_PROJECT = "aris-disease-classify"
DATASET_VERSIONS = {"v1": 1, "v2": 2}   # v1 = field only, v2 = field + Kaggle
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class PadToSquare:
    """Pads with black to a square, keeping the leaf's true shape."""

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        left, top = (s - w) // 2, (s - h) // 2
        return TF.pad(img, [left, top, s - w - left, s - h - top], fill=0)


def get_transforms(train):
    steps = [PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15, fill=0),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def get_data_root(data_tag):
    root = DATA_PROCESSED_DIR / "disease_classify" / f"labeled_{data_tag}"
    if not (root / "train").exists():
        download_roboflow_dataset(ROBOFLOW_PROJECT, DATASET_VERSIONS[data_tag], "folder", root)
    return root


def run_name(arch, data_tag, seed):
    """Folder name for a run. Seed 42 keeps the original names (no suffix)."""
    return f"{arch}_{data_tag}" + ("" if seed == 42 else f"_s{seed}")


def build_model(arch, num_classes, pretrained=True):
    if arch == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * x.size(0)
        preds += out.argmax(1).tolist()
        labels += y.tolist()

    n = len(labels)
    acc = sum(p == t for p, t in zip(preds, labels)) / n
    f1 = f1_score(labels, preds, average="macro")
    return total_loss / n, acc, f1, preds, labels


def field_only_f1(dataset, preds, labels):
    """Macro F1 on validation crops that are NOT from Kaggle (diagnostic only)."""
    keep = [i for i, (path, _) in enumerate(dataset.samples) if not Path(path).name.startswith("kaggle_")]
    if not keep:
        return float("nan")
    return f1_score([labels[i] for i in keep], [preds[i] for i in keep], average="macro")


def train(arch, data_tag, epochs=60, batch=32, lr=3e-4, patience=12, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_data_root(data_tag)

    train_ds = datasets.ImageFolder(root / "train", get_transforms(train=True))
    valid_ds = datasets.ImageFolder(root / "valid", get_transforms(train=False))
    assert train_ds.classes == valid_ds.classes, "Class folders differ between splits"
    classes = train_ds.classes

    counts = np.bincount(train_ds.targets, minlength=len(classes))
    weights = torch.tensor(len(train_ds) / (len(classes) * counts), dtype=torch.float32)
    print(f"Classes: {classes}")
    print(f"Train counts: {counts.tolist()} -> class weights: {[round(w, 2) for w in weights.tolist()]}")

    loader_kw = dict(batch_size=batch, num_workers=0, pin_memory=True)
    train_dl = DataLoader(train_ds, shuffle=True, **loader_kw)
    valid_dl = DataLoader(valid_ds, shuffle=False, **loader_kw)

    model = build_model(arch, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = MODELS_DIR / "disease_classify" / run_name(arch, data_tag, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc, va_f1, va_preds, va_labels = run_epoch(model, valid_dl, criterion, device)
        va_field_f1 = field_only_f1(valid_ds, va_preds, va_labels)
        scheduler.step()

        history.append([epoch, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1, va_field_f1])
        flag = ""
        if va_f1 > best_f1:
            best_f1, best_epoch, flag = va_f1, epoch, "  <- best"
            torch.save(
                {"arch": arch, "classes": classes, "img_size": IMG_SIZE, "data_tag": data_tag, "seed": seed,
                 "epoch": epoch, "val_macro_f1": va_f1, "state_dict": model.state_dict()},
                out_dir / "best.pt",
            )
        print(f"Epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} macroF1 {va_f1:.3f} fieldF1 {va_field_f1:.3f}{flag}")

        if epoch - best_epoch >= patience:
            print(f"Early stopping: no val macro F1 improvement for {patience} epochs.")
            break

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_macro_f1",
                    "val_loss", "val_acc", "val_macro_f1", "val_field_macro_f1"])
        w.writerows(history)

    print(f"\nBest val macro F1 {best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved: {out_dir / 'best.pt'}")