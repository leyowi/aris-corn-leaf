"""
deployment/onnx_export/export_models.py

Run on the DEV PC (not the Pi). Packs everything the Raspberry Pi 5 needs into
deployment/bundle/:

  models/leaf_detect.onnx      (+ leaf_detect_ncnn_model/   if NCNN export works)
  models/leaf_segment.onnx     (+ leaf_segment_ncnn_model/)
  models/baseline_seg.onnx     (+ baseline_seg_ncnn_model/)
  models/classifier.onnx       Stage 3, MobileNetV2 wide 128x512
  manifest.json                classes, input size, source runs, export date
  test_images/                 the 38 test photos, for the Pi speed test
  reference.json               what the PC pipeline predicts on those photos,
                               so the Pi can check it gives the same answers

Checks done here, before anything goes to the Pi:
  - classifier: ONNX output == PyTorch output
  - preprocessing: the Pi's numpy version == the training (torchvision) version

Run from the project root:
    python deployment/onnx_export/export_models.py
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "deployment" / "edge_inference"))

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR  # noqa: E402
from src.models.disease_classify.common import WIDE_SIZES, build_model, get_transforms  # noqa: E402
from src.pipeline.inference_pipeline import (  # noqa: E402
    DETECT_WEIGHTS, SEGMENT_WEIGHTS, ARISPipeline, pick_best_classifier)
from edge_pipeline import preprocess_crop  # noqa: E402

BASELINE_WEIGHTS = MODELS_DIR / "baseline_seg" / "v1" / "weights" / "best.pt"
TEST_IMAGES = DATA_PROCESSED_DIR / "baseline_seg" / "test" / "images"
BUNDLE = PROJECT_ROOT / "deployment" / "bundle"
IMGSZ = 640


def export_yolo(weights, name, out_models):
    from ultralytics import YOLO

    result = {"source": str(weights.relative_to(PROJECT_ROOT))}
    onnx_path = Path(YOLO(str(weights)).export(format="onnx", imgsz=IMGSZ, simplify=True))
    shutil.copy2(onnx_path, out_models / f"{name}.onnx")
    onnx_path.unlink()                     # keep models/ clean: the copy lives in the bundle
    result["onnx"] = f"models/{name}.onnx"
    print(f"[ok] {name}.onnx")

    try:
        ncnn_dir = Path(YOLO(str(weights)).export(format="ncnn", imgsz=IMGSZ))
        dest = out_models / f"{name}_ncnn_model"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(ncnn_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.rmtree(ncnn_dir)            # keep models/ clean
        result["ncnn"] = f"models/{name}_ncnn_model"
        print(f"[ok] {name}_ncnn_model")
    except Exception as e:  # NCNN is optional; ONNX is enough to run
        print(f"[skip] NCNN export for {name} failed ({e.__class__.__name__}); ONNX will be used")
    return result


def export_classifier(ckpt_path, out_models):
    import onnxruntime as ort

    ckpt = torch.load(ckpt_path, map_location="cpu")
    input_mode = ckpt.get("input_mode", "square")
    h, w = WIDE_SIZES.get(input_mode, (224, 224))
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = out_models / "classifier.onnx"
    dummy = torch.randn(1, 3, h, w)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)

    # Check 1: ONNX gives the same numbers as PyTorch
    x = torch.randn(4, 3, h, w)
    with torch.no_grad():
        ref = model(x).numpy()
    out = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    assert diff < 1e-3, f"ONNX classifier differs from PyTorch (max diff {diff})"
    print(f"[ok] classifier.onnx  (max difference vs PyTorch: {diff:.2e})")

    # Check 2: the Pi's numpy preprocessing matches training preprocessing
    crop = Image.fromarray(np.random.default_rng(0).integers(0, 255, (420, 70, 3), dtype=np.uint8))
    torch_x = get_transforms(train=False, input_mode=input_mode)(crop).numpy()
    numpy_x = preprocess_crop(crop, (h, w))
    pdiff = float(np.abs(torch_x - numpy_x).max())
    assert pdiff < 1e-4, f"Pi preprocessing differs from training preprocessing (max diff {pdiff})"
    print(f"[ok] Pi preprocessing matches training (max difference: {pdiff:.2e})")

    return {"source": str(Path(ckpt_path).relative_to(PROJECT_ROOT)), "onnx": "models/classifier.onnx",
            "arch": ckpt["arch"], "classes": ckpt["classes"], "input_mode": input_mode,
            "input_size_hw": [h, w], "val_macro_f1": round(float(ckpt["val_macro_f1"]), 4)}


def build_reference(clf_path, images):
    """Disease labels the PC pipeline gives on each test photo (for the Pi agreement check)."""
    import cv2

    pipe = ARISPipeline(classifier_path=clf_path)
    ref = {}
    for img_path in images:
        leaves = pipe.predict(cv2.imread(str(img_path)))
        ref[img_path.name] = sorted(l["class_name"] for l in leaves)
    return ref


def main():
    shutil.rmtree(BUNDLE, ignore_errors=True)
    out_models = BUNDLE / "models"
    out_models.mkdir(parents=True)

    clf_path, _ = pick_best_classifier()
    manifest = {
        "exported": datetime.now().isoformat(timespec="seconds"),
        "imgsz": IMGSZ,
        "leaf_detect": export_yolo(DETECT_WEIGHTS, "leaf_detect", out_models),
        "leaf_segment": export_yolo(SEGMENT_WEIGHTS, "leaf_segment", out_models),
        "baseline_seg": export_yolo(BASELINE_WEIGHTS, "baseline_seg", out_models),
        "classifier": export_classifier(clf_path, out_models),
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))

    images = sorted(TEST_IMAGES.glob("*.jpg"))
    (BUNDLE / "test_images").mkdir()
    for p in images:
        shutil.copy2(p, BUNDLE / "test_images" / p.name)
    print(f"[ok] copied {len(images)} test images")

    (BUNDLE / "reference.json").write_text(json.dumps(build_reference(clf_path, images), indent=2))
    print("[ok] reference.json (PC pipeline answers for the agreement check)")

    size_mb = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1e6
    print(f"\nBundle ready: {BUNDLE}  ({size_mb:.1f} MB)")
    print("Copy the whole 'bundle' folder to the Pi: ~/ARIS-CORN/deployment/bundle")


if __name__ == "__main__":
    main()