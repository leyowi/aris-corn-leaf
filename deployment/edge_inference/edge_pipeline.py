"""
deployment/edge_inference/edge_pipeline.py

The ARIS pipeline for the Raspberry Pi 5, using the exported models in
deployment/bundle/. Same logic as src/pipeline/inference_pipeline.py:

  Stage 1  leaf_detect   is there a corn leaf in the frame? (cheap gate)
  Stage 2  leaf_segment  exact mask of every leaf
  Stage 3  classifier    disease of every leaf (ONNX Runtime, no PyTorch needed)

Usage:
    from edge_pipeline import EdgePipeline
    pipe = EdgePipeline(fmt="ncnn")      # or "onnx"
    leaves, timings = pipe.predict(image_bgr)
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BUNDLE = Path(__file__).resolve().parents[1] / "bundle"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
CROP_PAD = 10


def preprocess_crop(crop_pil, size_hw):
    """Numpy copy of the training transform: leaf horizontal, fit inside (h, w)
    keeping its shape, pad with black, scale to 0-1, normalize. Returns CHW float32."""
    h, w = size_hw
    if crop_pil.height > crop_pil.width:
        crop_pil = crop_pil.rotate(90, expand=True)
    scale = min(w / crop_pil.width, h / crop_pil.height)
    nw, nh = max(1, round(crop_pil.width * scale)), max(1, round(crop_pil.height * scale))
    resized = np.asarray(crop_pil.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return (canvas.transpose(2, 0, 1) - MEAN) / STD


def crop_leaf(img_bgr, polygon):
    """Same crop as training: black background, 10 px padding, RGB PIL image."""
    h, w = img_bgr.shape[:2]
    pts = polygon.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    x, y, bw, bh = cv2.boundingRect(pts)
    x0, y0 = max(x - CROP_PAD, 0), max(y - CROP_PAD, 0)
    x1, y1 = min(x + bw + CROP_PAD, w), min(y + bh + CROP_PAD, h)
    crop = masked[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class EdgePipeline:
    def __init__(self, fmt="onnx", bundle=BUNDLE, threads=4):
        import onnxruntime as ort
        from ultralytics import YOLO

        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        self.fmt = fmt
        self.imgsz = self.manifest["imgsz"]

        self.detector = YOLO(self._model_path("leaf_detect"), task="detect")
        self.segmenter = YOLO(self._model_path("leaf_segment"), task="segment")

        clf = self.manifest["classifier"]
        self.classes = clf["classes"]
        self.cls_size = tuple(clf["input_size_hw"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.classifier = ort.InferenceSession(str(self.bundle / clf["onnx"]), opts,
                                               providers=["CPUExecutionProvider"])

    def _model_path(self, name):
        entry = self.manifest[name]
        if self.fmt not in entry:
            raise FileNotFoundError(f"{name} has no '{self.fmt}' export in the bundle")
        return str(self.bundle / entry[self.fmt])

    def load_baseline(self):
        from ultralytics import YOLO
        return YOLO(self._model_path("baseline_seg"), task="segment")

    def predict(self, img_bgr, gate_conf=0.25, seg_conf=0.25, max_det=50, use_gate=True):
        """Returns (leaves, timings_ms). Each leaf: polygon, class_name, cls_prob, seg_conf, conf."""
        t = {"stage1": 0.0, "stage2": 0.0, "stage3": 0.0}

        if use_gate:
            t0 = time.perf_counter()
            det = self.detector.predict(img_bgr, conf=gate_conf, imgsz=self.imgsz, verbose=False)[0]
            t["stage1"] = (time.perf_counter() - t0) * 1000
            if det.boxes is None or len(det.boxes) == 0:
                return [], t

        t0 = time.perf_counter()
        seg = self.segmenter.predict(img_bgr, conf=seg_conf, max_det=max_det, imgsz=self.imgsz, verbose=False)[0]
        t["stage2"] = (time.perf_counter() - t0) * 1000
        if seg.masks is None or len(seg.masks) == 0:
            return [], t

        t0 = time.perf_counter()
        polygons, seg_confs, batch = [], [], []
        for poly, c in zip(seg.masks.xy, seg.boxes.conf.tolist()):
            if len(poly) < 3:
                continue
            crop = crop_leaf(img_bgr, poly)
            if crop is None:
                continue
            polygons.append(poly)
            seg_confs.append(c)
            batch.append(preprocess_crop(crop, self.cls_size))

        leaves = []
        if batch:
            logits = self.classifier.run(None, {"input": np.stack(batch).astype(np.float32)})[0]
            probs = softmax(logits)
            for p, sc, pr in zip(polygons, seg_confs, probs):
                k = int(pr.argmax())
                leaves.append({"polygon": p, "class_name": self.classes[k], "cls_prob": float(pr[k]),
                               "seg_conf": float(sc), "conf": float(sc) * float(pr[k])})
        t["stage3"] = (time.perf_counter() - t0) * 1000
        return leaves, t