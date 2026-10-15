import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import MODELS_DIR  # noqa: E402
from src.models.disease_classify.common import build_model, get_transforms  # noqa: E402

DETECT_WEIGHTS = MODELS_DIR / "leaf_detect" / "v1" / "weights" / "best.pt"
SEGMENT_WEIGHTS = MODELS_DIR / "leaf_segment" / "v1-6" / "weights" / "best.pt"
CLASSIFIER_PATTERN = "mobilenet_v2_v1_wide*"   # final Stage 3 setup (3 seeds)
CROP_PAD = 10                                  # same padding as the training crops


def pick_best_classifier():
    """Chooses the Stage 3 seed with the highest VALIDATION macro F1 (never test)."""
    best = None
    for ckpt_path in sorted((MODELS_DIR / "disease_classify").glob(f"{CLASSIFIER_PATTERN}/best.pt")):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if best is None or ckpt["val_macro_f1"] > best[1]:
            best = (ckpt_path, ckpt["val_macro_f1"])
    if best is None:
        raise FileNotFoundError(f"No classifier found matching {CLASSIFIER_PATTERN}")
    return best


def crop_leaf(img_bgr, polygon):
    """Same crop as training: mask out the background, pad 10 px, return RGB PIL image."""
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


class ARISPipeline:
    def __init__(self, classifier_path=None, device=None):
        from ultralytics import YOLO

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.detector = YOLO(str(DETECT_WEIGHTS))
        self.segmenter = YOLO(str(SEGMENT_WEIGHTS))

        if classifier_path is None:
            classifier_path, _ = pick_best_classifier()
        ckpt = torch.load(classifier_path, map_location=self.device)
        self.classifier_path = Path(classifier_path)
        self.classes = ckpt["classes"]
        self.classifier = build_model(ckpt["arch"], len(self.classes), pretrained=False)
        self.classifier.load_state_dict(ckpt["state_dict"])
        self.classifier.to(self.device).eval()
        self.transform = get_transforms(train=False, input_mode=ckpt.get("input_mode", "square"))

    @torch.no_grad()
    def predict(self, img_bgr, gate_conf=0.25, seg_conf=0.25, max_det=50, use_gate=True):
        """Returns a list of leaves: {polygon, class_name, conf, seg_conf, cls_prob}."""
        if use_gate:
            det = self.detector.predict(img_bgr, conf=gate_conf, verbose=False)[0]
            if det.boxes is None or len(det.boxes) == 0:
                return []                         # no leaf in frame: skip Stages 2-3

        seg = self.segmenter.predict(img_bgr, conf=seg_conf, max_det=max_det, verbose=False)[0]
        if seg.masks is None or len(seg.masks) == 0:
            return []

        polygons, seg_confs, crops = [], [], []
        for poly, c in zip(seg.masks.xy, seg.boxes.conf.tolist()):
            if len(poly) < 3:
                continue
            crop = crop_leaf(img_bgr, poly)
            if crop is None:
                continue
            polygons.append(poly)
            seg_confs.append(c)
            crops.append(self.transform(crop))
        if not crops:
            return []

        probs = torch.softmax(self.classifier(torch.stack(crops).to(self.device)), dim=1).cpu()
        cls_prob, cls_id = probs.max(1)

        return [
            {"polygon": p, "class_name": self.classes[k], "cls_prob": float(cp),
             "seg_conf": float(sc), "conf": float(sc) * float(cp)}
            for p, k, cp, sc in zip(polygons, cls_id.tolist(), cls_prob.tolist(), seg_confs)
        ]