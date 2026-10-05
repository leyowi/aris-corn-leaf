# ARIS - Leaf Detection (Stage 1) - Test Set Evaluation

**Model:** YOLOv8n (detection) \
**Run:** v1  \
**Weights:** models\leaf_detect\v1\weights\best.pt \
**Evaluated:** 2026-10-05 22:06 \
**Test set:** 56 `corn-leaf` instances (same split as Stage 2)

## Detection (Bounding Box) Metrics

| Metric | Value |
|---|---|
| Precision | 0.8792 |
| Recall | 0.7800 |
| F1 score | 0.8267 |
| mAP@50 | 0.8666 |
| mAP@50-95 | 0.6721 |

## Inference Speed (dev GPU, per image)

| Stage | ms |
|---|---|
| Preprocess | 1.08 |
| Inference | 9.22 |
| Postprocess | 0.94 |
