# ARIS - Baseline (single-stage YOLOv8n-seg, 4 disease classes) - Test Set Evaluation

**Run:** v1
**Weights:** models\baseline_seg\v1\weights\best.pt
**Evaluated:** 2026-10-15 13:55
**Test set:** same field test images as Stages 1-3, 55 leaf instances

## Overall - Mask (segmentation)

| Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 0.7302 | 0.5423 | 0.6224 | 0.5876 | 0.5075 |

## Overall - Box (detection)

| Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| 0.7302 | 0.5423 | 0.6224 | 0.5944 | 0.4952 |

## Per class - Mask

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Instances |
|---|---|---|---|---|---|---|
| corn-rust | 0.7034 | 0.6667 | 0.6846 | 0.6034 | 0.5288 | 12 |
| gray-leaf-spot | 0.6043 | 0.3810 | 0.4673 | 0.3943 | 0.2873 | 21 |
| healthy | 0.6132 | 0.5294 | 0.5682 | 0.6243 | 0.5389 | 17 |
| northern-corn-leaf-blight | 1.0000 | 0.5924 | 0.7440 | 0.7283 | 0.6750 | 5 |

## Per class - Box

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Instances |
|---|---|---|---|---|---|---|
| corn-rust | 0.7034 | 0.6667 | 0.6846 | 0.6034 | 0.4850 | 12 |
| gray-leaf-spot | 0.6043 | 0.3810 | 0.4673 | 0.4133 | 0.3057 | 21 |
| healthy | 0.6132 | 0.5294 | 0.5682 | 0.6324 | 0.5085 | 17 |
| northern-corn-leaf-blight | 1.0000 | 0.5924 | 0.7440 | 0.7283 | 0.6817 | 5 |

## Speed (dev GPU, per image)

| Preprocess | Inference | Postprocess |
|---|---|---|
| 1.03 ms | 11.48 ms | 1.44 ms |
