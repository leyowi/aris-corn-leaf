# ARIS - 3-stage pipeline vs single-stage baseline (end to end)

**Evaluated:** 2026-10-15 14:04
**Test set:** 38 raw field photos, 55 labeled leaves
**Pipeline:** leaf_detect v1 (gate) -> leaf_segment v1-6 -> mobilenet_v2_v1_wide_s123
**Baseline:** baseline_seg v1 (YOLOv8n-seg, 4 disease classes)
**Scoring:** identical code for both systems, mask IoU. P/R/F1 and leaf-level at confidence 0.25.

## Overall

| System | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Leaves found | Correct (of found) | Found AND correct | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| 3-stage pipeline | 0.5191 | 0.5707 | 0.5363 | 0.5044 | 0.4607 | 0.8364 | 0.6304 | 0.5273 | 40.9 |
| Single-stage baseline | 0.5937 | 0.5562 | 0.5728 | 0.5319 | 0.4548 | 0.7636 | 0.6905 | 0.5273 | 12.5 |

## Per class - 3-stage pipeline

| Class | Leaves | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| corn-rust | 12 | 0.5625 | 0.7500 | 0.6429 | 0.6443 | 0.5668 |
| gray-leaf-spot | 21 | 0.2857 | 0.2857 | 0.2857 | 0.2116 | 0.1708 |
| healthy | 17 | 0.4783 | 0.6471 | 0.5500 | 0.5576 | 0.5012 |
| northern-corn-leaf-blight | 5 | 0.7500 | 0.6000 | 0.6667 | 0.6040 | 0.6040 |

## Per class - single-stage baseline

| Class | Leaves | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| corn-rust | 12 | 0.6667 | 0.6667 | 0.6667 | 0.5494 | 0.4980 |
| gray-leaf-spot | 21 | 0.4286 | 0.4286 | 0.4286 | 0.3327 | 0.2432 |
| healthy | 17 | 0.5294 | 0.5294 | 0.5294 | 0.5096 | 0.4475 |
| northern-corn-leaf-blight | 5 | 0.7500 | 0.6000 | 0.6667 | 0.7360 | 0.6304 |

Notes: precision/recall/F1/mAP are means over the 4 classes. "Leaves found" = a
prediction overlaps the leaf with mask IoU >= 0.5. Speed is measured on the dev GPU
at the operating threshold; Raspberry Pi 5 timing is measured separately.
