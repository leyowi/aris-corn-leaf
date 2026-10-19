# ARIS - 3-stage pipeline vs single-stage baseline (end to end)

**Evaluated:** 2026-10-19 12:16
**Test set:** 38 raw field photos, 55 labeled leaves
**Pipeline:** leaf_detect v1 (gate) -> leaf_segment v1-6 -> mobilenet_v2_v1_wide_hr_s123
**Baseline:** baseline_seg v1 (YOLOv8n-seg, 4 disease classes)
**Scoring:** identical code for both systems, mask IoU. P/R/F1 and leaf-level at confidence 0.25.

## Overall

| System | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Leaves found | Correct (of found) | Found AND correct | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| 3-stage pipeline | 0.5260 | 0.6271 | 0.5427 | 0.5310 | 0.4764 | 0.8364 | 0.6957 | 0.5818 | 41.2 |
| Single-stage baseline | 0.5937 | 0.5562 | 0.5728 | 0.5319 | 0.4548 | 0.7636 | 0.6905 | 0.5273 | 11.6 |

## Per class - 3-stage pipeline

| Class | Leaves | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| corn-rust | 12 | 0.5238 | 0.9167 | 0.6667 | 0.5963 | 0.5374 |
| gray-leaf-spot | 21 | 0.6000 | 0.2857 | 0.3871 | 0.2873 | 0.2057 |
| healthy | 17 | 0.4800 | 0.7059 | 0.5714 | 0.6363 | 0.5583 |
| northern-corn-leaf-blight | 5 | 0.5000 | 0.6000 | 0.5455 | 0.6040 | 0.6040 |

## Per class - single-stage baseline

| Class | Leaves | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| corn-rust | 12 | 0.6667 | 0.6667 | 0.6667 | 0.5494 | 0.4980 |
| gray-leaf-spot | 21 | 0.4286 | 0.4286 | 0.4286 | 0.3327 | 0.2432 |
| healthy | 17 | 0.5294 | 0.5294 | 0.5294 | 0.5096 | 0.4475 |
| northern-corn-leaf-blight | 5 | 0.7500 | 0.6000 | 0.6667 | 0.7360 | 0.6304 |

## Healthy vs diseased (test)

The farmer's first question is "is this leaf sick?". Here any disease counts as "diseased".

| System | Correct (of found) | Found AND correct |
|---|---|---|
| 3-stage pipeline | 0.8696 | 0.7273 |
| Single-stage baseline | 0.8571 | 0.6545 |

## Confidence filter: "uncertain, check manually" (test)

Leaves below the threshold are flagged for a person to check instead of being labeled.
The threshold is chosen on the VALIDATION set as the lowest one reaching 85% accuracy.

| System | Threshold (from validation) | Val: kept / accuracy | Test: kept / accuracy | Test leaves flagged |
|---|---|---|---|---|
| 3-stage pipeline | 0.60 | 87% / 89% | 93% / 67% | 3 |
| Single-stage baseline | 0.50 | 92% / 86% | 83% / 69% | 7 |

Full trade-off on test (kept = share of found leaves ARIS labels itself):

| System | >= 0.00 | >= 0.50 | >= 0.60 | >= 0.70 | >= 0.80 | >= 0.90 | >= 0.95 |
|---|---|---|---|---|---|---|---|
| 3-stage pipeline | 100% kept, 70% | 96% kept, 68% | 93% kept, 67% | 91% kept, 69% | 78% kept, 75% | 65% kept, 77% | 46% kept, 76% |
| Single-stage baseline | 100% kept, 69% | 83% kept, 69% | 79% kept, 73% | 74% kept, 74% | 43% kept, 67% | 21% kept, 78% | none kept |

Notes: precision/recall/F1/mAP are means over the 4 classes. "Leaves found" = a
prediction overlaps the leaf with mask IoU >= 0.5. Speed is measured on the dev GPU
at the operating threshold; Raspberry Pi 5 timing is measured separately.
