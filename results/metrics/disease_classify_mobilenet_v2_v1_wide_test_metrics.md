# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v1 (field only)
**Seed:** 42
**Input:** wide (128 x 512)
**Best epoch:** 7 (val macro F1 0.8260)
**Evaluated:** 2026-10-15 13:22
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.7273 |
| Macro precision | 0.7547 |
| Macro recall | 0.7688 |
| Macro F1 | 0.7427 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.7333 | 0.9167 | 0.8148 | 12 |
| gray-leaf-spot | 0.8333 | 0.4762 | 0.6061 | 21 |
| healthy | 0.6522 | 0.8824 | 0.7500 | 17 |
| northern-corn-leaf-blight | 0.8000 | 0.8000 | 0.8000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.32 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
