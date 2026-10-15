# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v1 (field only)
**Seed:** 123
**Best epoch:** 11 (val macro F1 0.8025)
**Evaluated:** 2026-10-15 12:17
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6364 |
| Macro precision | 0.6614 |
| Macro recall | 0.6774 |
| Macro F1 | 0.6604 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6000 | 0.7500 | 0.6667 | 12 |
| gray-leaf-spot | 0.5455 | 0.5714 | 0.5581 | 21 |
| healthy | 0.8333 | 0.5882 | 0.6897 | 17 |
| northern-corn-leaf-blight | 0.6667 | 0.8000 | 0.7273 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.64 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
