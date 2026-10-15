# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v1 (field only)
**Seed:** 123
**Input:** wide (128 x 512)
**Best epoch:** 12 (val macro F1 0.8379)
**Evaluated:** 2026-10-15 13:26
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6364 |
| Macro precision | 0.6453 |
| Macro recall | 0.6830 |
| Macro F1 | 0.6615 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6923 | 0.7500 | 0.7200 | 12 |
| gray-leaf-spot | 0.5556 | 0.4762 | 0.5128 | 21 |
| healthy | 0.6667 | 0.7059 | 0.6857 | 17 |
| northern-corn-leaf-blight | 0.6667 | 0.8000 | 0.7273 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.29 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
