# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 7
**Best epoch:** 14 (val macro F1 0.8651)
**Evaluated:** 2026-10-15 12:22
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6545 |
| Macro precision | 0.6448 |
| Macro recall | 0.6808 |
| Macro F1 | 0.6518 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6667 | 1.0000 | 0.8000 | 12 |
| gray-leaf-spot | 0.6250 | 0.4762 | 0.5405 | 21 |
| healthy | 0.6875 | 0.6471 | 0.6667 | 17 |
| northern-corn-leaf-blight | 0.6000 | 0.6000 | 0.6000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.65 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
