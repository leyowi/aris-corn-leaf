# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 123
**Best epoch:** 19 (val macro F1 0.8726)
**Evaluated:** 2026-10-15 12:38
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.5818 |
| Macro precision | 0.6136 |
| Macro recall | 0.5683 |
| Macro F1 | 0.5798 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6000 | 0.7500 | 0.6667 | 12 |
| gray-leaf-spot | 0.4545 | 0.4762 | 0.4651 | 21 |
| healthy | 0.7333 | 0.6471 | 0.6875 | 17 |
| northern-corn-leaf-blight | 0.6667 | 0.4000 | 0.5000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.91 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
