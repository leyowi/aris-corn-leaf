# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 42
**Best epoch:** 24 (val macro F1 0.8785)
**Evaluated:** 2026-10-15 12:17
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.5818 |
| Macro precision | 0.5796 |
| Macro recall | 0.5152 |
| Macro F1 | 0.5294 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.5385 | 0.5833 | 0.5600 | 12 |
| gray-leaf-spot | 0.4800 | 0.5714 | 0.5217 | 21 |
| healthy | 0.8000 | 0.7059 | 0.7500 | 17 |
| northern-corn-leaf-blight | 0.5000 | 0.2000 | 0.2857 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.49 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
