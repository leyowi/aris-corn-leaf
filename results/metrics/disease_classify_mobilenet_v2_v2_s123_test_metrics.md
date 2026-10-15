# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 123
**Best epoch:** 7 (val macro F1 0.8725)
**Evaluated:** 2026-10-15 12:26
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6182 |
| Macro precision | 0.6640 |
| Macro recall | 0.6152 |
| Macro F1 | 0.6356 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.7000 | 0.5833 | 0.6364 | 12 |
| gray-leaf-spot | 0.5000 | 0.5714 | 0.5333 | 21 |
| healthy | 0.7059 | 0.7059 | 0.7059 | 17 |
| northern-corn-leaf-blight | 0.7500 | 0.6000 | 0.6667 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.81 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
