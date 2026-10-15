# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 42
**Best epoch:** 9 (val macro F1 0.8753)
**Evaluated:** 2026-10-15 12:30
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6364 |
| Macro precision | 0.6075 |
| Macro recall | 0.6567 |
| Macro F1 | 0.6159 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.5882 | 0.8333 | 0.6897 | 12 |
| gray-leaf-spot | 0.6000 | 0.4286 | 0.5000 | 21 |
| healthy | 0.8667 | 0.7647 | 0.8125 | 17 |
| northern-corn-leaf-blight | 0.3750 | 0.6000 | 0.4615 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.62 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
