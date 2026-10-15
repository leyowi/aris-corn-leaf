# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v2 (field + Kaggle (train/valid only))
**Seed:** 7
**Best epoch:** 8 (val macro F1 0.8884)
**Evaluated:** 2026-10-15 12:33
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6364 |
| Macro precision | 0.7443 |
| Macro recall | 0.6046 |
| Macro F1 | 0.6311 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.7143 | 0.8333 | 0.7692 | 12 |
| gray-leaf-spot | 0.5357 | 0.7143 | 0.6122 | 21 |
| healthy | 0.7273 | 0.4706 | 0.5714 | 17 |
| northern-corn-leaf-blight | 1.0000 | 0.4000 | 0.5714 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.71 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
