# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v1 (field only)
**Seed:** 7
**Input:** square (224 x 224)
**Best epoch:** 13 (val macro F1 0.8096)
**Evaluated:** 2026-10-15 13:19
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6364 |
| Macro precision | 0.6357 |
| Macro recall | 0.6360 |
| Macro F1 | 0.6342 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.5714 | 0.6667 | 0.6154 | 12 |
| gray-leaf-spot | 0.5714 | 0.5714 | 0.5714 | 21 |
| healthy | 0.8000 | 0.7059 | 0.7500 | 17 |
| northern-corn-leaf-blight | 0.6000 | 0.6000 | 0.6000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.20 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
