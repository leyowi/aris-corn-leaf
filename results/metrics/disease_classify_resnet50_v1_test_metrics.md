# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v1 (field only)
**Seed:** 42
**Best epoch:** 14 (val macro F1 0.8597)
**Evaluated:** 2026-10-15 12:26
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.5636 |
| Macro precision | 0.5869 |
| Macro recall | 0.5738 |
| Macro F1 | 0.5763 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.5000 | 0.5833 | 0.5385 | 12 |
| gray-leaf-spot | 0.4783 | 0.5238 | 0.5000 | 21 |
| healthy | 0.7692 | 0.5882 | 0.6667 | 17 |
| northern-corn-leaf-blight | 0.6000 | 0.6000 | 0.6000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.48 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
