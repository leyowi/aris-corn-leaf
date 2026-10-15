# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v1 (field only)
**Seed:** 7
**Input:** square (224 x 224)
**Best epoch:** 33 (val macro F1 0.8155)
**Evaluated:** 2026-10-15 13:18
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.5455 |
| Macro precision | 0.5728 |
| Macro recall | 0.5709 |
| Macro F1 | 0.5709 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6154 | 0.6667 | 0.6400 | 12 |
| gray-leaf-spot | 0.4091 | 0.4286 | 0.4186 | 21 |
| healthy | 0.6667 | 0.5882 | 0.6250 | 17 |
| northern-corn-leaf-blight | 0.6000 | 0.6000 | 0.6000 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.59 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
