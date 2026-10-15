# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** mobilenet_v2
**Training data:** v1 (field only)
**Seed:** 42
**Input:** square (224 x 224)
**Best epoch:** 7 (val macro F1 0.8037)
**Evaluated:** 2026-10-15 13:18
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.7091 |
| Macro precision | 0.7164 |
| Macro recall | 0.7132 |
| Macro F1 | 0.7067 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.6875 | 0.9167 | 0.7857 | 12 |
| gray-leaf-spot | 0.7059 | 0.5714 | 0.6316 | 21 |
| healthy | 0.7222 | 0.7647 | 0.7429 | 17 |
| northern-corn-leaf-blight | 0.7500 | 0.6000 | 0.6667 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 2.23 M |
| Inference (dev GPU, batch 1) | 6.78 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
