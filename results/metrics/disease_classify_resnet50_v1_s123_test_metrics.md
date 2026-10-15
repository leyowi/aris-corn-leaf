# ARIS - Disease Classification (Stage 3) - Test Set Evaluation

**Model:** resnet50
**Training data:** v1 (field only)
**Seed:** 123
**Best epoch:** 10 (val macro F1 0.8651)
**Evaluated:** 2026-10-15 12:30
**Test set:** 55 leaf crops, 100% field images

## Overall

| Metric | Value |
|---|---|
| Accuracy | 0.6182 |
| Macro precision | 0.6321 |
| Macro recall | 0.5940 |
| Macro F1 | 0.6056 |

## Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| corn-rust | 0.5000 | 0.3333 | 0.4000 | 12 |
| gray-leaf-spot | 0.5417 | 0.6190 | 0.5778 | 21 |
| healthy | 0.7368 | 0.8235 | 0.7778 | 17 |
| northern-corn-leaf-blight | 0.7500 | 0.6000 | 0.6667 | 5 |

## Model cost

| Item | Value |
|---|---|
| Parameters | 23.52 M |
| Inference (dev GPU, batch 1) | 7.68 ms/image |

Note: classes with small support (e.g. northern-corn-leaf-blight) have high
variance per-class metrics; interpret them with caution.
