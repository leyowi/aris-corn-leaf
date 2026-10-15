# ARIS - Disease Classification (Stage 3) - Comparison

Field-only test set (55 crops). Values are mean ± standard deviation across seeds.
v1 = field only, v2 = field + Kaggle (train/valid only).

## Overall

| Model | Data | Seeds | Val macro F1 | Test accuracy | Test macro F1 | Params (M) | GPU ms |
|---|---|---|---|---|---|---|---|
| mobilenet_v2 | v1 | 3 | 0.807 ± 0.007 | 0.630 ± 0.082 | 0.646 ± 0.069 | 2.23 | 6.553 ± 0.090 |
| mobilenet_v2 | v2 | 3 | 0.872 ± 0.007 | 0.618 ± 0.036 | 0.606 ± 0.066 | 2.23 | 6.650 ± 0.160 |
| resnet50 | v1 | 3 | 0.845 ± 0.031 | 0.606 ± 0.038 | 0.605 ± 0.029 | 23.52 | 7.613 ± 0.115 |
| resnet50 | v2 | 3 | 0.879 ± 0.008 | 0.618 ± 0.032 | 0.609 ± 0.026 | 23.52 | 7.747 ± 0.148 |

## Per class test F1

| Model | Data | corn-rust | gray-leaf-spot | healthy | northern-corn-leaf-blight |
|---|---|---|---|---|---|
| mobilenet_v2 | v1 | 0.697 ± 0.078 | 0.536 ± 0.108 | 0.686 ± 0.059 | 0.665 ± 0.064 |
| mobilenet_v2 | v2 | 0.665 ± 0.123 | 0.532 ± 0.009 | 0.708 ± 0.042 | 0.517 ± 0.203 |
| resnet50 | v1 | 0.518 ± 0.109 | 0.550 ± 0.043 | 0.732 ± 0.058 | 0.622 ± 0.039 |
| resnet50 | v2 | 0.709 ± 0.054 | 0.526 ± 0.077 | 0.690 ± 0.121 | 0.511 ± 0.056 |
