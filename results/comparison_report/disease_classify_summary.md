# ARIS - Disease Classification (Stage 3) - Comparison

Field-only test set (55 crops). Values are mean ± standard deviation across seeds.
Input: square = pad to 224 x 224, wide = rotate leaf horizontal and fit to 128 x 512.
v1 = field only, v2 = field + Kaggle (train/valid only).

## Overall

| Model | Data | Input | Seeds | Val macro F1 | Test accuracy | Test macro F1 | Params (M) | GPU ms |
|---|---|---|---|---|---|---|---|---|
| mobilenet_v2 | v1 | square | 3 | 0.807 ± 0.007 | 0.630 ± 0.082 | 0.646 ± 0.069 | 2.23 | 6.543 ± 0.263 |
| mobilenet_v2 | v1 | wide | 3 | 0.831 ± 0.006 | 0.655 ± 0.066 | 0.670 ± 0.069 | 2.23 | 6.263 ± 0.074 |
| mobilenet_v2 | v2 | square | 3 | 0.872 ± 0.007 | 0.618 ± 0.036 | 0.606 ± 0.066 | 2.23 | 6.350 ± 0.314 |
| resnet50 | v1 | square | 3 | 0.845 ± 0.031 | 0.606 ± 0.038 | 0.605 ± 0.029 | 23.52 | 7.323 ± 0.137 |
| resnet50 | v2 | square | 3 | 0.879 ± 0.008 | 0.618 ± 0.032 | 0.609 ± 0.026 | 23.52 | 7.083 ± 0.081 |

## Per class test F1

| Model | Data | Input | corn-rust | gray-leaf-spot | healthy | northern-corn-leaf-blight |
|---|---|---|---|---|---|---|
| mobilenet_v2 | v1 | square | 0.697 ± 0.078 | 0.536 ± 0.108 | 0.686 ± 0.059 | 0.665 ± 0.064 |
| mobilenet_v2 | v1 | wide | 0.725 ± 0.088 | 0.527 ± 0.073 | 0.719 ± 0.032 | 0.709 ± 0.101 |
| mobilenet_v2 | v2 | square | 0.665 ± 0.123 | 0.532 ± 0.009 | 0.708 ± 0.042 | 0.517 ± 0.203 |
| resnet50 | v1 | square | 0.518 ± 0.109 | 0.550 ± 0.043 | 0.732 ± 0.058 | 0.622 ± 0.039 |
| resnet50 | v2 | square | 0.709 ± 0.054 | 0.526 ± 0.077 | 0.690 ± 0.121 | 0.511 ± 0.056 |
