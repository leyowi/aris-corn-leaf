# ARIS - Error analysis (VALIDATION set)

Classifier: mobilenet_v2_v1_wide_s123 (val macro F1 0.8379)

## Part A - Stage 3 on validation leaf crops (96 crops)

| True vs Predicted | RUST | GLS | HEALTHY | NCLB | Recall |
|---|---|---|---|---|---|
| RUST | 18 | 8 | 2 | 0 | 0.64 |
| GLS | 1 | 34 | 2 | 0 | 0.92 |
| HEALTHY | 0 | 4 | 15 | 0 | 0.79 |
| NCLB | 0 | 0 | 1 | 11 | 0.92 |

### Most common mistakes

| Mistake (true -> predicted) | Crops |
|---|---|
| corn-rust -> gray-leaf-spot | 8 |
| healthy -> gray-leaf-spot | 4 |
| corn-rust -> healthy | 2 |
| gray-leaf-spot -> healthy | 2 |
| gray-leaf-spot -> corn-rust | 1 |
| northern-corn-leaf-blight -> healthy | 1 |

Wrong crops and contact sheets: `results/error_analysis/classifier_val/`

## Part B - full pipeline on validation photos (confidence 0.25)

| Class | Correct | Wrong disease | Missed |
|---|---|---|---|
| RUST | 21 | 5 | 2 |
| GLS | 31 | 4 | 2 |
| HEALTHY | 13 | 3 | 3 |
| NCLB | 9 | 1 | 2 |
| **Total** | **74** | **13** | **9** |

Photos with mistakes (green correct, red wrong, yellow missed): `results/error_analysis/pipeline_val/`
