# Stage 3 held-out test evaluation

Evaluated 2026-09-29T07:46:20+00:00 at commit `df1d962a1c` (model-v1). Test split: 56,744 rows, 74 frauds, 56,670 legitimate. All thresholds were frozen on validation before this run.

## Frozen operating points

| Model / point | Threshold | TP | FP | FN | TN | Precision (95% CI) | Recall (95% CI) | F1 | False alerts / 10k |
|---|---|---|---|---|---|---|---|---|---|
| **xgboost @ 5/10k (production)** | 0.103254 | 58 | 28 | 16 | 56642 | 0.674 (0.58 to 0.77) | 0.784 (0.69 to 0.88) | 0.725 | 4.94 |
| xgboost @ 1/10k (precision-first alternative) | 0.414631 | 55 | 9 | 19 | 56661 | 0.859 (0.76 to 0.94) | 0.743 (0.63 to 0.84) | 0.797 | 1.59 |
| logistic_regression @ 5/10k (comparison) | 0.999725 | 52 | 19 | 22 | 56651 | 0.732 (0.63 to 0.84) | 0.703 (0.59 to 0.81) | 0.717 | 3.35 |
| isolation_forest @ 5/10k (comparison) | 0.69087 | 0 | 19 | 74 | 56651 | 0.000 (0.00 to 0.00) | 0.000 (0.00 to 0.00) | 0.000 | 3.35 |

## Threshold-free

| Model | AUPRC | ROC-AUC |
|---|---|---|
| xgboost | 0.7885 | 0.9821 |
| logistic_regression | 0.7442 | 0.9816 |
| isolation_forest | 0.0561 | 0.9523 |

## Validation reference (production threshold)

- Validation: recall 0.789, precision 0.616, 4.94 false alerts per 10k.

## Baseline

- Flag nothing: accuracy 99.8696%, recall 0.
