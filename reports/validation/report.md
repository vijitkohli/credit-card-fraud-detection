# Stage 2 validation report

Generated 2026-09-28T09:17:50+00:00. Splits used: train and validation only (validation: 56,746 rows, 57 frauds, 56,689 legitimate). The test split was not loaded.

## Selection rules (declared before results)

- Hyperparameters within each model family and time-feature variant are chosen by validation AUPRC (threshold-free, least noisy with 57 validation frauds).
- The time_within_daily_cycle feature is kept only if it improves validation AUPRC and does not reduce validation recall at the primary budget.
- The production model is chosen on validation recall at the agreed false-alert budget, with AUPRC as the tie-breaker.
- Isolation Forest is fitted without labels; labels are used only for its thresholds.
- The test split is not loaded by this experiment.

## Candidates (best configuration per family and time variant)

| Candidate | Params | AUPRC | ROC-AUC |
|---|---|---|---|
| isolation_forest | `{"n_estimators": 300}` | 0.0283 | 0.9397 |
| logistic_regression | `{"C": 1.0, "class_weight": "balanced"}` | 0.7532 | 0.9691 |
| logistic_regression+time_cycle | `{"C": 1.0, "class_weight": "balanced"}` | 0.7531 | 0.9730 |
| xgboost | `{"max_depth": 5, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 0.7823 | 0.9860 |
| xgboost+time_cycle | `{"max_depth": 5, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 0.7884 | 0.9814 |

## Operating points on validation

### isolation_forest

| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k | Precision 95% CI | Recall 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 per 10k budget | 0.729184 | 0 | 5 | 57 | 0.000 | 0.000 | 0.000 | 0.88 | 0.00 to 0.00 | 0.00 to 0.00 |
| 5 per 10k budget | 0.69087 | 0 | 28 | 57 | 0.000 | 0.000 | 0.000 | 4.94 | 0.00 to 0.00 | 0.00 to 0.00 |
| 10 per 10k budget | 0.6702 | 1 | 56 | 56 | 0.018 | 0.018 | 0.018 | 9.88 | 0.00 to 0.06 | 0.00 to 0.06 |
| 20 per 10k budget | 0.645224 | 1 | 113 | 56 | 0.009 | 0.018 | 0.012 | 19.93 | 0.00 to 0.03 | 0.00 to 0.06 |
| max F1 (ref) | 0.599687 | 18 | 389 | 39 | 0.044 | 0.316 | 0.078 | 68.62 | 0.03 to 0.07 | 0.20 to 0.43 |

### logistic_regression

| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k | Precision 95% CI | Recall 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 per 10k budget | 0.999982 | 38 | 5 | 19 | 0.884 | 0.667 | 0.760 | 0.88 | 0.79 to 0.97 | 0.54 to 0.79 |
| 5 per 10k budget | 0.999725 | 43 | 28 | 14 | 0.606 | 0.754 | 0.672 | 4.94 | 0.49 to 0.72 | 0.63 to 0.86 |
| 10 per 10k budget | 0.99474 | 44 | 56 | 13 | 0.440 | 0.772 | 0.561 | 9.88 | 0.34 to 0.54 | 0.65 to 0.88 |
| 20 per 10k budget | 0.969745 | 45 | 113 | 12 | 0.285 | 0.789 | 0.419 | 19.93 | 0.22 to 0.35 | 0.67 to 0.89 |
| max F1 (ref) | 0.999986 | 38 | 5 | 19 | 0.884 | 0.667 | 0.760 | 0.88 | 0.79 to 0.97 | 0.54 to 0.79 |

### logistic_regression+time_cycle

| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k | Precision 95% CI | Recall 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 per 10k budget | 0.999989 | 38 | 5 | 19 | 0.884 | 0.667 | 0.760 | 0.88 | 0.79 to 0.97 | 0.54 to 0.79 |
| 5 per 10k budget | 0.999892 | 43 | 28 | 14 | 0.606 | 0.754 | 0.672 | 4.94 | 0.49 to 0.72 | 0.63 to 0.86 |
| 10 per 10k budget | 0.996676 | 44 | 56 | 13 | 0.440 | 0.772 | 0.561 | 9.88 | 0.34 to 0.54 | 0.65 to 0.88 |
| 20 per 10k budget | 0.979081 | 45 | 113 | 12 | 0.285 | 0.789 | 0.419 | 19.93 | 0.22 to 0.35 | 0.67 to 0.89 |
| max F1 (ref) | 0.999996 | 38 | 4 | 19 | 0.905 | 0.667 | 0.768 | 0.71 | 0.81 to 0.98 | 0.54 to 0.79 |

### xgboost

| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k | Precision 95% CI | Recall 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 per 10k budget | 0.414631 | 42 | 5 | 15 | 0.894 | 0.737 | 0.808 | 0.88 | 0.80 to 0.98 | 0.61 to 0.85 |
| 5 per 10k budget | 0.103254 | 45 | 28 | 12 | 0.616 | 0.789 | 0.692 | 4.94 | 0.51 to 0.72 | 0.67 to 0.89 |
| 10 per 10k budget | 0.0310166 | 45 | 56 | 12 | 0.446 | 0.789 | 0.570 | 9.88 | 0.35 to 0.54 | 0.67 to 0.89 |
| 20 per 10k budget | 0.0106016 | 45 | 113 | 12 | 0.285 | 0.789 | 0.419 | 19.93 | 0.22 to 0.36 | 0.67 to 0.89 |
| max F1 (ref) | 0.637672 | 42 | 2 | 15 | 0.955 | 0.737 | 0.832 | 0.35 | 0.88 to 1.00 | 0.61 to 0.85 |

### xgboost+time_cycle

| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k | Precision 95% CI | Recall 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 per 10k budget | 0.504236 | 44 | 5 | 13 | 0.898 | 0.772 | 0.830 | 0.88 | 0.81 to 0.98 | 0.65 to 0.88 |
| 5 per 10k budget | 0.0932691 | 45 | 28 | 12 | 0.616 | 0.789 | 0.692 | 4.94 | 0.51 to 0.72 | 0.67 to 0.89 |
| 10 per 10k budget | 0.0228168 | 45 | 56 | 12 | 0.446 | 0.789 | 0.570 | 9.88 | 0.35 to 0.54 | 0.67 to 0.89 |
| 20 per 10k budget | 0.00886423 | 45 | 113 | 12 | 0.285 | 0.789 | 0.419 | 19.93 | 0.22 to 0.35 | 0.67 to 0.89 |
| max F1 (ref) | 0.572529 | 44 | 3 | 13 | 0.936 | 0.772 | 0.846 | 0.53 | 0.86 to 1.00 | 0.65 to 0.88 |

## Time-cycle coverage

Cycle positions are hours since an unknown offset, not clock time.

- Train: 33.44 hours, ending at cycle position 9.44 h.
- Validation: cycle positions 9.44 to 16.34 h only.

## Baseline

- Flag nothing: accuracy 99.8996%, recall 0.

## All grid results

| Family | Time cycle | Params | Rounds | AUPRC | ROC-AUC |
|---|---|---|---|---|---|
| logistic_regression | False | `{"C": 0.01, "class_weight": null}` |  | 0.6566 | 0.9722 |
| logistic_regression | False | `{"C": 0.01, "class_weight": "balanced"}` |  | 0.7295 | 0.9695 |
| logistic_regression | False | `{"C": 0.1, "class_weight": null}` |  | 0.6326 | 0.9697 |
| logistic_regression | False | `{"C": 0.1, "class_weight": "balanced"}` |  | 0.7519 | 0.9691 |
| logistic_regression | False | `{"C": 1.0, "class_weight": null}` |  | 0.6236 | 0.9681 |
| logistic_regression | False | `{"C": 1.0, "class_weight": "balanced"}` |  | 0.7532 | 0.9691 |
| xgboost | False | `{"max_depth": 3, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 27 | 0.7411 | 0.9611 |
| xgboost | False | `{"max_depth": 3, "scale_pos_weight": "sqrt_ratio", "scale_pos_weight_value": 22.2882}` | 10 | 0.7358 | 0.9492 |
| xgboost | False | `{"max_depth": 5, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 221 | 0.7823 | 0.9860 |
| xgboost | False | `{"max_depth": 5, "scale_pos_weight": "sqrt_ratio", "scale_pos_weight_value": 22.2882}` | 10 | 0.7298 | 0.9617 |
| logistic_regression | True | `{"C": 0.01, "class_weight": null}` |  | 0.6531 | 0.9720 |
| logistic_regression | True | `{"C": 0.01, "class_weight": "balanced"}` |  | 0.7340 | 0.9732 |
| logistic_regression | True | `{"C": 0.1, "class_weight": null}` |  | 0.6147 | 0.9707 |
| logistic_regression | True | `{"C": 0.1, "class_weight": "balanced"}` |  | 0.7520 | 0.9731 |
| logistic_regression | True | `{"C": 1.0, "class_weight": null}` |  | 0.6108 | 0.9703 |
| logistic_regression | True | `{"C": 1.0, "class_weight": "balanced"}` |  | 0.7531 | 0.9730 |
| xgboost | True | `{"max_depth": 3, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 44 | 0.7618 | 0.9679 |
| xgboost | True | `{"max_depth": 3, "scale_pos_weight": "sqrt_ratio", "scale_pos_weight_value": 22.2882}` | 50 | 0.7721 | 0.9716 |
| xgboost | True | `{"max_depth": 5, "scale_pos_weight": "1", "scale_pos_weight_value": 1.0}` | 230 | 0.7884 | 0.9814 |
| xgboost | True | `{"max_depth": 5, "scale_pos_weight": "sqrt_ratio", "scale_pos_weight_value": 22.2882}` | 43 | 0.7468 | 0.9797 |
| isolation_forest | False | `{"n_estimators": 300}` |  | 0.0283 | 0.9397 |
