"""Candidate model definitions. Every score is oriented so that higher means riskier."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260927

# The last part of the training period (by time) chooses the number of boosting rounds;
# validation is kept for model selection and thresholds only.
EARLY_STOPPING_FRACTION = 0.15
MAX_BOOSTING_ROUNDS = 2_000
EARLY_STOPPING_ROUNDS = 100

XGB_FIXED_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "random_state": SEED,
    "n_jobs": -1,
}


def logistic_regression(C: float, class_weight: str | None) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=2_000, random_state=SEED),
    )


def fit_xgboost(
    X: pd.DataFrame, y: pd.Series, max_depth: int, scale_pos_weight: float
) -> tuple[xgb.XGBClassifier, int]:
    """Pick the number of rounds on the time-ordered tail of train, then refit on all of train.

    `X` must be in time order.
    """
    cut = round(len(X) * (1 - EARLY_STOPPING_FRACTION))
    probe = xgb.XGBClassifier(
        n_estimators=MAX_BOOSTING_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        **XGB_FIXED_PARAMS,
    )
    probe.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], verbose=False)
    n_rounds = probe.best_iteration + 1

    model = xgb.XGBClassifier(
        n_estimators=n_rounds,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        **XGB_FIXED_PARAMS,
    )
    model.fit(X, y, verbose=False)
    return model, n_rounds


def isolation_forest() -> IsolationForest:
    return IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)


def risk_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, IsolationForest):
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]
