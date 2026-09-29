import numpy as np
import pandas as pd
import pytest

from fraud import models


@pytest.fixture
def separable():
    """Time-ordered synthetic data where fraud is shifted on two features."""
    rng = np.random.default_rng(0)
    n = 3_000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series((rng.uniform(size=n) < 0.03).astype(int))
    X.loc[y == 1, ["a", "b"]] += 3
    return X, y


def test_xgboost_learns_and_reports_rounds(separable):
    X, y = separable
    model, n_rounds = models.fit_xgboost(X, y, max_depth=3, scale_pos_weight=1.0)
    scores = models.risk_scores(model, X)
    assert 1 <= n_rounds <= models.MAX_BOOSTING_ROUNDS
    assert model.n_estimators == n_rounds
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_xgboost_is_deterministic(separable):
    X, y = separable
    a, _ = models.fit_xgboost(X, y, max_depth=3, scale_pos_weight=1.0)
    b, _ = models.fit_xgboost(X, y, max_depth=3, scale_pos_weight=1.0)
    np.testing.assert_array_equal(models.risk_scores(a, X), models.risk_scores(b, X))


def test_logistic_regression_scores_are_probabilities(separable):
    X, y = separable
    model = models.logistic_regression(C=1.0, class_weight=None).fit(X, y)
    scores = models.risk_scores(model, X)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_isolation_forest_scores_outliers_higher(separable):
    X, _ = separable
    model = models.isolation_forest().fit(X)
    outlier = pd.DataFrame([[8.0, 8.0, 8.0, 8.0]], columns=X.columns)
    assert models.risk_scores(model, outlier)[0] > np.median(models.risk_scores(model, X))
