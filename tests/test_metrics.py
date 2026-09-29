import numpy as np
import pytest

from fraud import metrics


@pytest.fixture
def scored():
    """20,000 legitimate and 50 fraud scores with overlap."""
    rng = np.random.default_rng(1)
    legit = rng.uniform(0, 0.9, 20_000)
    fraud = rng.uniform(0.5, 1.0, 50)
    y = np.r_[np.zeros_like(legit), np.ones_like(fraud)].astype(int)
    return y, np.r_[legit, fraud]


def test_confusion_counts_and_rates():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = metrics.confusion_at(y, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["false_alerts_per_10k_legit"] == pytest.approx(10_000 / 3)


def test_threshold_is_inclusive():
    m = metrics.confusion_at(np.array([1]), np.array([0.5]), 0.5)
    assert m["tp"] == 1


@pytest.mark.parametrize("budget", [1, 5, 10, 20])
def test_budget_threshold_respects_budget_and_maximises_recall(scored, budget):
    y, scores = scored
    allowed = int(budget * 20_000 / 10_000)
    threshold = metrics.threshold_for_fp_budget(y, scores, budget)

    assert metrics.confusion_at(y, scores, threshold)["fp"] <= allowed
    lower = np.max(scores[scores < threshold])
    assert metrics.confusion_at(y, scores, lower)["fp"] > allowed


def test_budget_threshold_handles_tied_legit_scores():
    y = np.array([0] * 10_000 + [1])
    scores = np.r_[np.full(10_000, 0.3), 0.9]
    threshold = metrics.threshold_for_fp_budget(y, scores, 1)
    m = metrics.confusion_at(y, scores, threshold)
    assert m["fp"] == 0
    assert m["tp"] == 1


def test_budget_thresholds_are_monotonic(scored):
    y, scores = scored
    thresholds = [metrics.threshold_for_fp_budget(y, scores, b) for b in (1, 5, 10, 20)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_max_f1_threshold_beats_neighbours(scored):
    y, scores = scored
    best = metrics.max_f1_threshold(y, scores)
    best_f1 = metrics.confusion_at(y, scores, best)["f1"]
    for t in np.quantile(scores, [0.5, 0.9, 0.99, 0.999]):
        assert metrics.confusion_at(y, scores, t)["f1"] <= best_f1 + 1e-12


def test_threshold_free_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert metrics.threshold_free(y, np.array([0.1, 0.2, 0.8, 0.9])) == {
        "auprc": 1.0,
        "roc_auc": 1.0,
    }


def test_bootstrap_ci_is_reproducible_and_brackets_point_estimate(scored):
    y, scores = scored
    t = metrics.threshold_for_fp_budget(y, scores, 10)
    a = metrics.bootstrap_ci(y, scores, t, n_resamples=200, seed=3)
    assert a == metrics.bootstrap_ci(y, scores, t, n_resamples=200, seed=3)
    point = metrics.confusion_at(y, scores, t)
    for name in ("precision", "recall"):
        lo, hi = a[name]
        assert lo <= point[name] <= hi
