"""Threshold selection and evaluation metrics.

A transaction is flagged when `score >= threshold`. False-positive budgets are expressed
as false alerts per 10,000 legitimate transactions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PER = 10_000


def confusion_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(scores) >= threshold
    tp = int((flagged & y).sum())
    fp = int((flagged & ~y).sum())
    fn = int((~flagged & y).sum())
    tn = int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_alerts_per_10k_legit": PER * fp / (fp + tn) if fp + tn else 0.0,
        "flagged": tp + fp,
    }


def threshold_for_fp_budget(y: np.ndarray, scores: np.ndarray, per_10k: float) -> float:
    """Lowest threshold whose false alerts stay within the budget, i.e. maximum recall."""
    legit = np.sort(np.asarray(scores)[~np.asarray(y).astype(bool)])[::-1]
    allowed = math.floor(per_10k * len(legit) / PER)
    if allowed >= len(legit):
        return float(np.min(scores))
    return float(np.nextafter(legit[allowed], np.inf))


def max_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_free(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auprc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def bootstrap_ci(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_resamples: int = 1_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Percentile bootstrap intervals for precision, recall and false alerts at a threshold."""
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(scores) >= threshold
    rng = np.random.default_rng(seed)
    stats = np.empty((n_resamples, 3))
    for i in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        yi, fi = y[idx], flagged[idx]
        tp = np.sum(fi & yi)
        fp = np.sum(fi & ~yi)
        pos, neg = yi.sum(), (~yi).sum()
        stats[i] = (
            tp / (tp + fp) if tp + fp else np.nan,
            tp / pos if pos else np.nan,
            PER * fp / neg if neg else np.nan,
        )
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    names = ("precision", "recall", "false_alerts_per_10k_legit")
    return {
        name: [float(np.nanpercentile(stats[:, j], lo)), float(np.nanpercentile(stats[:, j], hi))]
        for j, name in enumerate(names)
    }
