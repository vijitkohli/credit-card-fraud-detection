"""Model features, shared by training and serving.

Every transformation here is stateless and row-wise, so a single transaction at serving
time gets exactly the features it would have had in a training batch. Anything fitted
(e.g. scaling for Logistic Regression) lives inside the model pipeline, fitted on train only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PCA_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 29))
RAW_INPUT_COLUMNS: tuple[str, ...] = ("Time", *PCA_COLUMNS, "Amount")

SECONDS_PER_CYCLE = 86_400
# Position within an assumed 24-hour cycle whose start offset is unknown: Time = 0 is not
# known to be midnight, so these must never be presented as clock or local time.
TIME_CYCLE_COLUMNS: tuple[str, ...] = (
    "time_within_daily_cycle_sin",
    "time_within_daily_cycle_cos",
)

BASE_FEATURES: tuple[str, ...] = (*PCA_COLUMNS, "log_amount")


def feature_names(include_time_cycle: bool = False) -> tuple[str, ...]:
    return (*BASE_FEATURES, *TIME_CYCLE_COLUMNS) if include_time_cycle else BASE_FEATURES


def build_features(raw: pd.DataFrame, include_time_cycle: bool = False) -> pd.DataFrame:
    """Map raw transactions (Time, V1-V28, Amount) to model features.

    Raw `Time` is never a model input: it only encodes position in the two-day window.
    """
    missing = [c for c in RAW_INPUT_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw input columns: {missing}")
    if (raw["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative")

    features = raw.loc[:, list(PCA_COLUMNS)].astype("float64")
    features["log_amount"] = np.log1p(raw["Amount"].astype("float64"))
    if include_time_cycle:
        angle = 2 * np.pi * (raw["Time"].astype("float64") % SECONDS_PER_CYCLE) / SECONDS_PER_CYCLE
        features[TIME_CYCLE_COLUMNS[0]] = np.sin(angle)
        features[TIME_CYCLE_COLUMNS[1]] = np.cos(angle)
    return features.loc[:, list(feature_names(include_time_cycle))]
