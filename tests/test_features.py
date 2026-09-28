import numpy as np
import pandas as pd
import pytest

from fraud.features import (
    BASE_FEATURES,
    TIME_CYCLE_COLUMNS,
    build_features,
    feature_names,
)


def test_default_features_exclude_time(raw_frame):
    features = build_features(raw_frame)
    assert tuple(features.columns) == BASE_FEATURES
    assert not any("time" in c.lower() for c in features.columns)


def test_log_amount(raw_frame):
    features = build_features(raw_frame)
    np.testing.assert_allclose(features["log_amount"], np.log1p(raw_frame["Amount"]))


def test_pca_columns_pass_through_unchanged(raw_frame):
    features = build_features(raw_frame)
    pd.testing.assert_frame_equal(features[["V1", "V28"]], raw_frame[["V1", "V28"]])


def test_time_cycle_is_opt_in_bounded_and_periodic(raw_frame):
    features = build_features(raw_frame, include_time_cycle=True)
    assert tuple(features.columns) == feature_names(include_time_cycle=True)
    assert features[list(TIME_CYCLE_COLUMNS)].abs().le(1.0).all().all()

    shifted = raw_frame.assign(Time=raw_frame["Time"] + 86_400)
    pd.testing.assert_frame_equal(features, build_features(shifted, include_time_cycle=True))


def test_single_row_matches_batch(raw_frame):
    """Serving builds features one transaction at a time; it must match training batches."""
    batch = build_features(raw_frame, include_time_cycle=True)
    for i in (0, 17, len(raw_frame) - 1):
        single = build_features(raw_frame.iloc[[i]], include_time_cycle=True)
        pd.testing.assert_frame_equal(single, batch.iloc[[i]])


def test_ignores_extra_columns(raw_frame):
    extra = raw_frame.assign(transaction_id=1, split="train")
    pd.testing.assert_frame_equal(build_features(extra), build_features(raw_frame))


def test_missing_column_raises(raw_frame):
    with pytest.raises(ValueError, match="V7"):
        build_features(raw_frame.drop(columns="V7"))


def test_negative_amount_raises(raw_frame):
    with pytest.raises(ValueError, match="Amount"):
        build_features(raw_frame.assign(Amount=-1.0))
