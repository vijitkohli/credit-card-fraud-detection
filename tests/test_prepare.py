import json
from itertools import pairwise

import pandas as pd
import pytest

from conftest import make_raw
from fraud import prepare
from fraud.prepare import SPLIT_ORDER


def split_frame(rows: int = 1_000) -> pd.DataFrame:
    df, _ = prepare.deduplicate(prepare.add_transaction_ids(make_raw(rows)))
    df["split"] = prepare.time_ordered_split(df)
    return df


class TestDeduplicate:
    def test_removes_exact_duplicates_keeping_first(self, raw_frame):
        with_dups = pd.concat([raw_frame, raw_frame.iloc[[3, 3, 50]]], ignore_index=True)
        df, report = prepare.deduplicate(prepare.add_transaction_ids(with_dups))

        assert report["exact_duplicates_removed"] == 3
        assert report["rows_after"] == len(raw_frame)
        assert df["transaction_id"].tolist() == list(range(len(raw_frame)))

    def test_counts_fraud_duplicates(self, raw_frame):
        fraud_row = raw_frame[raw_frame["Class"] == 1].iloc[[0]]
        with_dups = pd.concat([raw_frame, fraud_row], ignore_index=True)
        _, report = prepare.deduplicate(prepare.add_transaction_ids(with_dups))
        assert report["fraud_duplicates_removed"] == 1

    def test_keeps_rows_that_differ_only_in_time(self, raw_frame):
        repeat = raw_frame.iloc[[5]].assign(Time=raw_frame["Time"].max() + 1)
        df, report = prepare.deduplicate(
            prepare.add_transaction_ids(pd.concat([raw_frame, repeat], ignore_index=True))
        )
        assert report["exact_duplicates_removed"] == 0
        assert len(df) == len(raw_frame) + 1

    def test_rejects_conflicting_labels(self, raw_frame):
        conflict = raw_frame.iloc[[1]].assign(Class=1 - raw_frame["Class"].iloc[1])
        with pytest.raises(ValueError, match="conflicting"):
            prepare.deduplicate(
                prepare.add_transaction_ids(pd.concat([raw_frame, conflict], ignore_index=True))
            )


class TestTimeOrderedSplit:
    def test_every_row_assigned_once_in_expected_proportions(self):
        df = split_frame(1_000)
        counts = df["split"].value_counts()
        assert counts.sum() == len(df)
        assert counts.to_dict() == {"train": 600, "validation": 200, "test": 200}

    def test_splits_are_strictly_ordered_in_time(self):
        df = split_frame()
        for earlier, later in pairwise(SPLIT_ORDER):
            assert df.loc[df["split"] == earlier, "Time"].max() < (
                df.loc[df["split"] == later, "Time"].min()
            )

    def test_rows_sharing_a_boundary_time_stay_together(self):
        raw = make_raw(100)
        raw.loc[55:65, "Time"] = raw.loc[55, "Time"]
        raw["Time"] = raw["Time"].cummax()
        df, _ = prepare.deduplicate(prepare.add_transaction_ids(raw))
        df["split"] = prepare.time_ordered_split(df)

        boundary = df[df["Time"] == raw.loc[55, "Time"]]
        assert boundary["split"].nunique() == 1

    def test_is_deterministic_and_index_aligned(self):
        df = split_frame()
        shuffled = df.drop(columns="split").sample(frac=1, random_state=1)
        again = prepare.time_ordered_split(shuffled)
        pd.testing.assert_series_equal(again.sort_index(), df["split"].sort_index())


class TestDemoSubset:
    def test_contains_only_validation_rows_and_all_validation_frauds(self):
        df = split_frame()
        demo = prepare.select_demo_subset(df, legit_sample=50)
        validation = df[df["split"] == "validation"]

        assert set(demo["split"]) == {"validation"}
        assert set(validation.loc[validation["Class"] == 1, "transaction_id"]) <= set(
            demo["transaction_id"]
        )
        assert (demo["Class"] == 0).sum() == 50
        assert demo["Time"].is_monotonic_increasing

    def test_is_deterministic(self):
        df = split_frame()
        pd.testing.assert_frame_equal(
            prepare.select_demo_subset(df, legit_sample=50),
            prepare.select_demo_subset(df, legit_sample=50),
        )


def test_cross_split_overlap_detects_time_shifted_repeats():
    df = split_frame()
    train_row = df[df["split"] == "train"].iloc[0]
    test_idx = df.index[df["split"] == "test"][0]
    df.loc[test_idx, prepare.MODEL_VISIBLE_COLUMNS] = train_row[prepare.MODEL_VISIBLE_COLUMNS]

    overlap = prepare.cross_split_overlap(df)
    assert overlap["test"]["rows_also_in_train"] == 1
    assert overlap["validation"]["rows_also_in_train"] == 0


@pytest.mark.data
@pytest.mark.skipif(not prepare.TRANSACTIONS_PATH.exists(), reason="run fraud.prepare first")
def test_prepared_real_data_is_consistent():
    summary = json.loads(prepare.SPLIT_SUMMARY_PATH.read_text())
    df = pd.read_parquet(prepare.TRANSACTIONS_PATH)

    assert len(df) == summary["deduplication"]["rows_after"]
    assert not df.drop(columns=["transaction_id", "split"]).duplicated().any()
    assert df["transaction_id"].is_unique
    for earlier, later in pairwise(SPLIT_ORDER):
        assert df.loc[df["split"] == earlier, "Time"].max() < (
            df.loc[df["split"] == later, "Time"].min()
        )
    for name, s in summary["splits"].items():
        assert s["fraud"] > 0, f"{name} has no fraud"
        assert (
            prepare.ids_sha256(df.loc[df["split"] == name, "transaction_id"])
            == (s["transaction_ids_sha256"])
        )
