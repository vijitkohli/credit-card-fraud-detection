"""De-duplicate, split in time order, and carve out the development demo subset.

Usage:
    uv run python -m fraud.prepare

Writes `data/processed/transactions.parquet` (gitignored; raw columns plus
`transaction_id` and `split`), `data/processed/demo_subset.parquet` (gitignored), and the
committed summary `reports/split_summary.json`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from fraud import data
from fraud.features import PCA_COLUMNS

SPLIT_FRACTIONS = {"train": 0.6, "validation": 0.2, "test": 0.2}
SPLIT_ORDER = tuple(SPLIT_FRACTIONS)
DEMO_LEGIT_SAMPLE = 2_000
DEMO_SEED = 20260927

PROCESSED_DIR = data.DATA_DIR / "processed"
TRANSACTIONS_PATH = PROCESSED_DIR / "transactions.parquet"
DEMO_SUBSET_PATH = PROCESSED_DIR / "demo_subset.parquet"
SPLIT_SUMMARY_PATH = data.ROOT / "reports" / "split_summary.json"

# Model inputs exclude raw Time, so rows equal on these look identical to the model.
MODEL_VISIBLE_COLUMNS = [*PCA_COLUMNS, "Amount"]


def add_transaction_ids(raw: pd.DataFrame) -> pd.DataFrame:
    """`transaction_id` is the 0-based row position in the source file, for traceability."""
    out = raw.copy()
    out.insert(0, "transaction_id", np.arange(len(raw), dtype="int64"))
    return out


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop exact duplicates (all source columns equal), keeping the first occurrence."""
    source_columns = [c for c in df.columns if c != "transaction_id"]
    feature_columns = [c for c in source_columns if c != "Class"]

    conflicting = df.groupby(feature_columns)["Class"].nunique().gt(1).sum()
    if conflicting:
        raise ValueError(f"{conflicting} feature-identical groups have conflicting labels")

    is_dup = df.duplicated(subset=source_columns, keep="first")
    report = {
        "rows_before": len(df),
        "exact_duplicates_removed": int(is_dup.sum()),
        "fraud_duplicates_removed": int(df.loc[is_dup, "Class"].sum()),
        "rows_after": int((~is_dup).sum()),
        "conflicting_label_groups": 0,
    }
    return df.loc[~is_dup].reset_index(drop=True), report


def time_ordered_split(df: pd.DataFrame) -> pd.Series:
    """Assign train/validation/test by time order.

    Cut points are pushed forward past any rows sharing the boundary `Time`, so every
    split boundary is a strict time boundary.
    """
    ordered = df.sort_values("Time", kind="stable")
    times = ordered["Time"].to_numpy()
    n = len(ordered)

    cuts = []
    cumulative = 0.0
    for name in SPLIT_ORDER[:-1]:
        cumulative += SPLIT_FRACTIONS[name]
        cut = round(n * cumulative)
        while 0 < cut < n and times[cut] == times[cut - 1]:
            cut += 1
        cuts.append(cut)

    labels = np.empty(n, dtype=object)
    bounds = [0, *cuts, n]
    for name, start, stop in zip(SPLIT_ORDER, bounds[:-1], bounds[1:], strict=True):
        labels[start:stop] = name
    return pd.Series(labels, index=ordered.index, name="split").reindex(df.index)


def select_demo_subset(
    df: pd.DataFrame, legit_sample: int = DEMO_LEGIT_SAMPLE, seed: int = DEMO_SEED
) -> pd.DataFrame:
    """All validation frauds plus a fixed random sample of validation legitimate rows.

    Used for API and dashboard development so the held-out test set stays untouched.
    """
    validation = df[df["split"] == "validation"]
    frauds = validation[validation["Class"] == 1]
    legit = validation[validation["Class"] == 0]
    sampled = legit.sample(n=min(legit_sample, len(legit)), random_state=seed)
    return pd.concat([frauds, sampled]).sort_values("Time", kind="stable").reset_index(drop=True)


def cross_split_overlap(df: pd.DataFrame) -> dict[str, Any]:
    """Count rows whose model-visible values also appear in train (same values, other Time)."""
    train = df.loc[df["split"] == "train", MODEL_VISIBLE_COLUMNS]
    train_keys = set(pd.util.hash_pandas_object(train, index=False))
    out: dict[str, Any] = {}
    for name in SPLIT_ORDER[1:]:
        part = df[df["split"] == name]
        keys = pd.util.hash_pandas_object(part[MODEL_VISIBLE_COLUMNS], index=False)
        seen = keys.isin(train_keys).to_numpy()
        out[name] = {
            "rows_also_in_train": int(seen.sum()),
            "fraud_rows_also_in_train": int(part.loc[seen, "Class"].sum()),
            "share_of_split": round(float(seen.mean()), 6),
        }
    return out


def ids_sha256(ids: pd.Series) -> str:
    return hashlib.sha256(np.sort(ids.to_numpy(dtype="int64")).tobytes()).hexdigest()


def summarise(
    df: pd.DataFrame,
    demo: pd.DataFrame,
    dedup_report: dict[str, Any],
    source_content_sha256: str,
) -> dict[str, Any]:
    splits = {}
    for name in SPLIT_ORDER:
        part = df[df["split"] == name]
        fraud = int(part["Class"].sum())
        splits[name] = {
            "rows": len(part),
            "fraud": fraud,
            "legit": len(part) - fraud,
            "fraud_rate": round(fraud / len(part), 6),
            "time_min_seconds": float(part["Time"].min()),
            "time_max_seconds": float(part["Time"].max()),
            "transaction_ids_sha256": ids_sha256(part["transaction_id"]),
        }
    return {
        "source_content_sha256": source_content_sha256,
        "deduplication": dedup_report,
        "split_fractions": SPLIT_FRACTIONS,
        "splits": splits,
        "model_visible_overlap_with_train": cross_split_overlap(df),
        "demo_subset": {
            "source_split": "validation",
            "rows": len(demo),
            "fraud": int(demo["Class"].sum()),
            "legit_sample": int((demo["Class"] == 0).sum()),
            "seed": DEMO_SEED,
            "transaction_ids_sha256": ids_sha256(demo["transaction_id"]),
        },
    }


def prepare() -> dict[str, Any]:
    source = data.verify_local()
    raw = add_transaction_ids(data.load_raw())
    deduped, dedup_report = deduplicate(raw)
    deduped["split"] = time_ordered_split(deduped)
    demo = select_demo_subset(deduped)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(TRANSACTIONS_PATH, index=False)
    demo.to_parquet(DEMO_SUBSET_PATH, index=False)

    summary = summarise(deduped, demo, dedup_report, source["content_sha256"])
    SPLIT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_split(name: str, allow_test: bool = False) -> pd.DataFrame:
    """Load one split in time order.

    The test split is held out until the model, preprocessing and threshold are frozen;
    only the one-time final evaluation passes `allow_test=True`.
    """
    if name not in SPLIT_FRACTIONS:
        raise ValueError(f"Unknown split {name!r}")
    if name == "test" and not allow_test:
        raise PermissionError(
            "The test split is held out; pass allow_test=True only for the final evaluation"
        )
    df = pd.read_parquet(TRANSACTIONS_PATH)
    return df[df["split"] == name].reset_index(drop=True)


def main() -> int:
    summary = prepare()
    for name, s in summary["splits"].items():
        print(f"{name:<10} {s['rows']:>7} rows  {s['fraud']:>3} fraud  rate {s['fraud_rate']:.4%}")
    d = summary["demo_subset"]
    print(f"demo       {d['rows']:>7} rows  {d['fraud']:>3} fraud  (from validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
