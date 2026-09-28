"""Acquire, verify and record provenance for the ULB Credit Card Fraud dataset.

Usage:
    uv run python -m fraud.data download   # fetch from OpenML, verify, write provenance
    uv run python -m fraud.data verify     # offline re-check of local files against provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

OPENML_DATASET_ID = 1597
OPENML_DESCRIPTION_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_DATASET_ID}"

EXPECTED_NAME = "creditcard"
EXPECTED_VERSION = "1"
EXPECTED_TARGET = "Class"
EXPECTED_ARFF_MD5 = "178bcf9bb1f31a3dfe12d0e577884add"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Time",
    *(f"V{i}" for i in range(1, 29)),
    "Amount",
    "Class",
)

# Checked against the Kaggle API (mlg-ulb/creditcardfraud, version 3) on 2026-09-27.
ORIGINAL_DISTRIBUTION = {
    "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    "licence": "Database: Open Database License (ODbL) v1.0; "
    "Contents: Database Contents License (DbCL) v1.0",
    "kaggle_version": 3,
}
CITATION = (
    "Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. "
    "Calibrating Probability with Undersampling for Unbalanced Classification. "
    "In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015."
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_ARFF = DATA_DIR / "raw" / "creditcard.arff"
RAW_PARQUET = DATA_DIR / "raw" / "creditcard.parquet"
PROVENANCE_PATH = DATA_DIR / "provenance.json"


class DatasetVerificationError(RuntimeError):
    """The source metadata or the downloaded data does not match what we expect."""


def fetch_openml_description(url: str = OPENML_DESCRIPTION_URL) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["data_set_description"]


def verify_description(desc: dict[str, Any]) -> None:
    """Confirm the OpenML record is the ULB dataset before downloading anything."""
    checks = {
        "id": (str(desc.get("id")), str(OPENML_DATASET_ID)),
        "name": (desc.get("name"), EXPECTED_NAME),
        "version": (str(desc.get("version")), EXPECTED_VERSION),
        "default_target_attribute": (desc.get("default_target_attribute"), EXPECTED_TARGET),
        "md5_checksum": (desc.get("md5_checksum"), EXPECTED_ARFF_MD5),
        "status": (desc.get("status"), "active"),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        details = ", ".join(f"{k}: got {g!r}, expected {e!r}" for k, (g, e) in mismatches.items())
        raise DatasetVerificationError(f"OpenML description mismatch: {details}")
    if "Dal Pozzolo" not in " ".join(desc.get("creator", [])):
        raise DatasetVerificationError("OpenML creator list does not include the ULB authors")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def file_digest(path: Path, algorithm: str) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def parse_arff(path: Path) -> pd.DataFrame:
    """Parse the dense numeric ARFF directly.

    OpenML marks `Time` as the row-id attribute, so generic loaders such as
    sklearn's fetch_openml drop it; reading the file ourselves keeps every column.
    """
    names: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("@attribute"):
                names.append(stripped.split(maxsplit=2)[1].strip("'\""))
            elif lowered.startswith("@data"):
                break
        df = pd.read_csv(f, header=None, names=names, quotechar="'", comment="%")
    df["Class"] = df["Class"].astype("int8")
    return df


def validate_frame(
    df: pd.DataFrame,
    expected_rows: int = EXPECTED_ROWS,
    expected_fraud: int = EXPECTED_FRAUD,
) -> dict[str, Any]:
    """Check schema, counts and time ordering; return a summary for the provenance record."""
    if tuple(df.columns) != EXPECTED_COLUMNS:
        raise DatasetVerificationError(f"Unexpected columns: {list(df.columns)}")
    if df.isna().any().any():
        raise DatasetVerificationError("Dataset contains missing values")
    features = df.drop(columns="Class")
    if not all(pd.api.types.is_float_dtype(t) for t in features.dtypes):
        raise DatasetVerificationError("All feature columns must be floating point")
    if not set(df["Class"].unique()) <= {0, 1}:
        raise DatasetVerificationError("Class must contain only 0 and 1")
    if len(df) != expected_rows:
        raise DatasetVerificationError(f"Expected {expected_rows} rows, got {len(df)}")
    fraud = int(df["Class"].sum())
    if fraud != expected_fraud:
        raise DatasetVerificationError(f"Expected {expected_fraud} frauds, got {fraud}")
    if not df["Time"].is_monotonic_increasing:
        raise DatasetVerificationError("Time must be non-decreasing for a time-ordered split")

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "fraud_count": fraud,
        "legit_count": len(df) - fraud,
        "fraud_rate": round(fraud / len(df), 6),
        "time_min_seconds": float(df["Time"].min()),
        "time_max_seconds": float(df["Time"].max()),
        "time_non_decreasing": True,
        "content_sha256": frame_sha256(df),
    }


def frame_sha256(df: pd.DataFrame) -> str:
    """Hash of the parsed values, independent of the on-disk file format."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def build_provenance(
    desc: dict[str, Any], arff_path: Path, frame_summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "dataset": "ULB Credit Card Fraud Detection",
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {
            "provider": "OpenML",
            "dataset_id": OPENML_DATASET_ID,
            "name": desc["name"],
            "version": desc["version"],
            "version_label": desc.get("version_label"),
            "description_url": OPENML_DESCRIPTION_URL,
            "download_url": desc["url"],
            "upload_date": desc.get("upload_date"),
            "openml_licence": desc.get("licence"),
        },
        "original_distribution": ORIGINAL_DISTRIBUTION,
        "citation": CITATION,
        "file": {
            "format": "ARFF",
            "bytes": arff_path.stat().st_size,
            "md5": file_digest(arff_path, "md5"),
            "md5_matches_openml": True,
            "sha256": file_digest(arff_path, "sha256"),
        },
        "frame": frame_summary,
        "notes": [
            "OpenML marks Time as its row_id_attribute, so the ARFF is parsed directly "
            "to keep the Time column.",
            "Time is seconds elapsed since the first transaction in the dataset. "
            "Time = 0 is not known to be midnight, so Time must never be presented as "
            "local time or clock hour.",
            "V1-V28 are anonymised PCA components with no published meaning.",
            "The raw data is not committed; recreate it with `python -m fraud.data download`.",
        ],
    }


def acquire(desc: dict[str, Any] | None = None) -> dict[str, Any]:
    desc = desc or fetch_openml_description()
    verify_description(desc)

    if not RAW_ARFF.exists() or file_digest(RAW_ARFF, "md5") != EXPECTED_ARFF_MD5:
        print(f"Downloading {desc['url']} -> {RAW_ARFF.relative_to(ROOT)}")
        download(desc["url"], RAW_ARFF)
    md5 = file_digest(RAW_ARFF, "md5")
    if md5 != EXPECTED_ARFF_MD5:
        raise DatasetVerificationError(f"ARFF md5 {md5} does not match OpenML checksum")

    df = parse_arff(RAW_ARFF)
    summary = validate_frame(df)
    df.to_parquet(RAW_PARQUET, index=False)

    provenance = build_provenance(desc, RAW_ARFF, summary)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def verify_local() -> dict[str, Any]:
    """Offline check that local files still match the committed provenance record."""
    if not PROVENANCE_PATH.exists():
        raise DatasetVerificationError("data/provenance.json missing; run `download` first")
    provenance = json.loads(PROVENANCE_PATH.read_text())
    if not RAW_ARFF.exists():
        raise DatasetVerificationError("Raw ARFF missing; run `download` first")
    sha = file_digest(RAW_ARFF, "sha256")
    if sha != provenance["file"]["sha256"]:
        raise DatasetVerificationError("Raw ARFF sha256 does not match provenance")
    df = load_raw()
    summary = validate_frame(df)
    if summary["content_sha256"] != provenance["frame"]["content_sha256"]:
        raise DatasetVerificationError("Parsed content hash does not match provenance")
    return summary


def load_raw() -> pd.DataFrame:
    """Load the verified dataset (parquet cache, falling back to the ARFF)."""
    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    return parse_arff(RAW_ARFF)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud.data")
    parser.add_argument("command", choices=["download", "verify"])
    args = parser.parse_args(argv)
    try:
        result = acquire() if args.command == "download" else verify_local()
    except DatasetVerificationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    frame = result.get("frame", result)
    print(
        f"OK: {frame['rows']} rows, {frame['fraud_count']} frauds, "
        f"content sha256 {frame['content_sha256'][:12]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
