import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_raw
from fraud import data
from fraud.data import DatasetVerificationError

GOOD_DESCRIPTION = {
    "id": "1597",
    "name": "creditcard",
    "version": "1",
    "default_target_attribute": "Class",
    "md5_checksum": data.EXPECTED_ARFF_MD5,
    "status": "active",
    "creator": ["Andrea Dal Pozzolo", "Olivier Caelen", "Gianluca Bontempi"],
}


def make_frame(rows: int = 10, frauds: int = 2) -> pd.DataFrame:
    df = make_raw(rows)
    df["Class"] = np.array([1] * frauds + [0] * (rows - frauds), dtype="int8")
    return df


def write_arff(path, rows: list[list[str]]) -> None:
    header = ["% comment line", "@relation creditcard"]
    header += [f"@attribute '{c}' numeric" for c in data.EXPECTED_COLUMNS[:-1]]
    header += ["@attribute 'Class' {'0','1'}", "@data"]
    path.write_text("\n".join([*header, *(",".join(r) for r in rows)]) + "\n")


class TestVerifyDescription:
    def test_accepts_expected_record(self):
        data.verify_description(GOOD_DESCRIPTION)

    @pytest.mark.parametrize(
        "field,value",
        [("md5_checksum", "0" * 32), ("name", "other"), ("id", "42"), ("status", "deactivated")],
    )
    def test_rejects_mismatched_field(self, field, value):
        with pytest.raises(DatasetVerificationError, match=field):
            data.verify_description({**GOOD_DESCRIPTION, field: value})

    def test_rejects_unexpected_creator(self):
        with pytest.raises(DatasetVerificationError, match="creator"):
            data.verify_description({**GOOD_DESCRIPTION, "creator": ["Someone Else"]})


class TestParseArff:
    def test_keeps_time_and_parses_quoted_class(self, tmp_path):
        path = tmp_path / "tiny.arff"
        values = [str(float(i)) for i in range(30)]
        write_arff(path, [[*values, "'0'"], [*values, "'1'"]])

        df = data.parse_arff(path)

        assert tuple(df.columns) == data.EXPECTED_COLUMNS
        assert df["Class"].tolist() == [0, 1]
        assert df["Class"].dtype == "int8"
        assert df["Time"].tolist() == [0.0, 0.0]


class TestValidateFrame:
    def test_valid_frame_returns_summary(self):
        summary = data.validate_frame(make_frame(), expected_rows=10, expected_fraud=2)
        assert summary["rows"] == 10
        assert summary["fraud_count"] == 2
        assert summary["legit_count"] == 8
        assert summary["time_non_decreasing"] is True
        assert len(summary["content_sha256"]) == 64

    def test_rejects_wrong_columns(self):
        df = make_frame().rename(columns={"V1": "X1"})
        with pytest.raises(DatasetVerificationError, match="columns"):
            data.validate_frame(df, expected_rows=10, expected_fraud=2)

    def test_rejects_missing_values(self):
        df = make_frame()
        df.loc[0, "V3"] = np.nan
        with pytest.raises(DatasetVerificationError, match="missing"):
            data.validate_frame(df, expected_rows=10, expected_fraud=2)

    def test_rejects_non_binary_class(self):
        df = make_frame()
        df.loc[0, "Class"] = 2
        with pytest.raises(DatasetVerificationError, match="Class"):
            data.validate_frame(df, expected_rows=10, expected_fraud=2)

    def test_rejects_wrong_row_count(self):
        with pytest.raises(DatasetVerificationError, match="rows"):
            data.validate_frame(make_frame(), expected_rows=11, expected_fraud=2)

    def test_rejects_wrong_fraud_count(self):
        with pytest.raises(DatasetVerificationError, match="frauds"):
            data.validate_frame(make_frame(), expected_rows=10, expected_fraud=3)

    def test_rejects_unordered_time(self):
        df = make_frame()
        df.loc[5, "Time"] = -1.0
        with pytest.raises(DatasetVerificationError, match="Time"):
            data.validate_frame(df, expected_rows=10, expected_fraud=2)


def test_frame_hash_is_value_sensitive():
    df = make_frame()
    changed = df.copy()
    changed.loc[0, "Amount"] += 0.01
    assert data.frame_sha256(df) == data.frame_sha256(df.copy())
    assert data.frame_sha256(df) != data.frame_sha256(changed)


def test_file_digest(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert data.file_digest(path, "md5") == "900150983cd24fb0d6963f7d28e17f72"


@pytest.mark.data
@pytest.mark.skipif(not data.RAW_ARFF.exists(), reason="raw dataset not downloaded")
def test_local_dataset_matches_provenance():
    summary = data.verify_local()
    provenance = json.loads(data.PROVENANCE_PATH.read_text())
    assert summary["rows"] == data.EXPECTED_ROWS == provenance["frame"]["rows"]
    assert summary["fraud_count"] == data.EXPECTED_FRAUD
    assert provenance["file"]["md5"] == data.EXPECTED_ARFF_MD5
