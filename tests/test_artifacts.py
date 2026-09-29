import shutil

import numpy as np
import pytest

from fraud import artifacts, metrics
from fraud.prepare import TRANSACTIONS_PATH, load_split

needs_artifacts = pytest.mark.skipif(
    not (artifacts.DEFAULT_DIR / "metadata.json").exists(), reason="run fraud.train first"
)


@needs_artifacts
def test_tampered_model_is_rejected(tmp_path):
    for f in ("metadata.json", "xgboost.ubj"):
        shutil.copy(artifacts.DEFAULT_DIR / f, tmp_path / f)
    with (tmp_path / "xgboost.ubj").open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(artifacts.ArtifactError, match="sha256"):
        artifacts.load(tmp_path)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(artifacts.ArtifactError, match="missing"):
        artifacts.load(tmp_path)


@needs_artifacts
def test_metadata_records_the_agreed_decisions():
    model = artifacts.load()
    meta = model.metadata
    assert meta["fp_budget_per_10k"] == 5
    assert model.include_time_cycle is False
    assert not any("time" in name for name in meta["features"]["names"])
    assert meta["data"]["splits_used"] == ["train", "validation"]


@pytest.mark.data
@needs_artifacts
@pytest.mark.skipif(not TRANSACTIONS_PATH.exists(), reason="run fraud.prepare first")
def test_frozen_model_reproduces_validation_operating_point():
    model = artifacts.load(include_comparators=True)
    validation = load_split("validation")
    scores = model.score(validation)

    point = metrics.confusion_at(validation["Class"], scores, model.threshold)
    recorded = model.metadata["production"]["validation_at_threshold"]
    for key in ("tp", "fp", "fn", "tn"):
        assert point[key] == recorded[key]
    assert point["false_alerts_per_10k_legit"] <= model.metadata["fp_budget_per_10k"]

    single = model.score(validation.iloc[[0]])
    np.testing.assert_allclose(single, scores[:1], rtol=1e-6)
    assert set(model.comparators) == {"logistic_regression", "isolation_forest"}
