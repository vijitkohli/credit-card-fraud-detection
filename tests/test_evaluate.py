import pytest

from fraud import evaluate


def test_refuses_to_overwrite_existing_test_report(tmp_path):
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(evaluate.AlreadyEvaluatedError, match="evaluated once"):
        evaluate.run(report_dir=tmp_path)
