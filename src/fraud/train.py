"""Freeze the production model, its threshold and the comparison models.

Usage:
    uv run python -m fraud.train

Uses train and validation only. The configuration below was chosen from the Stage 2
validation report (`reports/validation/`) and recorded in docs/adr.md.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import xgboost

from fraud import data, metrics, models
from fraud.features import build_features, feature_names
from fraud.prepare import SPLIT_SUMMARY_PATH, load_split

MODEL_VERSION = "model-v1"
ARTIFACT_DIR = data.ROOT / "artifacts" / MODEL_VERSION

INCLUDE_TIME_CYCLE = False
FP_BUDGET_PER_10K = 5
XGB_CONFIG = {"max_depth": 5, "scale_pos_weight": 1.0}
LR_CONFIG = {"C": 1.0, "class_weight": "balanced"}

PRODUCTION_FILE = "xgboost.ubj"
LR_FILE = "logistic_regression.joblib"
IFOREST_FILE = "isolation_forest.joblib"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_point(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    threshold = metrics.threshold_for_fp_budget(y, scores, FP_BUDGET_PER_10K)
    return {
        **metrics.confusion_at(y, scores, threshold),
        **metrics.threshold_free(y, scores),
    }


def freeze() -> dict[str, Any]:
    train, validation = load_split("train"), load_split("validation")
    X_train = build_features(train, include_time_cycle=INCLUDE_TIME_CYCLE)
    X_val = build_features(validation, include_time_cycle=INCLUDE_TIME_CYCLE)
    y_train, y_val = train["Class"], validation["Class"].to_numpy()

    xgb_model, n_rounds = models.fit_xgboost(X_train, y_train, **XGB_CONFIG)
    lr_model = models.logistic_regression(**LR_CONFIG).fit(X_train, y_train)
    iforest = models.isolation_forest().fit(X_train)

    points = {
        name: frozen_point(y_val, models.risk_scores(model, X_val))
        for name, model in (
            ("xgboost", xgb_model),
            ("logistic_regression", lr_model),
            ("isolation_forest", iforest),
        )
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    xgb_model.save_model(ARTIFACT_DIR / PRODUCTION_FILE)
    joblib.dump(lr_model, ARTIFACT_DIR / LR_FILE)
    joblib.dump(iforest, ARTIFACT_DIR / IFOREST_FILE)

    split_summary = json.loads(SPLIT_SUMMARY_PATH.read_text())
    files = {
        name: {"sha256": sha256(ARTIFACT_DIR / name), "bytes": (ARTIFACT_DIR / name).stat().st_size}
        for name in (PRODUCTION_FILE, LR_FILE, IFOREST_FILE)
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_record": "docs/adr.md (Stage 2 decisions)",
        "features": {
            "names": list(feature_names(INCLUDE_TIME_CYCLE)),
            "include_time_cycle": INCLUDE_TIME_CYCLE,
            "builder": "fraud.features.build_features",
        },
        "production": {
            "family": "xgboost",
            "file": PRODUCTION_FILE,
            "params": {
                **XGB_CONFIG,
                **models.XGB_FIXED_PARAMS,
                "n_estimators": n_rounds,
            },
            "threshold": points["xgboost"]["threshold"],
            "threshold_rule": (
                f"Lowest threshold with at most {FP_BUDGET_PER_10K} false alerts per 10,000 "
                "legitimate validation transactions (maximum recall within budget)."
            ),
            "flag_rule": "risk_score >= threshold",
            "score_semantics": "Uncalibrated risk score in [0, 1]; not a probability.",
            "validation_at_threshold": points["xgboost"],
        },
        "comparators": {
            "logistic_regression": {
                "file": LR_FILE,
                "params": LR_CONFIG,
                "threshold": points["logistic_regression"]["threshold"],
                "validation_at_threshold": points["logistic_regression"],
            },
            "isolation_forest": {
                "file": IFOREST_FILE,
                "params": {"n_estimators": iforest.n_estimators, "random_state": models.SEED},
                "threshold": points["isolation_forest"]["threshold"],
                "validation_at_threshold": points["isolation_forest"],
                "note": "Comparison only; fitted without labels; excluded from explanations.",
            },
        },
        "fp_budget_per_10k": FP_BUDGET_PER_10K,
        "data": {
            "source_content_sha256": split_summary["source_content_sha256"],
            "train_ids_sha256": split_summary["splits"]["train"]["transaction_ids_sha256"],
            "validation_ids_sha256": split_summary["splits"]["validation"][
                "transaction_ids_sha256"
            ],
            "splits_used": ["train", "validation"],
            "licence": data.ORIGINAL_DISTRIBUTION["licence"],
            "attribution": data.CITATION,
        },
        "commit_policy": (
            "metadata.json and the XGBoost model are committed. The joblib comparators are "
            "pickles (version-bound; the Isolation Forest exceeds 1 MB) and are regenerated "
            "byte-identically by `python -m fraud.train`."
        ),
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "files": files,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    metadata = freeze()
    p = metadata["production"]
    v = p["validation_at_threshold"]
    print(
        f"Froze {MODEL_VERSION}: threshold {p['threshold']:.6f}, validation recall "
        f"{v['recall']:.3f}, precision {v['precision']:.3f}, {v['fp']} false alerts"
    )
    for name, f in metadata["files"].items():
        print(f"  {name:<28} {f['bytes'] / 1024:>8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
