"""Load frozen model artefacts, verifying them against their recorded hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud import data
from fraud.features import build_features
from fraud.models import risk_scores

DEFAULT_DIR = data.ROOT / "artifacts" / "model-v1"


class ArtifactError(RuntimeError):
    """An artefact is missing or does not match its recorded hash."""


@dataclass(frozen=True)
class FrozenModel:
    metadata: dict[str, Any]
    production: xgb.XGBClassifier
    comparators: dict[str, Any]

    @property
    def threshold(self) -> float:
        return self.metadata["production"]["threshold"]

    @property
    def include_time_cycle(self) -> bool:
        return self.metadata["features"]["include_time_cycle"]

    def features(self, raw: pd.DataFrame) -> pd.DataFrame:
        return build_features(raw, include_time_cycle=self.include_time_cycle)

    def score(self, raw: pd.DataFrame) -> np.ndarray:
        return risk_scores(self.production, self.features(raw))


def _verified(directory: Path, name: str, expected: dict[str, Any]) -> Path:
    path = directory / name
    if not path.exists():
        raise ArtifactError(f"{path} is missing; run `python -m fraud.train`")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ArtifactError(f"{path} does not match the sha256 in metadata.json")
    return path


def load(directory: Path = DEFAULT_DIR, include_comparators: bool = False) -> FrozenModel:
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ArtifactError(f"{metadata_path} is missing; run `python -m fraud.train`")
    metadata = json.loads(metadata_path.read_text())
    files = metadata["files"]

    production = xgb.XGBClassifier()
    production.load_model(
        _verified(directory, metadata["production"]["file"], files[metadata["production"]["file"]])
    )

    comparators = {}
    if include_comparators:
        # joblib files are pickles: only load files whose hash matches our own metadata.
        for name, spec in metadata["comparators"].items():
            comparators[name] = joblib.load(_verified(directory, spec["file"], files[spec["file"]]))
    return FrozenModel(metadata=metadata, production=production, comparators=comparators)
