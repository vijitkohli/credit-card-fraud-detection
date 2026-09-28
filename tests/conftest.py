import numpy as np
import pandas as pd
import pytest

from fraud import data


def make_raw(rows: int = 100, fraud_every: int = 10, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame in the source schema, time-ordered."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(rows, len(data.EXPECTED_COLUMNS) - 1)),
        columns=data.EXPECTED_COLUMNS[:-1],
    )
    df["Time"] = np.arange(rows, dtype=float) * 100
    df["Amount"] = rng.uniform(0, 500, size=rows)
    df["Class"] = (np.arange(rows) % fraud_every == 0).astype("int8")
    return df


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw()
