# Real-Time Credit Card Fraud Detection POC

A proof of concept that scores card transactions for fraud risk, explains each flagged
decision in plain English grounded in the model's own evidence, and shows results in a
live dashboard. The ML model makes the decision; the explanation layer only describes it.

- Architecture decisions: [`docs/adr.md`](docs/adr.md)
- Staged plan and current status: [`docs/implementation-plan.md`](docs/implementation-plan.md)
- Dataset provenance: [`data/provenance.json`](data/provenance.json)
- Original brief: [`fraud-detection-poc-handoff.md`](fraud-detection-poc-handoff.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync                                   # create .venv and install locked dependencies
uv run pre-commit install                 # enable git hooks
uv run python -m fraud.data download      # fetch ULB dataset from OpenML (~150 MB), verify, record provenance
uv run python -m fraud.data verify        # offline re-check against data/provenance.json
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run pre-commit run --all-files
```

## Dataset

ULB Credit Card Fraud Detection (Dal Pozzolo et al., 2015): 284,807 European card
transactions over two days in September 2013, 492 of them fraudulent (0.172%).
`V1`–`V28` are anonymised PCA components with no published meaning; `Time` is seconds
since the first transaction, not a clock time. Licensed ODbL v1.0 (database) / DbCL v1.0
(contents). The raw data is not committed; the download command recreates it and checks
it against the OpenML checksum.
