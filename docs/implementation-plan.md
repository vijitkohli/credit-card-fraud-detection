# Staged Implementation Plan

Each stage ends with runnable, tested output and one commit. Decisions are in `adr.md`.

## Stage 0: Foundations ✅

- uv + Python 3.12, `pyproject.toml`, ruff, pytest, pre-commit, `.gitignore`.
- `fraud.data`: verify the OpenML record, download, check MD5, parse the ARFF, validate schema
  and counts, write `data/provenance.json`.
- No training, API or frontend work.

## Stage 1: Data preparation ✅

- `features.py`: stateless, row-wise features shared by training and serving:
  `V1`–`V28`, `log_amount`, and opt-in `time_within_daily_cycle_sin/cos` for the ablation
  only. Raw `Time` is never a model input.
- `prepare.py` (`python -m fraud.prepare`):
  - `transaction_id` = row position in the source file.
  - Exact de-duplication: 1,081 rows removed, 19 of them fraud; no conflicting labels.
  - Time-ordered 60/20/20 split with strict time boundaries; rows sharing a boundary
    `Time` stay together.
  - Demo subset: all 57 validation frauds plus 2,000 legitimate validation rows (fixed seed).
  - `reports/split_summary.json` (committed): sizes, fraud counts, time ranges, split ID
    hashes, and the overlap between model inputs and training rows.
- Tests: split coverage, strict time ordering, boundary ties, determinism, demo subset only
  from validation, feature parity between single rows and batches, overlap detection.

## Stage 2: Models and validation (test set untouched)

- Train Logistic Regression, Isolation Forest and XGBoost (small grid, early stopping,
  `scale_pos_weight` in {1, √ratio}).
- Time-feature ablation: with and without `time_within_daily_cycle`.
- Validation report: AUPRC per model, and recall, precision, F1 and confusion counts at 1, 5,
  10 and 20 false alerts per 10k legitimate, plus max-F1 for comparison.
- **Review checkpoint:** recommend the production model and final operating point. Once
  agreed, freeze model, preprocessing and threshold into `artifacts/model-v1/`.
- Inspect artefact sizes before committing (ADR D10).

## Stage 3: One-time test evaluation

- Evaluate the frozen model once on the test set: all metrics, bootstrap 95% confidence
  intervals, PR curves, confusion matrix, the "flag nothing" baseline.
- Write `reports/metrics.json` and plots. Any later change requires a new model version and
  is disclosed.

## Stage 4: Evidence and explanation layer (offline, no API key)

- `evidence.py`: SHAP top-k contributions, percentiles relative to legitimate and fraudulent
  training data, novelty percentile.
- `explain.py`: deterministic template explanation, output schema, grounding validator,
  `source` field, file cache.
- CLI that explains a validation or demo transaction; batch grounding pass-rate check.
- Tests: evidence determinism, template never mentions clock time or V-feature meanings,
  validator rejects ungrounded features and numbers.
- **Checkpoint:** choose the LLM provider, then add it behind the single `explain` seam with
  timeout, retry and fallback.

## Stage 5: API

- FastAPI: `POST /api/score`, `GET /api/transactions/{id}`, `GET /api/metrics/official`,
  `GET /api/demo/session`, `POST /api/replay/{start,stop}`, `POST /api/demo/inject-fraud`,
  SSE `/api/stream`.
- Replay from the validation or demo subset; demo-mode fraud boost is clearly flagged in
  every event.
- Tests: contract tests, source labelling, separation of official and demo counters. Plus a
  scoring-latency benchmark (p50/p95).

## Stage 6: Dashboard

- Scaffold Vite + React + TypeScript against the now-stable schemas.
- Live feed; alert detail with SHAP evidence, explanation and source badge; official metrics
  panel with model comparison and threshold slider (precomputed held-out scores); a separate,
  clearly labelled demo-session panel; inject-fraud control.
- Optional Playwright smoke test.

## Stage 7: Results and write-up

- `reports/results.md`: frozen metrics with confidence intervals, an example flagged
  transaction and its explanation, the time-ordered split rationale, a scaling-to-bank note,
  and Sparkov listed as a future extension.
- Resume bullet; final ADR review; optional Docker Compose.
- The test set may now be replayed for the presentation.
