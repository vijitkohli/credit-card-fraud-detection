# Architecture Decision Record

Status: **Accepted** (2026-09-27). Amendments from the architecture review are folded in.

## Context

Proof of concept for a bank fraud-detection Work-Integrated Learning application. The brief:
real-time fraud monitoring, minimising false positives, a working demo with clear results,
and a plain-English explanation for every flagged transaction.

Principles:

- A software engineering and AI integration POC, not novel ML research.
- Simple and defensible over sophisticated; every choice must be explainable out loud.
- **The ML system makes the fraud decision. The LLM only explains an existing decision.**
- False positives matter as much as detection.
- No infrastructure or abstraction without a concrete need.

## Decisions

### D1. Dataset: ULB Credit Card Fraud (OpenML 1597)

284,807 real transactions, 492 frauds (0.172%), two days, features `Time`, `V1`–`V28`
(PCA components), `Amount`, `Class`.

- Chosen for credible, well-known, fast-to-reproduce metrics.
- Accepted limitation: `V1`–`V28` are anonymised. The application must describe them as
  *anonymised model features* and must never invent a meaning for them.
- Acquisition verifies the OpenML record (id, name, version, target, creators, published
  MD5) before download, verifies the file MD5 after download, and validates schema, row
  count, fraud count and time ordering. Source URL, hashes, schema and counts are recorded
  in `data/provenance.json`.
- OpenML marks `Time` as a row identifier, so generic loaders drop it. We parse the ARFF
  directly.
- Licence: ODbL v1.0 (database) / DbCL v1.0 (contents). Raw data is never committed.
- Future extension (out of scope): the synthetic Sparkov dataset for behavioural,
  human-readable explanations such as spend relative to a card's history.

### D2. Supervised models: select on validation

Candidates: Logistic Regression (baseline) and XGBoost (expected production candidate).
XGBoost is **not** a predetermined winner. The production model is selected on validation
AUPRC and validation recall at the chosen false-positive budget. The test set plays no
part in choosing the model, hyperparameters or threshold. Tuning is a small grid with early
stopping, on validation only.

### D3. Anomaly detection: Isolation Forest as comparator

Fitted without labels. Labels are used only to set its comparison threshold on validation,
which is stated openly. It is shown in the model comparison and as a secondary novelty
signal. It does not make or blend into the production decision.

### D4. Preprocessing

- Remove exact duplicate rows before splitting, to stop identical rows leaking across splits.
- Raw `Time` is dropped from the model: it only encodes position in a two-day window and
  breaks under a time-ordered split.
- **Time ablation:** Stage 2 compares validation performance with and without a cyclical
  feature named `time_within_daily_cycle` (sin/cos of `Time mod 86400`). It is kept only if
  it improves validation results. If kept, the documented assumption is that the dataset
  spans roughly two 24-hour cycles, **with an unknown offset**. `Time = 0` is not known to be
  midnight, so neither the UI nor the LLM may ever describe it as local time, clock hour or
  time of day.
- `Amount`: `log1p`.
- `V1`–`V28`: unchanged.
- Scaling only for Logistic Regression, fitted on train only, inside an sklearn `Pipeline`.
- One shared feature module used by both training and serving, with a parity test.

### D5. Class imbalance

No resampling (no SMOTE, no undersampling). XGBoost `scale_pos_weight` in {1, √(neg/pos)},
chosen on validation. The operating point comes from threshold selection (D7). Model output
is presented as a **risk score**, not a calibrated probability.

### D6. Split methodology: time-ordered 60/20/20

Ordered by `Time`, after de-duplication: 60% train, 20% validation, 20% test.

Why this is more representative of deployment: a production model is trained on the past
and scores the future. Random splits let the model train on transactions that occur after
the ones it is evaluated on, and mix both days' fraud patterns into training, which
typically inflates metrics. The time-ordered split accepts possibly lower numbers in
exchange for an honest estimate.

- Actual split after de-duplication (`reports/split_summary.json`): train 170,236 rows /
  342 frauds (0.20%), validation 56,746 / 57 (0.10%), test 56,744 / 74 (0.13%). The fraud
  rate falls over time, which is a realistic prior shift that a random split would hide.
- The test set is evaluated **once**, after model, preprocessing and threshold are frozen.
  Each test fraud is ~1.4 percentage points of recall (validation: ~1.75), so precision and
  recall are reported with bootstrap 95% confidence intervals.
- Rows equal on every model input but at a different `Time` are not duplicates and are
  kept. About 1% of validation and test rows match a training row this way, none of them
  fraud, so they cannot inflate fraud recall or precision. Measured in the split summary.
- Development of the API and dashboard uses validation examples or a separate demo subset,
  never the test set.
- The test set may be replayed in the final presentation only after the official metrics
  are frozen.

### D7. Decision threshold: false-positive budget, set on validation

False-positive rate is measured against legitimate transactions (FP / legitimate). On
validation, report recall, precision, F1 and confusion counts at the threshold that
maximises recall subject to each budget:

| Budget (false alerts per 10,000 legitimate) | Role |
|---|---|
| 1 | tight |
| **5** | **initial operating constraint** |
| 10 | looser |
| 20 | loosest |
| max-F1 threshold | comparison only |

The final operating point is recommended after these trade-offs are reviewed, then
**frozen using validation only**, and reported once on test.

Known limitation: validation has 56,689 legitimate transactions and 57 frauds, so a
1-per-10k budget allows only 5 false positives, and every threshold estimate is coarse.

### D8. Evaluation metrics

- Primary, threshold-free: AUPRC.
- At the operating point: precision, recall, F1, false-positive rate, false alerts per
  10,000 legitimate transactions, confusion matrix.
- Secondary: ROC-AUC, de-emphasised because it looks near-perfect under imbalance.
- Baselines: Logistic Regression, Isolation Forest, and "flag nothing" (the accuracy trap).
- System metrics: scoring latency p50/p95, reported separately from explanation latency,
  and the explanation grounding pass rate.

### D9. Backend: FastAPI, single service

Python is needed for in-process inference. FastAPI provides Pydantic validation for API and
LLM output schemas, auto-generated OpenAPI docs, async LLM calls and SSE.

### D10. Model artefacts and serving

- Versioned directory `artifacts/model-vN/`: XGBoost native JSON, joblib files for the sklearn
  models, `metadata.json` (threshold, feature list, library versions, dataset hash, metrics),
  and `reference_stats.json` (percentiles for explanations).
- Loaded once at API startup, scored in-process, SHAP (tree explainer) computed per request.
- No MLflow, ONNX, model registry or Kubernetes.
- **Committing artefacts:** sizes and the dataset licence are inspected before committing.
  Metadata, reports, plots and small artefacts are committed. Anything large is either
  regenerated by a reproducible script or stored with Git LFS. `check-added-large-files`
  (1 MB) enforces this in pre-commit. The raw dataset is never committed.

### D11. LLM integration: deferred provider, fallback first

- The provider is chosen only after the offline evidence pipeline works.
- The **deterministic template explanation is implemented first**. Scoring, evidence and
  explanations all work without an LLM API key.
- The provider sits behind one small interface (a single `explain(evidence) -> Explanation`
  seam). There is no multi-provider framework.
- Small model tier, temperature 0, structured JSON output, timeout and one retry, then fall
  back to the template.
- Explanations are generated asynchronously and only for flagged transactions. Scoring
  latency never waits on the LLM.
- Every explanation carries a `source` field: `live_llm`, `cached_llm` or
  `deterministic_fallback`, shown in both the API response and the UI.
- Cached explanations are the default for known demo transactions; live generation is
  optional.

### D12. What the LLM receives

A structured evidence object only:

- the verdict (already decided), risk score and threshold
- the top 5 SHAP contributions (feature, value, direction, magnitude)
- percentiles of those features relative to legitimate and fraudulent training data
- `Amount`, the Isolation Forest novelty percentile, and `time_within_daily_cycle` only if D4
  keeps it (labelled as an offset cycle position, never as clock time)

It never receives the ground-truth label or the full raw row, and its output schema has no
verdict field.

### D13. Grounding

1. The evidence is computed deterministically from the model.
2. The prompt says: explain, don't decide; cite only the features provided; describe
   `V1`–`V28` only as anonymised model features; never describe time as clock or local time.
3. The output is Pydantic-validated JSON (`summary`, `reasons[{feature, statement}]`).
4. A post-check rejects outputs that reference features outside the evidence, quote numbers
   that don't match it, or use banned phrasing (clock time, invented V-feature meanings).
   A rejected output falls back to the template.
5. The UI shows the SHAP evidence next to the text.
6. An offline batch evaluation reports the grounding pass rate.

### D14. Real-time simulation

- A backend replay task scores transactions in time order at an adjustable rate, using the
  same scoring path as `POST /api/score`.
- **Demo mode** is clearly labelled and may deliberately increase fraud frequency.
- An "inject known fraud" action is provided.
- Demo-session counters are kept **visually and technically separate** from the official
  held-out metrics: different endpoints, different UI panels, and demo counters are never
  labelled as model performance.
- Replay source: validation or demo subset during development; test set only after metrics
  are frozen.

### D15. Frontend/backend communication

- SSE for the live feed (`transaction_scored`, `explanation_ready`); REST for scoring,
  metrics, transaction detail and replay control.
- Frontend: Vite + React + TypeScript, scaffolded only in the dashboard stage, after the API
  schemas, evidence object and metrics response are fixed.

### D16. Repository structure

A single Python package (`src/fraud`) shared by training and serving, to prevent
training/serving skew. Tooling: uv with Python 3.12, ruff, pytest, pre-commit.

```text
src/fraud/        data.py (Stage 0), features.py, train.py, evaluate.py,
                  evidence.py, explain.py, api/
tests/
data/             provenance.json committed; raw data gitignored
artifacts/        versioned model artefacts (committed selectively, D10)
reports/          metrics, plots, results summary
frontend/         added in the dashboard stage
docs/             adr.md, implementation-plan.md
```

### Deliberately excluded

MLflow, feature store, Kafka, Kubernetes, canary deploys, Docker (optional at the end),
multi-provider LLM framework, SMOTE, Next.js, probability calibration.
