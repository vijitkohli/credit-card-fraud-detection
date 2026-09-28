# Project Handoff: Real-Time Fraud Detection POC (with Explainable AI Layer)

## Context

This is a proof-of-concept being built to support a university Work-Integrated Learning (WIL) application for a bank technology partnership project focused on fraud detection. The brief asks for: leveraging AI/ML for real-time fraud monitoring, minimizing false positives, and demonstrating a working proof of concept with clear results.

The builder's background: full-stack engineering (TypeScript/JavaScript, React, Node, Python, Flask), applied LLM integration (multi-LLM orchestration, prompt engineering), and a security/testing background (encryption, vulnerability testing, high test coverage). The POC should lean on **systems and LLM-integration strength**, not deep ML research — the model itself should be a solid, well-known approach, not a novel one.

## Goal

Build a working system that:
1. Scores incoming transactions for fraud risk in near-real-time.
2. Generates a plain-English explanation for every flagged transaction (the differentiator vs. a plain classifier).
3. Displays live results, flags, and model performance metrics in a dashboard.
4. Produces clear precision/recall/false-positive metrics that can be quoted in a resume bullet and a slide deck.

## Dataset

Agent's choice — pick whichever gives the fastest path to a working, demonstrable result. Reasonable options:
- Kaggle Credit Card Fraud dataset (small, PCA-anonymized, well-known, imbalanced) — fastest to stand up.
- IEEE-CIS Fraud Detection dataset (larger, richer features) — more research depth if time allows.

State which one was chosen and why in the final write-up.

## Architecture

1. **Detection layer** — train a model on the chosen dataset. Suggested approach: start with an unsupervised anomaly detector (e.g. Isolation Forest) and a supervised classifier (e.g. XGBoost or Random Forest) trained on the labeled fraud/not-fraud column, then compare the two.
2. **Explanation layer** — for each transaction flagged as suspicious, call an LLM with the transaction's feature values and anomaly/classification score, and generate a short, human-readable reason for the flag (e.g. "3x usual spend, new device, unusual local time"). LLM provider is the agent's choice (Claude API, OpenAI, or whatever is easiest to wire up given available credentials/tools).
3. **API layer** — a backend endpoint (Flask or Node, agent's choice) that takes a transaction, returns a fraud verdict, score, and explanation.
4. **Dashboard** — a React front end showing:
   - A live/simulated transaction feed.
   - Flagged transactions highlighted with their generated explanation.
   - A metrics panel: precision, recall, false-positive rate (and ideally a toggle or comparison between the anomaly detector and the classifier).

## Build Steps

- [ ] Pick and load the dataset; document the choice and reasoning.
- [ ] Clean/prepare data; handle class imbalance (fraud datasets are heavily imbalanced).
- [ ] Train the anomaly detection model.
- [ ] Train the supervised classification model.
- [ ] Evaluate both — precision, recall, F1, false-positive rate. Pick (or blend) the better-performing approach for the live demo.
- [ ] Build the explanation layer — LLM call that takes a flagged transaction's data and returns a short natural-language reason.
- [ ] Build the API endpoint that ties scoring + explanation together into one request/response call.
- [ ] Build the React dashboard: transaction feed, flagged alerts with explanations, metrics panel.
- [ ] Write a short results summary: final metrics, example flagged transaction with its generated explanation, and a one-paragraph "how this could extend to real bank scale" note.

## Deliverables

1. Working codebase (model training script, API service, dashboard).
2. A results summary (metrics + example output) suitable for lifting into a slide deck.
3. A one-line resume-bullet-style summary of the outcome, e.g.:
   > "Built a real-time fraud detection proof-of-concept combining a [model] anomaly-scoring model with an LLM explanation layer, achieving [X]% precision at [Y]% false-positive rate."

## Notes for the agent

- Optimize for a demonstrable, working end-to-end system over model sophistication — the explanation layer and the false-positive-rate framing are what make this stand out, not model complexity.
- Where a technical decision isn't specified above (LLM provider, exact model type, dataset), make a reasonable choice and state it clearly rather than pausing to ask, unless it meaningfully changes the outcome.
- Keep the code and write-up in a state that could be walked through in a short presentation: the "why" behind each choice should be easy to explain out loud.
