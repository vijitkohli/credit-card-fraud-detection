"""Stage 2 validation experiment: train candidates on train, compare them on validation.

Usage:
    uv run python -m fraud.experiment

Never loads the test split. Writes `reports/validation/` (report.json, report.md,
pr_curves.png). Nothing is frozen here; the production model and operating point are
chosen after reviewing this report.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from fraud import data, metrics, models
from fraud.features import SECONDS_PER_CYCLE, build_features
from fraud.prepare import load_split

BUDGETS_PER_10K = (1, 5, 10, 20)
PRIMARY_BUDGET = 5
LR_GRID = {"C": (0.01, 0.1, 1.0), "class_weight": (None, "balanced")}
XGB_GRID = {"max_depth": (3, 5), "scale_pos_weight": ("1", "sqrt_ratio")}

REPORT_DIR = data.ROOT / "reports" / "validation"

SELECTION_RULES = [
    "Hyperparameters within each model family and time-feature variant are chosen by "
    "validation AUPRC (threshold-free, least noisy with 57 validation frauds).",
    "The time_within_daily_cycle feature is kept only if it improves validation AUPRC and "
    "does not reduce validation recall at the primary budget.",
    "The production model is chosen on validation recall at the agreed false-alert budget, "
    "with AUPRC as the tie-breaker.",
    "Isolation Forest is fitted without labels; labels are used only for its thresholds.",
    "The test split is not loaded by this experiment.",
]


def operating_points(y: np.ndarray, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    points = {}
    for budget in BUDGETS_PER_10K:
        threshold = metrics.threshold_for_fp_budget(y, scores, budget)
        points[f"budget_{budget}_per_10k"] = {
            "budget_per_10k": budget,
            **metrics.confusion_at(y, scores, threshold),
            "ci95": metrics.bootstrap_ci(y, scores, threshold, seed=models.SEED),
        }
    threshold = metrics.max_f1_threshold(y, scores)
    points["max_f1"] = {
        "budget_per_10k": None,
        **metrics.confusion_at(y, scores, threshold),
        "ci95": metrics.bootstrap_ci(y, scores, threshold, seed=models.SEED),
    }
    return points


def cycle_coverage(train, validation) -> dict[str, Any]:
    """Which part of the assumed 24-hour cycle each split covers (offset unknown)."""

    def span(df):
        start, end = float(df["Time"].min()), float(df["Time"].max())
        return {
            "time_seconds": [start, end],
            "cycle_hours_covered": round((end - start) / 3600, 2),
            "cycle_position_hours": [
                round((start % SECONDS_PER_CYCLE) / 3600, 2),
                round((end % SECONDS_PER_CYCLE) / 3600, 2),
            ],
        }

    return {"train": span(train), "validation": span(validation)}


def run() -> dict[str, Any]:
    started = time.perf_counter()
    train, validation = load_split("train"), load_split("validation")
    y_train, y_val = train["Class"], validation["Class"].to_numpy()
    sqrt_ratio = math.sqrt((y_train == 0).sum() / (y_train == 1).sum())

    grid_results: list[dict[str, Any]] = []
    val_scores: dict[tuple[str, bool, str], np.ndarray] = {}

    for include_time in (False, True):
        X_train = build_features(train, include_time_cycle=include_time)
        X_val = build_features(validation, include_time_cycle=include_time)

        for C, class_weight in itertools.product(*LR_GRID.values()):
            params = {"C": C, "class_weight": class_weight}
            model = models.logistic_regression(**params).fit(X_train, y_train)
            key = ("logistic_regression", include_time, json.dumps(params))
            val_scores[key] = models.risk_scores(model, X_val)
            grid_results.append(
                {
                    "family": "logistic_regression",
                    "include_time_cycle": include_time,
                    "params": params,
                    **metrics.threshold_free(y_val, val_scores[key]),
                }
            )
            print(f"  logreg time={include_time} {params} auprc={grid_results[-1]['auprc']:.4f}")

        for max_depth, spw_name in itertools.product(*XGB_GRID.values()):
            spw = 1.0 if spw_name == "1" else sqrt_ratio
            model, n_rounds = models.fit_xgboost(X_train, y_train, max_depth, spw)
            params = {"max_depth": max_depth, "scale_pos_weight": spw_name}
            key = ("xgboost", include_time, json.dumps(params))
            val_scores[key] = models.risk_scores(model, X_val)
            grid_results.append(
                {
                    "family": "xgboost",
                    "include_time_cycle": include_time,
                    "params": {**params, "scale_pos_weight_value": round(spw, 4)},
                    "n_boosting_rounds": n_rounds,
                    **metrics.threshold_free(y_val, val_scores[key]),
                }
            )
            print(
                f"  xgboost time={include_time} {params} rounds={n_rounds} "
                f"auprc={grid_results[-1]['auprc']:.4f}"
            )

    X_train = build_features(train)
    iforest = models.isolation_forest().fit(X_train)
    params = {"n_estimators": iforest.n_estimators}
    key = ("isolation_forest", False, json.dumps(params))
    val_scores[key] = models.risk_scores(iforest, build_features(validation))
    grid_results.append(
        {
            "family": "isolation_forest",
            "include_time_cycle": False,
            "params": params,
            **metrics.threshold_free(y_val, val_scores[key]),
        }
    )

    candidates: dict[str, dict[str, Any]] = {}
    for (family, include_time), group in itertools.groupby(
        sorted(grid_results, key=lambda r: (r["family"], r["include_time_cycle"])),
        key=lambda r: (r["family"], r["include_time_cycle"]),
    ):
        best = max(group, key=lambda r: r["auprc"])
        name = family + ("+time_cycle" if include_time else "")
        params = {k: v for k, v in best["params"].items() if k != "scale_pos_weight_value"}
        scores = val_scores[(family, include_time, json.dumps(params))]
        candidates[name] = {**best, "operating_points": operating_points(y_val, scores)}
        candidates[name]["_scores"] = scores

    flag_nothing_accuracy = float((y_val == 0).mean())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(y_val, candidates)

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "library_versions": library_versions(),
        "splits_used": ["train", "validation"],
        "validation": {
            "rows": len(y_val),
            "fraud": int(y_val.sum()),
            "legit": int((y_val == 0).sum()),
        },
        "selection_rules": SELECTION_RULES,
        "primary_budget_per_10k": PRIMARY_BUDGET,
        "time_cycle_coverage": cycle_coverage(train, validation),
        "baselines": {"flag_nothing": {"accuracy": flag_nothing_accuracy, "recall": 0.0}},
        "grid_results": grid_results,
        "candidates": {
            k: {kk: vv for kk, vv in v.items() if kk != "_scores"} for k, v in candidates.items()
        },
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORT_DIR / "report.md").write_text(render_markdown(report))
    return report


def library_versions() -> dict[str, str]:
    import sklearn
    import xgboost

    return {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}


def plot_pr_curves(y: np.ndarray, candidates: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, c in candidates.items():
        precision, recall, _ = precision_recall_curve(y, c["_scores"])
        ax.plot(recall, precision, label=f"{name} (AUPRC {c['auprc']:.3f})")
        point = c["operating_points"][f"budget_{PRIMARY_BUDGET}_per_10k"]
        ax.scatter(point["recall"], point["precision"], s=25)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Validation precision-recall (dots: {PRIMARY_BUDGET} false alerts per 10k)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)


def render_markdown(report: dict[str, Any]) -> str:
    v = report["validation"]
    lines = [
        "# Stage 2 validation report",
        "",
        f"Generated {report['generated_at']}. Splits used: train and validation only "
        f"(validation: {v['rows']:,} rows, {v['fraud']} frauds, {v['legit']:,} legitimate). "
        "The test split was not loaded.",
        "",
        "## Selection rules (declared before results)",
        "",
        *[f"- {rule}" for rule in report["selection_rules"]],
        "",
        "## Candidates (best configuration per family and time variant)",
        "",
        "| Candidate | Params | AUPRC | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for name, c in report["candidates"].items():
        params = json.dumps(c["params"])
        lines.append(f"| {name} | `{params}` | {c['auprc']:.4f} | {c['roc_auc']:.4f} |")

    lines += ["", "## Operating points on validation", ""]
    for name, c in report["candidates"].items():
        lines += [
            f"### {name}",
            "",
            "| Point | Threshold | TP | FP | FN | Precision | Recall | F1 | False alerts / 10k "
            "| Precision 95% CI | Recall 95% CI |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for p in c["operating_points"].values():
            label = (
                f"{p['budget_per_10k']} per 10k budget" if p["budget_per_10k"] else "max F1 (ref)"
            )
            ci = p["ci95"]
            lines.append(
                f"| {label} | {p['threshold']:.6g} | {p['tp']} | {p['fp']} | {p['fn']} "
                f"| {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} "
                f"| {p['false_alerts_per_10k_legit']:.2f} "
                f"| {ci['precision'][0]:.2f} to {ci['precision'][1]:.2f} "
                f"| {ci['recall'][0]:.2f} to {ci['recall'][1]:.2f} |"
            )
        lines.append("")

    cov = report["time_cycle_coverage"]
    lines += [
        "## Time-cycle coverage",
        "",
        "Cycle positions are hours since an unknown offset, not clock time.",
        "",
        f"- Train: {cov['train']['cycle_hours_covered']} hours, "
        f"ending at cycle position {cov['train']['cycle_position_hours'][1]} h.",
        f"- Validation: cycle positions {cov['validation']['cycle_position_hours'][0]} to "
        f"{cov['validation']['cycle_position_hours'][1]} h only.",
        "",
        "## Baseline",
        "",
        f"- Flag nothing: accuracy {report['baselines']['flag_nothing']['accuracy']:.4%}, "
        "recall 0.",
        "",
        "## All grid results",
        "",
        "| Family | Time cycle | Params | Rounds | AUPRC | ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["grid_results"]:
        lines.append(
            f"| {r['family']} | {r['include_time_cycle']} | `{json.dumps(r['params'])}` "
            f"| {r.get('n_boosting_rounds', '')} | {r['auprc']:.4f} | {r['roc_auc']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = run()
    print(f"\nDone in {report['runtime_seconds']}s -> {REPORT_DIR.relative_to(data.ROOT)}/")
    for name, c in report["candidates"].items():
        p = c["operating_points"][f"budget_{PRIMARY_BUDGET}_per_10k"]
        print(
            f"{name:<32} auprc {c['auprc']:.4f}  @5/10k: recall {p['recall']:.3f} "
            f"precision {p['precision']:.3f} fp {p['fp']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
