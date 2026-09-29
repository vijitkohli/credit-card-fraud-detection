"""Stage 3: one-time evaluation of the frozen model on the held-out test split.

Usage:
    uv run python -m fraud.evaluate

Thresholds come from the frozen artefacts (set on validation); nothing is tuned here.
Refuses to overwrite an existing test report unless `--rerun` is passed, and records the
git commit and artefact hashes that were evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from fraud import artifacts, data, metrics
from fraud.models import risk_scores
from fraud.prepare import SPLIT_SUMMARY_PATH, ids_sha256, load_split

REPORT_DIR = data.ROOT / "reports" / "test"
PRECISION_FIRST_BUDGET = 1


class AlreadyEvaluatedError(RuntimeError):
    """The held-out test report already exists."""


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=data.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "artifacts_match_commit": not run("status", "--porcelain", "--", "artifacts"),
        "uncommitted_src_changes": run("status", "--porcelain", "--", "src").splitlines(),
    }


def evaluate_at(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        **metrics.confusion_at(y, scores, threshold),
        "ci95": metrics.bootstrap_ci(y, scores, threshold, seed=0),
    }


def run(report_dir: Path = REPORT_DIR, rerun: bool = False) -> dict[str, Any]:
    if (report_dir / "report.json").exists() and not rerun:
        raise AlreadyEvaluatedError(
            f"{report_dir / 'report.json'} exists; the test set is evaluated once. "
            "Pass --rerun only to reproduce the same frozen evaluation."
        )

    git = git_state()
    if not git["artifacts_match_commit"]:
        raise RuntimeError("Frozen artefacts differ from the committed version; commit first")

    model = artifacts.load(include_comparators=True)
    meta = model.metadata
    validation = load_split("validation")
    test = load_split("test", allow_test=True)

    expected_ids = json.loads(SPLIT_SUMMARY_PATH.read_text())["splits"]["test"]
    if ids_sha256(test["transaction_id"]) != expected_ids["transaction_ids_sha256"]:
        raise RuntimeError("Test split does not match the recorded transaction ID hash")

    y = test["Class"].to_numpy()
    X = model.features(test)
    production_scores = risk_scores(model.production, X)

    # The precision-first alternative's threshold is also derived from validation only.
    precision_first_threshold = metrics.threshold_for_fp_budget(
        validation["Class"].to_numpy(), model.score(validation), PRECISION_FIRST_BUDGET
    )

    results: dict[str, Any] = {
        "xgboost": {
            "role": "production",
            "threshold_source": "validation (frozen)",
            **metrics.threshold_free(y, production_scores),
            "at_frozen_threshold": evaluate_at(y, production_scores, model.threshold),
            "precision_first_alternative": {
                "budget_per_10k_on_validation": PRECISION_FIRST_BUDGET,
                **evaluate_at(y, production_scores, precision_first_threshold),
            },
        }
    }
    scores_by_model = {"xgboost": production_scores}
    for name, comparator in model.comparators.items():
        scores = risk_scores(comparator, X)
        scores_by_model[name] = scores
        results[name] = {
            "role": "comparison",
            "threshold_source": "validation (frozen)",
            **metrics.threshold_free(y, scores),
            "at_frozen_threshold": evaluate_at(y, scores, meta["comparators"][name]["threshold"]),
        }

    report = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_version": meta["model_version"],
        "git": git,
        "artifact_hashes": {k: v["sha256"] for k, v in meta["files"].items()},
        "metadata_sha256": hashlib.sha256(
            (artifacts.DEFAULT_DIR / "metadata.json").read_bytes()
        ).hexdigest(),
        "fp_budget_per_10k": meta["fp_budget_per_10k"],
        "test": {
            "rows": len(y),
            "fraud": int(y.sum()),
            "legit": int((y == 0).sum()),
            "transaction_ids_sha256": expected_ids["transaction_ids_sha256"],
        },
        "baselines": {"flag_nothing": {"accuracy": float((y == 0).mean()), "recall": 0.0}},
        "validation_reference": meta["production"]["validation_at_threshold"],
        "results": results,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(report_dir, y, scores_by_model, report)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (report_dir / "report.md").write_text(render_markdown(report))
    return report


def plot_pr_curves(
    report_dir: Path, y: np.ndarray, scores_by_model: dict[str, np.ndarray], report: dict
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        r = report["results"][name]
        ax.plot(recall, precision, label=f"{name} (AUPRC {r['auprc']:.3f})")
        p = r["at_frozen_threshold"]
        ax.scatter(p["recall"], p["precision"], s=30, zorder=3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-out test precision-recall (dots: frozen validation thresholds)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(report_dir / "pr_curves.png", dpi=150)
    plt.close(fig)


def _row(label: str, p: dict[str, Any]) -> str:
    ci = p["ci95"]
    return (
        f"| {label} | {p['threshold']:.6g} | {p['tp']} | {p['fp']} | {p['fn']} | {p['tn']} "
        f"| {p['precision']:.3f} ({ci['precision'][0]:.2f} to {ci['precision'][1]:.2f}) "
        f"| {p['recall']:.3f} ({ci['recall'][0]:.2f} to {ci['recall'][1]:.2f}) "
        f"| {p['f1']:.3f} | {p['false_alerts_per_10k_legit']:.2f} |"
    )


def render_markdown(report: dict[str, Any]) -> str:
    t = report["test"]
    v = report["validation_reference"]
    lines = [
        "# Stage 3 held-out test evaluation",
        "",
        f"Evaluated {report['evaluated_at']} at commit `{report['git']['commit'][:10]}` "
        f"({report['model_version']}). Test split: {t['rows']:,} rows, {t['fraud']} frauds, "
        f"{t['legit']:,} legitimate. All thresholds were frozen on validation before this run.",
        "",
        "## Frozen operating points",
        "",
        "| Model / point | Threshold | TP | FP | FN | TN | Precision (95% CI) "
        "| Recall (95% CI) | F1 | False alerts / 10k |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    xgb = report["results"]["xgboost"]
    lines.append(
        _row(
            f"**xgboost @ {report['fp_budget_per_10k']}/10k (production)**",
            xgb["at_frozen_threshold"],
        )
    )
    lines.append(
        _row("xgboost @ 1/10k (precision-first alternative)", xgb["precision_first_alternative"])
    )
    for name, r in report["results"].items():
        if name != "xgboost":
            lines.append(
                _row(
                    f"{name} @ {report['fp_budget_per_10k']}/10k (comparison)",
                    r["at_frozen_threshold"],
                )
            )
    lines += [
        "",
        "## Threshold-free",
        "",
        "| Model | AUPRC | ROC-AUC |",
        "|---|---|---|",
        *[
            f"| {name} | {r['auprc']:.4f} | {r['roc_auc']:.4f} |"
            for name, r in report["results"].items()
        ],
        "",
        "## Validation reference (production threshold)",
        "",
        f"- Validation: recall {v['recall']:.3f}, precision {v['precision']:.3f}, "
        f"{v['false_alerts_per_10k_legit']:.2f} false alerts per 10k.",
        "",
        "## Baseline",
        "",
        f"- Flag nothing: accuracy {report['baselines']['flag_nothing']['accuracy']:.4%}, "
        "recall 0.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m fraud.evaluate")
    parser.add_argument("--rerun", action="store_true", help="reproduce an existing evaluation")
    args = parser.parse_args(argv)
    try:
        report = run(rerun=args.rerun)
    except AlreadyEvaluatedError as exc:
        print(f"REFUSED: {exc}")
        return 1
    p = report["results"]["xgboost"]["at_frozen_threshold"]
    print(
        f"Test ({report['test']['fraud']} frauds): recall {p['recall']:.3f}, "
        f"precision {p['precision']:.3f}, {p['fp']} false alerts "
        f"({p['false_alerts_per_10k_legit']:.2f} per 10k), commit {report['git']['commit'][:10]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
