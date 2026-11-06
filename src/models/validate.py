"""Robustness checks: state-grouped cross-validation and Stage-1 label sensitivity.

* **State-grouped CV** holds out whole states, so performance cannot lean on within-state similarity
  (shared NHM administration, reporting practice, survey design). It is compared with ordinary
  stratified CV using the *same* selected model and hyper-parameters. In every outer fold the class-1
  decision threshold is re-chosen on an inner CV of the training part only (no peeking at the test fold).
* **Label sensitivity** re-runs Stage 1 under alternative, defensible settings and reports how stable
  the Priority / High-Risk set is relative to the baseline configuration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.models.metrics import apply_threshold, choose_threshold, class_one_index
from src.models.train_evaluate import CANDIDATES, class_weights, make_pipeline
from src.target.ols_priority_classifier import assign_priority_classes

STAGE1_VARIANTS: dict[str, dict[str, Any]] = {
    "mortality_only": {"mortality_weight": 1.0, "morbidity_weight": 0.0},
    "care_threshold_80pct": {"care_threshold": 80.0},
    "residual_threshold_0": {"residual_threshold": 0.0},
    "residual_threshold_0.5": {"residual_threshold": 0.5},
    "eb_prior_1000": {"eb_prior_live_births": 1000},
    "eb_prior_10000": {"eb_prior_live_births": 10000},
}


def _task_frame(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "discordance_binary":
        return frame.loc[frame["priority_class"].isin([1, 3])].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _class_one_scores(y: np.ndarray, predicted: np.ndarray, p1: np.ndarray) -> dict[str, float | None]:
    precision, recall, _, _ = precision_recall_fscore_support(y, predicted, labels=[1], zero_division=0)
    try:
        auc = float(roc_auc_score((y == 1).astype(int), p1))
    except ValueError:
        auc = None
    return {
        "class_1_recall": float(recall[0]),
        "class_1_precision": float(precision[0]),
        "class_1_f2": float(fbeta_score(y, predicted, beta=2, labels=[1], average="macro", zero_division=0)),
        "class_1_roc_auc": auc,
    }


def cross_validate_scheme(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator,
    features: list[str],
    scheme: str,
    model_config: Mapping[str, Any],
    seed: int,
    zones: Mapping | None,
    n_neighbors: int,
    folds: int = 5,
) -> dict[str, Any]:
    if scheme == "state_grouped":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    classes = np.sort(y.unique())
    p1_index = class_one_index(classes)
    multiplier = float(model_config["priority_class_weight_multiplier"])
    target = float(model_config.get("target_class_1_recall", 0.82))
    pooled_p1 = np.zeros(len(y))
    pooled_pred = np.zeros(len(y), dtype=int)
    fold_rows = []
    for number, (train_idx, test_idx) in enumerate(splits, start=1):
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        # Inner CV on the training part only, to choose this fold's decision threshold.
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        inner_proba = np.zeros((len(train_idx), len(classes)))
        for inner_train, inner_valid in inner.split(train_x, train_y):
            model = make_pipeline(clone(estimator), features, zones, n_neighbors)
            model.fit(
                train_x.iloc[inner_train],
                train_y.iloc[inner_train],
                classifier__sample_weight=class_weights(train_y.iloc[inner_train], multiplier),
            )
            inner_proba[inner_valid] = model.predict_proba(train_x.iloc[inner_valid])
        threshold = choose_threshold(train_y.to_numpy(), inner_proba, classes, target)["threshold"]
        model = make_pipeline(clone(estimator), features, zones, n_neighbors)
        model.fit(train_x, train_y, classifier__sample_weight=class_weights(train_y, multiplier))
        proba = model.predict_proba(X.iloc[test_idx])
        predicted = apply_threshold(proba, classes, threshold)
        pooled_p1[test_idx] = proba[:, p1_index]
        pooled_pred[test_idx] = predicted
        test_y = y.iloc[test_idx].to_numpy()
        fold_rows.append(
            {
                "fold": number,
                "n_test": int(len(test_idx)),
                "n_priority": int((test_y == 1).sum()),
                "n_states": int(groups.iloc[test_idx].nunique()),
                "threshold": float(threshold),
                **_class_one_scores(test_y, predicted, proba[:, p1_index]),
            }
        )
    folds_frame = pd.DataFrame(fold_rows)
    pooled = _class_one_scores(y.to_numpy(), pooled_pred, pooled_p1)
    return {
        "pooled": pooled,
        "fold_mean": folds_frame[["class_1_recall", "class_1_precision", "class_1_f2", "class_1_roc_auc"]].mean().to_dict(),
        "fold_std": folds_frame[["class_1_recall", "class_1_precision", "class_1_f2", "class_1_roc_auc"]].std().to_dict(),
        "folds": fold_rows,
        "_pooled_predictions": pooled_pred,
        "_pooled_p1": pooled_p1,
    }


def per_state_recall(y: pd.Series, groups: pd.Series, predicted: np.ndarray, minimum_priority: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame({"state": groups.to_numpy(), "y": y.to_numpy(), "pred": predicted})
    priority = frame.loc[frame["y"] == 1]
    summary = priority.groupby("state").agg(n_priority=("y", "size"), detected=("pred", lambda s: int((s == 1).sum())))
    summary = summary.loc[summary["n_priority"] >= minimum_priority].copy()
    summary["recall_when_state_held_out"] = summary["detected"] / summary["n_priority"]
    return summary.sort_values("recall_when_state_held_out").reset_index()


def label_sensitivity(base_frame: pd.DataFrame, target_config: Mapping[str, Any]) -> pd.DataFrame:
    """Stability of Stage-1 labels under alternative settings (Jaccard / retention of the Priority set)."""
    baseline, _ = assign_priority_classes(base_frame, dict(target_config))
    base_priority = set(baseline.loc[baseline["priority_class"] == 1, "district_key"])
    rows = [
        {
            "variant": "baseline",
            "settings": "",
            "n_priority": len(base_priority),
            "priority_jaccard_vs_baseline": 1.0,
            "baseline_priority_retained": 1.0,
            "all_class_agreement": 1.0,
        }
    ]
    for name, override in STAGE1_VARIANTS.items():
        variant, _ = assign_priority_classes(base_frame, {**target_config, **override})
        priority = set(variant.loc[variant["priority_class"] == 1, "district_key"])
        union = base_priority | priority
        rows.append(
            {
                "variant": name,
                "settings": json.dumps(override),
                "n_priority": len(priority),
                "priority_jaccard_vs_baseline": len(base_priority & priority) / len(union) if union else 1.0,
                "baseline_priority_retained": len(base_priority & priority) / len(base_priority) if base_priority else 1.0,
                "all_class_agreement": float((variant["priority_class"].to_numpy() == baseline["priority_class"].to_numpy()).mean()),
            }
        )
    return pd.DataFrame(rows)


def save_validation_figures(report: dict[str, Any], states: pd.DataFrame, sensitivity: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    metrics = ["class_1_recall", "class_1_precision", "class_1_f2", "class_1_roc_auc"]
    labels = ["Recall", "Precision", "F2", "ROC-AUC"]
    figure, axis = plt.subplots(figsize=(7, 3.6))
    width = 0.38
    positions = np.arange(len(metrics))
    for offset, (scheme, colour) in zip((-width / 2, width / 2), (("random_stratified", "#94a3b8"), ("state_grouped", "#2a6f97"))):
        means = [report["cv"][scheme]["fold_mean"][m] for m in metrics]
        stds = [report["cv"][scheme]["fold_std"][m] for m in metrics]
        axis.bar(positions + offset, means, width, yerr=stds, capsize=3, color=colour, label=scheme.replace("_", " "))
    axis.axhline(0.82, color="#D32F2F", linestyle="--", linewidth=1, label="recall target 0.82")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1)
    axis.set_title("Class-1 performance: random vs state-grouped 5-fold CV (mean ± sd)")
    axis.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    path = figure_dir / "validation_random_vs_state_grouped.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    if not states.empty:
        figure, axis = plt.subplots(figsize=(6.5, max(2.5, 0.3 * len(states))))
        axis.barh(states["state"], states["recall_when_state_held_out"], color="#2a6f97")
        for y_pos, (value, n) in enumerate(zip(states["recall_when_state_held_out"], states["n_priority"])):
            axis.text(value + 0.01, y_pos, f"n={n}", va="center", fontsize=7)
        axis.axvline(0.82, color="#D32F2F", linestyle="--", linewidth=1)
        axis.set_xlim(0, 1.1)
        axis.set_xlabel("Priority recall when the state is held out")
        axis.set_title("Per-state recall under state-grouped CV")
        figure.tight_layout()
        path = figure_dir / "validation_per_state_recall.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        written.append(path)

    figure, axis = plt.subplots(figsize=(7, 3.4))
    shown = sensitivity.loc[sensitivity["variant"] != "baseline"]
    axis.barh(shown["variant"], shown["baseline_priority_retained"], color="#7c3aed", label="baseline Priority retained")
    axis.scatter(shown["priority_jaccard_vs_baseline"], shown["variant"], color="#111827", zorder=3, label="Jaccard vs baseline")
    axis.set_xlim(0, 1)
    axis.set_title("Stage-1 label sensitivity")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    path = figure_dir / "validation_label_sensitivity.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


def run_validation(
    modelling: pd.DataFrame,
    stage1_input: pd.DataFrame,
    features: list[str],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    output_path: Path,
    figure_dir: Path | None = None,
) -> dict[str, Any]:
    model_config = config["modelling"]
    seed = int(config["project"]["random_state"])
    task = metrics.get("task", model_config.get("task", "four_class"))
    selected = metrics["selected_model"]
    params = metrics["candidate_models"][selected]["best_params"]
    estimator = CANDIDATES[selected][0](params, seed)
    data = _task_frame(modelling, task)
    X = data[["state", *features]]
    y = data["priority_class"].astype(int)
    groups = data["state"].astype(str)
    zones = config.get("zones")
    neighbours = int(config["preprocessing"]["knn_neighbors"])

    cv = {
        scheme: cross_validate_scheme(X, y, groups, estimator, features, scheme, model_config, seed, zones, neighbours)
        for scheme in ("random_stratified", "state_grouped")
    }
    states = per_state_recall(y, groups, cv["state_grouped"]["_pooled_predictions"])
    sensitivity = label_sensitivity(stage1_input, config["target"])
    report = {
        "task": task,
        "model": selected,
        "n_districts": int(len(y)),
        "n_states": int(groups.nunique()),
        "cv": {scheme: {k: v for k, v in result.items() if not k.startswith("_")} for scheme, result in cv.items()},
        "generalisation_gap": {
            metric: cv["random_stratified"]["fold_mean"][metric] - cv["state_grouped"]["fold_mean"][metric]
            for metric in ("class_1_recall", "class_1_precision", "class_1_f2", "class_1_roc_auc")
        },
        "per_state_recall": states.to_dict(orient="records"),
        "label_sensitivity": sensitivity.to_dict(orient="records"),
        "notes": "Thresholds are re-chosen inside every outer training fold (inner 3-fold CV). Same model class and "
        "hyper-parameters as the selected production model.",
    }
    if figure_dir is not None:
        report["figures"] = [str(path.name) for path in save_validation_figures(report, states, sensitivity, figure_dir)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report


def main() -> None:
    """Re-run validation against the latest pipeline artefacts without retraining."""
    import joblib

    from src.configuration import configured_path, load_config

    config = load_config()
    metrics = json.loads(configured_path(config, "metrics").read_text(encoding="utf-8"))
    master = pd.read_csv(configured_path(config, "master_dataset"))
    master["district_key"] = master["state"].astype(str) + "|" + master["district"].astype(str)
    eligible = master.loc[master["modelling_eligible"].astype(bool)].reset_index(drop=True)
    modelling = eligible.copy()
    modelling["priority_class"] = modelling["priority_class"].astype(int)
    features = list(joblib.load(configured_path(config, "model")).named_steps["imputer"].feature_columns)
    report = run_validation(
        modelling, eligible, features, metrics, config, configured_path(config, "validation_report"), configured_path(config, "figures")
    )
    print(json.dumps({"cv": {k: v["pooled"] for k, v in report["cv"].items()}, "gap": report["generalisation_gap"]}, indent=2))


if __name__ == "__main__":
    main()
