"""Stage 3: TreeSHAP attribution, five-domain policy roll-ups and the master priority table."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.configuration import domain_by_feature, feature_domains, scheme_mapping
from src.models.metrics import apply_threshold
from src.target.ols_priority_classifier import CLASS_NAMES

BINARY_PREDICTION_LABELS = {1: "Discordant (Priority)", 3: "Not discordant"}


def predict_classes(model, X: pd.DataFrame) -> np.ndarray:
    """Operational class using the decision threshold stored with the model (argmax if none)."""
    return apply_threshold(model.predict_proba(X), np.asarray(model.classes_), getattr(model, "decision_threshold_", None))


def prediction_label(model, predicted: int) -> str:
    if getattr(model, "task_", "four_class") == "discordance_binary":
        return BINARY_PREDICTION_LABELS[int(predicted)]
    return CLASS_NAMES[int(predicted)]


def in_training_domain(model, observed_class: int) -> bool:
    """Binary models are trained only on high-access districts (classes 1 and 3)."""
    return getattr(model, "task_", "four_class") != "discordance_binary" or int(observed_class) in (1, 3)


def tree_model(model):
    """Return the fitted tree ensemble inside the Stage-2 pipeline."""
    classifier = model.named_steps["classifier"]
    return getattr(classifier, "estimator_", classifier)


def output_space(estimator) -> str:
    """TreeSHAP explains probabilities for sklearn forests and raw log-odds margins for boosters."""
    name = type(estimator).__name__
    if name == "GradientBoostingClassifier":
        return "log_odds"
    return "log_odds" if name in {"XGBClassifier", "LGBMClassifier"} else "probability"


def model_output(estimator, transformed: np.ndarray, class_index: int) -> np.ndarray:
    """The quantity TreeSHAP values add up to, for the requested class."""
    if output_space(estimator) == "probability":
        return estimator.predict_proba(transformed)[:, class_index]
    name = type(estimator).__name__
    if name == "XGBClassifier":
        margin = estimator.predict(transformed, output_margin=True)
    elif name == "GradientBoostingClassifier":
        margin = estimator.decision_function(transformed)
    else:
        margin = estimator.predict_proba(transformed, raw_score=True)
    margin = np.asarray(margin)
    # Binary boosters return a single margin for the positive (encoded-1 = Priority) class.
    return margin if margin.ndim == 1 else margin[:, class_index]


def class_shap_values(explainer, transformed: np.ndarray, class_index: int) -> np.ndarray:
    values = explainer.shap_values(transformed, check_additivity=False)
    if isinstance(values, list):
        return np.asarray(values[class_index])
    values = np.asarray(values)
    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    # SHAP 0.46 returns (rows, features, classes); older variants return (classes, rows, features).
    if values.shape[0] == transformed.shape[0] and values.shape[1] == transformed.shape[1]:
        return values[:, :, class_index]
    return values[class_index, :, :]


def base_value(explainer, class_index: int) -> float:
    expected = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    return float(expected[class_index] if len(expected) > class_index else expected[0])


def explain(model, X: pd.DataFrame) -> dict[str, Any]:
    """Compute class-1 TreeSHAP values for rows of ``X`` (state + feature columns)."""
    import shap

    classes = np.asarray(model.classes_)
    if 1 not in classes:
        raise ValueError("The trained classifier has no Priority / High-Risk class (class 1).")
    class_index = int(np.flatnonzero(classes == 1)[0])
    transformed = model.named_steps["imputer"].transform(X)
    estimator = tree_model(model)
    # The tree model's own column for Priority (EncodedClassifier may encode labels in a different order).
    codes = getattr(model.named_steps["classifier"], "codes_", None)
    tree_index = int(codes[1]) if codes is not None else class_index
    explainer = shap.TreeExplainer(estimator)
    values = class_shap_values(explainer, transformed, tree_index)
    if values.shape != transformed.shape:
        raise AssertionError(f"SHAP output {values.shape} does not match model input {transformed.shape}.")
    base = base_value(explainer, tree_index)
    reconstructed = base + values.sum(axis=1)
    target = model_output(estimator, transformed, tree_index)
    return {
        "values": values,
        "base_value": base,
        "transformed": transformed,
        "class_index": class_index,
        "tree_class_index": tree_index,
        "output_space": output_space(estimator),
        "max_additivity_error": float(np.max(np.abs(reconstructed - target))),
    }


def domain_contributions(shap_values: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    """Phi_{i,D_k} = sum of member-feature SHAP values (exact by additivity)."""
    lookup = domain_by_feature()
    frame = pd.DataFrame(shap_values, columns=feature_columns)
    domains = {}
    for domain in feature_domains():
        members = [feature for feature in feature_columns if lookup.get(feature) == domain]
        domains[domain] = frame[members].sum(axis=1) if members else pd.Series(0.0, index=frame.index)
    return pd.DataFrame(domains)


def national_domain_shares(domain_frame: pd.DataFrame, class_one_mask: np.ndarray) -> pd.DataFrame:
    """Domain Share(D_k) = sum_i |Phi_{i,D_k}| / sum_m sum_i |Phi_{i,D_m}| over class-1 districts."""
    subset = domain_frame.loc[class_one_mask].abs()
    if subset.empty:
        subset = domain_frame.abs()
    totals = subset.sum(axis=0)
    shares = totals / totals.sum() * 100 if totals.sum() else totals
    signed = domain_frame.loc[class_one_mask].mean(axis=0) if class_one_mask.any() else domain_frame.mean(axis=0)
    return (
        pd.DataFrame(
            {
                "domain": shares.index,
                "share_pct": shares.round(2).to_numpy(),
                "mean_signed_contribution": signed.reindex(shares.index).to_numpy(),
                "scheme": [scheme_mapping()[domain]["scheme"] for domain in shares.index],
            }
        )
        .sort_values("share_pct", ascending=False)
        .reset_index(drop=True)
    )


def _format_driver(feature: str, value: float, benchmark: float, shap_value: float) -> str:
    return f"{feature}: {value:.2f} (state median {benchmark:.2f}; SHAP +{shap_value:.3f})"


def generate_shap_artifacts(
    frame: pd.DataFrame,
    model,
    feature_columns: list[str],
    shap_path: Path,
    priority_table_path: Path,
    domain_share_path: Path | None = None,
    calibrated_model=None,
    low_access_audit_threshold: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-compute class-1 TreeSHAP values and the policy-oriented master priority action table."""
    X = frame[["state", *feature_columns]].copy()
    result = explain(model, X)
    shap_values, transformed = result["values"], result["transformed"]
    if result["max_additivity_error"] > 1e-3:
        raise AssertionError(f"TreeSHAP additivity violated (max error {result['max_additivity_error']:.2e}).")
    analysis = frame.reset_index(drop=True).copy()
    analysis.loc[:, feature_columns] = transformed
    class_index = result["class_index"]
    raw_probability = model.predict_proba(X)[:, class_index]
    risk_model = calibrated_model if calibrated_model is not None else model
    risk_index = int(np.flatnonzero(np.asarray(risk_model.classes_) == 1)[0])
    risk_probability = risk_model.predict_proba(X)[:, risk_index]
    predicted_class = predict_classes(model, X)
    domains = domain_contributions(shap_values, feature_columns)

    shap_frame = analysis[["district", "state", "priority_class", "priority_label"]].copy()
    shap_frame["predicted_class"] = predicted_class.astype(int)
    shap_frame["in_training_domain"] = [in_training_domain(model, c) for c in analysis["priority_class"]]
    shap_frame["priority_risk_score"] = np.round(risk_probability * 100, 1)
    shap_frame["raw_class_1_probability"] = raw_probability
    shap_frame["shap_base_value"] = result["base_value"]
    shap_frame["shap_output_space"] = result["output_space"]
    for index, feature in enumerate(feature_columns):
        shap_frame[f"shap__{feature}"] = shap_values[:, index]
    for domain in domains:
        shap_frame[f"domain__{domain}"] = domains[domain].to_numpy()
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    shap_frame.to_parquet(shap_path, index=False)

    if domain_share_path is not None:
        shares = national_domain_shares(domains, (analysis["priority_class"] == 1).to_numpy())
        domain_share_path.parent.mkdir(parents=True, exist_ok=True)
        shares.to_csv(domain_share_path, index=False)

    feature_domain = domain_by_feature()
    schemes = scheme_mapping()
    state_medians = analysis.groupby("state")[feature_columns].median(numeric_only=True)
    rows: list[dict[str, object]] = []
    for row_index, row in analysis.iterrows():
        contributions = shap_values[row_index]
        positive = np.flatnonzero(contributions > 0)
        ranked = positive[np.argsort(contributions[positive])[::-1]][:5]
        domain_row = domains.iloc[row_index]
        primary = str(domain_row.idxmax()) if (domain_row > 0).any() else "none (no domain raises class-1 risk)"
        benchmark = state_medians.loc[row["state"]]
        care = float(row.get("institutional_births_pct", np.nan))
        output: dict[str, object] = {
            "district": row["district"],
            "state": row["state"],
            "priority_class": int(row["priority_class"]),
            "priority_label": CLASS_NAMES[int(row["priority_class"])],
            "predicted_class": int(predicted_class[row_index]),
            "predicted_label": prediction_label(model, predicted_class[row_index]),
            "in_training_domain": in_training_domain(model, row["priority_class"]),
            "priority_risk_score": round(float(risk_probability[row_index]) * 100, 1),
            # Isotonic calibration is a step function, so tied risk scores are ordered by the raw probability.
            "raw_class_1_probability": round(float(raw_probability[row_index]), 4),
            "institutional_births_pct": care,
            "studentized_residual": round(float(row["studentized_residual"]), 3),
            "primary_domain_bottleneck": primary,
            "recommended_scheme": schemes[primary]["scheme"] if primary in schemes else "",
            "recommended_action": schemes[primary]["action"] if primary in schemes else "",
            "crs_verbal_autopsy_audit_flag": bool(
                int(row["priority_class"]) == 4 and np.isfinite(care) and care < low_access_audit_threshold
            ),
        }
        for domain in domains:
            output[f"domain__{domain}"] = round(float(domain_row[domain]), 4)
        for rank, feature_index in enumerate(ranked, start=1):
            feature = feature_columns[int(feature_index)]
            output[f"top_{rank}_driver"] = _format_driver(
                feature, float(row[feature]), float(benchmark[feature]), float(contributions[feature_index])
            )
            output[f"top_{rank}_scheme"] = schemes[feature_domain[feature]]["scheme"]
        rows.append(output)
    actions = pd.DataFrame(rows).sort_values(
        ["priority_class", "priority_risk_score", "raw_class_1_probability"], ascending=[True, False, False]
    )
    actions.insert(0, "rank", np.arange(1, len(actions) + 1))
    priority_table_path.parent.mkdir(parents=True, exist_ok=True)
    actions.to_csv(priority_table_path, index=False)
    return shap_frame, actions


def save_shap_figures(model, frame: pd.DataFrame, feature_columns: list[str], figure_dir: Path) -> list[Path]:
    """Global beeswarm and five-domain bar chart for the publication figure set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    figure_dir.mkdir(parents=True, exist_ok=True)
    X = frame[["state", *feature_columns]].copy()
    result = explain(model, X)
    written: list[Path] = []
    plt.figure()
    shap.summary_plot(
        result["values"], pd.DataFrame(result["transformed"], columns=feature_columns), show=False, max_display=20
    )
    plt.title(f"Class-1 TreeSHAP ({result['output_space']})")
    path = figure_dir / "shap_beeswarm_class1.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close("all")
    written.append(path)

    domains = domain_contributions(result["values"], feature_columns)
    shares = national_domain_shares(domains, (frame["priority_class"] == 1).to_numpy())
    figure, axis = plt.subplots(figsize=(7, 3.6))
    labels = [domain.replace("_", " ").title() for domain in shares["domain"]][::-1]
    axis.barh(labels, shares["share_pct"][::-1], color="#2a6f97")
    axis.set_xlabel("Share of |class-1 SHAP| among Priority districts (%)")
    axis.set_title("National domain attribution")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    path = figure_dir / "shap_domain_shares.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


def explain_single(model, row: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Live class-1 TreeSHAP for one (possibly perturbed) district row; used by the what-if tab."""
    result = explain(model, row[["state", *feature_columns]])
    return pd.Series(result["values"][0], index=feature_columns)
