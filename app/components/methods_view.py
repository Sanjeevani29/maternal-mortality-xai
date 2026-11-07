"""Tab 5 — model benchmark, calibration, data provenance and audit trails."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.target.ols_priority_classifier import CLASS_NAMES


def _benchmark_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, result in metrics["candidate_models"].items():
        holdout, oof = result["holdout"], result["out_of_fold"]
        rows.append(
            {
                "model": name + (" (selected)" if name == metrics["selected_model"] else ""),
                "TreeSHAP": "yes" if result["treeshap_compatible"] else "no",
                "Optuna trials": result["optuna_trials"],
                "CV objective": result["cv_selection_score"],
                "OOF class-1 recall": oof["class_1_recall"],
                "Holdout class-1 recall": holdout["class_1_recall"],
                "Holdout class-1 F1": holdout["class_1_f1"],
                "Holdout class-1 F2": holdout["class_1_f2"],
                "Holdout macro F1": holdout["macro_f1"],
                "Holdout OvR ROC-AUC": holdout["ovr_roc_auc"],
                "Holdout OvR PR-AUC": holdout["ovr_pr_auc"],
                "Holdout class-1 Brier": holdout.get("class_1_brier"),
                "Holdout class-1 recall @ threshold": result.get("holdout_at_threshold", {}).get("class_1_recall"),
                "Holdout class-1 precision @ threshold": result.get("holdout_at_threshold", {}).get("class_1_precision"),
            }
        )
    return pd.DataFrame(rows)


def _validation_section(validation: dict) -> None:
    if not validation:
        st.info("Validation results are not available for this build.")
        return
    st.subheader("Does the model generalise to unseen states?")
    rows = []
    for scheme, result in validation["cv"].items():
        mean, std = result["fold_mean"], result["fold_std"]
        rows.append(
            {
                "cross-validation": scheme.replace("_", " "),
                **{
                    name: f"{mean[key]:.2f} ± {std[key]:.2f}"
                    for key, name in [
                        ("class_1_recall", "Recall"),
                        ("class_1_precision", "Precision"),
                        ("class_1_f2", "F2"),
                        ("class_1_roc_auc", "ROC-AUC"),
                    ]
                },
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    gap = validation["generalisation_gap"]["class_1_roc_auc"]
    grouped_auc = validation["cv"]["state_grouped"]["fold_mean"]["class_1_roc_auc"]
    if grouped_auc < 0.6:
        st.error(
            f"When whole states are held out, ROC-AUC falls to {grouped_auc:.2f} (gap {gap:+.2f}). The model mainly "
            "recognises *which states* contain discordant districts rather than district-level determinants that transfer "
            "across states. Do not use it to rank districts in a state it was not trained on, and read SHAP drivers as "
            "within-sample associations."
        )
    elif grouped_auc < 0.7:
        st.warning(
            f"Held-out-state ROC-AUC is {grouped_auc:.2f} (vs {grouped_auc + gap:.2f} under random CV). Rankings transfer "
            "only moderately across states: use the Priority list as an audit screening aid, and be most cautious in states "
            "with few training districts."
        )
    left, right = st.columns(2)
    with left:
        states = pd.DataFrame(validation.get("per_state_recall", []))
        if not states.empty:
            st.markdown("**Priority recall when each state is held out** (states with ≥ 3 Priority districts)")
            st.dataframe(states.round(2), hide_index=True, use_container_width=True)
    with right:
        sensitivity = pd.DataFrame(validation.get("label_sensitivity", []))
        if not sensitivity.empty:
            st.markdown("**Stage-1 label sensitivity** (stability of the Priority set)")
            st.dataframe(sensitivity.drop(columns=["settings"]).round(2), hide_index=True, use_container_width=True)


def render(data, root: Path) -> None:
    metrics, provenance = data.metrics, data.provenance
    st.subheader("Model benchmark (Stage 2)")
    if metrics.get("task") == "discordance_binary":
        st.markdown(
            "**Task:** Priority / High-Risk vs Best Performer among high-access districts only "
            f"({metrics['train_size'] + metrics['holdout_size']} districts). Both classes have above-threshold institutional "
            "births, so access proxies cannot separate them and SHAP drivers reflect quality and structural determinants."
        )
    st.dataframe(_benchmark_table(metrics).round(3), hide_index=True, use_container_width=True)
    st.caption(metrics.get("selection_rule", ""))
    selected = metrics["candidate_models"][metrics["selected_model"]]
    evaluated = selected.get("holdout_at_threshold", selected["holdout"])
    recall = evaluated["class_1_recall"]
    if metrics.get("decision_threshold") is not None:
        info = metrics["threshold_selection"]
        st.markdown(
            f"**Class-1 decision threshold:** {metrics['decision_threshold']:.3f} ({info['rule']}; "
            f"OOF recall {info['oof_class_1_recall']:.2f}). Holdout at this threshold: recall {evaluated['class_1_recall']:.2f}, "
            f"precision {evaluated['class_1_precision']:.2f}, F2 {evaluated['class_1_f2']:.2f}."
        )
    if recall >= metrics.get("class_1_recall_target", 0.82):
        st.success(f"Holdout class-1 recall {recall:.2f} meets the 0.82 target.")
    else:
        st.warning(f"Holdout class-1 recall {recall:.2f} is below the blueprint's 0.82 target — treat Priority predictions as a screening aid, not a definitive list.")

    left, right = st.columns(2)
    with left:
        cm = evaluated["confusion_matrix"]
        names = [CLASS_NAMES[label] for label in cm["labels"]]
        figure = px.imshow(cm["matrix"], x=names, y=names, text_auto=True, color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Observed"})
        figure.update_layout(height=360, margin={"l": 0, "r": 0, "t": 30, "b": 0}, title="Holdout confusion matrix", coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        calibration = metrics.get("calibration") or {}
        if calibration:
            st.markdown("**Probability calibration** (" + calibration["method"] + ")")
            st.metric("Holdout class-1 Brier — raw", f"{calibration['holdout_class_1_brier_raw']:.4f}")
            st.metric("Holdout class-1 Brier — calibrated", f"{calibration['holdout_class_1_brier_calibrated']:.4f}")
            st.caption(calibration["note"])
        st.markdown("**Best hyper-parameters**")
        st.json(selected["best_params"], expanded=False)

    _validation_section(data.validation)

    st.subheader("Data provenance")
    cols = st.columns(4)
    cols[0].metric("NFHS-5 districts", provenance.get("districts_total", "—"))
    cols[1].metric("Matched to HMIS", provenance.get("districts_hmis_matched", "—"))
    cols[2].metric("Modelled", provenance.get("districts_modelled", "—"))
    cols[3].metric("Features kept", len(metrics.get("feature_columns", [])))
    if provenance.get("hmis_archives_used"):
        years = [
            name.replace("hmis_", "").replace("_official.zip", "").replace("_", "–")
            for name in provenance["hmis_archives_used"]
        ]
        st.write("HMIS financial years pooled:", ", ".join(years))
        if provenance.get("hmis_archives_missing"):
            st.warning("Some configured HMIS financial years were unavailable; the pooled window is shorter than planned.")
    with st.expander("Feature selection audit (missingness, anchor-proxy and VIF rules)"):
        st.dataframe(data.feature_audit, hide_index=True, use_container_width=True)
    with st.expander("District name reconciliation audit"):
        if not data.match_audit.empty:
            st.write(data.match_audit["method"].value_counts())
            st.dataframe(data.match_audit, hide_index=True, use_container_width=True)
    with st.expander("Stage-1 OLS anchor model"):
        st.json({k: v for k, v in data.stage1.items() if k != "feature_columns"})

    figures = sorted((root / "reports" / "figures").glob("*.png"))
    if figures:
        st.subheader("Publication figures")
        for left, right in zip(figures[::2], figures[1::2] + [None]):
            a, b = st.columns(2)
            a.image(str(left), use_column_width=True)
            if right is not None:
                b.image(str(right), use_column_width=True)
