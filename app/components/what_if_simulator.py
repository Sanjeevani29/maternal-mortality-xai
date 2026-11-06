"""Tab 3 — counterfactual 'what-if' policy simulator with live TreeSHAP."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.common import pick_district
from src.configuration import pretty
from src.explain.counterfactual_sim import simulate
from src.explain.shap_profiler import explain_single
from src.explain.shap_profiler import prediction_label


def _slider_features(data, shap_row: pd.Series, features: list[str]) -> list[str]:
    """Top-N modifiable features for this district, ranked by |class-1 SHAP|."""
    settings = data.config.get("simulator", {})
    modifiable = [f for f in settings.get("modifiable_features", []) if f in features]
    ranked = sorted(modifiable, key=lambda f: abs(float(shap_row.get(f"shap__{f}", 0.0))), reverse=True)
    return ranked[: int(settings.get("max_sliders", 8))]


def render(data, mask: pd.Series, model, calibrated) -> None:
    st.subheader("Counterfactual 'what-if' policy simulator")
    st.info("Directional model-sensitivity exercise only. It does not estimate the causal effect of a programme.")
    row = pick_district(data.master, mask, key="simulator_district")
    features = list(data.metrics["feature_columns"])
    shap_row = data.shap.loc[(data.shap["district"] == row["district"]) & (data.shap["state"] == row["state"])].iloc[0]
    original = row[["state", *features]].to_frame().T
    original[features] = original[features].astype(float)
    # Show sliders at the model's imputed value when the source value is missing.
    imputed = pd.Series(model.named_steps["imputer"].transform(original)[0], index=features)

    action = data.actions.loc[(data.actions["district"] == row["district"]) & (data.actions["state"] == row["state"])].iloc[0]
    if not bool(action.get("in_training_domain", True)):
        st.info(
            "This district is outside the model's training domain (it is below the institutional-birth threshold). "
            "The Stage-2 model was trained on high-access districts only, so its risk score and drivers here are an "
            "extrapolation: read them as 'quality-gap risk if access improved', not as a discordance finding."
        )
    sliders = _slider_features(data, shap_row, features)
    st.caption("Sliders show the eight modifiable indicators with the largest influence on this district's classification.")
    edits: dict[str, float] = {}
    defaults: dict[str, float] = {}
    columns = st.columns(2)
    for position, feature in enumerate(sliders):
        series = data.master[feature].dropna()
        low, high = float(series.quantile(0.01)), float(series.quantile(0.99))
        current = float(np.clip(imputed[feature], low, high))
        step = max((high - low) / 200, 0.01)
        defaults[feature] = round(current, 2)
        with columns[position % 2]:
            edits[feature] = st.slider(
                f"{pretty(feature)} (now {imputed[feature]:.1f})",
                min_value=round(low, 2),
                max_value=round(high, 2),
                value=defaults[feature],
                step=round(step, 2),
                key=f"slider_{row['state']}_{row['district']}_{feature}",
            )
    changed = {f: v for f, v in edits.items() if not np.isclose(v, defaults[f])}
    baseline_row = original.copy()
    baseline_row[features] = imputed.to_numpy()
    result = simulate(model, baseline_row, changed, risk_model=calibrated)

    a, b, c = st.columns(3)
    a.metric("Baseline priority risk", f"{result['baseline_probability'] * 100:.1f}%", help=prediction_label(model, result["baseline_class"]))
    b.metric(
        "Simulated priority risk",
        f"{result['simulated_probability'] * 100:.1f}%",
        delta=f"{result['delta_percentage_points']:+.1f} pp",
        delta_color="inverse",
    )
    c.metric(
        "Simulated model prediction",
        prediction_label(model, result["simulated_class"]),
        delta=None if result["simulated_class"] == result["baseline_class"] else f"from {prediction_label(model, result['baseline_class'])}",
        delta_color="off",
    )

    if changed:
        before = explain_single(model, baseline_row, features)
        after = explain_single(model, result["candidate_row"], features)
        delta = (after - before).loc[list(changed)]
        figure = go.Figure(
            go.Bar(
                x=delta.to_numpy(),
                y=[pretty(f) for f in delta.index],
                orientation="h",
                marker_color=["#D32F2F" if v > 0 else "#388E3C" for v in delta.to_numpy()],
            )
        )
        figure.update_layout(
            height=60 + 40 * len(delta),
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            title="Live TreeSHAP: change in each edited driver's class-1 contribution",
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.caption("Move a slider to simulate an intervention.")
