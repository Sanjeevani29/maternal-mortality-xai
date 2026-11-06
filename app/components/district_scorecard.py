"""Tab 2 — district XAI diagnostic scorecard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components.common import CLASS_COLOURS, pick_district
from src.configuration import domain_by_feature, feature_domains, pretty


def deprivation_percentiles(master: pd.DataFrame, features: list[str], higher_is_worse: set[str]) -> pd.DataFrame:
    """Per-district domain deprivation percentile (0 = least deprived, 100 = most deprived)."""
    lookup = domain_by_feature()
    ranks = pd.DataFrame(index=master.index)
    for feature in features:
        rank = master[feature].rank(pct=True) * 100
        ranks[feature] = rank if feature in higher_is_worse else 100 - rank
    domains = {}
    for domain in feature_domains():
        members = [f for f in features if lookup.get(f) == domain]
        if members:
            domains[domain] = ranks[members].mean(axis=1)
    return pd.DataFrame(domains)


def _gauge(score: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%"},
            title={"text": "Priority risk score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e293b"},
                "steps": [
                    {"range": [0, 33], "color": "#dcfce7"},
                    {"range": [33, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#fee2e2"},
                ],
            },
        )
    )
    figure.update_layout(height=230, margin={"l": 20, "r": 20, "t": 40, "b": 0})
    return figure


def waterfall(shap_row: pd.Series, values: pd.Series, top: int = 10) -> go.Figure:
    """SHAP waterfall from the national baseline to this district's class-1 model output."""
    columns = [c for c in shap_row.index if c.startswith("shap__")]
    contributions = shap_row[columns].astype(float)
    contributions.index = [c.replace("shap__", "") for c in columns]
    order = contributions.abs().sort_values(ascending=False).index
    shown, rest = order[:top], order[top:]
    labels = [f"{pretty(f)} = {values.get(f, np.nan):.1f}" for f in shown]
    measures = ["absolute"] + ["relative"] * len(shown)
    y = [float(shap_row["shap_base_value"])] + contributions[shown].tolist()
    x = ["National baseline"] + labels
    if len(rest):
        x.append(f"{len(rest)} other features")
        y.append(float(contributions[rest].sum()))
        measures.append("relative")
    x.append("District output")
    y.append(0)
    measures.append("total")
    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures[::-1],
            y=x[::-1],
            x=y[::-1],
            increasing={"marker": {"color": "#D32F2F"}},
            decreasing={"marker": {"color": "#388E3C"}},
            totals={"marker": {"color": "#334155"}},
            connector={"line": {"color": "#cbd5e1"}},
        )
    )
    space = shap_row.get("shap_output_space", "probability")
    figure.update_layout(
        height=420,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title=f"Class-1 TreeSHAP waterfall ({'probability' if space == 'probability' else 'log-odds'} scale)",
        xaxis_title="Contribution toward Priority / High-Risk",
    )
    return figure


def render(data, mask: pd.Series) -> None:
    st.subheader("District XAI diagnostic scorecard")
    row = pick_district(data.master, mask, key="scorecard_district")
    action = data.actions.loc[(data.actions["district"] == row["district"]) & (data.actions["state"] == row["state"])].iloc[0]
    shap_row = data.shap.loc[(data.shap["district"] == row["district"]) & (data.shap["state"] == row["state"])].iloc[0]
    features = list(data.metrics["feature_columns"])

    top_left, top_mid, top_right = st.columns([1.1, 1, 1])
    with top_left:
        st.plotly_chart(_gauge(float(action["priority_risk_score"])), use_container_width=True)
    with top_mid:
        colour = CLASS_COLOURS[action["priority_label"]]
        st.markdown(f"<div class='class-pill' style='background:{colour}'>{action['priority_label']}</div>", unsafe_allow_html=True)
        st.metric("Model prediction", action["predicted_label"])
        st.metric("Studentized residual r_i", f"{action['studentized_residual']:+.2f}")
    with top_right:
        st.metric("Institutional births", f"{row['institutional_births_pct']:.1f}%")
        st.metric("Facility-reported MMR (EB-smoothed)", f"{row['maternal_mortality_ratio']:.0f} / 100k")
        st.metric("Primary bottleneck", pretty(str(action["primary_domain_bottleneck"])))
    if not bool(action.get("in_training_domain", True)):
        st.info(
            "This district is outside the model's training domain (it is below the institutional-birth threshold). "
            "The Stage-2 model was trained on high-access districts only, so its risk score and drivers here are an "
            "extrapolation: read them as 'quality-gap risk if access improved', not as a discordance finding."
        )
    if bool(action.get("crs_verbal_autopsy_audit_flag", False)):
        st.error("Low Burden with low institutional births: flag for CRS & verbal-autopsy audit — reported risk may be under-counted.")
    st.caption("Risk score = calibrated probability of the Priority / High-Risk class. Observed class comes from Stage-1 (outcome data); predicted class from the Stage-2 model (determinants only).")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(waterfall(shap_row, row), use_container_width=True)
        st.caption("Red bars push the district toward Priority / High-Risk; green bars pull it away. Feature values shown are as reported (blank = imputed).")
    with right:
        higher_is_worse = set(data.config.get("dashboard", {}).get("higher_is_worse", []))
        percentiles = deprivation_percentiles(data.master, features, higher_is_worse)
        index = data.master.index[(data.master["district"] == row["district"]) & (data.master["state"] == row["state"])][0]
        state_median = percentiles.loc[data.master["state"] == row["state"]].median()
        best = percentiles.loc[data.master["priority_class"] == 3].median()
        labels = [d.replace("_", " ").title() for d in percentiles.columns]
        radar = go.Figure()
        for name, series, colour in [
            ("National best performers (median)", best, "#388E3C"),
            (f"{row['state']} median", state_median, "#64748b"),
            (row["district"], percentiles.loc[index], "#D32F2F"),
        ]:
            values = series.tolist()
            radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], name=name, line={"color": colour}, fill="toself" if name == row["district"] else None, opacity=0.85))
        radar.update_layout(
            polar={"radialaxis": {"range": [0, 100], "tickvals": [25, 50, 75]}},
            height=420,
            margin={"l": 40, "r": 40, "t": 30, "b": 0},
            legend={"orientation": "h", "y": -0.1},
            title="Domain deprivation percentile (higher = worse)",
        )
        st.plotly_chart(radar, use_container_width=True)

    st.markdown("#### Policy action card")
    card_left, card_right = st.columns([1, 1.4])
    with card_left:
        st.markdown(f"**Primary scheme:** {action['recommended_scheme']}")
        st.write(action["recommended_action"])
    with card_right:
        rows = []
        for rank in range(1, 6):
            driver = action.get(f"top_{rank}_driver")
            if isinstance(driver, str) and driver:
                rows.append({"#": rank, "Local driver (value vs state median, SHAP)": driver, "Scheme lever": action.get(f"top_{rank}_scheme", "")})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("No feature pushes this district toward the Priority class.")
    st.warning("SHAP indicates model associations, not clinical causation. Confirm with MDSR, facility audits and community evidence.")
