"""Tab 1 — national choropleth, OLS discordance quadrant plot and global domain attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.common import CLASS_COLOURS
from src.configuration import pretty


def _choropleth(data, frame: pd.DataFrame) -> None:
    if data.geojson is None:
        st.info("No district boundaries are available for this dataset (synthetic demo or GeoJSON missing).")
        return
    mapped_keys = {feature["id"] for feature in data.geojson["features"]}
    plotted = frame.loc[frame["district_key"].isin(mapped_keys)]
    figure = px.choropleth_mapbox(
        plotted,
        geojson=data.geojson,
        locations="district_key",
        color="priority_label",
        color_discrete_map=CLASS_COLOURS,
        category_orders={"priority_label": list(CLASS_COLOURS)},
        hover_name="district",
        hover_data={"state": True, "district_key": False, "priority_risk_score": ":.1f", "studentized_residual": ":.2f"},
        mapbox_style="carto-positron",
        center={"lat": 22.5, "lon": 80.5},
        zoom=3.4,
        opacity=0.82,
        labels={"priority_label": "Class", "priority_risk_score": "Risk score (%)"},
    )
    figure.update_traces(marker_line_width=0.2, marker_line_color="white")
    figure.update_layout(height=560, margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend={"orientation": "h", "y": -0.02})
    st.plotly_chart(figure, use_container_width=True)
    unmapped = len(frame) - len(plotted)
    if unmapped:
        st.caption(
            f"{unmapped} filtered district(s) have no polygon in the 2011-era boundary file (districts created later). "
            "They are included in every table and chart, but not on the map."
        )


def _quadrant(data, frame: pd.DataFrame) -> None:
    stage1 = data.stage1
    plot = frame.copy()
    plot["expected_risk"] = plot["ols_expected_risk"]
    figure = px.scatter(
        plot,
        x="institutional_births_pct",
        y="composite_maternal_risk",
        color="priority_label",
        color_discrete_map=CLASS_COLOURS,
        category_orders={"priority_label": list(CLASS_COLOURS)},
        hover_name="district",
        hover_data={
            "state": True,
            "composite_maternal_risk": ":.2f",
            "expected_risk": ":.2f",
            "studentized_residual": ":.2f",
            "priority_label": False,
        },
        labels={
            "institutional_births_pct": "Institutional births (%)",
            "composite_maternal_risk": "Composite maternal risk (z)",
            "expected_risk": "OLS expected risk",
            "studentized_residual": "Studentized residual r_i",
            "priority_label": "Class",
        },
        opacity=0.8,
    )
    if stage1:
        all_care = data.master["institutional_births_pct"]
        x = np.linspace(all_care.min(), all_care.max(), 120)
        expected = stage1["intercept"] + stage1["coefficient"] * x
        figure.add_trace(go.Scatter(x=x, y=expected, mode="lines", name="OLS expected risk", line={"color": "#334155", "width": 2}))
        # Residual-threshold boundary r_i = tau_r, using the leverage of each x.
        centred = x - all_care.mean()
        leverage = 1 / len(all_care) + centred**2 / np.sum((all_care - all_care.mean()) ** 2)
        boundary = expected + stage1["residual_threshold"] * stage1["residual_sigma"] * np.sqrt(1 - leverage)
        upper = x >= stage1["care_threshold"]
        figure.add_trace(
            go.Scatter(x=x[upper], y=boundary[upper], mode="lines", name=f"Residual threshold (r = {stage1['residual_threshold']})", line={"color": "#D32F2F", "dash": "dash", "width": 1.5})
        )
        figure.add_vline(x=stage1["care_threshold"], line_dash="dot", line_color="#64748b", annotation_text="care threshold")
        figure.add_hline(y=stage1["risk_median"], line_dash="dot", line_color="#64748b", annotation_text="median risk")
    figure.update_layout(height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend={"orientation": "h", "y": -0.15})
    st.plotly_chart(figure, use_container_width=True)


def render(data, mask: pd.Series) -> None:
    master = data.master.merge(
        data.actions[["district", "state", "priority_risk_score", "crs_verbal_autopsy_audit_flag"]],
        on=["district", "state"],
        how="left",
    )
    frame = master.loc[mask.to_numpy()] if mask.any() else master
    counts = frame["priority_label"].value_counts()
    audit_flags = int(frame["crs_verbal_autopsy_audit_flag"].fillna(False).astype(bool).sum())
    columns = st.columns(5)
    columns[0].metric("Districts shown", len(frame))
    for column, label in zip(columns[1:], CLASS_COLOURS):
        column.metric(label, int(counts.get(label, 0)))
    st.caption(f"CRS / verbal-autopsy audit queue (Low Burden with low institutional births): **{audit_flags}** districts — a verification list, not evidence of low need.")

    left, right = st.columns(2)
    with left:
        st.subheader("Priority classes by district")
        _choropleth(data, frame)
    with right:
        st.subheader("OLS discordance quadrants")
        _quadrant(data, frame)

    st.subheader("What drives Priority / High-Risk classification nationally?")
    left, right = st.columns(2)
    with left:
        if not data.domain_shares.empty:
            shares = data.domain_shares.copy()
            shares["Domain"] = shares["domain"].map(lambda d: d.replace("_", " ").title())
            figure = px.bar(
                shares.sort_values("share_pct"),
                x="share_pct",
                y="Domain",
                orientation="h",
                hover_data={"scheme": True},
                labels={"share_pct": "Share of |class-1 SHAP| among Priority districts (%)"},
                color_discrete_sequence=["#2a6f97"],
            )
            figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0})
            st.plotly_chart(figure, use_container_width=True)
            st.caption("Domain shares are exact sums of member-feature TreeSHAP values (additivity), computed over observed Priority districts.")
    with right:
        shap_columns = [c for c in data.shap.columns if c.startswith("shap__")]
        class_one = data.shap.loc[data.shap["priority_class"] == 1, shap_columns]
        if not class_one.empty:
            importance = class_one.abs().mean().sort_values(ascending=False).head(12)
            importance.index = [pretty(c.replace("shap__", "")) for c in importance.index]
            figure = px.bar(
                importance.sort_values(),
                orientation="h",
                labels={"value": "Mean |TreeSHAP| (class 1)", "index": ""},
                color_discrete_sequence=["#7c3aed"],
            )
            figure.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0}, showlegend=False)
            st.plotly_chart(figure, use_container_width=True)
