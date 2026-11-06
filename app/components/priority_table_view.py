"""Tab 4 — master priority action table with CSV / Excel / PDF policy-brief export."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.reports.policy_brief import to_excel, to_pdf

DISPLAY_COLUMNS = [
    "rank",
    "district",
    "state",
    "priority_label",
    "predicted_label",
    "priority_risk_score",
    "studentized_residual",
    "institutional_births_pct",
    "primary_domain_bottleneck",
    "recommended_scheme",
    "top_1_driver",
    "top_2_driver",
    "top_3_driver",
    "top_4_driver",
    "top_5_driver",
    "crs_verbal_autopsy_audit_flag",
]


def render(data, mask: pd.Series) -> None:
    st.subheader("Master priority action table")
    keys = data.master.loc[mask.to_numpy(), ["district", "state"]] if mask.any() else data.master[["district", "state"]]
    filtered = data.actions.merge(keys, on=["district", "state"], how="inner").sort_values("rank")
    only_priority = st.toggle("Show only observed Priority / High-Risk districts", value=False)
    if only_priority:
        filtered = filtered.loc[filtered["priority_class"] == 1]
    columns = [c for c in DISPLAY_COLUMNS if c in filtered]
    st.dataframe(
        filtered[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "priority_risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.1f%%"),
            "studentized_residual": st.column_config.NumberColumn("Residual r_i", format="%+.2f"),
            "institutional_births_pct": st.column_config.NumberColumn("Inst. births %", format="%.1f"),
            "crs_verbal_autopsy_audit_flag": st.column_config.CheckboxColumn("CRS audit flag"),
        },
    )
    state = st.session_state.get("filter_state", "All states")
    title = "Maternal-risk discordance policy brief" + ("" if state == "All states" else f" - {state}")
    stem = "maternal_priority_" + ("india" if state == "All states" else state.lower().replace(" ", "_").replace("&", "and"))
    a, b, c = st.columns(3)
    a.download_button("Download CSV", filtered.to_csv(index=False).encode("utf-8"), f"{stem}.csv", "text/csv")
    b.download_button(
        "Download Excel",
        to_excel(filtered, data.domain_shares),
        f"{stem}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    c.download_button("Download PDF policy brief", to_pdf(filtered, title), f"{stem}_brief.pdf", "application/pdf")
