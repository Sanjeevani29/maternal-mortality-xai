from __future__ import annotations

import pandas as pd
import streamlit as st

CLASS_COLOURS = {
    "Priority / High-Risk": "#D32F2F",
    "Coverage Gap": "#F57C00",
    "Best Performer": "#388E3C",
    "Low Burden": "#1976D2",
}


def district_options(master: pd.DataFrame, mask: pd.Series) -> list[str]:
    subset = master.loc[mask] if mask.any() else master
    ordered = subset.sort_values(["state", "district"])
    return (ordered["district"] + " — " + ordered["state"]).tolist()


def pick_district(master: pd.DataFrame, mask: pd.Series, key: str) -> pd.Series:
    """Shared district picker; the last choice is remembered across tabs."""
    options = district_options(master, mask)
    remembered = st.session_state.get("selected_district")
    index = options.index(remembered) if remembered in options else 0
    choice = st.selectbox("District", options, index=index, key=key)
    st.session_state["selected_district"] = choice
    district, state = choice.split(" — ", 1)
    return master.loc[(master["district"] == district) & (master["state"] == state)].iloc[0]
