from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components import (  # noqa: E402
    district_scorecard,
    map_quadrant_view,
    methods_view,
    priority_table_view,
    what_if_simulator,
)
from app.data_access import artefact_version, load_artefacts, load_models, missing_artefacts  # noqa: E402
from src.target.ols_priority_classifier import CLASS_NAMES  # noqa: E402

st.set_page_config(page_title="Maternal Risk XAI", page_icon="◈", layout="wide")


def _inject_css() -> None:
    css = ROOT / "app" / "assets" / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _sidebar_filters(master):
    st.sidebar.header("Filters")
    states = ["All states", *sorted(master["state"].unique())]
    state = st.sidebar.selectbox("State", states, key="filter_state")
    classes = ["All classes", *[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]]
    label = st.sidebar.selectbox("Priority class", classes, key="filter_class")
    query = st.sidebar.text_input("Search district", key="filter_query").strip().lower()
    mask = master["district"].notna()
    if state != "All states":
        mask &= master["state"] == state
    if label != "All classes":
        mask &= master["priority_label"] == label
    if query:
        mask &= master["district"].str.lower().str.contains(query, regex=False)
    st.sidebar.caption(f"{int(mask.sum())} of {len(master)} modelled districts match.")
    return mask


def main() -> None:
    _inject_css()
    st.title("India maternal-risk discordance explorer")
    st.caption(
        "Explainable decision support for LaQshya, PMSMA, JSY and POSHAN Abhiyaan planning — research prototype, "
        "not a clinical or resource-allocation authority."
    )
    missing = missing_artefacts()
    if missing:
        st.error(f"Pipeline artefacts are missing ({', '.join(missing)}). Run the pipeline first:")
        st.code("python -m src.run_pipeline          # official NFHS-5 + HMIS data\npython -m src.run_pipeline --demo   # synthetic demo", language="bash")
        return
    version = artefact_version()
    data = load_artefacts(version)
    model, calibrated = load_models(version)
    if data.provenance.get("mode") == "SYNTHETIC DEMO" or data.master["district"].astype(str).str.startswith("Demo District").all():
        st.warning("You are viewing SYNTHETIC demo records. They are not real district findings.")
    with st.expander("Guardrails — read before acting on any result", expanded=False):
        st.markdown(
            f"""
- **Under-reporting in *Low Burden*:** low reported risk with low institutional births (< {data.config['target']['low_access_audit_threshold']:.0f}%) may reflect unrecorded home-delivery deaths. These districts are flagged for **CRS & verbal-autopsy audits**, not treated as genuinely low risk.
- **Associative, not causal:** SHAP values explain the model's classification; they do not replace MDSR clinical audits.
- **Human in the loop:** what-if simulations are directional inputs for State NHM PIPs and must be paired with LaQshya facility audits.
- **Data window:** {data.provenance.get('hmis_window_note', 'see Methods tab')}
"""
        )
    mask = _sidebar_filters(data.master)
    tabs = st.tabs(["National map & quadrants", "District scorecard", "What-if simulator", "Priority action table", "Methods & audit"])
    with tabs[0]:
        map_quadrant_view.render(data, mask)
    with tabs[1]:
        district_scorecard.render(data, mask)
    with tabs[2]:
        what_if_simulator.render(data, mask, model, calibrated)
    with tabs[3]:
        priority_table_view.render(data, mask)
    with tabs[4]:
        methods_view.render(data, ROOT)


if __name__ == "__main__":
    main()
