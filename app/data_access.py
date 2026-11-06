"""Cached artefact loading for the Streamlit decision-support app."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.configuration import configured_path, load_config

REQUIRED = ["master_dataset", "priority_table", "shap_matrix", "model", "metrics"]


@dataclass
class Artefacts:
    master: pd.DataFrame
    actions: pd.DataFrame
    shap: pd.DataFrame
    metrics: dict[str, Any]
    stage1: dict[str, Any]
    domain_shares: pd.DataFrame
    provenance: dict[str, Any]
    feature_audit: pd.DataFrame
    match_audit: pd.DataFrame
    geojson: dict | None
    validation: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)


def missing_artefacts() -> list[str]:
    config = load_config()
    return [name for name in REQUIRED if not configured_path(config, name).exists()]


def _optional_csv(path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner="Loading pre-computed district artefacts...")
def load_artefacts(_version: float) -> Artefacts:
    config = load_config()
    path = lambda name: configured_path(config, name)  # noqa: E731
    master = pd.read_csv(path("master_dataset"))
    master = master.loc[master["priority_class"].notna()].copy()
    master["priority_class"] = master["priority_class"].astype(int)
    master["district_key"] = master["state"].astype(str).str.strip() + "|" + master["district"].astype(str).str.strip()
    geo = path("geojson_simplified")
    return Artefacts(
        master=master.reset_index(drop=True),
        actions=pd.read_csv(path("priority_table")),
        shap=pd.read_parquet(path("shap_matrix")),
        metrics=json.loads(path("metrics").read_text(encoding="utf-8")),
        stage1=joblib.load(path("stage1_model")) if path("stage1_model").exists() else {},
        domain_shares=_optional_csv(path("domain_shares")),
        provenance=json.loads(path("provenance").read_text(encoding="utf-8")) if path("provenance").exists() else {},
        feature_audit=_optional_csv(path("feature_audit")),
        match_audit=_optional_csv(path("match_audit")),
        geojson=json.loads(geo.read_text(encoding="utf-8")) if geo.exists() else None,
        validation=json.loads(path("validation_report").read_text(encoding="utf-8"))
        if path("validation_report").exists()
        else {},
        config=config,
    )


@st.cache_resource(show_spinner="Loading models...")
def load_models(_version: float):
    config = load_config()
    model = joblib.load(configured_path(config, "model"))
    calibrated_path = configured_path(config, "calibrated_model")
    calibrated = joblib.load(calibrated_path) if calibrated_path.exists() else None
    return model, calibrated


def artefact_version() -> float:
    """Changes whenever the pipeline rewrites the model, so caches refresh after a re-run."""
    config = load_config()
    return configured_path(config, "model").stat().st_mtime
