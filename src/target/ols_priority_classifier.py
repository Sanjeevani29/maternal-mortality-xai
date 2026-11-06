from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


FORBIDDEN_FEATURES = {
    "maternal_mortality_ratio",
    "severe_morbidity_rate",
    "composite_maternal_risk",
    "institutional_births_pct",
    "ols_expected_risk",
    "ols_residual",
    "studentized_residual",
    "priority_class",
    "hmis_maternal_deaths",
    "hmis_live_births",
    "hmis_institutional_deliveries",
    "hmis_obstetric_complications",
    "hmis_pph_cases",
    "hmis_eclampsia_cases",
    "hmis_sepsis_cases",
    "hmis_ruptured_uterus_cases",
    # The public-facility share is a sub-component of the institutional-birth anchor.
    "institutional_births_public_pct",
    "priority_label",
}

CLASS_NAMES = {
    1: "Priority / High-Risk",
    2: "Coverage Gap",
    3: "Best Performer",
    4: "Low Burden",
}


@dataclass
class Stage1Model:
    intercept: float
    coefficient: float
    residual_sigma: float
    care_threshold: float
    risk_median: float
    residual_threshold: float


def _z_score(values: pd.Series) -> pd.Series:
    deviation = values.std(ddof=0)
    if not np.isfinite(deviation) or deviation == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / deviation


def construct_maternal_risk(frame: pd.DataFrame, target_config: dict) -> pd.DataFrame:
    result = frame.copy()
    deaths = result["hmis_maternal_deaths"].clip(lower=0)
    births = result["hmis_live_births"].clip(lower=1)
    state_deaths = deaths.groupby(result["state"]).transform("sum")
    state_births = births.groupby(result["state"]).transform("sum").clip(lower=1)
    state_rate = state_deaths / state_births
    prior = float(target_config["eb_prior_live_births"])
    result["maternal_mortality_ratio"] = ((deaths + prior * state_rate) / (births + prior)) * 100_000

    component_columns = [
        "hmis_pph_cases",
        "hmis_eclampsia_cases",
        "hmis_sepsis_cases",
        "hmis_ruptured_uterus_cases",
    ]
    if "hmis_obstetric_complications" in result:
        # The current official report publishes this as a single all-obstetric-complications total.
        # It is used only in target construction and is explicitly forbidden from Stage 2 features.
        complications = result["hmis_obstetric_complications"].clip(lower=0)
    elif set(component_columns).issubset(result.columns):
        complications = sum(result[column].clip(lower=0) for column in component_columns)
    else:
        raise ValueError("HMIS source needs total obstetric complications or the four specified complication counts.")
    deliveries = result["hmis_institutional_deliveries"].clip(lower=1)
    result["severe_morbidity_rate"] = complications / deliveries * 100
    result["composite_maternal_risk"] = (
        float(target_config["mortality_weight"]) * _z_score(result["maternal_mortality_ratio"])
        + float(target_config["morbidity_weight"]) * _z_score(result["severe_morbidity_rate"])
    )
    return result


def assign_priority_classes(frame: pd.DataFrame, target_config: dict) -> tuple[pd.DataFrame, Stage1Model]:
    """Fit the Stage-1 OLS relationship and construct the four operational labels."""
    result = construct_maternal_risk(frame, target_config)
    care = result["institutional_births_pct"].to_numpy(dtype=float)
    risk = result["composite_maternal_risk"].to_numpy(dtype=float)
    if not np.isfinite(care).all() or not np.isfinite(risk).all() or len(result) < 8:
        raise ValueError("Stage 1 requires at least eight districts with finite care and risk values.")
    design = np.column_stack([np.ones(len(care)), care])
    intercept, coefficient = np.linalg.lstsq(design, risk, rcond=None)[0]
    expected = design @ np.array([intercept, coefficient])
    residuals = risk - expected
    centered = care - care.mean()
    denominator = np.square(centered).sum()
    leverage = 1 / len(care) + np.square(centered) / denominator if denominator else np.repeat(1 / len(care), len(care))
    sigma = float(np.sqrt(np.square(residuals).sum() / max(len(care) - 2, 1)))
    studentized = residuals / (max(sigma, 1e-9) * np.sqrt(np.maximum(1 - leverage, 1e-9)))
    care_threshold = float(np.nanmedian(care)) if target_config["care_threshold"] == "median" else float(target_config["care_threshold"])
    risk_median = float(np.nanmedian(risk))
    residual_threshold = float(target_config["residual_threshold"])

    result["ols_expected_risk"] = expected
    result["ols_residual"] = residuals
    result["studentized_residual"] = studentized
    result["priority_class"] = np.select(
        [
            (care >= care_threshold) & (studentized > residual_threshold),
            (care < care_threshold) & (risk > risk_median),
            (care >= care_threshold) & (studentized <= residual_threshold),
        ],
        [1, 2, 3],
        default=4,
    ).astype(int)
    result["priority_label"] = result["priority_class"].map(CLASS_NAMES)
    model = Stage1Model(intercept, coefficient, sigma, care_threshold, risk_median, residual_threshold)
    return result, model


def assert_no_leakage(feature_columns: list[str] | set[str]) -> None:
    leakage = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    if leakage:
        raise AssertionError(f"DATA LEAKAGE DETECTED: {', '.join(leakage)}")


def stage1_model_dict(model: Stage1Model) -> dict[str, float]:
    return asdict(model)
