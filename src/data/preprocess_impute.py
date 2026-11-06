from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


def to_numeric(series: pd.Series) -> pd.Series:
    """Parse common presentation formats without silently coercing malformed values to zero."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _ratio(numerator: pd.Series, denominator: pd.Series, scale: float, upper: float | None = None) -> pd.Series:
    values = numerator / denominator.where(denominator > 0)
    values = (values * scale).replace([np.inf, -np.inf], np.nan)
    return values.clip(lower=0, upper=upper)


# HMIS-derived explanatory rates: (output column, numerator count, denominator count, scale, cap)
HMIS_RATES = [
    ("hmis_c_section_rate_pct", "hmis_total_c_sections", "hmis_institutional_deliveries", 100, 100),
    ("hmis_severe_anaemia_pct", "hmis_hb_below_7", "hmis_anc_registered", 100, 100),
    ("hmis_blood_units_per_1000_deliveries", "hmis_blood_units_issued", "hmis_institutional_deliveries", 1000, None),
    ("hmis_jssk_transport_pct", "hmis_jssk_transport", "hmis_institutional_deliveries", 100, 100),
    ("hmis_early_discharge_pct", "hmis_discharged_within_48h", "hmis_institutional_deliveries", 100, 100),
    ("hmis_first_trimester_reg_pct", "hmis_anc_first_trimester", "hmis_anc_registered", 100, 100),
    ("hmis_ifa_180_pct", "hmis_ifa_180", "hmis_anc_registered", 100, 100),
]


def engineer_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    preprocessing: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalise declared fields and add transparent, policy-relevant derived variables."""
    preprocessing = preprocessing or {}
    result = frame.copy()
    numeric_candidates = [
        column
        for column in result.columns
        if column not in {"state", "district", "priority_label", "hmis_match_method"} and not column.startswith("_")
    ]
    for column in numeric_candidates:
        if result[column].dtype == object:
            parsed = to_numeric(result[column])
            if parsed.notna().sum() >= result[column].notna().sum() * 0.9:
                result[column] = parsed
    for column in preprocessing.get("zero_as_missing", []):
        if column in result:
            result[column] = result[column].mask(result[column] == 0)

    for output, numerator, denominator, scale, cap in HMIS_RATES:
        if numerator in result and denominator in result:
            result[output] = _ratio(result[numerator], result[denominator], scale, cap)
    if "public_c_section_pct" in result:
        # WHO: <10% caesarean rate signals unmet need for surgical obstetric care.
        result["public_c_section_deficit"] = (10 - result["public_c_section_pct"]).clip(lower=0)
    if "oop_expenditure_inr" in result:
        result["oop_expenditure_log"] = np.log10(1 + result["oop_expenditure_inr"].clip(lower=0))
    wash_fields = [
        field
        for field in ["electricity_pct", "improved_water_pct", "improved_sanitation_pct", "clean_cooking_fuel_pct"]
        if field in result
    ]
    if wash_fields:
        # Composite for EDA only: it is an exact linear combination of the raw WASH features.
        result["wash_deprivation_score"] = 100 - result[wash_fields].mean(axis=1)
    missing_required = [column for column in feature_columns if column not in result.columns]
    if missing_required:
        raise ValueError(f"Input is missing configured feature columns: {', '.join(missing_required)}")
    return result


def variance_inflation_factors(frame: pd.DataFrame) -> pd.Series:
    """VIF_j = diag(inverse correlation matrix) on median-filled, standardised features."""
    filled = frame.apply(pd.to_numeric, errors="coerce")
    filled = filled.fillna(filled.median())
    standardised = (filled - filled.mean()) / filled.std(ddof=0).replace(0, np.nan)
    standardised = standardised.dropna(axis=1, how="any")
    correlation = np.corrcoef(standardised.to_numpy(), rowvar=False)
    inverse = np.linalg.pinv(correlation)
    return pd.Series(np.diag(inverse), index=standardised.columns)


def select_features(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    preprocessing: Mapping[str, Any],
    anchor_column: str = "institutional_births_pct",
) -> tuple[list[str], pd.DataFrame]:
    """Apply the label-free feature rules and return the kept list plus an audit table.

    1. Drop candidates missing in more than ``max_missing_share`` of districts.
    2. Drop near-duplicates of the Stage-1 anchor (|r| > ``anchor_proxy_max_abs_corr``): they would
       re-introduce the excluded access variable through the back door.
    3. Iteratively drop the highest-VIF feature while any VIF exceeds ``vif_threshold``.
    None of these rules looks at the priority labels.
    """
    audit: list[dict[str, object]] = []
    kept: list[str] = []
    max_missing = float(preprocessing.get("max_missing_share", 0.20))
    for column in candidates:
        share = float(frame[column].isna().mean())
        if share > max_missing:
            audit.append({"feature": column, "decision": "dropped", "rule": "missingness", "value": round(share, 3)})
        else:
            kept.append(column)

    proxy_limit = float(preprocessing.get("anchor_proxy_max_abs_corr", 1.0))
    if anchor_column in frame:
        for column in list(kept):
            correlation = frame[[column, anchor_column]].corr().iloc[0, 1]
            if pd.notna(correlation) and abs(correlation) > proxy_limit:
                kept.remove(column)
                audit.append(
                    {"feature": column, "decision": "dropped", "rule": "anchor_proxy", "value": round(float(correlation), 3)}
                )

    threshold = float(preprocessing.get("vif_threshold", np.inf))
    while len(kept) > 2:
        vif = variance_inflation_factors(frame[kept])
        worst = vif.idxmax()
        if vif[worst] <= threshold:
            break
        kept.remove(worst)
        audit.append({"feature": worst, "decision": "dropped", "rule": "vif", "value": round(float(vif[worst]), 2)})
    final_vif = variance_inflation_factors(frame[kept]) if len(kept) > 1 else pd.Series(dtype=float)
    for column in kept:
        audit.append(
            {
                "feature": column,
                "decision": "kept",
                "rule": "",
                "value": round(float(final_vif.get(column, np.nan)), 2),
            }
        )
    return kept, pd.DataFrame(audit)


class StateKNNImputer(BaseEstimator, TransformerMixin):
    """Fit NaN-aware KNN imputers on training districts within each state.

    Cohort fallback chain: state (>= ``min_state_size`` training districts) -> geographic zone ->
    national. Columns that are entirely missing within a cohort are pre-filled with the national
    training median so KNNImputer keeps the column. State is used only to choose the cohort and is
    never passed to the classifier.
    """

    def __init__(
        self,
        feature_columns: Sequence[str],
        state_column: str = "state",
        n_neighbors: int = 5,
        zones: Mapping[str, Sequence[str]] | None = None,
        min_state_size: int = 3,
    ):
        self.feature_columns = feature_columns  # stored as passed (sklearn clone contract)
        self.state_column = state_column
        self.n_neighbors = n_neighbors
        self.zones = zones
        self.min_state_size = min_state_size

    def _numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StateKNNImputer requires a pandas DataFrame with the state column.")
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model feature columns: {sorted(missing)}")
        return X.loc[:, list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")

    def _zone_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for zone, states in (self.zones or {}).items():
            for state in states:
                lookup[str(state).strip().lower()] = zone
        return lookup

    def _fit_cohort(self, matrix: pd.DataFrame) -> KNNImputer:
        prepared = matrix.copy()
        all_missing = prepared.columns[prepared.isna().all()]
        prepared[all_missing] = self.global_medians_[all_missing].to_numpy()
        imputer = KNNImputer(n_neighbors=min(self.n_neighbors, max(1, len(prepared))), weights="distance")
        imputer.fit(prepared)
        return imputer

    def fit(self, X: pd.DataFrame, y: object = None):
        if self.state_column not in X:
            raise ValueError(f"Missing state column: {self.state_column}")
        features = self._numeric_features(X)
        self.global_medians_ = features.median(axis=0).fillna(0.0)
        self.global_imputer_ = self._fit_cohort(features)
        lookup = self._zone_lookup()
        states = X[self.state_column].astype(str).str.strip()
        zones = states.str.lower().map(lookup).fillna("unzoned")
        self.state_imputers_ = {}
        for state in states.unique():
            mask = (states == state).to_numpy()
            if mask.sum() >= self.min_state_size:
                self.state_imputers_[state] = self._fit_cohort(features.loc[mask])
        self.zone_imputers_ = {}
        for zone in zones.unique():
            mask = (zones == zone).to_numpy()
            if mask.sum() >= self.min_state_size:
                self.zone_imputers_[zone] = self._fit_cohort(features.loc[mask])
        self.zone_of_state_ = lookup
        return self

    def _imputer_for(self, state: str) -> KNNImputer:
        if state in self.state_imputers_:
            return self.state_imputers_[state]
        zone = self.zone_of_state_.get(state.lower(), "unzoned")
        return self.zone_imputers_.get(zone, self.global_imputer_)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        features = self._numeric_features(X)
        transformed = np.empty((len(X), len(self.feature_columns)), dtype=float)
        state_values = X[self.state_column].astype(str).str.strip().to_numpy()
        for state in pd.unique(state_values):
            positions = np.flatnonzero(state_values == state)
            block = features.iloc[positions]
            # Rows missing every feature have no neighbours: fall back to the national median.
            empty_rows = block.isna().all(axis=1).to_numpy()
            output = np.tile(self.global_medians_.to_numpy(), (len(block), 1))
            if (~empty_rows).any():
                output[~empty_rows] = self._imputer_for(state).transform(block.loc[~empty_rows])
            transformed[positions] = output
        if not np.isfinite(transformed).all():
            transformed = np.where(np.isfinite(transformed), transformed, self.global_medians_.to_numpy())
        return transformed
