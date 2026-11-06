import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.preprocess_impute import StateKNNImputer, engineer_features, select_features


def test_knn_imputer_uses_neighbours_not_just_the_median():
    # Two clusters inside one state; a missing value must come from its own cluster.
    frame = pd.DataFrame(
        {
            "state": ["S"] * 8,
            "a": [1, 1.1, 0.9, 1.0, 10, 10.1, 9.9, 10.0],
            "b": [5, 5.1, 4.9, np.nan, 50, 50.2, 49.8, 50.1],
        }
    )
    imputer = StateKNNImputer(feature_columns=["a", "b"], n_neighbors=3).fit(frame)
    filled = imputer.transform(frame)
    assert abs(filled[3, 1] - 5.0) < 0.5, "imputed value should come from the low cluster, not the median (~27)"


def test_small_state_falls_back_to_zone_and_all_missing_columns_survive():
    frame = pd.DataFrame(
        {
            "state": ["Big"] * 6 + ["Tiny"],
            "a": [1, 2, 3, 4, 5, 6, 3.5],
            "b": [np.nan] * 6 + [np.nan],
        }
    )
    imputer = StateKNNImputer(feature_columns=["a", "b"], zones={"z": ["Big", "Tiny"]}).fit(frame)
    output = imputer.transform(frame)
    assert output.shape == (7, 2)
    assert np.isfinite(output).all()
    assert "Tiny" not in imputer.state_imputers_


def test_imputer_respects_sklearn_clone_contract():
    imputer = StateKNNImputer(feature_columns=["a", "b"], zones={"z": ["S"]})
    assert clone(imputer).get_params()["feature_columns"] == ["a", "b"]


def test_engineer_features_builds_rates_and_treats_suppressed_zeros_as_missing():
    frame = pd.DataFrame(
        {
            "state": ["S", "S"],
            "district": ["x", "y"],
            "public_c_section_pct": [4.0, 0.0],
            "oop_expenditure_inr": [999.0, 3000.0],
            "hmis_total_c_sections": [10, 20],
            "hmis_institutional_deliveries": [100, 0],
        }
    )
    result = engineer_features(frame, [], {"zero_as_missing": ["public_c_section_pct"]})
    assert result.loc[0, "public_c_section_deficit"] == 6.0
    assert pd.isna(result.loc[1, "public_c_section_pct"])
    assert result.loc[0, "hmis_c_section_rate_pct"] == 10.0
    assert pd.isna(result.loc[1, "hmis_c_section_rate_pct"]), "zero denominator must not produce inf"
    assert abs(result.loc[0, "oop_expenditure_log"] - 3.0) < 1e-9


def test_feature_selection_rules():
    rng = np.random.default_rng(1)
    anchor = rng.uniform(40, 100, 300)
    base = rng.normal(size=300)
    frame = pd.DataFrame(
        {
            "institutional_births_pct": anchor,
            "proxy": anchor + rng.normal(0, 1, 300),
            "sparse": np.where(rng.random(300) < 0.4, np.nan, rng.normal(size=300)),
            "x1": base,
            "x1_copy": base + rng.normal(0, 0.05, 300),
            "x2": rng.normal(size=300),
        }
    )
    kept, audit = select_features(
        frame,
        ["proxy", "sparse", "x1", "x1_copy", "x2"],
        {"max_missing_share": 0.2, "anchor_proxy_max_abs_corr": 0.9, "vif_threshold": 7.5},
    )
    rules = dict(zip(audit.loc[audit["decision"] == "dropped", "feature"], audit.loc[audit["decision"] == "dropped", "rule"]))
    assert rules["sparse"] == "missingness"
    assert rules["proxy"] == "anchor_proxy"
    assert ("x1" in rules) != ("x1_copy" in rules), "exactly one of the collinear pair should be dropped by VIF"
    assert "x2" in kept
