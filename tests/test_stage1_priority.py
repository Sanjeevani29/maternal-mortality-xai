import numpy as np
import pandas as pd

from src.target.ols_priority_classifier import assign_priority_classes


def test_stage1_creates_finite_risk_and_valid_operational_labels():
    rows = 24
    care = np.linspace(45, 95, rows)
    frame = pd.DataFrame(
        {
            "state": ["A"] * 12 + ["B"] * 12,
            "institutional_births_pct": care,
            "hmis_maternal_deaths": np.linspace(70, 38, rows) + np.tile([20, -12], 12),
            "hmis_live_births": np.repeat(20_000, rows),
            "hmis_institutional_deliveries": 20_000 * care / 100,
            "hmis_pph_cases": np.linspace(200, 80, rows),
            "hmis_eclampsia_cases": np.linspace(120, 50, rows),
            "hmis_sepsis_cases": np.linspace(80, 20, rows),
            "hmis_ruptured_uterus_cases": np.linspace(20, 5, rows),
        }
    )
    result, model = assign_priority_classes(
        frame,
        {
            "mortality_weight": 0.7,
            "morbidity_weight": 0.3,
            "eb_prior_live_births": 3000,
            "care_threshold": "median",
            "residual_threshold": 0.25,
        },
    )
    assert set(result["priority_class"]).issubset({1, 2, 3, 4})
    assert result["composite_maternal_risk"].notna().all()
    assert model.care_threshold == np.median(care)
