"""End-to-end smoke test on SYNTHETIC data, writing every artefact to a temporary directory."""
import copy
import json

import joblib
import pandas as pd

from src.configuration import feature_columns, load_config
from src.reports.policy_brief import to_excel, to_pdf
from src.run_pipeline import run
from src.target.ols_priority_classifier import FORBIDDEN_FEATURES


def test_configured_candidates_contain_no_forbidden_columns():
    assert not set(feature_columns()) & FORBIDDEN_FEATURES


def test_demo_pipeline_runs_end_to_end(tmp_path):
    config = copy.deepcopy(load_config())
    for name, value in config["paths"].items():
        if not value.startswith("data/raw") and not value.startswith("data/interim"):
            config["paths"][name] = str(tmp_path / value)
    config["project"]["demo_rows"] = 160
    summary = run(demo=True, trials=0, skip_figures=True, config=config)

    assert summary["mode"] == "SYNTHETIC DEMO"
    actions = pd.read_csv(tmp_path / "data/processed/priority_action_table.csv")
    shap = pd.read_parquet(tmp_path / "data/processed/shap_matrix_class1.parquet")
    assert len(actions) == len(shap) == 160
    assert actions["priority_risk_score"].between(0, 100).all()
    class_one = actions.loc[actions["priority_class"] == 1]
    assert class_one["top_1_driver"].notna().all()
    model = joblib.load(tmp_path / "models/best_classifier_pipeline.joblib")
    assert not set(model.named_steps["imputer"].feature_columns) & FORBIDDEN_FEATURES
    shares = pd.read_csv(tmp_path / "data/processed/domain_shares.csv")
    assert abs(shares["share_pct"].sum() - 100) < 0.1

    validation = json.loads((tmp_path / "models/validation_report.json").read_text())
    assert set(validation["cv"]) == {"random_stratified", "state_grouped"}
    assert len(validation["label_sensitivity"]) == 7
    assert 0 <= validation["cv"]["state_grouped"]["pooled"]["class_1_recall"] <= 1

    assert to_pdf(actions.head(5), "Test brief").startswith(b"%PDF")
    assert to_excel(actions.head(5), shares)[:2] == b"PK"
