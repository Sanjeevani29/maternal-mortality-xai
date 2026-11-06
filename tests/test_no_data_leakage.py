import pytest

from src.target.ols_priority_classifier import FORBIDDEN_FEATURES, assert_no_leakage


def test_forbidden_stage_one_fields_are_rejected():
    with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
        assert_no_leakage(["female_literacy_pct", "institutional_births_pct"])


def test_declared_feature_examples_are_allowed():
    assert_no_leakage(["female_literacy_pct", "severe_anaemia_pct", "blood_storage_units_per_100k"])


def test_priority_class_is_a_forbidden_feature():
    assert "priority_class" in FORBIDDEN_FEATURES
