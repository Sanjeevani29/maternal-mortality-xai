import numpy as np
import pandas as pd
import pytest

from src.explain.shap_profiler import explain, model_output, tree_model
from src.models.train_evaluate import DEFAULT_PARAMS, available_candidates, class_weights, make_pipeline


def _frame(rows: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(rows, 6)), columns=[f"f{i}" for i in range(6)])
    X.iloc[::11, 2] = np.nan
    X.insert(0, "state", np.where(np.arange(rows) % 2, "A", "B"))
    score = X["f0"].fillna(0) + 0.6 * X["f1"] - 0.4 * X["f3"]
    y = pd.Series(np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75])) + 1)
    return X, y


@pytest.mark.parametrize("name", [n for n, spec in available_candidates().items() if spec[2]])
def test_shap_values_sum_to_model_output(name):
    """f_c(x) = phi_0,c + sum_j phi_j,c for every TreeSHAP-compatible candidate (blueprint section 3.3)."""
    X, y = _frame()
    features = [c for c in X.columns if c != "state"]
    factory = available_candidates()[name][0]
    pipeline = make_pipeline(factory(DEFAULT_PARAMS[name], 0), features, zones=None, n_neighbors=5)
    pipeline.fit(X, y, classifier__sample_weight=class_weights(y, 1.25))
    result = explain(pipeline, X)
    reconstructed = result["base_value"] + result["values"].sum(axis=1)
    target = model_output(tree_model(pipeline), result["transformed"], result["tree_class_index"])
    np.testing.assert_allclose(reconstructed, target, atol=1e-4)
    assert result["max_additivity_error"] < 1e-4


@pytest.mark.parametrize("name", list(available_candidates()))
def test_binary_shap_explains_priority_class(name):
    """Binary discordance task: SHAP must explain P(Priority) / its margin, for every candidate."""
    X, y = _frame(seed=3)
    y = y.map({1: 1, 2: 1, 3: 3, 4: 3})
    features = [c for c in X.columns if c != "state"]
    factory = available_candidates()[name][0]
    pipeline = make_pipeline(factory(DEFAULT_PARAMS[name], 0), features, zones=None, n_neighbors=5)
    pipeline.fit(X, y, classifier__sample_weight=class_weights(y, 1.25))
    assert list(pipeline.classes_) == [1, 3]
    result = explain(pipeline, X)
    assert result["max_additivity_error"] < 1e-4
    priority_probability = pipeline.predict_proba(X)[:, 0]
    margin = result["base_value"] + result["values"].sum(axis=1)
    # Whatever the output space, the explained quantity must rise with P(Priority).
    assert np.corrcoef(margin, priority_probability)[0, 1] > 0.95


def test_encoded_classifier_exposes_original_labels():
    X, y = _frame()
    features = [c for c in X.columns if c != "state"]
    factory = available_candidates()["random_forest"][0]
    pipeline = make_pipeline(factory(DEFAULT_PARAMS["random_forest"], 0), features, None, 5).fit(X, y)
    assert list(pipeline.classes_) == [1, 2, 3, 4]
    assert pipeline.predict_proba(X).shape == (len(X), 4)
