import numpy as np

from src.models.metrics import apply_threshold, choose_threshold, classification_metrics


def test_apply_threshold_promotes_class_one_and_falls_back_to_next_best():
    classes = np.array([1, 2, 3])
    proba = np.array([[0.30, 0.50, 0.20], [0.10, 0.20, 0.70]])
    assert list(apply_threshold(proba, classes, None)) == [2, 3]
    assert list(apply_threshold(proba, classes, 0.25)) == [1, 3]


def test_choose_threshold_meets_recall_target_with_highest_cutoff():
    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [3] * 60)
    p1 = np.clip(np.where(y == 1, 0.6, 0.35) + rng.normal(0, 0.15, 100), 0, 1)
    proba = np.column_stack([p1, 1 - p1])
    info = choose_threshold(y, proba, np.array([1, 3]), target_recall=0.82)
    assert info["oof_class_1_recall"] >= 0.82
    looser = classification_metrics(y, proba, np.array([1, 3]), info["threshold"] - 0.05)
    assert looser["class_1_recall"] >= info["oof_class_1_recall"]


def test_binary_metrics_report_auc():
    y = np.array([1, 1, 3, 3])
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
    metrics = classification_metrics(y, proba, np.array([1, 3]))
    assert metrics["ovr_roc_auc"] == 1.0
    assert metrics["class_1_recall"] == 1.0
