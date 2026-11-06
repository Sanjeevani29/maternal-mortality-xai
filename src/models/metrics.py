from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def class_one_index(classes: np.ndarray) -> int:
    matches = np.flatnonzero(np.asarray(classes) == 1)
    if not len(matches):
        raise ValueError("Class 1 (Priority / High-Risk) is absent from the fitted classes.")
    return int(matches[0])


def selection_objective(class_1_f1: float, macro_f1: float, class_1_weight: float = 0.60) -> float:
    return class_1_weight * class_1_f1 + (1 - class_1_weight) * macro_f1


def apply_threshold(probabilities: np.ndarray, classes: np.ndarray, threshold: float | None) -> np.ndarray:
    """Predict class 1 whenever P(class 1) >= threshold; otherwise the most likely other class.

    ``threshold=None`` gives the plain argmax rule.
    """
    classes = np.asarray(classes)
    if threshold is None or 1 not in classes:
        return classes[np.argmax(probabilities, axis=1)]
    index = class_one_index(classes)
    others = probabilities.copy()
    others[:, index] = -np.inf
    fallback = classes[np.argmax(others, axis=1)]
    return np.where(probabilities[:, index] >= threshold, 1, fallback)


def choose_threshold(y_true, probabilities: np.ndarray, classes: np.ndarray, target_recall: float) -> dict[str, Any]:
    """Pick the class-1 cut-off on out-of-fold probabilities (never the holdout).

    Rule: the *highest* threshold whose class-1 recall reaches ``target_recall`` (i.e. the best precision
    that still meets the screening target). If no threshold reaches it, fall back to the F2-optimal one.
    """
    y_true = np.asarray(y_true)
    index = class_one_index(classes)
    candidates = np.unique(np.round(probabilities[:, index], 4))
    rows = []
    for threshold in candidates:
        predicted = apply_threshold(probabilities, classes, float(threshold))
        _, recall, _, _ = precision_recall_fscore_support(y_true, predicted, labels=[1], zero_division=0)
        f2 = fbeta_score(y_true, predicted, beta=2, labels=[1], average="macro", zero_division=0)
        rows.append((float(threshold), float(recall[0]), float(f2)))
    meeting = [row for row in rows if row[1] >= target_recall]
    if meeting:
        threshold, recall, f2 = max(meeting, key=lambda row: row[0])
        rule = f"highest threshold with OOF class-1 recall >= {target_recall}"
    else:
        threshold, recall, f2 = max(rows, key=lambda row: row[2])
        rule = f"OOF recall target {target_recall} unreachable; F2-optimal threshold"
    return {"threshold": threshold, "oof_class_1_recall": recall, "oof_class_1_f2": f2, "rule": rule}


def classification_metrics(
    y_true, probabilities: np.ndarray, classes: np.ndarray, threshold: float | None = None
) -> dict[str, Any]:
    """Blueprint evaluation set: class-1 recall/F1/F2, macro F1, OvR ROC-AUC & PR-AUC, class-1 Brier.

    Works for the binary discordance task (classes {1, 3}) and the four-class task.
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    predicted = apply_threshold(probabilities, classes, threshold)
    labels = sorted(set(classes.tolist()) | set(y_true.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "class_metrics": {
            str(label): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": [int(label) for label in labels],
            "matrix": confusion_matrix(y_true, predicted, labels=labels).tolist(),
        },
    }
    one = labels.index(1) if 1 in labels else None
    metrics["class_1_precision"] = float(precision[one]) if one is not None else 0.0
    metrics["class_1_recall"] = float(recall[one]) if one is not None else 0.0
    metrics["class_1_f1"] = float(f1[one]) if one is not None else 0.0
    metrics["class_1_f2"] = float(
        fbeta_score(y_true, predicted, beta=2, labels=[1], average="macro", zero_division=0)
    )
    metrics["selection_objective"] = selection_objective(metrics["class_1_f1"], metrics["macro_f1"])
    if len(classes) == 2:
        # Binary task: OvR averages collapse to the positive-class (class 1) curves.
        positive = (y_true == 1).astype(int)
        p1 = probabilities[:, class_one_index(classes)] if 1 in classes else probabilities[:, 1]
        try:
            metrics["ovr_roc_auc"] = float(roc_auc_score(positive, p1))
        except ValueError:
            metrics["ovr_roc_auc"] = None
        try:
            metrics["ovr_pr_auc"] = float(average_precision_score(positive, p1))
        except ValueError:
            metrics["ovr_pr_auc"] = None
    else:
        expected = label_binarize(y_true, classes=classes)
        try:
            metrics["ovr_roc_auc"] = float(roc_auc_score(expected, probabilities, average="macro", multi_class="ovr"))
        except ValueError:
            metrics["ovr_roc_auc"] = None
        try:
            metrics["ovr_pr_auc"] = float(average_precision_score(expected, probabilities, average="macro"))
        except ValueError:
            metrics["ovr_pr_auc"] = None
    metrics["decision_threshold"] = threshold
    if 1 in classes:
        index = class_one_index(classes)
        metrics["class_1_brier"] = float(brier_score_loss((y_true == 1).astype(int), probabilities[:, index]))
        try:
            metrics["class_1_pr_auc"] = float(average_precision_score((y_true == 1).astype(int), probabilities[:, index]))
        except ValueError:
            metrics["class_1_pr_auc"] = None
    return metrics
