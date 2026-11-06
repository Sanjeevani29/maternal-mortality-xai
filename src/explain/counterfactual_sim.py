from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.metrics import apply_threshold


def priority_probability(model, row: pd.DataFrame) -> float:
    classes = np.asarray(model.classes_)
    class_index = int(np.flatnonzero(classes == 1)[0])
    return float(model.predict_proba(row)[:, class_index][0])


def simulate(model, original_row: pd.DataFrame, edits: dict[str, float], risk_model=None) -> dict[str, object]:
    """Return directional model sensitivity for user-provided feature changes.

    ``model`` (the explained tree pipeline) determines the operational class; ``risk_model`` (the
    calibrated pipeline, if supplied) determines the displayed class-1 probability. This is a model
    sensitivity exercise and does not estimate causal policy impact.
    """
    candidate = original_row.copy()
    for feature, value in edits.items():
        if feature not in candidate:
            raise KeyError(f"Unknown simulator feature: {feature}")
        candidate.loc[candidate.index[0], feature] = value
    scorer = risk_model if risk_model is not None else model
    baseline = priority_probability(scorer, original_row)
    simulated = priority_probability(scorer, candidate)
    classes = np.asarray(model.classes_)
    threshold = getattr(model, "decision_threshold_", None)

    def operational(row: pd.DataFrame) -> int:
        return int(apply_threshold(model.predict_proba(row), classes, threshold)[0])

    return {
        "baseline_probability": baseline,
        "simulated_probability": simulated,
        "delta_percentage_points": (simulated - baseline) * 100,
        "baseline_class": operational(original_row),
        "simulated_class": operational(candidate),
        "candidate_row": candidate,
    }
