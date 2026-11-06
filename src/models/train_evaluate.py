"""Stage 2: leakage-guarded, cost-sensitive, Optuna-tuned tree ensembles."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.data.preprocess_impute import StateKNNImputer
from src.models.metrics import choose_threshold, classification_metrics, selection_objective
from src.target.ols_priority_classifier import assert_no_leakage


class EncodedClassifier(BaseEstimator, ClassifierMixin):
    """Fit any classifier on 0..K-1 labels while exposing the original labels as ``classes_``.

    XGBoost requires contiguous zero-based labels; wrapping every candidate keeps the interface uniform.
    For a two-class problem containing class 1, Priority (1) is encoded as the *positive* class and
    boosters are switched to a binary objective, so binary SHAP values / margins refer to Priority.
    The fitted tree model is available as ``estimator_`` for TreeSHAP.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        labels = np.unique(np.asarray(y))
        if len(labels) == 2 and 1 in labels:
            order = np.array([labels[labels != 1][0], 1])
        else:
            order = labels
        # classes_ stays sorted (sklearn calibration requires it); codes_ maps label -> encoded column.
        self.classes_ = labels
        lookup = {label: code for code, label in enumerate(order)}
        self.codes_ = lookup
        encoded = np.array([lookup[label] for label in np.asarray(y)])
        self.estimator_ = clone(self.estimator)
        if len(order) == 2:
            params = self.estimator_.get_params()
            objective = str(params.get("objective", ""))
            if objective.startswith("multi"):
                if type(self.estimator_).__name__ == "XGBClassifier":
                    self.estimator_.set_params(objective="binary:logistic", eval_metric="logloss")
                else:
                    self.estimator_.set_params(objective="binary")
        self.estimator_.fit(X, encoded, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        probabilities = self.estimator_.predict_proba(X)
        return probabilities[:, [self.codes_[label] for label in self.classes_]]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# name -> (factory(params, random_state), search space(trial), TreeSHAP-compatible for multiclass)
def _rf(params: dict, seed: int):
    return RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1, **params)


def _rf_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
    }


def _gb(params: dict, seed: int):
    return GradientBoostingClassifier(random_state=seed, **params)


def _gb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _xgb(params: dict, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=2, verbosity=0, **params
    )


def _xgb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _lgbm(params: dict, seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass", class_weight="balanced", subsample_freq=1, random_state=seed, verbosity=-1, n_jobs=2, **params
    )


def _lgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 25),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


DEFAULT_PARAMS = {
    "random_forest": {"n_estimators": 350, "max_depth": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    "gradient_boosting": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.85},
    "xgboost": {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    "lightgbm": {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 10},
}

CANDIDATES: dict[str, tuple[Callable, Callable, bool]] = {
    "random_forest": (_rf, _rf_space, True),
    # sklearn's multiclass GradientBoostingClassifier is not supported by shap.TreeExplainer, so it is
    # benchmarked but cannot be selected as the explained production model (ADR-03 requires TreeSHAP).
    "gradient_boosting": (_gb, _gb_space, False),
    "xgboost": (_xgb, _xgb_space, True),
    "lightgbm": (_lgbm, _lgbm_space, True),
}


def available_candidates() -> dict[str, tuple[Callable, Callable, bool]]:
    available = {}
    for name, spec in CANDIDATES.items():
        try:
            spec[0](DEFAULT_PARAMS[name], 0)
        except Exception as error:  # missing package or native library (e.g. libomp)
            warnings.warn(f"Skipping {name}: {error}")
            continue
        available[name] = spec
    return available


def class_weights(labels: pd.Series | np.ndarray, priority_multiplier: float) -> np.ndarray:
    labels = np.asarray(labels)
    values = compute_sample_weight(class_weight="balanced", y=labels)
    return values * np.where(labels == 1, priority_multiplier, 1.0)


def make_pipeline(classifier: object, feature_columns: list[str], zones: Mapping | None, n_neighbors: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", StateKNNImputer(feature_columns=feature_columns, zones=zones, n_neighbors=n_neighbors)),
            ("classifier", EncodedClassifier(classifier)),
        ]
    )


def _prepare_folds(X, y, feature_columns, zones, n_neighbors, folds, seed):
    """Impute each CV fold once (fitted on the fold's training part only) and reuse across trials."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prepared = []
    for train_idx, valid_idx in splitter.split(X, y):
        imputer = StateKNNImputer(feature_columns=feature_columns, zones=zones, n_neighbors=n_neighbors)
        imputer.fit(X.iloc[train_idx])
        prepared.append(
            (
                imputer.transform(X.iloc[train_idx]),
                y.iloc[train_idx].to_numpy(),
                imputer.transform(X.iloc[valid_idx]),
                y.iloc[valid_idx].to_numpy(),
                valid_idx,
            )
        )
    return prepared


def _cross_validate(factory, params, folds_data, multiplier, seed, class_1_weight, classes):
    oof = np.zeros((sum(len(fold[3]) for fold in folds_data), len(classes)))
    oof_index = np.concatenate([fold[4] for fold in folds_data])
    scores = []
    position = 0
    for train_x, train_y, valid_x, valid_y, _ in folds_data:
        model = EncodedClassifier(factory(params, seed)).fit(train_x, train_y, sample_weight=class_weights(train_y, multiplier))
        probabilities = model.predict_proba(valid_x)
        predicted = model.classes_[np.argmax(probabilities, axis=1)]
        class_1 = f1_score(valid_y, predicted, labels=[1], average="macro", zero_division=0)
        macro = f1_score(valid_y, predicted, average="macro", zero_division=0)
        scores.append(selection_objective(class_1, macro, class_1_weight))
        aligned = np.zeros((len(valid_y), len(classes)))
        for column, label in enumerate(model.classes_):
            aligned[:, list(classes).index(label)] = probabilities[:, column]
        oof[position : position + len(valid_y)] = aligned
        position += len(valid_y)
    order = np.argsort(oof_index)
    return float(np.mean(scores)), oof[order]


def tune_candidate(name, spec, folds_data, model_config, seed, classes) -> dict[str, Any]:
    factory, space, _ = spec
    multiplier = float(model_config["priority_class_weight_multiplier"])
    weight = float(model_config.get("class_1_objective_weight", 0.60))
    trials = int(model_config.get("optuna_trials", 0))
    started = time.time()
    if trials > 0:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(
            lambda trial: _cross_validate(factory, space(trial), folds_data, multiplier, seed, weight, classes)[0],
            n_trials=trials,
            timeout=model_config.get("optuna_timeout_seconds"),
        )
        params, completed = study.best_params, len(study.trials)
    else:
        params, completed = DEFAULT_PARAMS[name], 0
    cv_score, oof = _cross_validate(factory, params, folds_data, multiplier, seed, weight, classes)
    return {
        "params": params,
        "trials": completed,
        "cv_selection_score": cv_score,
        "oof_probabilities": oof,
        "tuning_seconds": round(time.time() - started, 1),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_config: dict[str, Any],
    random_state: int,
    model_path: Path,
    metrics_path: Path,
    calibrated_path: Path | None = None,
    zones: Mapping | None = None,
    n_neighbors: int = 5,
    progress: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Tune candidate tree ensembles, select by CV objective, evaluate on the untouched holdout."""
    assert_no_leakage(feature_columns)
    if "state" not in frame or "priority_class" not in frame:
        raise ValueError("Training frame requires state and priority_class columns.")
    task = model_config.get("task", "four_class")
    if task == "discordance_binary":
        # Priority (1) vs Best Performer (3): both are high-access, so the access anchor's proxies
        # cannot separate them and the model must learn quality/structural determinants.
        frame = frame.loc[frame["priority_class"].isin([1, 3])]
    elif task != "four_class":
        raise ValueError(f"Unknown modelling task: {task}")
    y = frame["priority_class"].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("At least two examples per priority class are required for a stratified holdout.")
    X = frame[["state", *feature_columns]].copy()
    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=float(model_config["test_size"]), stratify=y, random_state=random_state
    )
    folds = min(int(model_config["cv_folds"]), int(train_y.value_counts().min()))
    if folds < 2:
        raise ValueError("Training split is too small for stratified cross-validation.")
    multiplier = float(model_config["priority_class_weight_multiplier"])
    classes = np.sort(train_y.unique())
    folds_data = _prepare_folds(train_x, train_y, feature_columns, zones, n_neighbors, folds, random_state)

    benchmark: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    oof_store: dict[str, np.ndarray] = {}
    candidates = available_candidates()
    for name, spec in candidates.items():
        if progress:
            print(f"Tuning {name} ({model_config.get('optuna_trials', 0)} Optuna trials x {folds} folds)...", flush=True)
        tuned = tune_candidate(name, spec, folds_data, model_config, random_state, classes)
        pipeline = make_pipeline(spec[0](tuned["params"], random_state), feature_columns, zones, n_neighbors)
        pipeline.fit(train_x, train_y, classifier__sample_weight=class_weights(train_y, multiplier))
        fitted[name] = pipeline
        oof_store[name] = tuned["oof_probabilities"]
        # OOF probabilities are returned in the positional order of train_y.
        oof_metrics = classification_metrics(train_y.to_numpy(), tuned["oof_probabilities"], classes)
        benchmark[name] = {
            "best_params": tuned["params"],
            "optuna_trials": tuned["trials"],
            "tuning_seconds": tuned["tuning_seconds"],
            "cv_selection_score": tuned["cv_selection_score"],
            # shap.TreeExplainer supports sklearn GradientBoosting only for binary problems.
            "treeshap_compatible": spec[2] or len(classes) == 2,
            "out_of_fold": oof_metrics,
            "holdout": classification_metrics(test_y, pipeline.predict_proba(test_x), pipeline.classes_),
        }
        if progress:
            print(
                f"  {name}: CV objective {tuned['cv_selection_score']:.3f} | holdout class-1 recall "
                f"{benchmark[name]['holdout']['class_1_recall']:.3f}",
                flush=True,
            )

    eligible = [name for name in benchmark if benchmark[name]["treeshap_compatible"]]
    if not eligible:
        raise RuntimeError("No TreeSHAP-compatible candidate model could be trained.")
    selected = max(eligible, key=lambda name: benchmark[name]["cv_selection_score"])
    overall_best = max(benchmark, key=lambda name: benchmark[name]["cv_selection_score"])
    final_pipeline = fitted[selected]

    threshold_info: dict[str, Any] = {}
    threshold = None
    if model_config.get("threshold_tuning", False):
        threshold_info = choose_threshold(
            train_y.to_numpy(), oof_store[selected], classes, float(model_config.get("target_class_1_recall", 0.82))
        )
        threshold = threshold_info["threshold"]
        benchmark[selected]["holdout_at_threshold"] = classification_metrics(
            test_y, final_pipeline.predict_proba(test_x), final_pipeline.classes_, threshold
        )
    # Persist the decision rule with the model so the app and SHAP profiler apply the same cut-off.
    final_pipeline.decision_threshold_ = threshold
    final_pipeline.task_ = task

    calibration: dict[str, Any] = {}
    if calibrated_path is not None:
        calibrated = CalibratedClassifierCV(
            clone(final_pipeline),
            method=model_config.get("calibration_method", "isotonic"),
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated.fit(train_x, train_y, classifier__sample_weight=class_weights(train_y, multiplier))
        calibrated_metrics = classification_metrics(test_y, calibrated.predict_proba(test_x), calibrated.classes_)
        calibration = {
            "method": model_config.get("calibration_method", "isotonic"),
            "holdout_class_1_brier_raw": benchmark[selected]["holdout"].get("class_1_brier"),
            "holdout_class_1_brier_calibrated": calibrated_metrics.get("class_1_brier"),
            "holdout_calibrated": calibrated_metrics,
            "note": "Priority risk scores use the calibrated class-1 probability; class assignments and "
            "TreeSHAP explanations use the uncalibrated tree model.",
        }
        calibrated_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, calibrated_path)

    report = {
        "selected_model": selected,
        "task": task,
        "decision_threshold": threshold,
        "threshold_selection": threshold_info,
        "selection_rule": "highest mean 5-fold CV objective (0.60 x class-1 F1 + 0.40 x macro F1) among "
        "TreeSHAP-compatible models; holdout is reported, never used for selection",
        "best_cv_model_overall": overall_best,
        "candidate_models": benchmark,
        "feature_columns": feature_columns,
        "train_size": int(len(train_y)),
        "holdout_size": int(len(test_y)),
        "class_distribution": {str(key): int(value) for key, value in class_counts.sort_index().items()},
        "calibration": calibration,
        "class_1_recall_target": float(model_config.get("target_class_1_recall", 0.82)),
        "class_1_recall_target_met": benchmark[selected].get("holdout_at_threshold", benchmark[selected]["holdout"])[
            "class_1_recall"
        ]
        >= float(model_config.get("target_class_1_recall", 0.82)),
        "holdout_index": [int(index) for index in test_x.index],
        "methodology_note": "Labels are outcome-derived research labels. Metrics are research diagnostics, "
        "not deployment evidence.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    metrics_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return final_pipeline, report
