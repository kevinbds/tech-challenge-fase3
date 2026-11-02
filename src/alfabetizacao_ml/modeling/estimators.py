"""Estimator factories and calibrated inference without training-row persistence."""

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from alfabetizacao_ml.preprocessing.features import make_preprocessor


def inference_only_copy(estimator):
    """Drop sklearn RF's retained per-training-row weights from exported copies."""
    copied = copy.deepcopy(estimator)
    visited = set()

    def clean(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, (RandomForestClassifier, RandomForestRegressor)):
            if hasattr(node, "_sample_weight"):
                node._sample_weight = None
        if isinstance(node, dict):
            children = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        elif hasattr(node, "__dict__") and type(node).__module__.startswith("sklearn."):
            children = vars(node).values()
        else:
            return
        for child in children:
            clean(child)

    clean(copied)
    return copied


class ContextRiskClassifier(ClassifierMixin, BaseEstimator):
    """Known last-year risk, with a prevalence learned only from the fit population."""

    def __init__(self, use_context=True):
        self.use_context = use_context

    def fit(self, X, y, sample_weight=None):
        self.prevalence_ = float(np.average(y, weights=sample_weight))
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        values = np.full(len(X), self.prevalence_, dtype=float)
        if self.use_context:
            rate = X["taxa_alfabetizacao_lag1"].to_numpy(dtype=float)
            valid = np.isfinite(rate)
            values[valid] = 1 - rate[valid] / 100
        return np.column_stack([1 - values, values])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        values = X["taxa_alfabetizacao_lag1"].to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, self.mean_)


def make_classifier(name: str, params: dict | None = None, *, seed: int = 42, threads: int = 8):
    params = params or {}
    if name == "prevalence":
        return ContextRiskClassifier(use_context=False)
    if name == "contextual":
        return ContextRiskClassifier(use_context=True)
    if name == "logistic":
        estimator = LogisticRegression(C=1, max_iter=2000, class_weight=None, random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=200, class_weight=None, oob_score=False,
                                           random_state=seed, n_jobs=threads)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    estimator.set_params(**params)
    return Pipeline([("preprocess", make_preprocessor(scale=name == "logistic")), ("model", estimator)])


def make_regressor(name: str, params: dict | None = None, *, seed: int = 42, threads: int = 8):
    params = params or {}
    if name == "persistence":
        return PersistenceRegressor()
    if name == "ridge":
        estimator = Ridge(alpha=1)
    elif name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=200, oob_score=False, random_state=seed, n_jobs=threads)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(early_stopping=False, random_state=seed)
    else:
        raise ValueError(f"Unknown regressor: {name}")
    estimator.set_params(**params)
    return Pipeline([("preprocess", make_preprocessor(scale=name == "ridge")), ("model", estimator)])


def fit_weighted(estimator, X, y, weights=None):
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, **({"model__sample_weight": weights} if weights is not None else {}))
    return estimator.fit(X, y, sample_weight=weights)


def risk_probability(estimator, X) -> np.ndarray:
    index = np.flatnonzero(np.asarray(estimator.classes_) == 1)
    if len(index) != 1:
        raise ValueError("Estimator must expose the positive risk class 1")
    p = np.asarray(estimator.predict_proba(X)[:, index[0]], dtype=float)
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Invalid predicted risk probabilities")
    return p


def calibrate_frozen(estimator, X, y, weights):
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration reserve must contain both classes")
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid", ensemble=False)
    return calibrated.fit(X, y, sample_weight=weights)
