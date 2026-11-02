"""Bounded, auditable candidate search using only development municipalities."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from threadpoolctl import threadpool_limits

from .estimators import fit_weighted, make_classifier, make_regressor, risk_probability


def candidates(*, classification, maximum=12, seed=42, smoke=False):
    linear = "logistic" if classification else "ridge"
    if smoke:
        return [(linear, {"C": 1} if classification else {"alpha": 1}),
                ("random_forest", {"n_estimators": 12, "max_depth": 4, "min_samples_leaf": 5}),
                ("hist_gradient_boosting", {"max_iter": 12, "max_leaf_nodes": 7, "l2_regularization": 1})]
    result = [(linear, p) for p in ParameterGrid({"C" if classification else "alpha": [0.01, 0.1, 1, 10] if classification else [0.1, 1, 10, 100]})]
    grids = {
        "random_forest": {"n_estimators": [200], "max_depth": [8, 16, None],
                          "min_samples_leaf": [5, 20, 50],
                          "max_features": ["sqrt", 0.7] if classification else [0.7, 1.0]},
        "hist_gradient_boosting": {"max_iter": [100, 250, 500], "max_leaf_nodes": [15, 31],
                                   "l2_regularization": [0, 1, 10], "learning_rate": [0.03, 0.1]},
    }
    for name, grid in grids.items():
        result.extend((name, p) for p in ParameterSampler(grid, n_iter=min(maximum, len(list(ParameterGrid(grid)))), random_state=seed))
    return result


def search_candidates(X, y, weights, folds, *, classification, seed=42, threads=8,
                      maximum=12, smoke=False, checkpoint: Path | None = None) -> dict:
    rows = []
    factory = make_classifier if classification else make_regressor
    for name, params in candidates(classification=classification, maximum=maximum, seed=seed, smoke=smoke):
        started = time.perf_counter()
        scores, train_scores = [], []
        for train, valid in folds:
            estimator = factory(name, params, seed=seed, threads=threads)
            with threadpool_limits(limits=threads):
                fit_weighted(estimator, X.iloc[train], y[train], weights[train] if weights is not None else None)
                if classification:
                    pred = risk_probability(estimator, X.iloc[valid])
                    scores.append(float(average_precision_score(y[valid], pred, sample_weight=weights[valid])))
                    train_scores.append(float(average_precision_score(y[train], risk_probability(estimator, X.iloc[train]), sample_weight=weights[train])))
                else:
                    pred = np.clip(estimator.predict(X.iloc[valid]), 0, 100)
                    scores.append(float(mean_absolute_error(y[valid], pred)))
                    train_scores.append(float(mean_absolute_error(y[train], np.clip(estimator.predict(X.iloc[train]), 0, 100))))
        rows.append({"model": name, "params": params, "mean_score": float(np.mean(scores)),
                     "std_score": float(np.std(scores)), "fold_scores": scores,
                     "mean_training_score": float(np.mean(train_scores)),
                     "elapsed_seconds": time.perf_counter() - started})
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"completed_candidates": rows, "future_test_opened": False}, indent=2), encoding="utf-8")
    order = {"logistic": 0, "ridge": 0, "hist_gradient_boosting": 1, "random_forest": 2}
    best = min(rows, key=lambda r: ((-1 if classification else 1) * r["mean_score"], order[r["model"]], json.dumps(r["params"], sort_keys=True)))
    return {"metric": "average_precision" if classification else "mae", "cv_aggregation": "arithmetic mean across municipality-disjoint folds",
            "best": best, "cv_results": rows}
