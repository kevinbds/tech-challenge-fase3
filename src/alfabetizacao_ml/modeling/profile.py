"""Measure a real development fit without ever opening the temporal test."""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from alfabetizacao_ml.modeling.estimators import fit_weighted, make_classifier, risk_probability
from alfabetizacao_ml.preprocessing.features import select_features, validate_student_labels
from alfabetizacao_ml.preprocessing.splits import municipality_split


def profile_development(mart_path: Path, output: Path, model: str = "logistic") -> dict:
    started = time.perf_counter()
    frame = pd.read_parquet(mart_path)
    if not frame["ano"].eq(2024).all():
        raise ValueError("Runtime profiling is restricted to the declared development year")
    split = municipality_split(frame)
    frame = frame.loc[split.mask(frame)].reset_index(drop=True)
    features = select_features(frame)
    y, w = validate_student_labels(frame)
    params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20} if model == "random_forest" else {}
    estimator = make_classifier(model, params)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=8):
        fit_weighted(estimator, features, y, w)
        fit_seconds = time.perf_counter() - fit_start
        p = risk_probability(estimator, features)
    result = {"model": model, "params": params, "n_development": len(frame),
              "fit_seconds": fit_seconds, "elapsed_seconds": time.perf_counter() - started,
              "training_ap_only_not_validation": float(average_precision_score(y, p, sample_weight=w)),
              "test_year_opened": False, "feature_memory_bytes": int(features.memory_usage(deep=True).sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
