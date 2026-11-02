"""Exact decision reproducibility at the F2 boundary, including exported forests."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from alfabetizacao_ml.evaluation.metrics import classification_metrics, select_f2_threshold
from alfabetizacao_ml.modeling import experiments
from alfabetizacao_ml.modeling.estimators import (
    calibrate_frozen,
    fit_weighted,
    inference_only_copy,
    make_classifier,
    make_regressor,
    risk_probability,
)
from alfabetizacao_ml.preprocessing.features import select_features


def synthetic_contexts(seed):
    rng = np.random.default_rng(seed)
    context = np.repeat(np.arange(100), 60)
    z = context / 99
    frame = pd.DataFrame({"ano": 2024, "ano_contexto": 2023,
        "taxa_alfabetizacao_lag1": 100*(1-z), "media_portugues_lag1": 400-50*z,
        "media_inse_2021": 3+z, "log_populacao_2022": 9+z, "densidade_2022": 20+z,
        "sigla_uf": "SP", "regiao": "Sudeste", "rede": "municipal"})
    frame.loc[context % 7 == 0, ["taxa_alfabetizacao_lag1", "media_portugues_lag1"]] = np.nan
    frame.loc[context % 9 == 0, ["sigla_uf", "regiao"]] = None
    X = select_features(frame)
    X["sigla_uf"] = X["sigla_uf"].astype("category").cat.add_categories(["UNUSED"])
    return X, (rng.random(len(z)) < z).astype(int), rng.uniform(.1, 10, len(z)), context


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_serial_forest_dedup_reload_preserves_exact_probabilities_and_decisions(seed, tmp_path):
    X, y, w, context = synthetic_contexts(seed)
    development = context % 5 != 0
    reserve = ~development
    base = make_classifier("random_forest", {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 20}, threads=8)
    assert base.named_steps["model"].n_jobs == 8
    fit_weighted(base, X.loc[development], y[development], w[development])
    assert base.named_steps["model"].n_jobs == 1
    model = calibrate_frozen(base, X.loc[reserve], y[reserve], w[reserve])
    full = model.predict_proba(X.loc[reserve])[:, 1]
    optimized = risk_probability(model, X.loc[reserve])
    np.testing.assert_array_equal(full, optimized)
    threshold = select_f2_threshold(y[reserve], full, w[reserve])["threshold"]
    assert np.any(full == threshold)
    expected = classification_metrics(y[reserve], full, w[reserve], threshold)
    path = tmp_path / "forest.joblib"
    joblib.dump(inference_only_copy(model), path)
    restored = joblib.load(path)
    for _ in range(10):
        for candidate in (model, restored):
            p = risk_probability(candidate, X.loc[reserve])
            np.testing.assert_array_equal(p, full)
            np.testing.assert_array_equal(p >= threshold, full >= threshold)
            assert classification_metrics(y[reserve], p, w[reserve], threshold) == expected


def test_regression_forest_inference_is_serial_and_export_is_exact():
    X, _, _, _ = synthetic_contexts(4)
    y = np.linspace(0, 100, len(X))
    model = fit_weighted(make_regressor("random_forest", {"n_estimators": 20, "max_depth": 5}, threads=8), X, y)
    assert model.named_steps["model"].n_jobs == 1
    np.testing.assert_array_equal(model.predict(X), inference_only_copy(model).predict(X))


@pytest.mark.parametrize("invalid_policy", [0, 2, 8, True, 1.0, "1", None])
def test_protocol_rejects_parallel_inference_policy(tmp_path, monkeypatch, invalid_policy):
    monkeypatch.setattr(experiments, "verify_preparation", lambda _: "verified-fixture")
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"forest_inference_jobs": invalid_policy}), encoding="utf-8")
    with pytest.raises(ValueError, match="forest_inference_jobs=1"):
        experiments.run_experiments(tmp_path, tmp_path / "run", path)
