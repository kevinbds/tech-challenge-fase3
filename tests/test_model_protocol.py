"""Adversarial checks of the temporal, weighting and artifact boundaries."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score, recall_score

from alfabetizacao_ml.evaluation.metrics import (
    goal_attained,
    regression_metrics,
    select_f2_threshold,
    weighted_calibration_curve,
)
from alfabetizacao_ml.modeling import experiments
from alfabetizacao_ml.modeling.estimators import (
    calibrate_frozen,
    fit_weighted,
    inference_only_copy,
    make_classifier,
    risk_probability,
)
from alfabetizacao_ml.preprocessing.features import select_features, validate_student_labels
from alfabetizacao_ml.preprocessing.splits import grouped_folds, municipality_split


def cohorts():
    rng = np.random.default_rng(9)
    students, municipal = [], []
    for year in (2024, 2025, 2026):
        for i in range(45):
            for network in ("municipal", "estadual"):
                rate = 35 + i + (year - 2024) * 2
                row = dict(ano=year, ano_contexto=year-1, id_municipio=str(1000000+i),
                    sigla_uf="SP" if i % 2 else "BA", regiao="Sudeste" if i % 2 else "Nordeste",
                    rede=network, taxa_alfabetizacao_lag1=float(rate), media_portugues_lag1=400.+i,
                    media_inse_2021=4.+i/30, log_populacao_2022=9.+i/50,
                    densidade_2022=10.+i, taxa_alfabetizacao=float(rate+3),
                    meta_limite=60., meta_operador=">=", serie=2)
                if network == "municipal":
                    municipal.append(row.copy())
                for j in range(12):
                    label = int(j < 3 or (j >= 6 and rng.random() < rate/100))
                    students.append({**row, "alfabetizado": label, "risco": 1-label,
                                     "peso_aluno": float(1+j%4), "id_aluno": f"private-{i}-{j}"})
    return pd.DataFrame(students), pd.DataFrame(municipal)


def seal_snapshot(marts):
    coverage = marts.parent / "coverage.json"
    coverage.write_text('{"source": "synthetic test fixture"}', encoding="utf-8")
    digest = hashlib.sha256(json.dumps(json.loads(coverage.read_text()), ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"status": "passed", "coverage_hash": digest, "outputs": {
        "marts/" + p.name: {"sha256": experiments._digest(p), "bytes": p.stat().st_size}
        for p in marts.glob("*.parquet")}}
    (marts.parent / "preparation-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_preparation_receipt_rejects_partial_and_tampered_snapshot(tmp_path):
    marts = tmp_path / "marts"
    marts.mkdir()
    path = marts / "fixture.parquet"
    path.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        experiments.verify_preparation(marts)
    seal_snapshot(marts)
    experiments.verify_preparation(marts)
    marker = tmp_path / "preparation.in-progress"
    marker.touch()
    with pytest.raises(ValueError):
        experiments.verify_preparation(marts)
    marker.unlink()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        experiments.verify_preparation(marts)


def test_feature_boundary_and_temporal_provenance():
    students, _ = cohorts()
    selected = select_features(students)
    assert not {"id_aluno", "alfabetizado", "taxa_alfabetizacao", "risco"} & set(selected)
    for field in ("ano", "ano_contexto"):
        with pytest.raises(ValueError):
            select_features(students.drop(columns=field))
    students.loc[0, "ano_contexto"] = 2022
    with pytest.raises(ValueError):
        select_features(students)


def test_group_boundaries_order_independence_and_labels():
    students, _ = cohorts()
    split = municipality_split(students)
    assert split == municipality_split(students.sample(frac=1, random_state=11))
    assert not set(split.development) & set(split.reserve)
    dev = students.loc[split.mask(students)]
    y, _ = validate_student_labels(dev)
    for train, valid in grouped_folds(dev, y, classification=True):
        assert not set(dev.iloc[train].id_municipio) & set(dev.iloc[valid].id_municipio)
    for field, value in (("peso_aluno", 0), ("peso_aluno", np.inf), ("alfabetizado", 2)):
        bad = students.copy()
        bad.loc[0, field] = value
        with pytest.raises(ValueError):
            validate_student_labels(bad)


def test_weighted_bins_exact_f2_and_domain_fail_closed():
    bins = weighted_calibration_curve([0, 1], [.21, .29], [1, 9])
    assert bins[0]["observed"] == pytest.approx(.9)
    assert bins[0]["predicted"] == pytest.approx(.282)
    for p in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            weighted_calibration_curve([0], [p], [1])
    rng = np.random.default_rng(6)
    y, p, w = rng.integers(0, 2, 40), rng.choice([.1, .3, .7, .9], 40), rng.uniform(.1, 9, 40)
    exact = select_f2_threshold(y, p, w)
    candidates = [0, *np.unique(p)]
    brute = max(candidates, key=lambda t: (fbeta_score(y, p >= t, beta=2, sample_weight=w),
                                          recall_score(y, p >= t, sample_weight=w), -t))
    assert exact["threshold"] == brute
    assert regression_metrics([0, 100], [-4, 107])["mae"] == 0
    assert goal_attained(60., 60., ">") is False
    assert goal_attained(60., 60., ">=") is True
    assert goal_attained(60., None, None) is None


def test_calibration_freezes_base_and_positive_label_is_explicit():
    students, _ = cohorts()
    X = select_features(students)
    y, w = validate_student_labels(students)
    model = fit_weighted(make_classifier("logistic"), X, y, w)
    before = joblib.hash(model)
    calibrated = calibrate_frozen(model, X, y, w)
    assert before == joblib.hash(model)
    assert len(risk_probability(calibrated, X)) == len(X)
    class Reversed:
        classes_ = np.array([1, 0])
        def predict_proba(self, X):
            return np.tile([.8, .2], (len(X), 1))
    assert np.all(risk_probability(Reversed(), X) == .8)


def test_weighted_rf_export_removes_training_weights_without_prediction_change():
    students, _ = cohorts()
    X = select_features(students)
    y, w = validate_student_labels(students)
    base = fit_weighted(make_classifier("random_forest", {"n_estimators": 5, "max_depth": 3}), X, y, w)
    model = calibrate_frozen(base, X, y, w)
    exported = inference_only_copy(model)
    assert base.named_steps["model"]._sample_weight is not None
    assert exported.estimator.estimator.named_steps["model"]._sample_weight is None
    assert base.named_steps["model"].n_jobs == 1
    np.testing.assert_array_equal(risk_probability(model, X), risk_probability(exported, X))


def test_full_smoke_lock_reload_and_future_label_attack(tmp_path, monkeypatch):
    students, municipal = cohorts()
    marts = tmp_path / "marts"
    marts.mkdir()
    for year in (2024, 2025):
        students.loc[students.ano.eq(year)].to_parquet(marts / f"alunos_{year}.parquet", index=False)
    municipal.loc[municipal.ano.le(2025)].to_parquet(marts / "municipios.parquet", index=False)
    municipal.loc[municipal.ano.eq(2026)].to_parquet(marts / "municipios_2026.parquet", index=False)
    seal_snapshot(marts)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"seed": 42, "cv_folds": 5}), encoding="utf-8")
    original_load = experiments._load_year
    output = tmp_path / "first"
    def guarded_load(directory, stem, year):
        if year == 2025:
            assert (output / "protocol_lock.json").exists(), "Future labels opened before lock"
        return original_load(directory, stem, year)
    monkeypatch.setattr(experiments, "_load_year", guarded_load)
    first = experiments.run_experiments(marts, output, protocol, smoke=True)
    assert first["status"] == "smoke_completed"
    assert len(first["municipal"]["forecast_rows"]) == 45
    assert json.loads((output / "run_receipt.json").read_text())["evaluated_artifact_reload_verified"]
    first_lock = json.loads((output / "protocol_lock.json").read_text())
    first_model = joblib.load(output / "models/modelo_avaliado_2025_classificacao.joblib")
    assert set(first_model) == {"estimator", "metadata"}
    future_model = joblib.load(output / "models/modelo_projecao_2026_classificacao.joblib")
    assert first_model["metadata"]["role"] != future_model["metadata"]["role"]
    assert future_model["metadata"]["independent_future_test"] is None
    future = pd.read_parquet(marts / "alunos_2025.parquet")
    future["alfabetizado"] = np.random.default_rng(3).permutation(future.alfabetizado)
    future["risco"] = 1 - future.alfabetizado
    future.to_parquet(marts / "alunos_2025.parquet", index=False)
    seal_snapshot(marts)
    output = tmp_path / "attacked"
    second = experiments.run_experiments(marts, output, protocol, smoke=True)
    second_lock = json.loads((output / "protocol_lock.json").read_text())
    assert first_lock["classifier"] == second_lock["classifier"]
    assert first_lock["regressor"] == second_lock["regressor"]
    assert first["classification"]["prediction_fingerprint"] == second["classification"]["prediction_fingerprint"]
    assert first["classification"]["test"]["average_precision"] != second["classification"]["test"]["average_precision"]
    with pytest.raises(FileExistsError):
        experiments.run_experiments(marts, output, protocol, smoke=True)
