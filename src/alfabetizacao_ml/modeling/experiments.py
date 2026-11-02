"""End-to-end modeling with an explicit lock before the future test is read."""

import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from alfabetizacao_ml.evaluation.interpretation import (
    block_permutation_importance,
    cluster_municipalities,
    leakage_demonstration,
    shap_summary,
)
from alfabetizacao_ml.evaluation.metrics import (
    band_metrics,
    classification_metrics,
    goal_attained,
    regression_metrics,
    residual_band,
    select_f2_threshold,
)
from alfabetizacao_ml.evaluation.uncertainty import (
    paired_classification_bootstrap,
    paired_regression_bootstrap,
)
from alfabetizacao_ml.preprocessing.features import select_features, validate_student_labels
from alfabetizacao_ml.preprocessing.splits import grouped_folds, municipality_split

from .estimators import (
    calibrate_frozen,
    fit_weighted,
    inference_only_copy,
    make_classifier,
    make_regressor,
    risk_probability,
)
from .selection import search_candidates


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported public JSON value: {type(value).__name__}")


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default, allow_nan=False).encode()).hexdigest()


def _load_year(marts_dir, stem, year):
    if (marts_dir.parent / "preparation.in-progress").exists():
        raise ValueError("Snapshot preparation is still in progress")
    partition = marts_dir / f"{stem}_{year}.parquet"
    path = partition if partition.exists() else marts_dir / f"{stem}.parquet"
    result = pd.read_parquet(path, filters=[("ano", "==", year)])
    if result.empty or not result["ano"].eq(year).all():
        raise ValueError(f"Missing or invalid {stem} cohort for {year}")
    return result.reset_index(drop=True)


def verify_preparation(marts_dir):
    """Require a completed, hash-verified snapshot before reading any labels."""
    root = marts_dir.parent.resolve()
    receipt_path = root / "preparation-receipt.json"
    if (root / "preparation.in-progress").exists() or not receipt_path.is_file():
        raise ValueError("A completed preparation receipt is required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or not receipt.get("outputs"):
        raise ValueError("Preparation receipt did not pass")
    registered = set()
    for relative, entry in receipt["outputs"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Preparation receipt points outside the snapshot or to a missing file")
        if path.stat().st_size != entry["bytes"] or _digest(path) != entry["sha256"]:
            raise ValueError(f"Preparation artifact integrity mismatch: {relative}")
        registered.add(path)
    if any(path.resolve() not in registered for path in marts_dir.glob("*.parquet")):
        raise ValueError("Unregistered mart in prepared snapshot")
    coverage = root / "coverage.json"
    if not coverage.is_file():
        raise ValueError("Coverage report is missing")
    coverage_hash = hashlib.sha256(json.dumps(json.loads(coverage.read_text(encoding="utf-8")),
                                   ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if coverage_hash != receipt.get("coverage_hash"):
        raise ValueError("Coverage report integrity mismatch")
    if (root / "preparation.in-progress").exists():
        raise ValueError("Snapshot changed while preparation was being verified")
    return _digest(receipt_path)


def _input_manifest(marts_dir):
    candidates = [marts_dir.parent / "manifest.json", marts_dir.parent / "source_manifest.json"]
    snapshot = marts_dir.parent.name.removeprefix("local-")
    candidates.extend(marts_dir.parent.parent.parent.glob(f"manifests/{snapshot}*.json"))
    for candidate in candidates:
        if candidate.is_file():
            return _digest(candidate)
    # Fixtures and independently provided marts can be replayed by their own file manifest.
    return _hash_object({p.name: _digest(p) for p in sorted(marts_dir.glob("*.parquet"))})


def _source_fingerprint():
    package = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(package)): _digest(p)
              for name in ("preprocessing", "modeling", "evaluation")
              for p in sorted((package / name).glob("*.py"))}
    return _hash_object(source)


def _save_model(path, estimator, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": inference_only_copy(estimator), "metadata": metadata}, path, compress=3)
    return {"file": str(path.name), "sha256": _digest(path), "role": metadata["role"]}


def _subgroups(frame, y, probability, weights, threshold):
    result = []
    contexts = frame["taxa_alfabetizacao_lag1"].notna().map({True: "available", False: "missing"})
    dimensions = {"sigla_uf": frame["sigla_uf"], "regiao": frame["regiao"], "lag_context": contexts}
    for dimension, values in dimensions.items():
        for name in sorted(values.fillna("missing").astype(str).unique()):
            mask = values.fillna("missing").astype(str).eq(name).to_numpy()
            metrics = classification_metrics(y[mask], probability[mask], weights[mask], threshold)
            result.append({"dimension": dimension, "group": name,
                           **{key: metrics[key] for key in ("n", "weight", "average_precision", "roc_auc", "recall", "brier")}})
    return result


def _number(value):
    return None if value is None or pd.isna(value) else float(value)


def _municipal_rows(frame, predicted, quantiles=None, *, forecast=False):
    rows = []
    for offset, (_, row) in enumerate(frame.iterrows()):
        value = float(predicted[offset])
        limit = _number(row.get("meta_limite"))
        operator = row.get("meta_operador")
        operator = None if pd.isna(operator) else operator
        record = {"id_municipio": str(row["id_municipio"]), "sigla_uf": str(row["sigla_uf"]),
                  "ano": int(row["ano"]), "meta_limite": limit, "meta_operador": operator,
                  "goal_predicted": goal_attained(value, limit, operator)}
        if "nome_municipio" in frame:
            record["nome_municipio"] = str(row["nome_municipio"])
        if forecast:
            record.update({"forecast": value, "gap": value - limit if limit is not None else None})
        else:
            observed = float(row["taxa_alfabetizacao"])
            record.update({"observed": observed, "predicted": value,
                           "lower": float(np.clip(value + quantiles[0], 0, 100)),
                           "upper": float(np.clip(value + quantiles[1], 0, 100)),
                           "goal_observed": goal_attained(observed, limit, operator)})
        rows.append(record)
    return rows


def _municipal_cohort(frame):
    result = frame.loc[frame["rede"].eq("municipal")].reset_index(drop=True)
    if result.empty or result.duplicated(["ano", "id_municipio", "rede"]).any():
        raise ValueError("Municipal cohort requires unique municipality/year/network observations")
    y = pd.to_numeric(result["taxa_alfabetizacao"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(y).all() and ((y >= 0) & (y <= 100)).all()):
        raise ValueError("Municipal targets must be observed finite rates in [0,100]")
    return result, y


def run_experiments(marts_dir: Path, output_dir: Path, protocol_path: Path, *, smoke: bool = False) -> dict:
    """Train, freeze, evaluate, explain and create separately identified forecast models.

    All real rows are used. Smoke only reduces candidate trees/iterations and
    bootstrap repetitions; it does not silently subsample the population.
    """
    marts_dir, output_dir, protocol_path = map(Path, (marts_dir, output_dir, protocol_path))
    if (output_dir / "metrics.json").exists():
        raise FileExistsError("A completed run is immutable; choose a new output directory")
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    preparation_hash = verify_preparation(marts_dir)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = _hash_object(protocol)
    input_hash = _input_manifest(marts_dir)
    source_hash = _source_fingerprint()
    run_id = "run-" + _hash_object({"protocol": protocol_hash, "inputs": input_hash, "source": source_hash, "smoke": smoke})[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed, threads = int(protocol.get("seed", 42)), int(protocol.get("max_threads", 8))
    year, test_year = int(protocol.get("development_year", 2024)), int(protocol.get("test_year", 2025))
    fraction = float(protocol.get("reserve_municipality_fraction", 0.2))
    folds_count = int(protocol.get("cv_folds", 5))
    maximum = int(protocol.get("max_search_candidates_per_family", 12))
    if (year, test_year) != (2024, 2025):
        raise ValueError("This audited protocol requires development2024 and test2025")

    def progress(stage):
        _write(output_dir / "progress.json", {"run_id": run_id, "stage": stage,
               "elapsed_seconds": time.perf_counter() - started, "smoke": smoke})

    progress("load_development_2024")
    students = _load_year(marts_dir, "alunos", year)
    split = municipality_split(students, fraction, seed)
    dev = students.loc[split.mask(students)].reset_index(drop=True)
    reserve = students.loc[split.mask(students, reserve=True)].reset_index(drop=True)
    X, X_reserve = select_features(dev), select_features(reserve)
    y, weights = validate_student_labels(dev)
    y_reserve, w_reserve = validate_student_labels(reserve)
    folds = grouped_folds(dev, y, classification=True, n_splits=folds_count, seed=seed)
    progress("classification_search_development_only")
    selection = search_candidates(X, y, weights, folds, classification=True, seed=seed, threads=threads,
                                  maximum=maximum, smoke=smoke, checkpoint=output_dir / "classification_search.json")
    best = selection["best"]
    base = make_classifier(best["model"], best["params"], seed=seed, threads=threads)
    with threadpool_limits(limits=threads):
        fit_weighted(base, X, y, weights)
        base_before = joblib.hash(base)
        calibrated = calibrate_frozen(base, X_reserve, y_reserve, w_reserve)
        if joblib.hash(base) != base_before:
            raise AssertionError("Calibration changed the frozen estimator")
        threshold = select_f2_threshold(y_reserve, risk_probability(calibrated, X_reserve), w_reserve)
    baselines = {name: fit_weighted(make_classifier(name), X, y, weights)
                 for name in ("prevalence", "contextual")}

    progress("municipal_search_development_only")
    municipal, _ = _municipal_cohort(_load_year(marts_dir, "municipios", year))
    # Both tasks must reuse exactly the same municipality assignment.
    if not set(municipal["id_municipio"].astype(str)).issubset(set(split.development) | set(split.reserve)):
        raise ValueError("Municipal cohort includes groups absent from the frozen shared partition")
    municipal_split = split
    mdev = municipal.loc[municipal_split.mask(municipal)].reset_index(drop=True)
    mreserve = municipal.loc[municipal_split.mask(municipal, reserve=True)].reset_index(drop=True)
    Xm, Xmr = select_features(mdev), select_features(mreserve)
    ym, ymr = (part["taxa_alfabetizacao"].to_numpy(dtype=float) for part in (mdev, mreserve))
    mfolds = grouped_folds(mdev, ym, classification=False, n_splits=folds_count, seed=seed)
    mselection = search_candidates(Xm, ym, None, mfolds, classification=False, seed=seed, threads=threads,
                                   maximum=maximum, smoke=smoke, checkpoint=output_dir / "municipal_search.json")
    mbest = mselection["best"]
    municipal_model = make_regressor(mbest["model"], mbest["params"], seed=seed, threads=threads)
    with threadpool_limits(limits=threads):
        fit_weighted(municipal_model, Xm, ym)
        quantiles = residual_band(ymr, municipal_model.predict(Xmr))
    persistence = fit_weighted(make_regressor("persistence"), Xm, ym)

    lock = {"run_id": run_id, "protocol_hash": protocol_hash, "input_manifest_hash": input_hash,
            "preparation_receipt_hash": preparation_hash,
            "source_fingerprint": source_hash, "student_partition_hash": split.fingerprint,
            "municipal_partition_hash": municipal_split.fingerprint,
            "classifier": {"name": best["model"], "params": best["params"], "base_fingerprint": joblib.hash(base),
                           "calibrated_fingerprint": joblib.hash(calibrated), "threshold": threshold},
            "regressor": {"name": mbest["model"], "params": mbest["params"], "fingerprint": joblib.hash(municipal_model)},
            "residual_quantiles": list(quantiles), "future_test_opened": False}
    lock["lock_hash"] = _hash_object(lock)
    _write(output_dir / "protocol_lock.json", lock)
    metadata = {"run_id": run_id, "protocol_hash": protocol_hash, "input_manifest_hash": input_hash,
                "lock_hash": lock["lock_hash"], "training_year": year, "test_year": test_year}
    artifacts = {
        "classifier_evaluated": _save_model(output_dir / "models/modelo_avaliado_2025_classificacao.joblib", calibrated,
                          {**metadata, "role": "evaluated_2025_classifier", "threshold": threshold["threshold"]}),
        "regressor_evaluated": _save_model(output_dir / "models/modelo_avaliado_2025_municipal.joblib", municipal_model,
                         {**metadata, "role": "evaluated_2025_regressor", "residual_quantiles": list(quantiles)}),
    }

    # This is the first access to the future cohort, after the immutable lock.
    progress("evaluate_frozen_temporal_test_2025")
    test = _load_year(marts_dir, "alunos", test_year)
    Xt = select_features(test)
    yt, wt = validate_student_labels(test)
    with threadpool_limits(limits=threads):
        probability = risk_probability(calibrated, Xt)
        baseline_probability = {name: risk_probability(model, Xt) for name, model in baselines.items()}
    classification = {
        "selected_model": best["model"], "threshold": threshold["threshold"], "threshold_selection": threshold,
        "selection": selection, "population": {"development": len(dev), "calibration": len(reserve), "test": len(test),
                                               "development_municipalities": len(split.development), "reserve_municipalities": len(split.reserve)},
        "test": classification_metrics(yt, probability, wt, threshold["threshold"]),
        "test_at_0_5": classification_metrics(yt, probability, wt, 0.5),
        "baselines": {name: classification_metrics(yt, p, wt, threshold["threshold"]) for name, p in baseline_probability.items()},
        "subgroups": _subgroups(test, yt, probability, wt, threshold["threshold"]),
        "bootstrap": paired_classification_bootstrap(yt, probability, baseline_probability["contextual"], wt,
                       test["id_municipio"].to_numpy(), iterations=30 if smoke else 500, seed=seed),
        "prediction_fingerprint": hashlib.sha256(probability.tobytes()).hexdigest(),
    }
    mtest, ymt = _municipal_cohort(_load_year(marts_dir, "municipios", test_year))
    Xmt = select_features(mtest)
    with threadpool_limits(limits=threads):
        mraw = municipal_model.predict(Xmt)
        mprob = np.clip(mraw, 0, 100)
        pref = np.clip(persistence.predict(Xmt), 0, 100)
    evaluation_rows = _municipal_rows(mtest, mprob, quantiles)
    comparable = [row for row in evaluation_rows if row["goal_observed"] is not None]
    municipal_result = {"selected_model": mbest["model"], "selection": mselection,
        "population": {"development": len(mdev), "reserve": len(mreserve), "test": len(mtest)},
        "test": {**regression_metrics(ymt, mraw), **band_metrics(ymt, mprob, quantiles),
                 "goal_accuracy": float(np.mean([row["goal_observed"] == row["goal_predicted"] for row in comparable])) if comparable else None,
                 "goal_evaluation_n": len(comparable)},
        "baselines": {"persistence": regression_metrics(ymt, pref)},
        "residual_band": {"q05": quantiles[0], "q95": quantiles[1], "population": "independent 2024 reserve",
                          "scope": "exploratory historical band; no future coverage guarantee"},
        "bootstrap": paired_regression_bootstrap(ymt, mprob, pref, iterations=30 if smoke else 500, seed=seed),
        "evaluation_rows": evaluation_rows, "forecast_rows": []}

    progress("interpret_frozen_models_and_describe_2025")
    with threadpool_limits(limits=threads):
        interpretation = {"permutation_importance": block_permutation_importance(calibrated, test, Xt, yt, wt, seed=seed, repeats=2 if smoke else 5),
                          "shap": shap_summary(base, X, seed=seed, maximum=32 if smoke else 256),
                          "clusters": cluster_municipalities(mtest, seed=seed, smoke=smoke),
                          "leakage_demo": leakage_demonstration(X, y, weights, X_reserve, y_reserve, w_reserve)}

    progress("refit_separate_2026_artifacts_after_evaluation")
    test_reserve_mask = split.mask(test, reserve=True)
    # Previously unseen municipalities enter development, never the old reserve.
    refit_frame = pd.concat([dev, test.loc[~test_reserve_mask]], ignore_index=True)
    refit_reserve = test.loc[test_reserve_mask].reset_index(drop=True)
    Xrefit, Xcal = select_features(refit_frame), select_features(refit_reserve)
    yrefit, wrefit = validate_student_labels(refit_frame)
    ycal, wcal = validate_student_labels(refit_reserve)
    future_base = make_classifier(best["model"], best["params"], seed=seed, threads=threads)
    with threadpool_limits(limits=threads):
        fit_weighted(future_base, Xrefit, yrefit, wrefit)
        future_calibrated = calibrate_frozen(future_base, Xcal, ycal, wcal)
        future_threshold = select_f2_threshold(ycal, risk_probability(future_calibrated, Xcal), wcal)
    future_meta = {"run_id": run_id, "protocol_hash": protocol_hash, "input_manifest_hash": input_hash,
                   "evaluated_predecessor_lock": lock["lock_hash"], "training_years": [2024, 2025],
                   "independent_future_test": None, "forecast_year": 2026}
    artifacts["classifier_forecast"] = _save_model(output_dir / "models/modelo_projecao_2026_classificacao.joblib", future_calibrated,
                    {**future_meta, "role": "refit_2026_classifier_not_independently_evaluated", "threshold": future_threshold["threshold"]})
    mall = pd.concat([municipal, mtest], ignore_index=True)
    future_regressor = make_regressor(mbest["model"], mbest["params"], seed=seed, threads=threads)
    with threadpool_limits(limits=threads):
        fit_weighted(future_regressor, select_features(mall), mall["taxa_alfabetizacao"].to_numpy(dtype=float))
    artifacts["regressor_forecast"] = _save_model(output_dir / "models/modelo_projecao_2026_municipal.joblib", future_regressor,
                   {**future_meta, "role": "refit_2026_regressor_not_independently_evaluated"})
    forecast_path = marts_dir / "municipios_2026.parquet"
    if forecast_path.exists():
        forecast = pd.read_parquet(forecast_path)
        forecast = forecast.loc[forecast["rede"].eq("municipal")].reset_index(drop=True)
        if not forecast["ano"].eq(2026).all() or forecast.duplicated(["ano", "id_municipio", "rede"]).any():
            raise ValueError("Forecast mart must contain unique 2026 municipal contexts")
        forecast_values = np.clip(future_regressor.predict(select_features(forecast)), 0, 100)
        municipal_result["forecast_rows"] = _municipal_rows(forecast, forecast_values, forecast=True)
        municipal_result["forecast_status"] = "completed; one-step 2026 projection without independent 2026 evaluation"
    else:
        municipal_result["forecast_status"] = "not_available: no official-context 2026 feature mart supplied"

    coverage_path = marts_dir.parent / "coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {
        "classification": classification["population"], "municipal": municipal_result["population"],
        "source_coverage_report": "not supplied; only modeling-cohort counts available"}
    result = {"run_id": run_id, "protocol_hash": protocol_hash, "input_manifest_hash": input_hash,
              "status": "smoke_completed" if smoke else "completed", "classification": classification,
              "municipal": municipal_result, "coverage": coverage, "interpretation": interpretation,
              "artifacts": artifacts, "protocol_lock_hash": lock["lock_hash"],
              "limitations": [
                  "Individual classification uses shared territorial context; it is not an individual educational diagnosis.",
                  "Retrospective backtest uses currently available snapshots, not verified historical data vintages.",
                  "Preprocessing medians and scaling are unweighted, fit on development only; objectives and individual metrics use official weights.",
                  "Model performance does not identify causal effects of educational interventions.",
                  "Refit2026 models have no independent 2026 outcome evaluation; historical intervals do not guarantee future coverage.",
                  "Bootstrap describes conditional municipal variability, not the full official sampling design.",
              ]}
    # Re-load the frozen evaluated model and reproduce its actual test probabilities.
    reloaded = joblib.load(output_dir / "models/modelo_avaliado_2025_classificacao.joblib")["estimator"]
    with threadpool_limits(limits=threads):
        restored = risk_probability(reloaded, Xt)
    if not np.allclose(restored, probability, rtol=1e-12, atol=1e-12):
        raise AssertionError("Reloaded evaluated artifact does not reproduce published predictions")
    if _hash_object({key: value for key, value in lock.items() if key != "lock_hash"}) != lock["lock_hash"]:
        raise AssertionError("Frozen protocol changed during evaluation/refit")
    receipt = {"run_id": run_id, "started_at_utc": started_at,
               "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
               "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "source_fingerprint": source_hash, "protocol_hash": protocol_hash, "input_manifest_hash": input_hash,
               "temporal_lock_verified": True, "evaluated_artifact_reload_verified": True,
               "status": result["status"], "smoke": smoke}
    try:
        revision = subprocess.run(["rtk", "git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                  capture_output=True, text=True, check=True).stdout.strip()
        receipt["git_revision"] = revision
    except (OSError, subprocess.SubprocessError):
        receipt["git_revision"] = None
    _write(output_dir / "run_receipt.json", receipt)
    _write(output_dir / "metrics.json", result)
    progress("completed")
    return result
