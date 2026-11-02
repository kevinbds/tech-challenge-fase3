"""Aggregate explanations; no individual SHAP records or backgrounds are saved."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from alfabetizacao_ml.modeling.estimators import fit_weighted, make_classifier, risk_probability
from alfabetizacao_ml.preprocessing.features import (
    FEATURES,
    NUMERIC_FEATURES,
    make_preprocessor,
)


def compressed_context_examples(frame, features, y, weights):
    """Lossless weighted evaluation compression for identical contextual inputs."""
    keys = frame["id_municipio"].astype(str) + "|" + frame["rede"].astype(str)
    check = features.copy()
    check["__context"] = keys.to_numpy()
    if check.groupby("__context", sort=False).nunique(dropna=False).gt(1).any().any():
        raise ValueError("Predictors vary within municipality/network context")
    codes, unique = pd.factorize(keys, sort=True)
    contexts = check.groupby("__context", sort=True).first().reset_index(drop=True)
    masses = np.bincount(codes * 2 + np.asarray(y, dtype=int), weights=weights, minlength=2 * len(unique))
    active = masses > 0
    expanded_codes = np.repeat(np.arange(len(unique)), 2)[active]
    labels = np.tile([0, 1], len(unique))[active]
    return contexts.iloc[expanded_codes].reset_index(drop=True), labels, masses[active], expanded_codes, contexts


def block_permutation_importance(model, frame, features, y, weights, *, seed=42, repeats=5):
    X, labels, mass, context_codes, contexts = compressed_context_examples(frame, features, y, weights)
    reference = average_precision_score(labels, risk_probability(model, X), sample_weight=mass)
    rng = np.random.default_rng(seed)
    result = []
    for feature in FEATURES:
        scores = []
        for _ in range(repeats):
            changed = X.copy()
            values = contexts[feature].to_numpy()[rng.permutation(len(contexts))]
            changed[feature] = values[context_codes]
            score = average_precision_score(labels, risk_probability(model, changed), sample_weight=mass)
            scores.append(float(reference - score))
        result.append({"feature": feature, "importance_mean": float(np.mean(scores)),
                       "importance_std": float(np.std(scores)), "repeats": repeats,
                       "unit": "municipality/network block", "metric": "weighted average precision",
                       "function": "calibrated risk probability", "causal": False})
    return sorted(result, key=lambda row: row["importance_mean"], reverse=True)


def shap_summary(base, development_features, *, seed=42, maximum=256):
    import shap

    distinct = development_features.drop_duplicates().reset_index(drop=True)
    if len(distinct) > maximum:
        distinct = distinct.sample(maximum, random_state=seed).reset_index(drop=True)
    transformed = base.named_steps["preprocess"].transform(distinct)
    estimator = base.named_steps["model"]
    model_name = type(estimator).__name__
    if isinstance(estimator, LogisticRegression):
        explainer = shap.LinearExplainer(estimator, transformed)
        values = np.asarray(explainer.shap_values(transformed))
        output_scale = "base estimator log-odds of risk=1; before sigmoid calibration"
        expected = np.asarray(explainer.expected_value).reshape(-1)[0]
        explained_output = estimator.decision_function(transformed)
    else:
        explainer = shap.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(transformed, check_additivity=True))
        expected_values = np.asarray(explainer.expected_value).reshape(-1)
        if values.ndim == 3:
            class_index = int(np.flatnonzero(estimator.classes_ == 1)[0])
            values = values[:, :, class_index]
            expected = expected_values[class_index]
            explained_output = estimator.predict_proba(transformed)[:, class_index]
            output_scale = "base estimator risk probability; before sigmoid calibration"
        else:
            expected = expected_values[0]
            explained_output = estimator.decision_function(transformed)
            output_scale = "base estimator log-odds of risk=1; before sigmoid calibration"
    reconstructed = expected + values.sum(axis=1)
    if not np.allclose(reconstructed, explained_output, atol=1e-4, rtol=1e-4):
        raise ValueError("SHAP additivity does not match the declared base-estimator output")
    names = base.named_steps["preprocess"].get_feature_names_out()
    aggregate = dict.fromkeys(FEATURES, 0.0)
    for transformed_name, importance in zip(names, np.mean(np.abs(values), axis=0), strict=True):
        source = transformed_name.split("__", 1)[-1].removeprefix("missingindicator_")
        matches = [name for name in FEATURES if source == name or source.startswith(name + "_")]
        if not matches:
            raise ValueError("Cannot associate transformed SHAP feature with the public contract")
        aggregate[max(matches, key=len)] += float(importance)
    return {"status": "completed", "model": model_name, "class_label": "risco=1 (nao alfabetizado)",
            "output_scale": output_scale, "sample_unit": "distinct development contexts, equally weighted",
            "sample_contexts": len(distinct), "additivity_max_error": float(np.max(np.abs(reconstructed - explained_output))),
            "importance": [{"feature": k, "mean_abs_shap": v} for k, v in sorted(aggregate.items(), key=lambda kv: kv[1], reverse=True)],
            "limitation": "Explains the base estimator, not the calibrated probability; associations are not causal."}


def cluster_municipalities(frame, *, seed=42, smoke=False):
    columns = ["taxa_alfabetizacao", "media_inse_2021", "log_populacao_2022", "densidade_2022"]
    values = frame[columns].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Nonfinite observed clustering feature")
    transform = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    X = transform.fit_transform(values)
    distinct_count = len(np.unique(X, axis=0))
    if distinct_count < 3 or len(X) < 3:
        return {"status": "not_applicable", "reason": "Too few distinct municipal profiles for clustering"}
    records, models = [], {}
    with threadpool_limits(limits=8):
        for k in range(2, min(8, distinct_count - 1, len(X) - 1) + 1):
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            assignment = model.fit_predict(X)
            if len(np.unique(assignment)) < 2:
                continue
            score = float(silhouette_score(X, assignment, sample_size=min(len(X), 2000), random_state=seed))
            records.append({"k": k, "silhouette": score, "inertia": float(model.inertia_)})
            models[k] = model
        if not records:
            return {"status": "not_applicable", "reason": "No nondegenerate clustering candidate"}
        winner = max(records, key=lambda row: (row["silhouette"], -row["k"]))
        model = models[winner["k"]]
        assigned = model.predict(X)
        rng = np.random.default_rng(seed)
        stability = []
        for iteration in range(3 if smoke else 10):
            draw = rng.integers(0, len(X), len(X))
            alternate = KMeans(n_clusters=winner["k"], random_state=seed + iteration + 1, n_init=10).fit(X[draw])
            stability.append(float(adjusted_rand_score(assigned, alternate.predict(X))))
    described = values.copy()
    described["cluster"] = assigned
    profiles = []
    for cluster, group in described.groupby("cluster", sort=True):
        profiles.append({"cluster": int(cluster), "municipalities": len(group),
                         **{column: (float(group[column].mean()) if group[column].notna().any() else None) for column in columns}})
    return {"status": "completed", "selected_k": winner["k"], "silhouette": winner["silhouette"],
            "stability_ari": float(np.mean(stability)), "candidates": records,
            "profiles": profiles, "observed_year": int(frame["ano"].max()),
            "assignments": [{"id_municipio": str(m), "cluster": int(c)} for m, c in zip(frame["id_municipio"], assigned, strict=True)],
            "scope": "descriptive latest observed municipal profiles, after predictive protocol freeze"}


def leakage_demonstration(X_development, y_development, w_development, X_reserve, y_reserve, w_reserve):
    clean = make_classifier("logistic", {"C": 1})
    fit_weighted(clean, X_development, y_development, w_development)
    train, valid = X_development.copy(), X_reserve.copy()
    field = "leaked_outcome_for_demonstration"
    train[field], valid[field] = y_development, y_reserve
    original = make_preprocessor(scale=True)
    numeric_name, numeric_pipeline, _ = original.transformers[0]
    leaky_transform = ColumnTransformer([
        (numeric_name, numeric_pipeline, [*NUMERIC_FEATURES, field]), original.transformers[1],
    ], remainder="drop")
    leaky = Pipeline([("preprocess", leaky_transform), ("model", LogisticRegression(C=1, max_iter=2000, class_weight=None, random_state=42))])
    fit_weighted(leaky, train, y_development, w_development)
    return {"status": "completed", "estimator": "LogisticRegression(C=1)", "year": 2024,
            "same_split": True, "same_sample_weights": True, "metric": "weighted average precision",
            "without_leakage": float(average_precision_score(y_reserve, risk_probability(clean, X_reserve), sample_weight=w_reserve)),
            "with_leakage": float(average_precision_score(y_reserve, risk_probability(leaky, valid), sample_weight=w_reserve)),
            "injected_feature": "official outcome itself, deliberately unavailable at prediction time",
            "scope": "isolated educational experiment; excluded from candidate selection and deployment"}
