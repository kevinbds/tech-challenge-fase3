"""A closed feature contract prevents accidental target/identifier leakage."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = (
    "taxa_alfabetizacao_lag1", "media_portugues_lag1", "media_inse_2021",
    "log_populacao_2022", "densidade_2022",
)
CATEGORICAL_FEATURES = ("sigla_uf", "regiao", "rede")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def select_features(frame: pd.DataFrame, *, require_context: bool = True) -> pd.DataFrame:
    """Ignore every non-allowlisted column; reject malformed observed numbers."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if require_context and not {"ano", "ano_contexto"}.issubset(frame.columns):
        raise ValueError("A training/evaluation mart must contain year and context-year provenance")
    if "ano_contexto" in frame:
        context = pd.to_numeric(frame["ano_contexto"], errors="raise")
        target = pd.to_numeric(frame["ano"], errors="raise")
        has_lag = frame[["taxa_alfabetizacao_lag1", "media_portugues_lag1"]].notna().any(axis=1)
        if ((context.notna() & (context != target - 1)) | (has_lag & context.isna())).any():
            raise ValueError("Context must come from exactly the preceding year")
    result = frame.loc[:, FEATURES].copy()
    for name in NUMERIC_FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        if np.isinf(result[name].to_numpy()).any():
            raise ValueError(f"Nonfinite observed feature: {name}")
    rate = result["taxa_alfabetizacao_lag1"].dropna()
    if not rate.between(0, 100).all():
        raise ValueError("Lagged literacy rate must use percentage points in [0,100]")
    for name in ("log_populacao_2022", "densidade_2022"):
        if (result[name].dropna() < 0).any():
            raise ValueError(f"Negative contextual feature: {name}")
    for name in CATEGORICAL_FEATURES:
        result[name] = result[name].astype(object).where(result[name].notna(), np.nan)
    return result


def make_preprocessor(*, scale: bool) -> ColumnTransformer:
    numeric = [("impute", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True,
    ))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="ausente", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", verbose_feature_names_out=True)


def validate_student_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.to_numeric(frame["alfabetizado"], errors="raise").to_numpy(dtype=float)
    weights = pd.to_numeric(frame["peso_aluno"], errors="raise").to_numpy(dtype=float)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Official literacy label must belong to {0,1}; missing is not risk")
    if not (np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError("Student weights must be finite and strictly positive")
    risk = (1 - labels).astype(np.int8)
    if "risco" in frame and not np.array_equal(frame["risco"].to_numpy(), risk):
        raise ValueError("Risk mapping differs from 1 - official literacy label")
    if not frame["rede"].isin(["estadual", "municipal"]).all():
        raise ValueError("Student population is restricted to state and municipal schools")
    if "serie" in frame and not pd.to_numeric(frame["serie"], errors="raise").eq(2).all():
        raise ValueError("Student population is restricted to second grade")
    return risk, weights
