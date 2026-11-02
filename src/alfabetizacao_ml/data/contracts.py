"""Typed source rules shared by preparation and adversarial tests."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

RESULT_NETWORKS = {0: "total", 2: "estadual", 3: "municipal", 5: "publica"}
STUDENT_NETWORKS = {1: "federal", 2: "estadual", 3: "municipal", 4: "privada"}
MISSING_MARKERS = {"", "-", "--", "...", "..", "X", "NA", "N/A", "nan", "None", "<NA>"}
LEVEL_COLUMNS = [f"proporcao_aluno_nivel_{level}" for level in range(9)]


def is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() in MISSING_MARKERS


def decimal_value(value: object, *, nullable: bool = True) -> Decimal | None:
    if is_missing(value):
        if nullable:
            return None
        raise ValueError("Required numeric value is missing")
    try:
        result = Decimal(str(value).strip().replace(",", "."))
    except InvalidOperation as exc:
        raise ValueError("Malformed numeric value") from exc
    if not result.is_finite():
        raise ValueError("Non-finite numeric value")
    return result


def municipality_code(value: object) -> str:
    """Accept a seven-digit ID or an exactly integral spreadsheet number."""
    number = decimal_value(value, nullable=False)
    if number != number.to_integral_value():
        raise ValueError("Municipality code is not integral")
    text = str(int(number))
    if not re.fullmatch(r"[1-5][0-9]{6}", text):
        raise ValueError("Municipality code must contain seven digits")
    return text


def parse_meta(value: object) -> dict:
    raw = None if is_missing(value) else str(value).strip()
    if raw is None:
        return {"meta_raw": None, "meta_operador": None, "meta_limite": None,
                "meta_status": "missing_official"}
    match = re.fullmatch(r"(>=|≥|>)?\s*([0-9]+(?:[.,][0-9]+)?)\s*%?", raw)
    if match is None:
        raise ValueError("Malformed official target; not an official absence")
    limit = decimal_value(match.group(2), nullable=False)
    if not Decimal(0) <= limit <= Decimal(100):
        raise ValueError("Target outside percentage range")
    return {"meta_raw": raw, "meta_operador": ">" if match.group(1) == ">" else ">=",
            "meta_limite": limit, "meta_status": "available"}


def target_met(rate: object, operator: object, limit: object) -> bool | None:
    if is_missing(rate) or is_missing(limit) or operator is None:
        return None
    observed, target = decimal_value(rate), decimal_value(limit)
    if operator == ">":
        return observed > target
    if operator == ">=":
        return observed >= target
    raise ValueError("Unknown target operator")


def assert_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame[keys].isna().any(axis=None):
        raise ValueError(f"{name}: null key")
    if frame.duplicated(keys, keep=False).any():
        raise ValueError(f"{name}: duplicate dimension/key (exact or conflicting)")


def finite_numeric(series: pd.Series, *, nullable: bool = True) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    missing = text.isna() | text.isin(MISSING_MARKERS)
    numeric = pd.to_numeric(text.mask(missing), errors="coerce").astype("float64")
    if ((~missing) & numeric.isna()).any():
        raise ValueError(f"Malformed numeric column: {series.name}")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError(f"Non-finite numeric column: {series.name}")
    if not nullable and numeric.isna().any():
        raise ValueError(f"Required numeric column missing: {series.name}")
    return numeric


def integral_numeric(series: pd.Series) -> pd.Series:
    numeric = finite_numeric(series, nullable=False)
    if (numeric % 1 != 0).any():
        raise ValueError(f"Non-integral code/year/grade column: {series.name}")
    return numeric.astype("int64")


def validate_levels(frame: pd.DataFrame, year: int) -> None:
    if year == 2023:
        if frame[LEVEL_COLUMNS].notna().any(axis=None):
            raise ValueError("2023 levels are structurally unavailable")
        return
    if frame[LEVEL_COLUMNS].isna().any(axis=None):
        raise ValueError("Unexpected missing literacy level")
    numbers = frame[LEVEL_COLUMNS].astype(float)
    if not np.isfinite(numbers.to_numpy()).all() or ((numbers < 0) | (numbers > 100)).any(axis=None):
        raise ValueError("Literacy level outside finite percentage range")
    totals = numbers.sum(axis=1)
    if ((totals < 99.5) | (totals > 100.5)).any():
        raise ValueError("Literacy levels do not sum to approximately 100")


def safe_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    assert_unique(right, keys, name)
    result = left.merge(right, on=keys, how="left", validate="many_to_one")
    if len(result) != len(left):
        raise ValueError(f"{name}: join changed the number of records")
    return result


def exact_previous_year_context(frame: pd.DataFrame, historic: pd.DataFrame) -> pd.DataFrame:
    context = historic[["ano", "id_municipio", "rede", "taxa_alfabetizacao", "media_portugues"]].copy()
    context = context.rename(columns={
        "ano": "ano_contexto", "taxa_alfabetizacao": "taxa_alfabetizacao_lag1",
        "media_portugues": "media_portugues_lag1",
    })
    context["ano"] = context["ano_contexto"] + 1
    return safe_join(frame, context, ["ano", "id_municipio", "rede"], "exact previous-year context")
