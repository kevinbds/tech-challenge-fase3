"""Adversarial data cases use only explicitly synthetic student records."""

from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
import pytest

from alfabetizacao_ml.data.contracts import (
    LEVEL_COLUMNS,
    assert_unique,
    decimal_value,
    exact_previous_year_context,
    finite_numeric,
    municipality_code,
    parse_meta,
    safe_join,
    target_met,
    validate_levels,
)
from alfabetizacao_ml.data.preparation import _prepare_students, _write_frame


@pytest.mark.parametrize("raw,operator,limit", [
    (" > 80 ", ">", Decimal("80")), ("80,0", ">=", Decimal("80.0")),
    ("80.0004", ">=", Decimal("80.0004")), ("≥80", ">=", Decimal("80")),
])
def test_target_operator_and_precision(raw, operator, limit):
    result = parse_meta(raw)
    assert result["meta_operador"] == operator
    assert result["meta_limite"] == limit
    assert result["meta_status"] == "available"


@pytest.mark.parametrize("raw", [None, "", "-", np.nan])
def test_official_absence_is_not_a_zero_target(raw):
    result = parse_meta(raw)
    assert result["meta_status"] == "missing_official"
    assert result["meta_limite"] is None


@pytest.mark.parametrize("raw", ["bad", ">=unknown", "101", "-1", "inf"])
def test_malformed_target_is_not_missing(raw):
    with pytest.raises(ValueError):
        parse_meta(raw)


def test_strict_target_and_decimal_storage_boundary(tmp_path):
    assert target_met(80, ">", 80) is False
    assert target_met(80, ">=", 80) is True
    assert target_met("80.0004", ">", 80) is True
    assert target_met(None, ">", 80) is None
    path = tmp_path / "decimal.parquet"
    _write_frame(pd.DataFrame({"taxa": [Decimal("80.0004"), Decimal("100")]}), path, ["taxa"])
    with duckdb.connect() as con:
        actual = con.execute("SELECT taxa, taxa>80 FROM read_parquet(?) ORDER BY taxa", [str(path)]).fetchall()
    assert actual == [(Decimal("80.000400000"), True), (Decimal("100.000000000"), True)]


def test_decimal_inference_does_not_truncate_large_population(tmp_path):
    path = tmp_path / "population.parquet"
    _write_frame(pd.DataFrame({"population": [Decimal(1)] * 1001 + [Decimal(11451999)]}), path, ["population"])
    with duckdb.connect() as con:
        assert con.execute("SELECT max(population) FROM read_parquet(?)", [str(path)]).fetchone()[0] == Decimal(11451999)


def test_levels_structurally_absent_only_for_2023():
    missing = pd.DataFrame([{column: None for column in LEVEL_COLUMNS}])
    validate_levels(missing, 2023)
    with pytest.raises(ValueError, match="Unexpected missing"):
        validate_levels(missing, 2024)
    valid = pd.DataFrame([{column: Decimal(100) if index == 0 else Decimal(0)
                           for index, column in enumerate(LEVEL_COLUMNS)}])
    validate_levels(valid, 2024)
    with pytest.raises(ValueError, match="structurally unavailable"):
        validate_levels(valid, 2023)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), "NaN", "infinity"])
def test_nonfinite_observed_values_rejected(value):
    with pytest.raises(ValueError):
        decimal_value(value, nullable=False)
    with pytest.raises(ValueError):
        finite_numeric(pd.Series([value]), nullable=False)


@pytest.mark.parametrize("second", ["same", "conflicting"])
def test_dimension_duplicates_never_multiply_records(second):
    left = pd.DataFrame({"id": [1, 1]})
    right = pd.DataFrame({"id": [1, 1], "name": ["same", second]})
    with pytest.raises(ValueError, match="duplicate"):
        safe_join(left, right, ["id"], "dimension")


def test_exact_previous_year_does_not_bridge_a_gap_or_network():
    historic = pd.DataFrame({"ano": [2023, 2024], "id_municipio": ["1100015"] * 2,
                             "rede": ["municipal", "estadual"],
                             "taxa_alfabetizacao": [60, 70], "media_portugues": [720, 750]})
    target = pd.DataFrame({"ano": [2024, 2025], "id_municipio": ["1100015"] * 2,
                          "rede": ["municipal"] * 2})
    result = exact_previous_year_context(target, historic)
    assert result.loc[0, "ano_contexto"] == 2023
    assert pd.isna(result.loc[1, "ano_contexto"])
    assert pd.isna(result.loc[1, "taxa_alfabetizacao_lag1"])
    assert len(result) == len(target)


@pytest.mark.parametrize("value", ["3550308.1", "355030", "35503089", "x", None])
def test_municipality_code_rejects_nonintegral_or_bad_shape(value):
    with pytest.raises(ValueError):
        municipality_code(value)


def test_excel_integral_municipality_code_is_supported():
    assert municipality_code("3550308.0") == "3550308"
    assert municipality_code(3550308.0) == "3550308"


def _synthetic_students_zip(path: Path, rows: list[dict]) -> Path:
    columns = {"NU_ANO_AVALIACAO": "2024", "CO_MUNICIPIO": "1100015", "SG_UF": "RO",
               "ID_ESCOLA": "synthetic_school", "TP_SERIE": "2", "TP_DEPENDENCIA": "3",
               "IN_ALFABETIZADO": "1", "VL_PESO_ALUNO_LP": "1.5", "VL_PROFICIENCIA_LP": "800",
               "IN_PRESENCA_LP": "1", "IN_PREENCHIMENTO_LP": "1", "CO_CADERNO_LP": "1"}
    frame = pd.DataFrame([{**columns, "ID_ALUNO": f"synthetic_{index}", **row} for index, row in enumerate(rows)])
    with ZipFile(path, "w") as archive:
        archive.writestr("DADOS/TS_ALUNO.csv", frame.to_csv(index=False, sep=";").encode("latin1"))
    return path


def test_student_mart_retains_missing_context_and_excludes_leakage(tmp_path):
    archive = _synthetic_students_zip(tmp_path / "synthetic.zip", [
        {}, {"VL_PESO_ALUNO_LP": "0"}, {"IN_ALFABETIZADO": ""}, {"TP_DEPENDENCIA": "4"},
        {"CO_MUNICIPIO": "", "ID_ESCOLA": "", "TP_DEPENDENCIA": ""},
        {"ID_ESCOLA": ""},
    ])
    directory = pd.DataFrame({"id_municipio": ["1100015"], "sigla_uf": ["RO"], "regiao": ["Norte"]})
    historic = pd.DataFrame(columns=["ano", "id_municipio", "rede", "taxa_alfabetizacao", "media_portugues"])
    historic["ano"] = historic["ano"].astype(int)
    inse = pd.DataFrame({"id_municipio": ["1100015"], "rede": ["estadual"], "media_inse_2021": [5.0]})
    census = pd.DataFrame({"id_municipio": ["1100015"], "populacao_2022": [Decimal(100)], "densidade_2022": [Decimal(1)]})
    counts = _prepare_students(archive, 2024, directory, historic, inse, census,
                               tmp_path / "silver.parquet", tmp_path / "mart.parquet")
    mart = pd.read_parquet(tmp_path / "mart.parquet")
    assert counts["eligible_rows"] == 1
    assert counts["excluded_missing_identity_context"] == 2
    quarantine = pd.read_parquet(tmp_path / "quarantine_identity_2024.parquet")
    assert len(quarantine) == 2
    assert quarantine["quarantine_reason"].eq("missing_mandatory_identity_context").all()
    assert counts["excluded_network"] == 1
    assert counts["excluded_missing_label"] == 1
    assert counts["excluded_nonpositive_or_missing_weight"] == 1
    assert mart["ano_contexto"].isna().all()
    assert mart["media_inse_2021"].isna().all()
    assert mart["risco"].tolist() == [0]
    assert not {"proficiencia", "presenca", "caderno", "preenchimento_caderno"}.intersection(mart.columns)
    # A failed rebuild must leave the previously validated mart readable.
    original = (tmp_path / "mart.parquet").read_bytes()
    invalid = _synthetic_students_zip(tmp_path / "invalid.zip", [{"SG_UF": "SP"}])
    with pytest.raises(ValueError, match="disagrees"):
        _prepare_students(invalid, 2024, directory, historic, inse, census,
                          tmp_path / "silver.parquet", tmp_path / "mart.parquet")
    assert (tmp_path / "mart.parquet").read_bytes() == original
    assert len(pd.read_parquet(tmp_path / "mart.parquet")) == 1


def test_null_dimension_key_rejected():
    with pytest.raises(ValueError, match="null key"):
        assert_unique(pd.DataFrame({"id": [None]}), ["id"], "dimension")


def test_geography_contradiction_is_not_silently_quarantined():
    from alfabetizacao_ml.data.preparation import _validate_geography
    directory = pd.DataFrame({"id_municipio": ["1100015"], "sigla_uf": ["RO"], "regiao": ["Norte"]})
    frame = pd.DataFrame({"id_municipio": ["1100015"], "sigla_uf": ["SP"]})
    with pytest.raises(ValueError, match="disagrees"):
        _validate_geography(frame, directory)


def test_inse_uses_total_location_and_zero_quantity_is_missing(tmp_path):
    from alfabetizacao_ml.data.preparation import _inse
    path = tmp_path / "inse.xlsx"
    pd.DataFrame({"NU_ANO_SAEB": [2021]*4, "CO_MUNICIPIO": [1100015]*4,
                  "TP_TIPO_REDE": [3, 3, 2, 1], "TP_LOCALIZACAO": [0, 1, 0, 0],
                  "QTD_ALUNOS_INSE": [0, 10, 20, 30], "MEDIA_INSE": [0, 5, 6, 7]}).to_excel(path, index=False)
    result = _inse(path).set_index("rede")
    assert len(result) == 2
    assert pd.isna(result.loc["municipal", "media_inse_2021"])
    assert result.loc["estadual", "media_inse_2021"] == 6


def test_new_municipality_missing_census_is_retained():
    from alfabetizacao_ml.data.preparation import _enrich
    frame = pd.DataFrame({"id_municipio": ["1100015"], "rede": ["municipal"]})
    inse = pd.DataFrame(columns=["id_municipio", "rede", "media_inse_2021"])
    census = pd.DataFrame(columns=["id_municipio", "populacao_2022", "densidade_2022"])
    result = _enrich(frame, inse, census)
    assert len(result) == 1
    assert result["populacao_2022"].isna().all()


def test_fractional_year_or_grade_is_not_truncated():
    from alfabetizacao_ml.data.contracts import integral_numeric
    with pytest.raises(ValueError, match="Non-integral"):
        integral_numeric(pd.Series([2.9], name="grade"))


def test_target_workbook_horizons_and_network_contract(tmp_path):
    from alfabetizacao_ml.data.preparation import _targets
    row = {"ANO": 2024, "NOME_UF": "Brasil", "REDE": "PÚBLICA",
           "PC_ALUNO_ALFABETIZADO_2023": 60, "PC_ALUNO_ALFABETIZADO_2024": 65,
           **{f"META_FINAL_{year}": ">80" for year in range(2024, 2031)}}
    path = tmp_path / "targets.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False, startrow=1)
    targets, observed = _targets(path, municipal=False)
    assert set(targets["ano_meta"]) == set(range(2024, 2031))
    assert set(observed["ano"]) == {2023, 2024}
    row["REDE"] = "PRIVADA"
    pd.DataFrame([row]).to_excel(path, index=False, startrow=1)
    with pytest.raises(ValueError, match="unexpected network"):
        _targets(path, municipal=False)
