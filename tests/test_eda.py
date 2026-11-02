import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from alfabetizacao_ml.eda import (
    FEATURES,
    _assert_development,
    _municipal_analysis,
    _municipal_development,
    _student_aggregates,
    run_eda,
)


def fixture_marts(tmp_path):
    marts = tmp_path / "snapshot" / "marts"
    marts.mkdir(parents=True)
    coverage = {"students": {}}
    for year in (2023, 2024):
        frame = pd.DataFrame({"ano": [year] * 40, "serie": [2] * 40, "rede": ["municipal"] * 40,
                              "sigla_uf": ["RO"] * 40, "alfabetizado": [1, 0] * 20,
                              "peso_aluno": [1.0, 3.0] * 20,
                              **{name: [None if year == 2023 else 60.0] * 40 for name in FEATURES}})
        frame.to_parquet(marts / f"alunos_{year}.parquet", index=False)
        coverage["students"][str(year)] = {"raw_rows": 42, "eligible_rows": 40, "excluded_grade": 0,
            "excluded_network": 0, "excluded_missing_identity_context": 1,
            "excluded_missing_label": 0, "excluded_nonpositive_or_missing_weight": 1,
            "municipalities": 2, "municipality_network_contexts": 2}
    # Poison files demonstrate that neither all-year students nor test partitions
    # are opened. Municipal test-year target is also invalid and must be filtered.
    (marts / "alunos_2025.parquet").write_bytes(b"DO NOT OPEN HELD OUT LABELS")
    (marts / "alunos.parquet").write_bytes(b"DO NOT OPEN ALL YEAR LABELS")
    rows = []
    for year in (2023, 2024, 2025):
        for index, operator in enumerate((">", ">=")):
            rows.append({"ano": year, "id_municipio": f"110000{index}", "sigla_uf": "RO", "rede": "municipal",
                         "taxa_alfabetizacao": 999 if year == 2025 else 80,
                         "meta_operador": operator if year == 2024 else None,
                         "meta_limite": 80 if year == 2024 else None,
                         "meta_status": "available" if year == 2024 else "not_published_for_period",
                         **{name: None if year == 2023 else float(index + 1) for name in FEATURES}})
    pd.DataFrame(rows).to_parquet(marts / "municipios.parquet", index=False)
    (marts.parent / "coverage.json").write_text(json.dumps(coverage))
    return marts


def test_weights_development_filter_and_strict_targets(tmp_path):
    marts = fixture_marts(tmp_path)
    with duckdb.connect() as con:
        students, missing = _student_aggregates(con, marts)
        municipal = _municipal_development(con, marts)
    assert {row["ano"] for row in students["year"]} == {2023, 2024}
    assert all(row["literacy_pct"] == 25 and row["nonliteracy_pct"] == 75 for row in students["year"])
    assert all(row["unweighted_literacy_pct"] == 50 for row in students["year"])
    assert all(row["weighted_missing_pct"] == 100 for row in missing if row["ano"] == 2023)
    assert set(municipal["ano"]) == {2023, 2024}
    report = _municipal_analysis(municipal)
    assert report["targets_2024_total"] == {"municipalities": 2, "assessable": 2, "met": 1, "missing": 0}
    assert report["targets_2024_by_uf"][0]["strict_boundary_not_met"] == 1
    assert report["balanced_panel"]["municipalities"] == 2


def test_aggregate_artifacts_no_identifiers_and_no_test_data(tmp_path):
    marts = fixture_marts(tmp_path)
    output = tmp_path / "eda"
    report = run_eda(marts, output)
    serialized = (output / "eda_summary.json").read_text(encoding="utf-8")
    assert '"id_municipio"' not in serialized
    assert '"id_aluno"' not in serialized
    assert '"id_escola"' not in serialized
    assert '"ano": 2025' not in serialized
    assert len(report["figures"]) == 7
    assert all((output / filename).stat().st_size > 1000 for filename in report["figures"])
    assert report["test_labels_used"] is False


def test_mislabelled_student_partition_blocks_analysis(tmp_path):
    marts = fixture_marts(tmp_path)
    path = marts / "alunos_2024.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "ano"] = 2025
    frame.to_parquet(path)
    with duckdb.connect() as con, pytest.raises(ValueError, match="unexpected year"):
        _student_aggregates(con, marts)


@pytest.mark.parametrize("column,value", [("rede", None), ("serie", 3), ("serie", None)])
def test_ineligible_network_or_grade_blocks_analysis_independently(tmp_path, column, value):
    marts = fixture_marts(tmp_path)
    path = marts / "alunos_2024.parquet"
    frame = pd.read_parquet(path)
    frame[column] = value
    frame.to_parquet(path, index=False)
    with duckdb.connect() as con, pytest.raises(ValueError, match="validated eligible"):
        _student_aggregates(con, marts)


def test_active_snapshot_cannot_be_an_output(tmp_path):
    marts = fixture_marts(tmp_path)
    with pytest.raises(ValueError, match="outside the active"):
        run_eda(marts, marts.parent / "eda")


def test_year_assert_rejects_test_rows():
    with pytest.raises(ValueError, match="2023/2024"):
        _assert_development(pd.DataFrame({"ano": [2025]}))


def test_absent_target_is_not_a_failure_and_no_correlation_without_pairs(tmp_path):
    marts = fixture_marts(tmp_path)
    with duckdb.connect() as con:
        frame = _municipal_development(con, marts)
    frame.loc[frame["ano"].eq(2024), "meta_limite"] = np.nan
    result = _municipal_analysis(frame)
    assert result["targets_2024_total"]["assessable"] == 0
    assert result["targets_2024_total"]["missing"] == 2
    assert all(item["spearman"] is None for item in result["correlations"])


def test_municipal_test_labels_cannot_change_development_analysis(tmp_path):
    marts = fixture_marts(tmp_path)
    with duckdb.connect() as con:
        before = _municipal_analysis(_municipal_development(con, marts))
    path = marts / "municipios.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["ano"].eq(2025), "taxa_alfabetizacao"] = -999
    frame.to_parquet(path, index=False)
    with duckdb.connect() as con:
        after = _municipal_analysis(_municipal_development(con, marts))
    assert before == after


def test_notebook_source_is_output_free_and_has_no_test_metrics():
    import nbformat
    notebook = nbformat.read(Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb", as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "metrics.json" not in code
    assert "FASE3_MARTS" in code and "FASE3_EDA_OUTPUT" in code
    assert all(not cell.outputs and cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
