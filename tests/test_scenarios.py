import json
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader
from test_reports import metrics_fixture

from alfabetizacao_ml.scenarios import build_goal_scenarios, cluster_geography
from alfabetizacao_ml.visualization import build_reports
from alfabetizacao_ml.visualization.reports import _browser


def fixture_inputs(tmp_path):
    metrics = metrics_fixture.__wrapped__()
    metrics["municipal"]["forecast_rows"] = [
        {"id_municipio": str(1000000 + i), "sigla_uf": uf, "ano": 2026, "forecast": rate}
        for i, (uf, rate) in enumerate([("SP", 80), ("BA", 80.0004), ("AM", 80), ("RS", None)])
    ]
    metrics["municipal"]["evaluation_rows"] = [
        {"id_municipio": row["id_municipio"], "sigla_uf": row["sigla_uf"]}
        for row in metrics["municipal"]["forecast_rows"]
    ]
    metrics["interpretation"]["clusters"]["assignments"] = [
        {"id_municipio": str(1000000 + i), "cluster": i % 2} for i in range(4)
    ]
    metrics["interpretation"]["clusters"]["profiles"] = [
        {"cluster": 0, "municipalities": 2, "taxa_alfabetizacao": 80},
        {"cluster": 1, "municipalities": 2, "taxa_alfabetizacao": 70},
    ]
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    rows = [
        {
            "ano_meta": year,
            "nivel_geografico": "municipio",
            "id_geografia": str(1000000 + i),
            "rede": "municipal",
            "meta_raw": ">80" if i < 2 else ">=80",
            "meta_operador": ">" if i < 2 else ">=",
            "meta_limite": 80.0,
            "meta_status": "available",
        }
        for year in range(2027, 2031)
        for i in range(4)
        if not (year == 2030 and i == 2)
    ]
    # Irrelevant observed label is deliberately poisonous and never projected/read.
    for row in rows:
        row["taxa_alfabetizacao_2025"] = "DO_NOT_READ_OBSERVED_LABEL"
    target = tmp_path / "metas.parquet"
    pd.DataFrame(rows).to_parquet(target)
    return metrics, path, target


def test_scenarios_strict_targets_absence_and_frozen_input(tmp_path):
    metrics, path, target = fixture_inputs(tmp_path)
    original = path.read_bytes()
    result = build_goal_scenarios(path, target, tmp_path / "supplement")
    data = json.loads(Path(result["scenarios_json"]).read_text(encoding="utf-8"))
    assert path.read_bytes() == original
    assert data["synthetic_fixture"]
    assert len(data["municipal_rows"]) == 16
    for row in data["municipal_rows"]:
        code = row["id_municipio"]
        if code == "1000000":
            assert row["goal_met"] is False and row["scenario_rate"] == 80
        if code == "1000001":
            assert row["goal_met"] is True and row["scenario_rate"] == 80.0004
        if code == "1000002":
            assert row["goal_met"] is (None if row["ano"] == 2030 else True)
        if code == "1000003":
            assert row["status"] == "sem_projecao"
    assert data["summary"][-1] == {
        "ano": 2030,
        "atinge": 1,
        "nao_atinge": 1,
        "sem_meta": 1,
        "sem_projecao": 1,
    }
    geography = data["cluster_geography"]
    assert sum(row["municipalities"] for row in geography["by_region"]) == 4
    assert {r["regiao"] for r in geography["by_region"]} == {"Norte", "Nordeste", "Sudeste", "Sul"}
    assert geography["profiles"] == metrics["interpretation"]["clusters"]["profiles"]
    assert "DO_NOT_READ" not in Path(result["scenarios_json"]).read_text()


@pytest.mark.parametrize("field", ["evaluation_rows", "assignments"])
def test_cluster_join_rejects_duplicate_keys(tmp_path, field):
    data, _, _ = fixture_inputs(tmp_path)
    rows = (
        data["municipal"][field]
        if field == "evaluation_rows"
        else data["interpretation"]["clusters"][field]
    )
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="Duplicate"):
        cluster_geography(data)


def test_supplement_report_integration_and_hash_binding(tmp_path):
    _, path, target = fixture_inputs(tmp_path)
    result = build_goal_scenarios(path, target, tmp_path / "supplement")
    reports = build_reports(
        path,
        tmp_path / "reports",
        "https://github.com/kevinbds/tech-challenge-fase3",
        scenarios_path=result["scenarios_json"],
    )
    pdf = PdfReader(reports["technical_pdf"])
    assert len(pdf.pages) < 40
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert (
        "2027" in text
        and "2030" in text
        and "Nordeste" in text
        and "caracterização dos perfis" in text
    )
    assert all("TESTE" in page.extract_text() for page in pdf.pages)
    for page in pdf.pages:
        for ref in page.get("/Annots", []):
            uri = ref.get_object().get("/A", {}).get("/URI")
            if uri:
                assert str(uri).startswith("https://")
    with _browser() as browser:
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(Path(reports["deck_html"]).as_uri())
        page.evaluate("document.documentElement.dataset.mode='frame'")
        for index in (8, 9):
            page.evaluate("i=>window.fase3Deck.show(i)", index)
            body = page.locator(".slide.active .slide-body").bounding_box()
            footer = page.locator(".slide.active .source").bounding_box()
            assert body["y"] + body["height"] <= footer["y"]
            assert body["x"] + body["width"] <= 1848
            page.screenshot(path=str(tmp_path / f"supplement-slide-{index + 1}.png"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="match frozen"):
        build_reports(
            path,
            tmp_path / "bad",
            "https://github.com/kevinbds/tech-challenge-fase3",
            scenarios_path=result["scenarios_json"],
        )


def test_duplicate_targets_and_forecast_year_rejected(tmp_path):
    metrics, path, target = fixture_inputs(tmp_path)
    frame = pd.read_parquet(target)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(target)
    with pytest.raises(ValueError, match="Duplicate official"):
        build_goal_scenarios(path, target, tmp_path / "bad")
    frame.to_parquet(target)
    metrics["municipal"]["forecast_rows"][0]["ano"] = 2025
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="2026"):
        build_goal_scenarios(path, target, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cluster_missing_geography_preserves_cardinality(tmp_path):
    metrics, _, _ = fixture_inputs(tmp_path)
    metrics["municipal"]["evaluation_rows"].pop()
    result = cluster_geography(metrics)
    assert result["assignment_count"] == 4
    assert result["missing_geography_count"] == 1
    assert sum(row["municipalities"] for row in result["by_uf"]) == 4


def test_accepts_preparation_marts_directory_layout(tmp_path):
    _, source, target = fixture_inputs(tmp_path)
    snapshot = tmp_path / "snapshot"
    (snapshot / "marts").mkdir(parents=True)
    (snapshot / "silver").mkdir()
    (snapshot / "gold").mkdir()
    (snapshot / "silver/metas.parquet").write_bytes(target.read_bytes())
    result = build_goal_scenarios(source, snapshot / "marts", tmp_path / "output")
    assert (
        len(
            json.loads(Path(result["scenarios_json"]).read_text(encoding="utf-8"))["municipal_rows"]
        )
        == 16
    )
