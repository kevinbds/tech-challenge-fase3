"""Synthetic orchestration probes; training is stubbed, analyses/rendering are real."""

import json
import shutil
from unittest.mock import Mock

import pandas as pd
import pytest
from test_eda import fixture_marts
from test_scenarios import fixture_inputs

from alfabetizacao_ml.workflow import run_workflow


def setup_pipeline(tmp_path, monkeypatch):
    from alfabetizacao_ml import data, modeling

    marts = fixture_marts(tmp_path)
    _, metrics, targets = fixture_inputs(tmp_path)
    silver = marts.parent / "silver"
    silver.mkdir()
    shutil.copyfile(targets, silver / "metas.parquet")
    project = tmp_path / "public-project"
    project.mkdir()
    protocol = project / "protocol.json"
    protocol.write_text('{"seed": 42}', encoding="utf-8")
    manifest = tmp_path / "synthetic-manifest.json"
    manifest.write_text('{"synthetic_fixture": true}', encoding="utf-8")
    monkeypatch.setattr(data, "acquire_sources", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(data, "prepare_data", lambda *args: {"marts_dir": str(marts)})

    def trained_fixture(unused_marts, output, unused_protocol, *, smoke):
        # Model selection cannot start if development EDA did not complete.
        assert (output / "eda" / "eda_summary.json").is_file()
        value = json.loads(metrics.read_text(encoding="utf-8"))
        (output / "metrics.json").write_text(json.dumps(value), encoding="utf-8")
        return value

    training = Mock(side_effect=trained_fixture)
    monkeypatch.setattr(modeling, "run_experiments", training)
    options = dict(
        project_root=project,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        protocol_path=protocol,
        sources_manifest=manifest,
        mode="reproduce",
        smoke=True,
    )
    return marts, training, options


def test_invalid_development_population_stops_before_model_selection(tmp_path, monkeypatch):
    marts, training, options = setup_pipeline(tmp_path, monkeypatch)
    path = marts / "alunos_2024.parquet"
    frame = pd.read_parquet(path)
    frame["serie"] = 3
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="validated eligible"):
        run_workflow(**options)
    training.assert_not_called()
    receipt = json.loads((options["output_dir"] / "execution-receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error_type"] == "ValueError"
    assert not options["reports_dir"].exists()


def test_workflow_renders_the_supplement_bound_to_its_actual_metrics(tmp_path, monkeypatch):
    _, training, options = setup_pipeline(tmp_path, monkeypatch)
    receipt = run_workflow(**options)
    training.assert_called_once()
    assert receipt["status"] == "smoke_completed"
    reports = options["reports_dir"]
    supplement = json.loads((reports / "TEST_scenarios.json").read_text(encoding="utf-8"))
    assert supplement["metrics_sha256"] == receipt["metrics_sha256"]
    assert len(supplement["municipal_rows"]) == 16
    assert {row["ano"] for row in supplement["summary"]} == {2027, 2028, 2029, 2030}
    text = (reports / "TEST_APRESENTACAO_FASE3.html").read_text(encoding="utf-8")
    assert "2027" in text and "Sudeste" in text
    assert (reports / "TEST_RELATORIO_TECNICO_FASE3.pdf").stat().st_size > 1000
