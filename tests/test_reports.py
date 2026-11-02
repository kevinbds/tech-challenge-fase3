"""Public reports, numeric interpretation and output layout checks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from alfabetizacao_ml.visualization import build_reports
from alfabetizacao_ml.visualization.reports import (
    _band_sentence,
    _comparison_sentence,
    _read_metrics,
    _signal_sentence,
)


def test_small_ap_loss_and_operational_limitations_remain_visible(metrics_fixture):
    metrics_fixture["classification"]["test"]["average_precision"] = 0.45908839446
    metrics_fixture["classification"]["baselines"]["contextual"]["average_precision"] = (
        0.45957146391
    )
    assert "menor que a do baseline contextual em 0,000483" in _comparison_sentence(metrics_fixture)
    metrics_fixture["classification"]["bootstrap"] = {
        "ap_delta_vs_contextual_ci95": [-0.0029426, 0.0018078]
    }
    comparison = _comparison_sentence(metrics_fixture)
    assert (
        "não houve ganho demonstrado" in comparison
        and "0,4591" in comparison
        and "0,4596" in comparison
    )
    assert "IC 95%" in comparison and "-0,000483" in comparison
    metrics_fixture["classification"]["test"]["confusion_count"] = [
        [100000, 1200000],
        [26617, 639478],
    ]
    sentence = _signal_sentence(metrics_fixture)
    assert "93,56%" in sentence and "baixa seletividade" in sentence and "sem pesos" in sentence
    metrics_fixture["municipal"]["test"]["band_coverage"] = 0.7733
    assert "77,33%" in _band_sentence(metrics_fixture) and "subcobertura" in _band_sentence(
        metrics_fixture
    )
    metrics_fixture["municipal"]["test"]["band_coverage"] = 0.95
    assert "subcobertura" not in _band_sentence(metrics_fixture)


def test_long_audit_rows_keep_identification_on_each_pdf_page(tmp_path, metrics_fixture):
    metrics_fixture["classification"]["selection"] = {
        "HistGradientBoostingClassifier.cross_validation": {
            f"fold_{i}_average_precision_weighted_long_scientific_name": "0.63719924415"
            for i in range(18)
        },
        "long_parameter_description": "long scientific parameter text " * 60,
    }
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metrics_fixture), encoding="utf-8")
    result = build_reports(
        source, tmp_path / "rendered", "https://github.com/kevinbds/tech-challenge-fase3"
    )
    reader = PdfReader(result["technical_pdf"])
    for page in reader.pages:
        text = page.extract_text()
        assert "TESTE" in text and "FIAP" in text and "Relatório técnico" in text
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "JSON da execução" in text
    assert (tmp_path / "rendered/TEST_audit-details.json").read_bytes() == source.read_bytes()
    assert "fold_17" in (tmp_path / "rendered/TEST_audit-details.html").read_text(encoding="utf-8")


def test_realistic_audit_volume_is_compact_and_lossless(tmp_path, metrics_fixture):
    for task in ("classification", "municipal"):
        metrics_fixture[task]["selection"] = {
            "cv_results": [
                {
                    "model": "HistGradientBoostingClassifier" + str(i),
                    "params": {"max_iter": 150, "learning_rate": 0.05},
                    "mean_training_score": 0.67,
                    "mean_score": 0.63,
                    "std_score": 0.02,
                    "fold_scores": [0.61 + fold / 1000 for fold in range(10)],
                    "fold_training_scores": [0.67] * 10,
                }
                for i in range(28)
            ]
        }
    metrics_fixture["coverage"] = {
        str(year): {
            f"UF{i:02d}": {"n": 5000, "eligible": 4900, "missing_context": 100, "rate": 0.98}
            for i in range(50)
        }
        for year in (2023, 2024, 2025)
    }
    metrics_fixture["classification"]["subgroups"] = [
        {
            "dimension": "UF",
            "group": f"UF{i}",
            "n": 800,
            "average_precision": 0.63,
            "recall": 0.8,
            "brier": 0.19,
            **{f"extra_metric_{j}": j / 100 for j in range(30)},
        }
        for i in range(34)
    ]
    metrics_fixture["interpretation"]["shap"] = {
        "status": "completed",
        "importance": [{"feature": f"feature_{i}", "mean_abs_shap": i / 100} for i in range(30)],
    }
    metrics_fixture["interpretation"]["clusters"]["profiles"] = [
        {
            "cluster": i,
            "municipalities": 100,
            "taxa_alfabetizacao": 55 + i,
            "densidade_2022": 40 + i,
        }
        for i in range(8)
    ]
    source = tmp_path / "metrics.large.synthetic.json"
    source.write_text(json.dumps(metrics_fixture), encoding="utf-8")
    result = build_reports(
        source, tmp_path / "rendered", "https://github.com/kevinbds/tech-challenge-fase3"
    )
    pages = PdfReader(result["technical_pdf"]).pages
    assert len(pages) < 40
    text = "\n".join(page.extract_text() for page in pages)
    assert all("TESTE" in page.extract_text() for page in pages)
    assert "HistGradientBoostingClassifier27" in "".join(text.split()) and "UF33" in text
    assert "126 itens adicionais" in text and "6 itens adicionais" in text
    assert "Treino · AP pond." in text and "Treino · MAE (p.p.)" in " ".join(text.split())
    assert (tmp_path / "rendered/TEST_audit-details.json").read_bytes() == source.read_bytes()
    audit = (tmp_path / "rendered/TEST_audit-details.csv").read_text(encoding="utf-8-sig")
    assert "extra_metric_29" in audit and "fold_training_scores[9]" in audit


def test_pdf_links_are_portable_and_assets_do_not_disclose_workspace(built_reports):
    output = Path(built_reports["technical_pdf"]).parent
    expected = {
        "https://github.com/kevinbds/tech-challenge-fase3/blob/main/reports/TEST_audit-details."
        + extension
        for extension in ("json", "csv", "html")
    }
    found = set()
    for pdf in output.glob("*.pdf"):
        reader = PdfReader(pdf)
        texts = [str(reader.metadata)]
        for page in reader.pages:
            texts.append(page.extract_text())
            for ref in page.get("/Annots", []):
                action = ref.get_object().get("/A", {})
                if action.get("/URI"):
                    uri = str(action["/URI"])
                    assert uri.startswith("https://"), uri
                    found.add(uri)
        combined = "\n".join(texts).lower()
        assert "file://" not in combined and "workspace-fiap" not in combined
        assert "c:/" not in combined and "c:\\" not in combined
    assert expected.issubset(found)
    offline = (output / "TEST_RELATORIO_TECNICO_FASE3.html").read_text(encoding="utf-8")
    assert "href='TEST_audit-details.json'" in offline
    assert "/blob/main/reports/TEST_audit-details" not in offline
    for svg in output.rglob("*.svg"):
        text = svg.read_text(encoding="utf-8").lower()
        assert "file://" not in text and "workspace-fiap" not in text
        assert "c:/" not in text and "c:\\" not in text


def test_deck_pdf_keeps_metrics_in_three_columns(built_reports):
    reader = PdfReader(built_reports["deck_pdf"])
    assert len(reader.pages) == 10
    assert float(reader.pages[5].mediabox.width) == pytest.approx(1440)
    assert float(reader.pages[5].mediabox.height) == pytest.approx(810)
    positions = {}

    def capture(text, matrix, text_matrix, *_):
        if text.strip() in {"80,0%", "60,0%", "0,310"}:
            x, y = text_matrix[4:6]
            positions[text.strip()] = (
                matrix[0] * x + matrix[2] * y + matrix[4],
                matrix[1] * x + matrix[3] * y + matrix[5],
            )

    reader.pages[5].extract_text(visitor_text=capture)
    assert set(positions) == {"80,0%", "60,0%", "0,310"}
    columns = [positions[value] for value in ("80,0%", "60,0%", "0,310")]
    assert all(right[0] - left[0] > 200 for left, right in zip(columns, columns[1:], strict=False))
    assert max(y for _, y in columns) - min(y for _, y in columns) < 1


@pytest.fixture
def metrics_fixture() -> dict:
    return {
        "run_id": "SYNTHETIC-FIXTURE-ONLY",
        "protocol_hash": "a" * 64,
        "input_manifest_hash": "b" * 64,
        "status": "smoke_completed",
        "data_kind": "synthetic_fixture",
        "classification": {
            "selected_model": "LogisticRegression",
            "threshold": 0.31,
            "population": {"development": 600, "calibration": 150, "test": 400},
            "test": {
                "average_precision": 0.63,
                "roc_auc": 0.75,
                "precision": 0.6,
                "recall": 0.8,
                "f1": 0.686,
                "f2": 0.75,
                "brier": 0.19,
                "confusion_count": [[160, 80], [32, 128]],
                "calibration_curve": [
                    {
                        "lower": 0,
                        "upper": 0.5,
                        "predicted": 0.2,
                        "observed": 0.23,
                        "weight": 800,
                        "count": 200,
                    },
                    {
                        "lower": 0.5,
                        "upper": 1,
                        "predicted": 0.8,
                        "observed": 0.79,
                        "weight": 500,
                        "count": 200,
                    },
                ],
            },
            "baselines": {
                "contextual": {
                    "average_precision": 0.61,
                    "recall": 0.7,
                    "precision": 0.58,
                    "f1": 0.63,
                    "f2": 0.67,
                    "roc_auc": 0.73,
                    "brier": 0.2,
                },
                "prevalence": {"average_precision": 0.4},
            },
        },
        "municipal": {
            "selected_model": "Ridge",
            "population": {"development": 80, "reserve": 20, "test": 100},
            "test": {
                "mae": 4.25,
                "rmse": 5.7,
                "r2": 0.51,
                "band_coverage": 0.87,
                "band_mean_width": 14.2,
            },
            "baselines": {"persistence": {"mae": 4.8}},
            "forecast_rows": [
                {
                    "id_municipio": "0000001",
                    "sigla_uf": "XX",
                    "ano": 2026,
                    "forecast": 61.2,
                    "meta_limite": 70,
                    "meta_operador": ">=",
                    "gap": -8.8,
                }
            ],
        },
        "coverage": {"synthetic_fixture": True},
        "interpretation": {
            "permutation_importance": [
                {
                    "feature": "taxa_alfabetizacao_lag1",
                    "importance_mean": 0.03,
                    "importance_std": 0.002,
                }
            ],
            "clusters": {
                "status": "completed",
                "selected_k": 3,
                "silhouette": 0.42,
                "stability_ari": 0.81,
            },
        },
        "artifacts": {},
        "limitations": ["Fixture de teste; não representa o desempenho do projeto."],
    }


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("reports-synthetic")


@pytest.fixture(scope="module")
def built_reports(artifact_dir) -> dict:
    # A fixed fixture file allows the real browser/PDF scenario to run once.
    fixture = metrics_fixture.__wrapped__()
    source = artifact_dir / "metrics.synthetic.json"
    source.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    return build_reports(
        source,
        artifact_dir / "rendered",
        "https://github.com/kevinbds/tech-challenge-fase3",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(status="running"),
        lambda data: data.update(protocol_hash="missing"),
        lambda data: data["classification"]["test"].update(average_precision=None),
        lambda data: data["classification"]["test"].update(average_precision=1.1),
        lambda data: data["municipal"]["test"].update(mae=-1),
        lambda data: data["classification"].update(student_predictions=[{"score": 0.2}]),
        lambda data: data["artifacts"].update(
            nested={"id_aluno": "synthetic-CANARY_PRIVATE_RECORD"}
        ),
    ],
)
def test_invalid_scientific_or_private_input_fails_before_writing(
    tmp_path, metrics_fixture, mutation
):
    mutation(metrics_fixture)
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metrics_fixture), encoding="utf-8")
    target = tmp_path / "not-created"
    with pytest.raises(ValueError):
        build_reports(source, target, "https://github.com/kevinbds/tech-challenge-fase3")
    assert not target.exists()


def test_nonfinite_json_is_rejected(tmp_path, metrics_fixture):
    metrics_fixture["classification"]["test"]["brier"] = float("nan")
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metrics_fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        _read_metrics(source)


def test_baseline_winning_is_reported_honestly(metrics_fixture):
    metrics_fixture["classification"]["test"]["average_precision"] = 0.5
    sentence = _comparison_sentence(metrics_fixture)
    assert "menor que a do baseline contextual" in sentence
    assert "0,110" in sentence


def test_fixture_outputs_are_marked_on_every_pdf_page(built_reports):
    for field in ("deck_pdf", "technical_pdf"):
        path = Path(built_reports[field])
        assert path.name.startswith("TEST_")
        pages = PdfReader(path).pages
        assert len(pages) > 0
        assert all("TESTE" in page.extract_text() for page in pages)
    assert len(PdfReader(built_reports["deck_pdf"]).pages) == 10


def test_manifest_hashes_match_each_generated_artifact(built_reports):
    path = Path(built_reports["manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["synthetic_fixture"] is True
    for artifact in manifest["artifacts"]:
        target = path.parent / artifact["path"]
        assert target.is_file() and target.stat().st_size > 0
        assert artifact["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
        assert artifact["run_id"] == "SYNTHETIC-FIXTURE-ONLY"


def test_export_contains_only_the_public_documents(built_reports):
    assert set(built_reports) == {
        "manifest",
        "deck_html",
        "deck_pdf",
        "technical_pdf",
        "synthetic_fixture",
    }
    output = Path(built_reports["manifest"]).parent
    assert {path.name for path in output.glob("*.pdf")} == {
        "TEST_APRESENTACAO_FASE3.pdf",
        "TEST_RELATORIO_TECNICO_FASE3.pdf",
    }
    assert {path.name for path in output.glob("*.html")} == {
        "TEST_APRESENTACAO_FASE3.html",
        "TEST_RELATORIO_TECNICO_FASE3.html",
        "TEST_audit-details.html",
    }


def test_bad_public_urls_are_rejected(tmp_path, metrics_fixture):
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metrics_fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        build_reports(source, tmp_path / "bad", "https://example.com/fake")
    with pytest.raises(ValueError, match="credentials"):
        build_reports(
            source,
            tmp_path / "bad",
            "https://secret:token@github.com/kevinbds/tech-challenge-fase3",
        )


def test_html_values_are_escaped(metrics_fixture):
    from alfabetizacao_ml.visualization.reports import _technical

    data = copy.deepcopy(metrics_fixture)
    data["_synthetic"] = True
    data["classification"]["selected_model"] = "<img src=x onerror=alert(1)>"
    source = _technical(data, {}, "https://github.com/kevinbds/tech-challenge-fase3")
    assert "<img src=x" not in source
    assert "&lt;img src=x" in source
