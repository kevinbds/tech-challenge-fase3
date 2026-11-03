"""Scientific reports and presentation from aggregate evaluation results."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import re
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SOURCE_NOTE = "Fontes: INEP AEEB 2023–2025; metas oficiais; INSE 2021; IBGE Censo 2022."
RISK_NOTE = "Risco = não alfabetizado. Predição contextual; não é diagnóstico individual."
DECK_AUTHOR = "Kevin Barreto da Silva · RM 374637"
FORBIDDEN_RECORD_KEYS = {
    "id_aluno",
    "id_escola",
    "student_id",
    "student_predictions",
    "student_records",
}
FEATURE_LABELS = {
    "taxa_alfabetizacao_lag1": "Alfabetização no ano anterior",
    "media_portugues_lag1": "Média de português no ano anterior",
    "media_inse_2021": "Nível socioeconômico municipal (2021)",
    "populacao_2022": "População municipal (2022)",
    "log_populacao_2022": "Logaritmo da população (2022)",
    "densidade_2022": "Densidade demográfica (2022)",
    "sigla_uf": "Unidade da Federação",
    "regiao": "Região",
    "rede": "Rede de ensino",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value: Any, places: int = 3, suffix: str = "") -> str:
    number = _number(value)
    if number is None:
        return "Não disponível"
    return f"{number:,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".") + suffix


def _pct(value: Any) -> str:
    number = _number(value)
    return _fmt(number * 100 if number is not None else None, 1, "%")


def _reject_individuals(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN_RECORD_KEYS.intersection(str(key).lower() for key in value):
            raise ValueError("Report input contains a forbidden individual-record field")
        for nested in value.values():
            _reject_individuals(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_individuals(nested)


def _valid_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{label} must be a public HTTPS URL without credentials")
    if parsed.hostname in {"example.com", "localhost", "127.0.0.1"}:
        raise ValueError(f"{label} cannot be a placeholder URL")
    return value


def _read_metrics(path: Path) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(data, dict):
        raise ValueError("Metrics must be a JSON object")
    _reject_individuals(data)
    if data.get("status") not in {"completed", "smoke_completed"}:
        raise ValueError("Reports require a completed scientific run or an explicit smoke fixture")
    if not isinstance(data.get("run_id"), str) or not data["run_id"].strip():
        raise ValueError("Missing run_id")
    for field in ("protocol_hash", "input_manifest_hash"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(data.get(field, ""))):
            raise ValueError(f"Missing or invalid {field}")
    for section, metric in (("classification", "average_precision"), ("municipal", "mae")):
        component = data.get(section, {})
        if not isinstance(component, dict) or not isinstance(component.get("test"), dict):
            raise ValueError(f"Missing {section}.test")
        if not isinstance(component.get("selected_model"), str) or not component["selected_model"]:
            raise ValueError(f"Missing {section}.selected_model")
        if _number(component["test"].get(metric)) is None:
            raise ValueError(f"Missing finite metric {section}.test.{metric}")
    ap = data["classification"]["test"]["average_precision"]
    if not 0 <= ap <= 1 or data["municipal"]["test"]["mae"] < 0:
        raise ValueError("Primary metrics are outside their valid ranges")
    data["_synthetic"] = (
        data["status"] == "smoke_completed"
        or data.get("data_kind") in {"synthetic_fixture", "synthetic", "test"}
        or data.get("synthetic_fixture") is True
    )
    return data


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th scope='col'>{_escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _metric_strip(items: list[tuple[str, str]]) -> str:
    return (
        "<div class='metric-strip'>"
        + "".join(
            f"<div><b>{_escape(value)}</b><span>{_escape(label)}</span></div>"
            for value, label in items
        )
        + "</div>"
    )


def _fixture_mark(data: dict[str, Any]) -> str:
    return (
        "<div class='fixture-mark'>TESTE — DADOS SINTÉTICOS — NÃO É RESULTADO ACADÊMICO</div>"
        if data["_synthetic"]
        else ""
    )


def _chart_html(charts: dict[str, str], key: str, alt: str) -> str:
    if key not in charts:
        return "<p class='small'>Análise não disponível nesta execução; nenhuma evidência foi substituída por valores simulados.</p>"
    return f"<img class='chart' src='{_escape(charts[key])}' alt='{_escape(alt)}'>"


def _charts(data: dict[str, Any], output: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    target = output / "images"
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#111827",
            "text.color": "#111827",
        }
    )

    def save(fig: Any, key: str, note: str) -> None:
        fig.text(0.01, 0.01, note, fontsize=9, color="#5e6675")
        if data["_synthetic"]:
            fig.text(
                0.99,
                0.99,
                "TESTE — DADOS SINTÉTICOS",
                ha="right",
                va="top",
                color="#9b2800",
                fontsize=10,
            )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        for extension in ("png", "svg"):
            fig.savefig(target / f"{key}.{extension}", dpi=160, facecolor="#f7f5f0")
        plt.close(fig)
        paths[key] = f"images/{key}.png"

    classification = data["classification"]
    candidates = [("Modelo selecionado", classification["test"].get("average_precision"))]
    for key, label in (("contextual", "Contexto anterior"), ("prevalence", "Prevalência")):
        candidates.append(
            (label, classification.get("baselines", {}).get(key, {}).get("average_precision"))
        )
    candidates = [(name, float(value)) for name, value in candidates if _number(value) is not None]
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.barh(
        [name for name, _ in candidates][::-1],
        [value for _, value in candidates][::-1],
        color=["#aab5c5"] * (len(candidates) - 1) + ["#1769e0"],
    )
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Average precision ponderada · risco = não alfabetizado")
    for index, (_, value) in enumerate(candidates[::-1]):
        ax.text(value + 0.02, index, _fmt(value), va="center")
    save(
        fig,
        "classification_comparison",
        "Teste 2025. Mesma população e pesos em todas as comparações disponíveis.",
    )

    municipal = data["municipal"]
    candidates = [
        ("Modelo selecionado", municipal["test"].get("mae")),
        ("Persistência", municipal.get("baselines", {}).get("persistence", {}).get("mae")),
    ]
    candidates = [(name, float(value)) for name, value in candidates if _number(value) is not None]
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.barh(
        [name for name, _ in candidates][::-1],
        [value for _, value in candidates][::-1],
        color=["#aab5c5"] * (len(candidates) - 1) + ["#1769e0"],
    )
    maximum = max(value for _, value in candidates)
    ax.set_xlim(0, max(1, maximum * 1.3))
    ax.set_xlabel("Erro absoluto médio · pontos percentuais · menor é melhor")
    for index, (_, value) in enumerate(candidates[::-1]):
        ax.text(value + max(0.03, maximum * 0.025), index, _fmt(value, 2), va="center")
    save(fig, "municipal_comparison", "Teste 2025. Rede municipal; peso igual por município.")

    matrix = classification["test"].get("confusion_count")
    if (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        array = np.asarray(matrix, dtype=float)
        if np.isfinite(array).all() and (array >= 0).all():
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            ax.imshow(array, cmap="Blues", vmin=0)
            for row in range(2):
                for column in range(2):
                    ax.text(
                        column,
                        row,
                        _fmt(array[row, column], 0),
                        ha="center",
                        va="center",
                        color="white" if array[row, column] > array.max() * 0.55 else "#111827",
                        fontsize=19,
                    )
            labels = ["Alfabetizado", "Não alfabetizado"]
            ax.set_xticks([0, 1], labels)
            ax.set_yticks([0, 1], labels)
            ax.set_xlabel("Classe prevista")
            ax.set_ylabel("Classe observada")
            save(
                fig,
                "confusion_counts",
                "Contagem de registros; não representa massa ponderada. Limiar selecionado na reserva de 2024.",
            )

    curve = classification["test"].get("calibration_curve", [])
    points = [
        point
        for point in curve
        if isinstance(point, dict)
        and _number(point.get("predicted")) is not None
        and _number(point.get("observed")) is not None
    ]
    if points:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        ax.plot([0, 1], [0, 1], "--", color="#aab5c5", label="Referência ideal")
        ax.plot(
            [point["predicted"] for point in points],
            [point["observed"] for point in points],
            "o-",
            color="#1769e0",
            label="Modelo",
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Probabilidade média prevista",
            ylabel="Frequência observada ponderada",
        )
        ax.legend(frameon=False)
        save(
            fig,
            "weighted_calibration",
            "Teste 2025. Médias ponderadas por bin; calibração ajustada somente na reserva de 2024.",
        )

    importance = data.get("interpretation", {}).get("permutation_importance", [])
    items = [
        item
        for item in importance
        if isinstance(item, dict) and _number(item.get("importance_mean")) is not None
    ]
    if items:
        items = sorted(items, key=lambda item: item["importance_mean"], reverse=True)[:8]
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        ax.barh(
            [
                FEATURE_LABELS.get(item.get("feature"), str(item.get("feature", "Atributo")))
                for item in items
            ][::-1],
            [item["importance_mean"] for item in items][::-1],
            color="#1769e0",
        )
        ax.axvline(0, color="#5e6675", linewidth=0.8)
        ax.set_xlabel("Queda de average precision ponderada")
        save(
            fig,
            "context_importance",
            "Importância preditiva marginal; não estima efeito causal de intervenções.",
        )
    return paths


def _ranking(data: dict[str, Any], count: int = 5) -> list[list[str]]:
    rows = data["municipal"].get("forecast_rows", [])
    valid = [row for row in rows if isinstance(row, dict) and _number(row.get("gap")) is not None]
    valid.sort(key=lambda row: row["gap"])
    return [
        [
            str(row.get("id_municipio", "")),
            str(row.get("sigla_uf", "")),
            _fmt(row.get("forecast"), 1, "%"),
            str(row.get("meta_operador", "≥")) + " " + _fmt(row.get("meta_limite"), 1, "%"),
            _fmt(row.get("gap"), 1, " p.p."),
        ]
        for row in valid[:count]
    ]


def _comparison_sentence(data: dict[str, Any]) -> str:
    current = data["classification"]["test"]["average_precision"]
    baseline = _number(
        data["classification"].get("baselines", {}).get("contextual", {}).get("average_precision")
    )
    if baseline is None:
        return "A métrica do baseline contextual não está disponível nesta execução."
    delta = current - baseline
    interval = data["classification"].get("bootstrap", {}).get("ap_delta_vs_contextual_ci95")
    if (
        isinstance(interval, list)
        and len(interval) == 2
        and all(_number(value) is not None for value in interval)
        and interval[0] <= 0 <= interval[1]
    ):
        return f"AP semelhante ao baseline contextual; não houve ganho demonstrado. Modelo: {_fmt(current, 4)}; baseline: {_fmt(baseline, 4)}. Diferença observada: {_fmt(delta, 6)}; IC 95%: [{_fmt(interval[0], 6)}; {_fmt(interval[1], 6)}]."
    if delta > 0:
        return f"A AP observada do modelo foi maior que a do baseline contextual em {_fmt(delta, 6)} neste teste."
    if delta < 0:
        return f"A AP observada do modelo foi menor que a do baseline contextual em {_fmt(abs(delta), 6)} neste teste."
    return "Modelo e baseline contextual tiveram a mesma average precision no teste."


def _signal_sentence(data: dict[str, Any]) -> str:
    matrix = data["classification"]["test"].get("confusion_count")
    if not matrix or sum(map(sum, matrix)) == 0:
        return "Fração de registros sinalizados não disponível nesta execução."
    fraction = (matrix[0][1] + matrix[1][1]) / sum(map(sum, matrix))
    return (
        f"O limiar F2 sinaliza {_fmt(fraction * 100, 2, '%')} dos registros (contagem sem pesos)."
        + (
            " A baixa seletividade exige avaliar a capacidade de atendimento."
            if fraction >= 0.8
            else " A capacidade de atendimento precisa ser avaliada."
        )
    )


def _band_sentence(data: dict[str, Any]) -> str:
    coverage = _number(data["municipal"]["test"].get("band_coverage"))
    if coverage is None:
        return "Cobertura das bandas não disponível nesta execução."
    return (
        f"A banda dos quantis 5–95 tem cobertura observada de {_fmt(coverage * 100, 2, '%')} frente ao nominal exploratório de 90%."
        + (
            " Há subcobertura no teste; a banda não garante proteção futura."
            if coverage < 0.9
            else " A cobertura observada não garante proteção futura."
        )
    )


def _slides(data: dict[str, Any], charts: dict[str, str]) -> list[dict[str, str]]:
    classification, municipal = data["classification"], data["municipal"]
    test = classification["test"]
    importance = data.get("interpretation", {}).get("permutation_importance", [])
    valid_importance = [
        item
        for item in importance
        if isinstance(item, dict) and _number(item.get("importance_mean")) is not None
    ]
    top = sorted(valid_importance, key=lambda item: item["importance_mean"], reverse=True)[:3]
    top_text = ", ".join(
        FEATURE_LABELS.get(item.get("feature"), str(item.get("feature"))) for item in top
    )
    ranking = _ranking(data)
    ranking_body = (
        _table(["Código IBGE", "UF", "Projeção 2026", "Meta", "Diferença"], ranking)
        if ranking
        else "<p>Não há projeções municipais com meta e diferença disponíveis nesta execução.</p>"
    )
    clusters = data.get("interpretation", {}).get("clusters", {})
    cluster_body = _metric_strip(
        [
            (_fmt(clusters.get("selected_k"), 0), "Grupos descritivos"),
            (_fmt(clusters.get("silhouette")), "Silhouette"),
            (_fmt(clusters.get("stability_ari")), "Estabilidade ARI"),
        ]
    )
    cluster_body += "<p class='small'>Grupos resumem semelhanças entre territórios. A escolha de ações exige investigação local.</p>"
    if clusters.get("status") not in {"completed", "available", "ok"}:
        cluster_body += f"<p class='caption'>Status: {_escape(clusters.get('status', 'não disponível'))}. {_escape(clusters.get('reason', ''))}</p>"
    return [
        {
            "title": "Alfabetização no Brasil: contexto e metas",
            "eyebrow": "FIAP · Tech Challenge · Fase 3",
            "body": "<p class='lead'>Dados oficiais, validação temporal e contexto territorial para apoiar decisões educacionais.</p><p>Uma análise de risco por aluno e de metas por município.</p>",
            "source": RISK_NOTE,
            "theme": "dark",
        },
        {
            "title": "Dados e perguntas da análise",
            "eyebrow": "01 · Problema educacional",
            "body": "<div class='track'><div><b>2023–2025</b><span>Avaliações estaduais de alfabetização</span></div><div><b>2021 / 2022</b><span>Contexto socioeconômico e populacional</span></div><div><b>Metas</b><span>Comparação na rede e no ano corretos</span></div></div><p>Quais contextos concentram risco e quais municípios merecem análise prioritária?</p>",
            "source": SOURCE_NOTE,
        },
        {
            "title": "Treino em 2024, teste em 2025",
            "eyebrow": "02 · Validação",
            "body": "<div class='track'><div><b>2023 → 2024</b><span>Contexto anterior e desenvolvimento</span></div><div><b>80% / 20%</b><span>Municípios para ajuste e reserva</span></div><div><b>2024 → 2025</b><span>Contexto anterior e teste final</span></div></div><p>O teste não escolhe o modelo, os atributos ou o limiar.</p>",
            "source": "Backtest retrospectivo com as versões disponíveis das fontes; sua disponibilidade nas datas históricas não foi verificada.",
        },
        {
            "title": "Atributos usados na previsão",
            "eyebrow": "03 · Contexto e interpretação",
            "body": "<ul><li>Histórico municipal do ano anterior.</li><li>Rede, território e contexto socioeconômico.</li><li>População e densidade demográfica.</li></ul><p class='small'>Alunos com os mesmos atributos podem receber o mesmo risco.</p>"
            + (
                f"<p class='caption'>Maior importância nesta execução: {_escape(top_text)}.</p>"
                if top_text
                else ""
            ),
            "source": "Proficiência individual, respostas e resultados contemporâneos ficam fora dos atributos preditivos.",
        },
        {
            "title": "Classificação: comparação com o baseline",
            "eyebrow": "04 · Resultado da classificação",
            "body": _chart_html(
                charts,
                "classification_comparison",
                "Average precision ponderada do modelo e dos baselines no teste 2025",
            )
            + f"<p class='small'>{_escape(_comparison_sentence(data))}</p>",
            "source": f"Modelo selecionado na validação: {classification['selected_model']}. Maior average precision é melhor.",
        },
        {
            "title": "Recall, precisão e seletividade",
            "eyebrow": "05 · Risco e capacidade de atendimento",
            "body": _metric_strip(
                [
                    (_pct(test.get("recall")), "Recall de não alfabetizados"),
                    (_pct(test.get("precision")), "Precisão das sinalizações"),
                    (_fmt(classification.get("threshold")), "Limiar escolhido na reserva"),
                ]
            )
            + f"<p>{_escape(_signal_sentence(data))}</p><p class='small'>Recall e precisão ponderados descrevem erros diferentes; o limiar F2 não demonstra benefício de uma intervenção.</p>",
            "source": "Teste 2025; métricas ponderadas. A regra de limiar foi fixada antes de avaliar o teste.",
        },
        {
            "title": "Previsão das taxas municipais",
            "eyebrow": "06 · Modelo municipal",
            "body": _chart_html(
                charts,
                "municipal_comparison",
                "MAE do modelo municipal e da persistência em pontos percentuais",
            )
            + f"<p class='small'>Modelo municipal: {_escape(municipal['selected_model'])}. O erro é medido em pontos percentuais, com o mesmo peso para cada município.</p>",
            "source": "Rede municipal; uma linha por município e ano. Probabilidade individual não é taxa municipal prevista.",
        },
        {
            "title": "Municípios com maior déficit projetado",
            "eyebrow": "07 · Projeção municipal",
            "body": ranking_body
            + "<p class='small'>A diferença entre projeção e meta organiza os municípios para investigação e conferência local.</p>",
            "source": "Projeção 2026 após reajuste. As métricas de teste 2025 pertencem ao modelo avaliado, preservado separadamente.",
        },
        {
            "title": "Perfis territoriais semelhantes",
            "eyebrow": "08 · Padrões descritivos",
            "body": cluster_body,
            "source": "Clusterização descritiva após congelar a avaliação. Semelhança não demonstra efeito causal.",
        },
        {
            "title": "Uso dos resultados na gestão",
            "eyebrow": "09 · Valor estratégico",
            "body": "<ul><li>Priorizar a análise dos territórios sinalizados.</li><li>Validar explicações com gestores e profissionais locais.</li><li>Acompanhar cobertura, calibração e mudanças de contexto.</li></ul><p class='small'>O modelo apoia a investigação. O efeito de uma intervenção precisa ser medido.</p>",
            "source": "2027–2030: cenários explícitos frente às metas, sem garantia de desempenho futuro. "
            + RISK_NOTE,
            "theme": "dark",
        },
    ]


def _document(title: str, body: str, css: str, *, script: str = "", body_class: str = "") -> str:
    return f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{_escape(title)}</title><style>{css}</style></head><body class='{_escape(body_class)}'>{body}{f'<script>{script}</script>' if script else ''}</body></html>"


def _deck(data: dict[str, Any], slides: list[dict[str, str]], css: str, javascript: str) -> str:
    sections = []
    for index, slide in enumerate(slides, 1):
        sections.append(
            f"<section id='slide-{index}' class='slide {_escape(slide.get('theme', ''))} {'active' if index == 1 else ''}' aria-labelledby='title-{index}' aria-hidden='{'false' if index == 1 else 'true'}'>{_fixture_mark(data)}<p class='eyebrow' data-evidence>{_escape(slide['eyebrow'])}</p><span class='slide-number'>{index:02d} / {len(slides):02d}</span><h1 id='title-{index}' tabindex='-1' data-evidence>{_escape(slide['title'])}</h1><div class='slide-body' data-evidence>{slide['body']}</div><p class='source' data-evidence>{_escape(slide['source'])}</p><p class='author' data-evidence>{_escape(DECK_AUTHOR)}</p></section>"
        )
    controls = "<nav class='toolbar' aria-label='Controles da apresentação'><p>Alfabetização · Fase 3</p><div><button id='previous' type='button'>Anterior</button> <span id='position' aria-live='polite'>1 / 10</span> <button id='next' type='button'>Próximo</button> <button id='fullscreen' type='button'>Tela cheia</button></div></nav>"
    return _document(
        "Alfabetização — apresentação executiva",
        controls
        + "<main class='stage-shell'><div class='stage'>"
        + "".join(sections)
        + "</div></main>",
        css,
        script=javascript,
    )


def _compact(value: Any, limit: int = 70) -> str:
    if value is None:
        return "não disponível"
    text = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(value, (dict, list))
        else str(value)
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _audit_fields(value: Any, path: str = "") -> list[list[Any]]:
    if isinstance(value, dict) and value:
        return [
            row
            for key, item in value.items()
            for row in _audit_fields(item, f"{path}.{key}".strip("."))
        ]
    if isinstance(value, list) and value:
        return [
            row
            for index, item in enumerate(value)
            for row in _audit_fields(item, f"{path}[{index}]")
        ]
    return [[path, value]]


def _audit_summaries(data: dict[str, Any]) -> list[tuple]:
    c, m, interpretation = data["classification"], data["municipal"], data.get("interpretation", {})
    # Group coverage scalars by their parent record (e.g. year/UF), never one PDF row per leaf.
    coverage = {}
    for path, value in _audit_fields(data.get("coverage", {})):
        parent, _, field = path.rpartition(".")
        coverage.setdefault(parent or "cobertura", {})[field or path] = value
    result = [
        (
            "Cobertura por registro de ano/UF",
            ["Ano / UF / registro", "Contagens e cobertura"],
            [[key, value] for key, value in coverage.items()],
            24,
        )
    ]
    for name, section in (("Classificação", c), ("Municipal", m)):
        candidates = section.get("selection", {}).get("cv_results", [])
        result.append(
            (
                f"CV — {name}",
                [
                    "Candidato",
                    "Parâmetros",
                    "Treino · AP pond." if name == "Classificação" else "Treino · MAE (p.p.)",
                    "Validação · AP pond." if name == "Classificação" else "Validação · MAE (p.p.)",
                    "DP · AP pond." if name == "Classificação" else "DP · MAE (p.p.)",
                ],
                [
                    [
                        row.get("model"),
                        row.get("params"),
                        row.get("mean_training_score"),
                        row.get("mean_score"),
                        row.get("std_score"),
                    ]
                    for row in candidates
                ],
                32,
            )
        )
    result.append(
        (
            "Subgrupos",
            ["Dimensão / grupo", "n", "AP", "Recall", "Brier"],
            [
                [
                    str(row.get("dimension")) + " / " + str(row.get("group")),
                    row.get("n"),
                    row.get("average_precision"),
                    row.get("recall"),
                    row.get("brier"),
                ]
                for row in c.get("subgroups", [])
            ],
            40,
        )
    )
    result.append(
        (
            "Incerteza bootstrap",
            ["Medida", "Valor / intervalo"],
            [[key, value] for key, value in c.get("bootstrap", {}).items()],
            8,
        )
    )
    shap = interpretation.get("shap", {})
    result.append(
        (
            "SHAP — " + str(shap.get("status", "não disponível")),
            ["Atributo", "Média |SHAP|"],
            [[row.get("feature"), row.get("mean_abs_shap")] for row in shap.get("importance", [])],
            24,
        )
    )
    clusters = interpretation.get("clusters", {})
    result.append(
        (
            "Clusters — perfis",
            ["Cluster", "Municípios", "Perfil agregado"],
            [
                [
                    row.get("cluster"),
                    row.get("municipalities"),
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("cluster", "municipalities")
                    },
                ]
                for row in clusters.get("profiles", [])
            ],
            16,
        )
    )
    result.append(
        (
            "Experimento de vazamento",
            ["Medida", "Valor"],
            [[key, value] for key, value in interpretation.get("leakage_demo", {}).items()],
            8,
        )
    )
    return result


def _technical(data: dict[str, Any], charts: dict[str, str], repo_url: str) -> str:
    c, m = data["classification"], data["municipal"]
    pages: list[tuple[str, str]] = []
    pages.append(
        (
            "Predição contextual de alfabetização",
            "<p class='lead'>Classificação por aluno, análise municipal e interpretação dos resultados para apoiar a investigação educacional.</p>"
            + _metric_strip(
                [
                    (
                        _fmt(c["test"].get("average_precision")),
                        "Average precision ponderada · teste 2025",
                    ),
                    (_fmt(m["test"].get("mae"), 2, " p.p."), "MAE municipal · teste 2025"),
                ]
            )
            + f"<p>{_escape(_comparison_sentence(data))}</p><div class='callout'>{_escape(RISK_NOTE)}</div><h3>Continuidade do projeto</h3><p>A fase 3 reconstrói localmente a camada Gold, seguindo a linhagem e os contratos da fase 2 e incorporando o histórico necessário à análise.</p><h3>Reprodução da análise</h3><p>As configurações, as fontes e os resultados completos acompanham o repositório, identificados pela execução {_escape(data['run_id'])}.</p>",
        )
    )
    populations = []
    for label, section, keys in (
        ("Alunos", c, ("development", "calibration", "test")),
        ("Municípios", m, ("development", "reserve", "test")),
    ):
        for key in keys:
            value = section.get("population", {}).get(key)
            if isinstance(value, dict):
                value = value.get("n", value.get("rows", value.get("count")))
            populations.append([label, key, _fmt(value, 0)])
    pages.append(
        (
            "Dados, cobertura e granularidade",
            "<p>As avaliações estaduais de 2023–2025 oferecem rótulos de alfabetização do segundo ano. O contexto reúne indicadores municipais anteriores, INSE 2021 e população/densidade do Censo 2022.</p>"
            + _table(["Unidade", "Conjunto", "Registros"], populations)
            + "<h3>Regras de elegibilidade</h3><p>Alunos do segundo ano, redes estadual e municipal, rótulo oficial válido e peso finito positivo. As contagens completas de filtros e perdas de contexto pertencem ao relatório de qualidade da execução.</p><h3>Ausências com significado</h3><p>Ausência estrutural em 2023, avaliação não realizada, supressão, erro de leitura e falta de correspondência são situações distintas. O identificador mascarado de escola não serve para cruzamento com o Censo Escolar.</p><p>Metas mantêm texto original, operador e limite. Uma meta estritamente maior que 80 não é atingida por resultado igual a 80.</p>",
        )
    )
    pages.append(
        (
            "Protocolo de validação",
            _table(
                ["Etapa", "Dados", "Decisão permitida"],
                [
                    [
                        "Desenvolvimento",
                        "2024 com contexto 2023",
                        "Atributos, modelos e hiperparâmetros",
                    ],
                    [
                        "Validação cruzada",
                        "Cinco folds por município, nos 80% de desenvolvimento",
                        "Comparação de candidatos",
                    ],
                    [
                        "Reserva",
                        "20% dos municípios de 2024",
                        "Calibração, limiar e resíduos municipais",
                    ],
                    ["Teste final", "2025 com contexto 2024", "Avaliação do protocolo congelado"],
                ],
            )
            + "<p>O pré-processamento integra o pipeline e aprende somente no treino. A classificação usa pesos no ajuste, seleção, calibração e métricas. O modelo municipal atribui o mesmo peso a cada município.</p><h3>Prevenção de vazamento</h3><p>Proficiência individual, respostas, gabaritos, presença, preenchimento, caderno, identificadores, resultados contemporâneos e as nove proporções indisponíveis em 2023 ficam fora dos atributos principais. A defasagem exige o ano anterior exato.</p><p>O backtest utiliza as versões atuais das fontes. Não foi verificado se essas mesmas versões estavam disponíveis no instante histórico da previsão.</p>",
        )
    )
    labels = [
        ("average_precision", "Average precision", False),
        ("roc_auc", "ROC-AUC", False),
        ("precision", "Precisão", True),
        ("recall", "Recall de risco", True),
        ("f1", "F1", False),
        ("f2", "F2", False),
        ("brier", "Brier", False),
    ]
    rows = [
        [
            label,
            _pct(c["test"].get(key)) if percentage else _fmt(c["test"].get(key)),
            _pct(c.get("baselines", {}).get("contextual", {}).get(key))
            if percentage
            else _fmt(c.get("baselines", {}).get("contextual", {}).get(key)),
        ]
        for key, label, percentage in labels
    ]
    pages.append(
        (
            "Classificação: resultados do teste",
            f"<p>Modelo selecionado: <strong>{_escape(c['selected_model'])}</strong>. Limiar: {_fmt(c.get('threshold'))}. Classe positiva: não alfabetizado.</p>"
            + _table(["Métrica", "Modelo", "Baseline contextual"], rows)
            + _chart_html(charts, "classification_comparison", "Comparação de average precision")
            + "<p class='caption'>Métricas ponderadas no teste temporal de 2025. Valores maiores de AP, ROC-AUC, F1 e F2 são melhores; para Brier, valores menores são melhores.</p>",
        )
    )
    pages.append(
        (
            "Erros e calibração",
            _chart_html(charts, "confusion_counts", "Matriz de confusão em contagens de registros")
            + _chart_html(charts, "weighted_calibration", "Diagrama de calibração ponderado")
            + f"<p class='caption'>{_escape(_signal_sentence(data))}</p>"
            + "<p class='caption'>A matriz em contagens não é uma matriz de massas ponderadas. Probabilidades calibradas não dispensam validação da capacidade de atendimento e dos custos de erro.</p>",
        )
    )
    rows = [
        ["MAE", _fmt(m["test"].get("mae"), 2, " p.p.")],
        ["RMSE", _fmt(m["test"].get("rmse"), 2, " p.p.")],
        ["R²", _fmt(m["test"].get("r2"))],
        ["Cobertura da banda exploratória", _pct(m["test"].get("band_coverage"))],
        ["Largura média da banda", _fmt(m["test"].get("band_mean_width"), 2, " p.p.")],
    ]
    pages.append(
        (
            "Municípios e metas",
            f"<p>Modelo selecionado: {_escape(m['selected_model'])}. Uma linha por município, ano e rede municipal; comparação com persistência.</p>"
            + _table(["Medida", "Teste 2025"], rows)
            + _chart_html(charts, "municipal_comparison", "Comparação de erro municipal")
            + f"<p class='caption'>{_escape(_band_sentence(data))}</p>"
            + "<p>As bandas usam resíduos da reserva, após congelar a seleção. A cobertura em 2025 é observada; não garante cobertura futura. Projeções de 2026 usam artefato reajustado distinto daquele avaliado em 2025.</p>",
        )
    )
    ranking = _ranking(data)
    pages.append(
        (
            "Interpretação e aplicação estratégica",
            _chart_html(charts, "context_importance", "Importância de atributos contextuais")
            + "<h3>Fatores relevantes</h3><p>Importância preditiva descreve a relação com o comportamento do modelo. Não estima o efeito causal de alterar uma política pública.</p><h3>Municípios para investigação</h3>"
            + (
                _table(["Código IBGE", "UF", "Projeção 2026", "Meta", "Diferença"], ranking)
                if ranking
                else "<p>Não há ranking de projeções com metas disponível nesta execução.</p>"
            )
            + "<p>Regiões semelhantes são examinadas pela clusterização descritiva. Priorizações precisam ser combinadas com conhecimento local, cobertura dos dados e avaliação das intervenções.</p>",
        )
    )
    defaults = [
        "Os resultados individuais refletem o contexto territorial e não constituem diagnóstico pedagógico.",
        "O backtest utiliza as versões disponíveis das fontes; sua disponibilidade nas datas históricas não foi verificada.",
        "Medianas e escalas foram ajustadas sem pesos no desenvolvimento. Treinamento, calibração e métricas individuais utilizam os pesos oficiais.",
        "Os intervalos descrevem a variabilidade entre municípios, condicionada aos dados observados; não reproduzem todo o desenho amostral oficial.",
        "A importância preditiva descreve associações e não estima efeitos causais de políticas públicas.",
        "As projeções de 2026 ainda não têm avaliação independente. Os cenários de 2027–2030 mantêm a taxa projetada para 2026 e não são previsões validadas desses anos.",
        "As bandas históricas não garantem a mesma cobertura em anos futuros.",
    ]
    source_notes = {
        "Individual classification uses shared territorial context; it is not an individual educational diagnosis.": defaults[
            0
        ],
        "Retrospective backtest uses currently available snapshots, not verified historical data vintages.": defaults[
            1
        ],
        "Preprocessing medians and scaling are unweighted, fit on development only; objectives and individual metrics use official weights.": defaults[
            2
        ],
        "Bootstrap describes conditional municipal variability, not the full official sampling design.": defaults[
            3
        ],
        "Model performance does not identify causal effects of educational interventions.": defaults[
            4
        ],
        "Refit2026 models have no independent 2026 outcome evaluation; historical intervals do not guarantee future coverage.": defaults[
            5
        ],
    }
    limitations = [
        source_notes.get(value, value)
        for value in data.get("limitations", [])
        if isinstance(value, str)
    ]
    limits = (
        "<ul>"
        + "".join(f"<li>{_escape(value)}</li>" for value in dict.fromkeys(defaults + limitations))
        + "</ul>"
    )
    pages.append(
        (
            "Limitações, reprodução e fontes",
            limits
            + f"<h3>Reprodução</h3><p>O repositório documenta o ambiente, os comandos, o manifest de fontes e as verificações. A reprodução confere os hashes das fontes. Atualizações das entradas geram uma nova versão dos resultados.</p><p><a href='{_escape(repo_url)}'>{_escape(repo_url)}</a></p><h3>Fontes</h3><p>{_escape(SOURCE_NOTE)}</p><p>Referências metodológicas: CRISP-DM e documentação das bibliotecas do projeto. O repositório reúne os contratos dos dados e o protocolo de avaliação.</p>",
        )
    )

    supplement = data.get("_scenarios")
    if supplement:
        prefix = "TEST_" if data["_synthetic"] else ""
        link = f"<p>Todos os municípios e a composição por UF estão em <a href='{prefix}scenarios.json'>scenarios.json</a> e <a href='{prefix}scenarios.csv'>scenarios.csv</a>, também incluídos no ZIP.</p>"
        pages.append(
            (
                "Cenário explícito · 2027–2030",
                f"<p>{_escape(supplement['scenario'])}</p>"
                + _table(
                    ["Ano", "Atingem", "Não atingem", "Sem meta", "Sem projeção"],
                    [
                        [r["ano"], r["atinge"], r["nao_atinge"], r["sem_meta"], r["sem_projecao"]]
                        for r in supplement["summary"]
                    ],
                )
                + "<p>Comparação sem arredondar: 80 não satisfaz uma meta estritamente maior que 80. Metas ausentes não são preenchidas. A população é limitada às linhas da projeção de 2026.</p>"
                + link,
            )
        )
        geography = supplement["cluster_geography"]
        region_rows = geography["by_region"]
        pages.append(
            (
                "Clusters · composição territorial",
                f"<p>{_escape(geography['interpretation'])}</p><p>{geography['assignment_count']} municípios atribuídos; {geography['missing_geography_count']} sem correspondência regional. Mostrados {min(12, len(region_rows))} de {len(region_rows)} registros cluster/região; restantes no JSON.</p>"
                + _table(
                    ["Cluster", "Região", "Municípios"],
                    [[r["cluster"], r["regiao"], r["municipalities"]] for r in region_rows[:12]],
                )
                + link,
            )
        )
        profiles = geography["profiles"]
        pages.append(
            (
                "Clusters · caracterização dos perfis",
                "<p>Valores agregados observados; atributos distinguem os perfis sem identificar efeitos de intervenções. Mostrados até 8 perfis; demais valores completos no JSON.</p>"
                + _table(
                    ["Cluster", "Municípios", "Perfil observado"],
                    [
                        [
                            r.get("cluster"),
                            r.get("municipalities"),
                            _compact(
                                {
                                    k: v
                                    for k, v in r.items()
                                    if k not in ("cluster", "municipalities")
                                },
                                110,
                            ),
                        ]
                        for r in profiles[:8]
                    ],
                )
                + link,
            )
        )
    prefix = "TEST_" if data["_synthetic"] else ""
    audit_link = f"<p>Detalhamento agregado integral: <a href='{prefix}audit-details.json'>JSON da execução</a>, <a href='{prefix}audit-details.csv'>CSV de campos</a> e <a href='{prefix}audit-details.html'>apêndice HTML externo</a>. Os mesmos arquivos acompanham o ZIP em reports/. O HTML usa arquivos locais; os links do PDF apontam para o repositório. Os anexos preservam os valores completos do arquivo de resultados.</p>"
    for title, headers, rows, limit in _audit_summaries(data):
        shown = rows[:limit]
        note = f"<p>{len(rows)} itens nesta seção; {len(shown)} exibidos; {max(0, len(rows) - len(shown))} itens adicionais somente no apêndice externo. Cada candidato ou grupo ocupa uma linha.</p>"
        for start in range(0, max(1, len(shown)), 8):
            compact = [[_compact(value) for value in row] for row in shown[start : start + 8]]
            pages.append(
                (
                    f"Síntese · {title} · {start // 8 + 1}",
                    note
                    + _table(
                        headers,
                        compact or [["Não disponível nesta execução"] + [""] * (len(headers) - 1)],
                    )
                    + audit_link,
                )
            )
    return (
        "<main class='document'>"
        + "".join(
            f"<section class='report-page'>{_fixture_mark(data)}<p class='eyebrow'>FIAP · Fase 3 · Relatório técnico · {index:02d}</p><h{1 if index == 1 else 2}>{_escape(title)}</h{1 if index == 1 else 2}>{body}</section>"
            for index, (title, body) in enumerate(pages, 1)
        )
        + "</main>"
    )


@contextmanager
def _browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        executable = os.environ.get("FASE3_BROWSER_EXECUTABLE")
        if executable:
            browser = playwright.chromium.launch(headless=True, executable_path=executable)
        elif (
            os.name == "nt"
            and Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe").exists()
        ):
            browser = playwright.chromium.launch(headless=True, channel="msedge")
        else:
            browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _render_pdf(browser: Any, source: Path, target: Path, repo_url: str) -> None:
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
    try:
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        # Modify only the print DOM, never the offline HTML or its local image paths.
        # Chromium otherwise resolves relative link annotations to private file:// URIs.
        page.evaluate(
            """base => document.querySelectorAll('a[href]').forEach(link => {
                const href = link.getAttribute('href');
                if (/^(TEST_)?(audit-details|scenarios)\\.(json|csv|html)$/.test(href)) {
                    link.setAttribute('href', `${base}/blob/main/reports/${href}`);
                }
            })""",
            repo_url.rstrip("/"),
        )
        page.emulate_media(media="print")
        options = {}
        if "RELATORIO_TECNICO" in source.name:
            marker = "TESTE — DADOS SINTÉTICOS · " if source.name.startswith("TEST_") else ""
            options = {
                "display_header_footer": True,
                "header_template": f"<div style='font:8px sans-serif;width:100%;text-align:center'>{marker}FIAP · FASE 3 · Relatório técnico</div>",
                "footer_template": "<div style='font:8px sans-serif;width:100%;text-align:center'>FIAP · FASE 3 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
            }
        page.pdf(path=str(target), print_background=True, prefer_css_page_size=True, **options)
    finally:
        page.close()


def prepare_deck_frames(deck_path: Path | str, output_dir: Path | str) -> list[Path]:
    """Export each slide as a 1920×1080 PNG image."""
    source, output = Path(deck_path), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with _browser() as browser:
        page = browser.new_page(
            viewport={"width": 1920, "height": 1080}, device_scale_factor=1, reduced_motion="reduce"
        )
        page.goto(source.resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.documentElement.dataset.mode='frame'")
        page.evaluate("document.fonts.ready")
        count = page.locator(".slide").count()
        for index in range(count):
            page.evaluate("index => window.fase3Deck.show(index)", index)
            path = output / f"slide-{index + 1:02d}.png"
            page.screenshot(path=str(path), animations="disabled")
            paths.append(path)
        page.close()
    return paths


def build_reports(
    metrics_path: Path | str,
    output_dir: Path | str,
    repo_url: str,
    *,
    scenarios_path: Path | str | None = None,
) -> dict[str, Any]:
    """Create the technical report, slide deck and complete aggregate appendix.

    Synthetic inputs create TEST_ files with visible marks on every PDF page.
    Missing primary metrics fail before file creation.
    """
    metrics_path, output = Path(metrics_path), Path(output_dir)
    data = _read_metrics(metrics_path)
    if scenarios_path:
        supplement = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
        if (
            supplement.get("metrics_sha256") != _hash(metrics_path)
            or supplement.get("run_id") != data["run_id"]
        ):
            raise ValueError("Scenario supplement does not match frozen metrics")
        if bool(supplement.get("synthetic_fixture")) != data["_synthetic"]:
            raise ValueError("Scenario synthetic status does not match metrics")
        data["_scenarios"] = supplement
    repo_url = _valid_url(repo_url, "repo_url")
    output.mkdir(parents=True, exist_ok=True)
    prefix = "TEST_" if data["_synthetic"] else ""
    assets = files("alfabetizacao_ml.visualization").joinpath("assets")
    deck_css = assets.joinpath("deck.css").read_text(encoding="utf-8")
    report_css = assets.joinpath("report.css").read_text(encoding="utf-8")
    javascript = assets.joinpath("deck.js").read_text(encoding="utf-8")
    charts = _charts(data, output)
    slides = _slides(data, charts)
    if scenarios_path:
        supplement = data["_scenarios"]
        geography = supplement["cluster_geography"]
        regional = sorted(
            geography["by_region"],
            key=lambda row: (-row["municipalities"], row["cluster"], row["regiao"]),
        )
        slides[8]["body"] = (
            _table(
                ["Cluster", "Região", "Municípios"],
                [[r["cluster"], r["regiao"], r["municipalities"]] for r in regional[:5]],
            )
            + f"<p class='small'>Composição descritiva: {len(regional)} pares cluster/região; até cinco maiores acima. Perfis e UF completos no suplemento.</p>"
        )
        largest_profiles = sorted(
            geography["profiles"],
            key=lambda row: (-row["municipalities"], row["cluster"]),
        )[:2]
        profile_text = "; ".join(
            f"cluster {row['cluster']}: {_fmt(row['municipalities'], 0)} municípios, taxa observada {_fmt(row.get('taxa_alfabetizacao'), 1, '%')}"
            for row in largest_profiles
        )
        if profile_text:
            slides[8]["body"] += (
                f"<p class='small'>{_escape(profile_text)}. Associação descritiva, sem inferência causal.</p>"
            )
        slides[9]["body"] = (
            _table(
                ["Ano", "Atingem", "Não atingem", "Sem meta", "Sem projeção"],
                [
                    [r["ano"], r["atinge"], r["nao_atinge"], r["sem_meta"], r["sem_projecao"]]
                    for r in supplement["summary"]
                ],
            )
            + "<p class='small'>Hipótese: manter a taxa prevista de 2026. Sem novo treino nem garantia futura; usar para investigar distância às metas.</p>"
        )
        slides[9]["title"] = "Metas de 2027–2030 sob taxa constante"
    documents = {
        "APRESENTACAO_FASE3": _deck(data, slides, deck_css, javascript),
        "RELATORIO_TECNICO_FASE3": _document(
            "Alfabetização no Brasil — relatório técnico",
            _technical(data, charts, repo_url),
            report_css,
        ),
    }
    generated: list[tuple[Path, str, str]] = []
    if scenarios_path:
        scenario_json = output / f"{prefix}scenarios.json"
        scenario_json.write_bytes(Path(scenarios_path).read_bytes())
        scenario_csv = output / f"{prefix}scenarios.csv"
        rows = data["_scenarios"]["municipal_rows"]
        with scenario_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        generated.extend(
            [
                (scenario_json, "json", "aggregate_scenario"),
                (scenario_csv, "csv", "aggregate_scenario"),
            ]
        )
    # Preserve the entire validated source, including folds and all subgroup metrics.
    audit_json = output / f"{prefix}audit-details.json"
    audit_json.write_bytes(metrics_path.read_bytes())
    fields = _audit_fields(json.loads(metrics_path.read_text(encoding="utf-8")))
    audit_csv = output / f"{prefix}audit-details.csv"
    with audit_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field_path", "json_value"])
        writer.writerows((key, json.dumps(value, ensure_ascii=False)) for key, value in fields)
    audit_html = output / f"{prefix}audit-details.html"
    audit_html.write_text(
        _document(
            "Apêndice agregado integral",
            "<main class='document'>"
            + _fixture_mark(data)
            + f"<h1>Resultados completos</h1><p>Execução: {_escape(data['run_id'])}. {len(fields)} campos; valores integrais, sem seleção ou imputação. Este apêndice externo não integra a paginação do relatório.</p>"
            + _table(
                ["Campo", "Valor JSON"],
                [[key, json.dumps(value, ensure_ascii=False)] for key, value in fields],
            )
            + "</main>",
            report_css,
        ),
        encoding="utf-8",
    )
    generated.extend(
        [
            (audit_json, "json", "aggregate_audit"),
            (audit_csv, "csv", "aggregate_audit"),
            (audit_html, "html", "aggregate_audit"),
        ]
    )
    with _browser() as browser:
        for name, document in documents.items():
            source, pdf = output / f"{prefix}{name}.html", output / f"{prefix}{name}.pdf"
            source.write_text(document, encoding="utf-8")
            _render_pdf(browser, source, pdf, repo_url)
            generated.extend(
                [(source, "html", "aggregate_report"), (pdf, "pdf", "aggregate_report")]
            )
    generated.extend(
        (output / relative.replace(".png", f".{extension}"), extension, "aggregate_chart")
        for relative in charts.values()
        for extension in ("png", "svg")
    )
    manifest = {
        "schema_version": 1,
        "run_id": data["run_id"],
        "metrics_sha256": _hash(metrics_path),
        "protocol_hash": data["protocol_hash"],
        "input_manifest_hash": data["input_manifest_hash"],
        "synthetic_fixture": data["_synthetic"],
        "artifacts": [
            {
                "path": path.relative_to(output).as_posix(),
                "kind": kind,
                "sensitivity": sensitivity,
                "sha256": _hash(path),
                "size_bytes": path.stat().st_size,
                "run_id": data["run_id"],
                "synthetic_fixture": data["_synthetic"],
            }
            for path, kind, sensitivity in generated
        ],
    }
    manifest_path = output / f"{prefix}artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "manifest": str(manifest_path),
        "deck_html": str(output / f"{prefix}APRESENTACAO_FASE3.html"),
        "deck_pdf": str(output / f"{prefix}APRESENTACAO_FASE3.pdf"),
        "technical_pdf": str(output / f"{prefix}RELATORIO_TECNICO_FASE3.pdf"),
        "synthetic_fixture": data["_synthetic"],
    }
