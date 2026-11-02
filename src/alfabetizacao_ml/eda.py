"""Development-only EDA: aggregated outputs, explicit weights and no test metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from alfabetizacao_ml.data.contracts import target_met

matplotlib.use("Agg")
import matplotlib.pyplot as plt

YEARS = (2023, 2024)
FEATURES = ("taxa_alfabetizacao_lag1", "media_portugues_lag1", "media_inse_2021",
            "populacao_2022", "densidade_2022", "log_populacao_2022")


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _assert_development(frame: pd.DataFrame) -> None:
    if frame.empty or not set(frame["ano"].unique()).issubset(YEARS):
        raise ValueError("EDA accepts nonempty development rows from 2023/2024 only")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _student_aggregates(con: duckdb.DuckDBPyConnection, marts: Path) -> tuple[dict, list[dict]]:
    paths = [marts / f"alunos_{year}.parquet" for year in YEARS]
    # Never use alunos.parquet or a wildcard: both would include the held-out year.
    for year, path in zip(YEARS, paths, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"Required development partition: {path.name}")
        observed = con.execute("SELECT DISTINCT ano FROM read_parquet(?)", [str(path)]).fetchall()
        if observed != [(year,)]:
            raise ValueError("Student development partition contains an unexpected year")
    relation = con.read_parquet([str(path) for path in paths], union_by_name=True)
    relation.project(",".join(["ano", "rede", "sigla_uf", "alfabetizado", "peso_aluno", *FEATURES])).create_view("development")
    invalid = con.execute("""SELECT count(*) FROM development WHERE
        alfabetizado IS NULL OR alfabetizado NOT IN (0,1) OR peso_aluno IS NULL
        OR peso_aluno<=0 OR NOT isfinite(peso_aluno)
        OR rede NOT IN ('estadual','municipal')""").fetchone()[0]
    if invalid:
        raise ValueError("EDA requires the validated eligible student population")
    summaries = {}
    for name, dimensions in {
        "year": ["ano"], "year_network": ["ano", "rede"],
        "year_uf": ["ano", "sigla_uf"], "year_uf_network": ["ano", "sigla_uf", "rede"],
    }.items():
        keys = ",".join(dimensions)
        frame = con.execute(f"""SELECT {keys},count(*) AS eligible_n,
            sum(peso_aluno) AS weight_sum,
            100*sum(peso_aluno*alfabetizado)/sum(peso_aluno) AS literacy_pct,
            100*sum(peso_aluno*(1-alfabetizado))/sum(peso_aluno) AS nonliteracy_pct,
            100*avg(alfabetizado) AS unweighted_literacy_pct
            FROM development GROUP BY {keys} ORDER BY {keys}""").df()
        _assert_development(frame)
        # Do not release rates for very small geographic/network strata.
        frame["rate_suppressed"] = frame["eligible_n"] < 30
        frame.loc[frame["rate_suppressed"], ["literacy_pct", "nonliteracy_pct", "unweighted_literacy_pct"]] = np.nan
        summaries[name] = _records(frame)
    missing = []
    for feature in FEATURES:
        frame = con.execute(f"""SELECT ano,count(*) AS eligible_n,
            count(*) FILTER (WHERE {feature} IS NULL) AS missing_n,
            100*sum(CASE WHEN {feature} IS NULL THEN peso_aluno ELSE 0 END)/sum(peso_aluno) AS weighted_missing_pct
            FROM development GROUP BY ano ORDER BY ano""").df()
        frame["feature"] = feature
        missing.extend(_records(frame))
    return summaries, missing


def _municipal_development(con: duckdb.DuckDBPyConnection, marts: Path) -> pd.DataFrame:
    # SQL predicate/projection precede collection into Python. No 2025 row is
    # returned or used by any EDA calculation, chart, hypothesis or correlation.
    columns = ["ano", "id_municipio", "sigla_uf", "rede", "taxa_alfabetizacao",
               "meta_operador", "meta_limite", "meta_status", *FEATURES]
    frame = con.execute(
        f"SELECT {','.join(columns)} FROM read_parquet(?) WHERE ano IN (2023,2024) ORDER BY ano,id_municipio,rede",
        [str(marts / "municipios.parquet")],
    ).df()
    _assert_development(frame)
    if set(frame["ano"].unique()) != set(YEARS) or not frame["rede"].eq("municipal").all():
        raise ValueError("Municipal EDA requires both development years and municipal network")
    if frame.duplicated(["ano", "id_municipio", "rede"]).any():
        raise ValueError("Duplicate municipal development key")
    for column in ["taxa_alfabetizacao", "meta_limite", *FEATURES]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError("Nonfinite municipal context")
    if frame["taxa_alfabetizacao"].isna().any() or not frame["taxa_alfabetizacao"].between(0, 100).all():
        raise ValueError("Invalid municipal literacy percentage")
    return frame


def _municipal_analysis(frame: pd.DataFrame) -> dict:
    _assert_development(frame)
    distributions, correlations, missing = [], [], []
    for year, group in frame.groupby("ano"):
        counts, edges = np.histogram(group["taxa_alfabetizacao"], bins=np.arange(0, 105, 5))
        distributions.extend({"ano": int(year), "lower_pct": int(low), "upper_pct": int(high), "municipalities": int(n)}
                             for n, low, high in zip(counts, edges[:-1], edges[1:], strict=True))
        for feature in FEATURES:
            missing.append({"ano": int(year), "feature": feature, "municipalities": len(group),
                            "missing_n": int(group[feature].isna().sum())})
            paired = group[[feature, "taxa_alfabetizacao"]].dropna()
            usable = len(paired) >= 3 and paired.nunique().min() > 1
            correlations.append({"ano": int(year), "feature": feature, "paired_municipalities": len(paired),
                                 "pearson": float(paired.corr(method="pearson").iloc[0, 1]) if usable else None,
                                 "spearman": float(paired.corr(method="spearman").iloc[0, 1]) if usable else None})
    panel = frame.loc[frame["ano"].eq(2023), ["id_municipio", "rede", "taxa_alfabetizacao"]].merge(
        frame.loc[frame["ano"].eq(2024), ["id_municipio", "rede", "sigla_uf", "taxa_alfabetizacao"]],
        on=["id_municipio", "rede"], suffixes=("_2023", "_2024"), validate="one_to_one")
    panel["change_pp"] = panel["taxa_alfabetizacao_2024"] - panel["taxa_alfabetizacao_2023"]
    changes = panel.groupby("sigla_uf")["change_pp"].agg(municipalities="size", mean_change_pp="mean", median_change_pp="median").reset_index()
    targets = frame.loc[frame["ano"].eq(2024)].copy()
    targets["assessable"] = targets["meta_status"].eq("available") & targets["meta_limite"].notna() & targets["meta_operador"].isin([">", ">="])
    targets["met"] = [bool(target_met(rate, op, limit)) if available else False for rate, op, limit, available in
                      zip(targets["taxa_alfabetizacao"], targets["meta_operador"], targets["meta_limite"], targets["assessable"], strict=True)]
    targets["strict_boundary_not_met"] = targets["assessable"] & targets["meta_operador"].eq(">") & targets["taxa_alfabetizacao"].eq(targets["meta_limite"])
    target_summary = targets.groupby("sigla_uf").agg(municipalities=("ano", "size"), assessable=("assessable", "sum"),
                                                         met=("met", "sum"), strict_boundary_not_met=("strict_boundary_not_met", "sum")).reset_index()
    target_summary["met_pct_of_assessable"] = 100 * target_summary["met"] / target_summary["assessable"].replace(0, np.nan)
    n2023, n2024 = (int(frame["ano"].eq(year).sum()) for year in YEARS)
    return {"histogram": distributions, "correlations": correlations, "missing": missing,
            "balanced_panel": {"municipalities": len(panel), "only_2023": n2023-len(panel), "only_2024": n2024-len(panel),
                               "mean_change_pp": float(panel["change_pp"].mean()) if len(panel) else None,
                               "median_change_pp": float(panel["change_pp"].median()) if len(panel) else None,
                               "share_increased": float(panel["change_pp"].gt(0).mean()) if len(panel) else None},
            "changes_by_uf": _records(changes), "targets_2024_by_uf": _records(target_summary),
            "targets_2024_total": {"municipalities": len(targets), "assessable": int(targets["assessable"].sum()),
                                   "met": int(targets["met"].sum()), "missing": int((~targets["assessable"]).sum())}}


def _coverage(marts: Path) -> dict:
    path = marts.parent / "coverage.json"
    if not path.exists():
        raise FileNotFoundError("coverage.json is required to describe the eligible population")
    coverage = json.loads(path.read_text(encoding="utf-8"))
    fields = ("raw_rows", "eligible_rows", "excluded_grade", "excluded_network", "excluded_missing_identity_context",
              "excluded_missing_label", "excluded_nonpositive_or_missing_weight", "municipalities", "municipality_network_contexts")
    # Coverage is aggregate metadata; only development-year counts are returned.
    result = {str(year): {key: coverage["students"][str(year)][key] for key in fields} for year in YEARS}
    for values in result.values():
        excluded = sum(value for key, value in values.items() if key.startswith("excluded_"))
        if values["raw_rows"] != excluded + values["eligible_rows"]:
            raise ValueError("Development eligibility accounting does not reconcile")
    return result


def _figures(report: dict, output: Path) -> list[str]:
    files = []

    def save(name):
        plt.tight_layout()
        plt.savefig(output / name, dpi=150, bbox_inches="tight")
        plt.close()
        files.append(name)

    frame = pd.DataFrame(report["students"]["year_network"])
    labels = frame["ano"].astype(str) + " / " + frame["rede"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, frame["literacy_pct"], label="Alfabetizados", color="#147d92")
    ax.bar(labels, frame["nonliteracy_pct"], bottom=frame["literacy_pct"], label="Não alfabetizados", color="#d18b32")
    ax.set(ylabel="Percentual ponderado (%)", ylim=(0, 100), title="População elegível por ano e rede")
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1))
    save("01_weighted_network.png")

    frame = pd.DataFrame(report["students"]["year_uf"])
    pivot = frame.pivot(index="sigla_uf", columns="ano", values="nonliteracy_pct")
    pivot.plot.barh(figsize=(9, 10), color=["#a5cbd3", "#147d92"])
    plt.xlabel("Não alfabetizados — percentual ponderado (%)")
    plt.title("Cobertura observada por UF; ausência de barra não significa zero")
    save("02_weighted_uf.png")

    missing = pd.DataFrame(report["student_missing"])
    missing.pivot(index="feature", columns="ano", values="weighted_missing_pct").plot.barh(figsize=(10, 5))
    plt.xlabel("Peso dos alunos sem contexto / peso total (%)")
    plt.title("Ausências de contexto: 2023 não possui antecedentes 2022 nesta fonte")
    save("03_context_missing.png")

    histogram = pd.DataFrame(report["municipal"]["histogram"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for year, group in histogram.groupby("ano"):
        ax.stairs(group["municipalities"], np.r_[group["lower_pct"], group["upper_pct"].iloc[-1]], label=str(year))
    ax.set(xlabel="Alfabetização municipal (%) — intervalos de 5 p.p.", ylabel="Municípios (peso igual)", title="Distribuições municipais observadas")
    ax.legend()
    save("04_municipal_distribution.png")

    correlations = pd.DataFrame(report["municipal"]["correlations"])
    correlation_plot = correlations.pivot(index="feature", columns="ano", values="spearman").astype(float)
    if correlation_plot.notna().any(axis=None):
        correlation_plot.plot.barh(figsize=(10, 5))
    else:
        _, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Correlações indisponíveis: pares ou variação insuficientes", ha="center", transform=ax.transAxes)
    plt.xlim(-1, 1)
    plt.xlabel("Correlação de Spearman com taxa municipal (pares completos)")
    plt.title("Associação ecológica descritiva; não causal nem individual")
    save("05_context_correlations.png")

    changes = pd.DataFrame(report["municipal"]["changes_by_uf"])
    if len(changes):
        changes.set_index("sigla_uf")["mean_change_pp"].plot.bar(figsize=(10, 4), color="#147d92")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Mudança média 2023→2024 (p.p.)")
        plt.title("Painel municipal com peso igual; alunos não são um painel individual")
        save("06_balanced_changes.png")

    targets = pd.DataFrame(report["municipal"]["targets_2024_by_uf"])
    targets.set_index("sigla_uf")["met_pct_of_assessable"].plot.bar(figsize=(10, 4), color="#147d92")
    plt.ylim(0, 100)
    plt.ylabel("Metas atingidas entre municípios avaliáveis (%)")
    plt.title("Metas 2024, edição 2024: operadores > e ≥ preservados")
    save("07_targets_2024.png")
    return files


def run_eda(marts_dir: Path, output_dir: Path) -> dict:
    """Read development partitions and emit private aggregate JSON/PNG artifacts."""
    marts, output = Path(marts_dir).resolve(), Path(output_dir).resolve()
    if output == marts.parent or output.is_relative_to(marts.parent):
        raise ValueError("EDA output must be outside the active data snapshot")
    with duckdb.connect() as con:
        students, missing = _student_aggregates(con, marts)
        municipal = _municipal_analysis(_municipal_development(con, marts))
    coverage = _coverage(marts)
    for row in students["year"]:
        if row["eligible_n"] != coverage[str(row["ano"])]["eligible_rows"]:
            raise ValueError("EDA mart population differs from coverage metadata")
    report = {"schema_version": 1, "status": "completed", "development_years": list(YEARS),
              "test_labels_used": False, "test_metrics_used": False, "students": students,
              "student_missing": missing, "municipal": municipal, "coverage": coverage,
              "input_audit": [{"filename": f"alunos_{year}.parquet", "allowed_years": [year],
                               "sha256": _digest(marts / f"alunos_{year}.parquet")} for year in YEARS]
                              + [{"filename": "municipios.parquet", "predicate": "ano IN (2023,2024)",
                                  "allowed_years": list(YEARS), "sha256": _digest(marts / "municipios.parquet")}],
              "interpretation": [
                  "Student rates use positive official weights within the observed eligible population, not a causal or population-universal claim.",
                  "Municipal correlations, histograms, changes and targets give each municipality equal weight; they are not individual effects.",
                  "Published municipal estimates and the eligible student mart can have different coverage/denominators; their means are not interchangeable with an official national indicator.",
                  "Cross-year population coverage changes; balanced municipality comparisons do not follow the same students.",
                  "Missing 2023 lag context is structural: the frozen AEEB series starts in 2023. Do not replace it with zero.",
                  "The 2024 target edition may differ from later revisions; >80 is not achieved by a value exactly equal to80.",
                  "Development-only means restricted data, not a claim that this notebook predates the frozen modeling protocol. Do not reopen model selection.",
                  "2024 municipality-grouped development/CV and reserve precede a frozen 2025 test. Identifiers and contemporaneous outcomes are not features.",
              ],
              "observational_hypotheses": [
                  "Prior municipal literacy may be associated with subsequent municipal literacy and contextual student risk; temporal association is not causation.",
                  "INSE may be associated with municipal outcomes; ecological associations cannot diagnose a student's abilities or family circumstances.",
                  "UF/network contrasts may reflect coverage, composition and structural conditions; this analysis does not identify treatment effects.",
              ]}
    output.mkdir(parents=True, exist_ok=True)
    report["figures"] = _figures(report, output)
    (output / "eda_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
