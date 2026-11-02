"""Validated local reconstruction of the phase-2 lineage.

Only aggregate diagnostics are emitted. Student identifiers stay in private
Silver/marts and are never printed, returned as examples or written to reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .acquisition import canonical_hash, sha256_file, verify_manifest
from .contracts import (
    LEVEL_COLUMNS,
    RESULT_NETWORKS,
    STUDENT_NETWORKS,
    assert_unique,
    decimal_value,
    exact_previous_year_context,
    finite_numeric,
    integral_numeric,
    municipality_code,
    parse_meta,
    safe_join,
    target_met,
    validate_levels,
)

STUDENT_RENAME = {
    "NU_ANO_AVALIACAO": "ano", "CO_MUNICIPIO": "id_municipio", "SG_UF": "sigla_uf",
    "ID_ESCOLA": "id_escola", "ID_ALUNO": "id_aluno", "TP_SERIE": "serie",
    "TP_DEPENDENCIA": "rede_codigo", "IN_ALFABETIZADO": "alfabetizado",
    "VL_PESO_ALUNO_LP": "peso_aluno", "VL_PROFICIENCIA_LP": "proficiencia",
    "IN_PRESENCA_LP": "presenca", "IN_PREENCHIMENTO_LP": "preenchimento_caderno",
    "CO_CADERNO_LP": "caderno",
}


def _read_csv(archive: Path, member: str, **kwargs):
    # pandas closes the ZipExtFile after reading; ZipFile must live for chunks.
    zipped = ZipFile(archive)
    stream = zipped.open(member)
    return pd.read_csv(stream, sep=";", encoding="latin1", dtype=str, **kwargs)


def _write_frame(frame: pd.DataFrame, path: Path, decimal_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path
    path = path.with_suffix(".parquet.partial")
    decimal_columns = [c for c in (decimal_columns or []) if c in frame]
    # Prevent DuckDB's sampled pandas-object inference from selecting a narrow
    # Decimal precision based on the first municipalities (e.g. population).
    frame = frame.copy()
    for column in decimal_columns:
        frame[column] = frame[column].map(lambda value: None if pd.isna(value) else str(value))
    con = duckdb.connect()
    try:
        con.register("source_frame", frame)
        replacements = ", ".join(f'CAST("{c}" AS DECIMAL(38,9)) AS "{c}"' for c in decimal_columns)
        query = "SELECT *" + (f" REPLACE ({replacements})" if replacements else "") + " FROM source_frame"
        con.execute(f"COPY ({query}) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()
    path.replace(destination)


def _directory(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in records:
        micro = item.get("microrregiao")
        if micro:
            uf = micro["mesorregiao"]["UF"]
        else:
            uf = item["regiao-imediata"]["regiao-intermediaria"]["UF"]
        code = municipality_code(item["id"])
        if not code.startswith(str(uf["id"])):
            raise ValueError("Directory municipality/UF code mismatch")
        rows.append({"id_municipio": code, "sigla_uf": uf["sigla"],
                     "regiao": uf["regiao"]["nome"], "nome_municipio": item["nome"]})
    frame = pd.DataFrame(rows)
    assert_unique(frame, ["id_municipio"], "IBGE municipality directory")
    return frame


def _validate_geography(frame: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    reference = directory.rename(columns={"sigla_uf": "uf_directory"})
    result = safe_join(frame, reference[["id_municipio", "uf_directory", "regiao"]],
                       ["id_municipio"], "IBGE geographic relationship")
    if result["uf_directory"].isna().any():
        raise ValueError("Official record references a municipality absent from the directory")
    if (result["sigla_uf"] != result["uf_directory"]).any():
        raise ValueError("Official record UF disagrees with its municipality")
    return result.drop(columns="uf_directory")


def _results(archive: Path, year: int, directory: pd.DataFrame, *, state: bool = False) -> pd.DataFrame:
    frame = _read_csv(archive, "DADOS/TS_ESTADO.csv" if state else "DADOS/TS_MUNICIPIO.csv")
    rename = {
        "NU_ANO_AVALIACAO": "ano", "SG_UF": "sigla_uf", "TP_SERIE": "serie",
        "ID_TIPO_REDE": "rede_codigo", "PC_ALUNO_ALFABETIZADO": "taxa_alfabetizacao",
        "VL_MEDIA_LP": "media_portugues", "CO_MUNICIPIO": "id_municipio",
        "NO_MUNICIPIO": "nome_municipio",
        **{f"PC_ALUNO_NIVEL_{n}_LP": f"proporcao_aluno_nivel_{n}" for n in range(9)},
    }
    frame = frame.rename(columns=rename)
    required = ["ano", "sigla_uf", "serie", "rede_codigo", "taxa_alfabetizacao", "media_portugues"]
    if not state:
        required.append("id_municipio")
    if not set(required).issubset(frame.columns):
        raise ValueError(f"AEEB {year}: missing contracted result columns")
    frame["ano"] = integral_numeric(frame["ano"])
    frame["serie"] = integral_numeric(frame["serie"])
    if not frame["ano"].eq(year).all() or not frame["serie"].eq(2).all():
        raise ValueError("Result observation year or school grade differs from contract")
    codes = finite_numeric(frame["rede_codigo"], nullable=False)
    if not codes.isin(RESULT_NETWORKS).all():
        raise ValueError("Unknown result network code")
    frame["rede"] = codes.map(RESULT_NETWORKS)
    for column in ["taxa_alfabetizacao", "media_portugues"]:
        frame[column] = frame[column].map(lambda value: decimal_value(value, nullable=False))
        if (frame[column] < 0).any():
            raise ValueError(f"Negative result field: {column}")
    if (frame["taxa_alfabetizacao"] > 100).any():
        raise ValueError("Result outside percentage range; no automatic rescaling")
    for column in LEVEL_COLUMNS:
        if year == 2023:
            if column in frame and frame[column].notna().any():
                raise ValueError("2023 literacy levels are structurally unavailable")
            frame[column] = None
        elif column not in frame:
            raise ValueError("Unexpected missing literacy-level column")
        else:
            frame[column] = frame[column].map(decimal_value)
    validate_levels(frame, year)
    frame["levels_applicability"] = "not_applicable" if year == 2023 else "applicable"
    if state:
        regions = directory[["sigla_uf", "regiao"]].groupby("sigla_uf", as_index=False).first()
        frame = safe_join(frame, regions, ["sigla_uf"], "UF directory")
        if frame["regiao"].isna().any():
            raise ValueError("Official state result references an unknown UF")
        keys = ["ano", "sigla_uf", "rede"]
    else:
        frame["id_municipio"] = frame["id_municipio"].map(municipality_code)
        frame = _validate_geography(frame, directory)
        keys = ["ano", "id_municipio", "rede"]
    assert_unique(frame, keys, "official result")
    return frame.drop(columns=["rede_codigo", "CO_UF"], errors="ignore")


def _targets(path: Path, *, municipal: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_excel(path, header=1, dtype=object)
    frame = frame.loc[pd.to_numeric(frame["ANO"], errors="coerce").eq(2024)].copy()
    if frame.empty:
        raise ValueError("Target workbook contains no contracted reference year")
    targets, observed = [], []
    for row in frame.to_dict("records"):
        if not municipal and str(row["REDE"]).strip().upper() != "PÚBLICA":
            raise ValueError("State/national target has an unexpected network")
        if municipal:
            geography = municipality_code(row["CO_MUNICIPIO"])
            level, network = "municipio", "municipal"
            if str(row["NO_TP_REDE"]).strip().upper() != "MUNICIPAL":
                raise ValueError("Municipal target has an unexpected network")
        elif str(row["NOME_UF"]).strip().lower() == "brasil":
            geography, level, network = "BR", "brasil", "publica"
        else:
            geography, level, network = str(row["SIGLA_UF"]).strip(), "uf", "publica"
        for year in range(2024, 2031):
            targets.append({"ano_referencia": 2024, "ano_meta": year,
                            "nivel_geografico": level, "id_geografia": geography,
                            "rede": network, **parse_meta(row[f"META_FINAL_{year}"])})
        # The workbook is the official national observation source. Municipal
        # and state facts instead use the AEEB microdata, not mixed revisions.
        if level == "brasil":
            for year in (2023, 2024):
                value = decimal_value(row[f"PC_ALUNO_ALFABETIZADO_{year}"])
                if value is not None and not 0 <= value <= 100:
                    raise ValueError("National result outside percentage range")
                observed.append({"ano": year, "nivel_geografico": level,
                                 "id_geografia": geography, "rede": network,
                                 "taxa_alfabetizacao": value})
    targets_frame = pd.DataFrame(targets)
    assert_unique(targets_frame, ["ano_meta", "nivel_geografico", "id_geografia", "rede"], "official targets")
    return targets_frame, pd.DataFrame(observed)


def _inse(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object)
    if not finite_numeric(frame["NU_ANO_SAEB"], nullable=False).eq(2021).all():
        raise ValueError("INSE observation year differs from 2021")
    codes = pd.to_numeric(frame["TP_TIPO_REDE"], errors="raise")
    location = pd.to_numeric(frame["TP_LOCALIZACAO"], errors="raise")
    frame = frame.loc[codes.isin([2, 3]) & location.eq(0)].copy()
    frame["id_municipio"] = frame["CO_MUNICIPIO"].map(municipality_code)
    frame["rede"] = pd.to_numeric(frame["TP_TIPO_REDE"]).map({2: "estadual", 3: "municipal"})
    count = finite_numeric(frame["QTD_ALUNOS_INSE"], nullable=False)
    mean = finite_numeric(frame["MEDIA_INSE"])
    if (count < 0).any() or ((count > 0) & (mean <= 0)).any():
        raise ValueError("Inconsistent INSE quantity/mean")
    frame["media_inse_2021"] = mean.where(count > 0, np.nan)
    result = frame[["id_municipio", "rede", "media_inse_2021"]]
    assert_unique(result, ["id_municipio", "rede"], "INSE municipality/network/total location")
    return result


def _census(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text(encoding="utf-8"))
    if len(records) != 2 or {str(item["id"]) for item in records} != {"93", "614"}:
        raise ValueError("Both unique census population and density variables are required")
    variable_names = {"93": "populacao_2022", "614": "densidade_2022"}
    frames = []
    for variable in records:
        name = variable_names.get(str(variable["id"]))
        if name is None:
            raise ValueError("Unexpected census variable")
        if len(variable["resultados"]) != 1:
            raise ValueError("Census total unexpectedly has multiple classification slices")
        rows = []
        for item in variable["resultados"][0]["series"]:
            value = decimal_value(item["serie"]["2022"])
            if value is not None and value < 0:
                raise ValueError("Negative census population/density")
            rows.append({"id_municipio": municipality_code(item["localidade"]["id"]), name: value})
        current = pd.DataFrame(rows)
        assert_unique(current, ["id_municipio"], f"IBGE {name}")
        frames.append(current)
    if len(frames) != 2:
        raise ValueError("Both census population and density are required")
    return frames[0].merge(frames[1], on="id_municipio", how="outer", validate="one_to_one")


def _enrich(frame: pd.DataFrame, inse: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    frame = safe_join(frame, inse, ["id_municipio", "rede"], "INSE context")
    frame = safe_join(frame, census, ["id_municipio"], "Census context")
    frame["log_populacao_2022"] = np.log1p(pd.to_numeric(frame["populacao_2022"], errors="raise").astype(float))
    return frame


def _comparative(targets: pd.DataFrame, municipal: pd.DataFrame,
                 states: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    muni = municipal.loc[municipal["rede"].eq("municipal"), ["ano", "id_municipio", "rede", "taxa_alfabetizacao"]].rename(columns={"id_municipio": "id_geografia"})
    muni["nivel_geografico"] = "municipio"
    uf = states.loc[states["rede"].eq("publica"), ["ano", "sigla_uf", "rede", "taxa_alfabetizacao"]].rename(columns={"sigla_uf": "id_geografia"})
    uf["nivel_geografico"] = "uf"
    observations = pd.concat([muni, uf, national], ignore_index=True)
    observations["ano_resultado"] = observations["ano"]
    observations = observations.rename(columns={"ano": "ano_meta"})
    out = safe_join(targets, observations,
                    ["ano_meta", "nivel_geografico", "id_geografia", "rede"], "target/results")
    out["atingiu_meta"] = [target_met(rate, op, limit) for rate, op, limit in
                           zip(out["taxa_alfabetizacao"], out["meta_operador"], out["meta_limite"], strict=True)]
    out["gap_pp"] = [None if pd.isna(rate) or pd.isna(limit) else decimal_value(rate) - decimal_value(limit)
                     for rate, limit in zip(out["taxa_alfabetizacao"], out["meta_limite"], strict=True)]
    out["resultado_status"] = np.where(out["taxa_alfabetizacao"].notna(), "available",
                                      np.where(out["ano_meta"] > 2025, "future_period", "absent_from_frozen_source"))
    return out


def _prepare_students(archive: Path, year: int, directory: pd.DataFrame,
                      municipal: pd.DataFrame, inse: pd.DataFrame, census: pd.DataFrame,
                      silver_path: Path, mart_path: Path) -> dict:
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    silver_destination, mart_destination = silver_path, mart_path
    silver_path = silver_path.with_suffix(".parquet.partial")
    mart_path = mart_path.with_suffix(".parquet.partial")
    silver_writer = mart_writer = None
    counts = {"raw_rows": 0, "excluded_grade": 0, "excluded_network": 0,
              "excluded_missing_identity_context": 0,
              "excluded_missing_label": 0, "excluded_nonpositive_or_missing_weight": 0,
              "eligible_rows": 0, "rows_missing_previous_year": 0, "rows_missing_inse": 0,
              "rows_missing_census": 0}
    geographic_counts: dict[tuple[str, str], dict[str, int]] = {}
    quarantine = []
    try:
        chunks = _read_csv(archive, "DADOS/TS_ALUNO.csv", usecols=list(STUDENT_RENAME), chunksize=200_000)
        for raw in chunks:
            frame = raw.rename(columns=STUDENT_RENAME)
            counts["raw_rows"] += len(frame)
            missing_identity = frame[["id_municipio", "id_escola", "id_aluno", "rede_codigo", "sigla_uf"]].isna().any(axis=1)
            counts["excluded_missing_identity_context"] += int(missing_identity.sum())
            if missing_identity.any():
                rejected = frame.loc[missing_identity].copy()
                rejected["quarantine_reason"] = "missing_mandatory_identity_context"
                quarantine.append(rejected)
            frame = frame.loc[~missing_identity].copy()
            if frame.empty:
                continue
            frame["ano"] = integral_numeric(frame["ano"]).astype("int16")
            frame["serie"] = integral_numeric(frame["serie"]).astype("int16")
            if not frame["ano"].eq(year).all():
                raise ValueError("Student observation year mismatch")
            frame["id_municipio"] = frame["id_municipio"].map(municipality_code)
            code = finite_numeric(frame["rede_codigo"], nullable=False)
            if not code.isin(STUDENT_NETWORKS).all():
                raise ValueError("Unknown student network code")
            frame["rede"] = code.map(STUDENT_NETWORKS)
            frame = frame.drop(columns="rede_codigo")
            for column in ("peso_aluno", "alfabetizado", "proficiencia", "presenca", "preenchimento_caderno"):
                frame[column] = finite_numeric(frame[column])
            if not frame["alfabetizado"].dropna().isin([0, 1]).all():
                raise ValueError("Official student label outside {0,1}")
            if frame[["id_aluno", "id_escola"]].isna().any(axis=None):
                raise ValueError("Missing private student identity")
            frame = _validate_geography(frame, directory)
            silver_table = pa.Table.from_pandas(frame, preserve_index=False)
            if silver_writer is None:
                silver_writer = pq.ParquetWriter(silver_path, silver_table.schema, compression="zstd")
            silver_writer.write_table(silver_table)
            counts["excluded_grade"] += int((~frame["serie"].eq(2)).sum())
            eligible = frame.loc[frame["serie"].eq(2)].copy()
            counts["excluded_network"] += int((~eligible["rede"].isin(["estadual", "municipal"])).sum())
            eligible = eligible.loc[eligible["rede"].isin(["estadual", "municipal"])].copy()
            counts["excluded_missing_label"] += int(eligible["alfabetizado"].isna().sum())
            eligible = eligible.loc[eligible["alfabetizado"].notna()].copy()
            counts["excluded_nonpositive_or_missing_weight"] += int((~eligible["peso_aluno"].gt(0)).sum())
            eligible = eligible.loc[eligible["peso_aluno"].gt(0)].copy()
            if eligible.empty:
                continue
            eligible["alfabetizado"] = eligible["alfabetizado"].astype("int8")
            eligible["risco"] = (1 - eligible["alfabetizado"]).astype("int8")
            # Contemporaneous outcomes survive only in private Silver. The mart
            # carries the official label, identity/grouping and antecedent X.
            eligible = eligible.drop(columns=["proficiencia", "presenca", "preenchimento_caderno", "caderno"])
            eligible = exact_previous_year_context(eligible, municipal)
            eligible = _enrich(eligible, inse, census)
            for column in ["taxa_alfabetizacao_lag1", "media_portugues_lag1", "populacao_2022", "densidade_2022"]:
                eligible[column] = pd.to_numeric(eligible[column], errors="raise").astype(float)
            eligible["ano_contexto"] = eligible["ano_contexto"].astype("float64")
            counts["eligible_rows"] += len(eligible)
            counts["rows_missing_previous_year"] += int(eligible["ano_contexto"].isna().sum())
            counts["rows_missing_inse"] += int(eligible["media_inse_2021"].isna().sum())
            counts["rows_missing_census"] += int(eligible["populacao_2022"].isna().sum())
            for (uf, network), group in eligible.groupby(["sigla_uf", "rede"], observed=True):
                entry = geographic_counts.setdefault((uf, network), {"rows": 0, "missing_previous_year": 0})
                entry["rows"] += len(group)
                entry["missing_previous_year"] += int(group["ano_contexto"].isna().sum())
            table = pa.Table.from_pandas(eligible, preserve_index=False)
            if mart_writer is None:
                mart_writer = pq.ParquetWriter(mart_path, table.schema, compression="zstd")
            mart_writer.write_table(table)
    finally:
        if silver_writer is not None:
            silver_writer.close()
        if mart_writer is not None:
            mart_writer.close()
    if counts["eligible_rows"] == 0:
        raise ValueError("Student eligibility produced zero rows")
    con = duckdb.connect()
    try:
        duplicated = con.execute("SELECT count(*) FROM (SELECT ano,id_municipio,id_escola,id_aluno,count(*) n FROM read_parquet(?) GROUP BY ALL HAVING n>1)", [str(silver_path)]).fetchone()[0]
        if duplicated:
            raise ValueError(f"Student private key duplicates: {duplicated} groups")
        municipal_count, contexts = con.execute("SELECT count(DISTINCT id_municipio),count(DISTINCT (id_municipio,rede)) FROM read_parquet(?)", [str(mart_path)]).fetchone()
    finally:
        con.close()
    counts["municipalities"] = municipal_count
    counts["municipality_network_contexts"] = contexts
    counts["by_uf_network"] = [{"sigla_uf": uf, "rede": network, **values}
                                for (uf, network), values in sorted(geographic_counts.items())]
    quarantine_frame = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=list(STUDENT_RENAME.values()))
    counts["quarantine_by_uf"] = quarantine_frame.groupby("sigla_uf", dropna=False).size().rename("rows").reset_index().fillna("missing").to_dict("records")
    _write_frame(quarantine_frame, silver_destination.with_name(f"quarantine_identity_{year}.parquet"))
    silver_path.replace(silver_destination)
    mart_path.replace(mart_destination)
    return counts


def prepare_data(cache_dir: Path, manifest_path: Path) -> dict:
    root = Path(cache_dir).resolve()
    manifest = verify_manifest(root, Path(manifest_path))
    files = {item["id"]: root / item["relative_path"] for item in manifest["sources"]}
    output = root / "snapshots" / manifest["snapshot_id"]
    for folder in ("silver", "gold", "marts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    in_progress = output / "preparation.in-progress"
    in_progress.write_text("Preparing frozen sources; completed Parquets remain readable.", encoding="utf-8")
    directory = _directory(files["ibge_municipios"])
    inse = _inse(files["inse_2021"])
    census = _census(files["ibge_censo_2022"])
    _write_frame(directory, output / "silver" / "diretorio_municipio.parquet")
    _write_frame(inse, output / "silver" / "inse_municipio.parquet")
    _write_frame(census, output / "silver" / "censo_municipio.parquet", ["populacao_2022", "densidade_2022"])
    municipal = pd.concat([_results(files[f"aeeb_{year}"], year, directory) for year in (2023, 2024, 2025)], ignore_index=True)
    states = pd.concat([_results(files[f"aeeb_{year}"], year, directory, state=True) for year in (2023, 2024, 2025)], ignore_index=True)
    municipal["release_id"] = states["release_id"] = manifest["snapshot_id"]
    decimals = ["taxa_alfabetizacao", "media_portugues", *LEVEL_COLUMNS]
    _write_frame(municipal, output / "gold" / "indicador_municipio.parquet", decimals)
    _write_frame(states, output / "silver" / "uf.parquet", decimals)
    targets_municipal, _ = _targets(files["metas_municipios_2024"], municipal=True)
    targets_uf, national = _targets(files["metas_uf_brasil_2024"], municipal=False)
    targets = pd.concat([targets_municipal, targets_uf], ignore_index=True)
    targets["release_id"] = manifest["snapshot_id"]
    _write_frame(targets, output / "silver" / "metas.parquet", ["meta_limite"])
    comparative = _comparative(targets, municipal, states, national)
    _write_frame(comparative, output / "gold" / "comparativo_meta_resultado.parquet", ["meta_limite", "taxa_alfabetizacao", "gap_pp"])
    # Keep the phase-2 last-observation LAG product distinct from exact t-1 ML.
    con = duckdb.connect()
    try:
        con.execute("COPY (SELECT *,LAG(ano) OVER w AS ano_anterior_disponivel,LAG(taxa_alfabetizacao) OVER w AS taxa_anterior_disponivel,taxa_alfabetizacao-LAG(taxa_alfabetizacao) OVER w AS variacao_pp FROM read_parquet($source) WINDOW w AS (PARTITION BY id_municipio,rede ORDER BY ano)) TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)", {"source": str(output / "gold" / "indicador_municipio.parquet"), "destination": str(output / "gold" / "evolucao_alfabetizacao.parquet.partial")})
    finally:
        con.close()
    (output / "gold" / "evolucao_alfabetizacao.parquet.partial").replace(output / "gold" / "evolucao_alfabetizacao.parquet")
    mart_municipal = municipal.loc[municipal["rede"].eq("municipal")].copy()
    mart_municipal = exact_previous_year_context(mart_municipal, municipal)
    mart_municipal = _enrich(mart_municipal, inse, census)
    target_context = targets_municipal.rename(columns={"id_geografia": "id_municipio", "ano_meta": "ano"})
    target_context = target_context[["ano", "id_municipio", "rede", "meta_raw", "meta_operador", "meta_limite", "meta_status"]]
    mart_municipal = safe_join(mart_municipal, target_context, ["ano", "id_municipio", "rede"], "municipal target context")
    mart_municipal["meta_status"] = mart_municipal["meta_status"].fillna("not_published_for_period")
    _write_frame(mart_municipal, output / "marts" / "municipios.parquet", [*decimals, "meta_limite", "taxa_alfabetizacao_lag1", "media_portugues_lag1", "populacao_2022", "densidade_2022"])
    projection = municipal.loc[municipal["ano"].eq(2025) & municipal["rede"].eq("municipal"),
                               ["id_municipio", "sigla_uf", "regiao", "rede"]].copy()
    projection["ano"] = 2026
    projection["taxa_alfabetizacao"] = None
    projection = _enrich(exact_previous_year_context(projection, municipal), inse, census)
    projection = safe_join(projection, target_context, ["ano", "id_municipio", "rede"], "projection target context")
    projection["meta_status"] = projection["meta_status"].fillna("not_published_for_period")
    _write_frame(projection, output / "marts" / "municipios_2026.parquet",
                 ["taxa_alfabetizacao", "meta_limite", "taxa_alfabetizacao_lag1", "media_portugues_lag1", "populacao_2022", "densidade_2022"])
    coverage = {"snapshot_id": manifest["snapshot_id"], "manifest_hash": manifest["content_id"],
                "directory_municipalities": len(directory), "census_municipalities": len(census),
                "municipal_rows_by_year_network": municipal.groupby(["ano", "rede"]).size().rename("rows").reset_index().to_dict("records"),
                "students": {},
                "limitations": ["Retrospective backtest uses the frozen currently available source revisions.",
                                "Target workbook reference is 2024, with targets for 2024–2030; no target is invented for 2023.",
                                "National observed results in this source snapshot cover 2023/2024 only.",
                                "2024 AEEB documentation reports no RR evaluation and no RR/DF state-result rows; DF student/municipal data are retained.",
                                "Current municipality directory has one more municipality than the 2022 census; absent historical context remains missing.",
                                "2023 literacy-level proportions are structurally unavailable."]}
    # 2024 first to let the modeling lane start profiling while other years run.
    for year in (2024, 2023, 2025):
        silver = output / "silver" / f"alunos_{year}.parquet"
        mart = output / "marts" / f"alunos_{year}.parquet"
        counts = _prepare_students(files[f"aeeb_{year}"], year, directory, municipal, inse, census, silver, mart)
        coverage["students"][str(year)] = counts
        (output / "coverage.partial.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"prepared students {year}: {counts['eligible_rows']} eligible, {counts['municipalities']} municipalities", flush=True)
    con = duckdb.connect()
    try:
        student_parts = [str(output / "marts" / f"alunos_{year}.parquet") for year in (2023, 2024, 2025)]
        con.execute("COPY (SELECT * FROM read_parquet($sources, union_by_name=true)) TO $destination (FORMAT PARQUET, COMPRESSION ZSTD)", {"sources": student_parts, "destination": str(output / "marts" / "alunos.parquet.partial")})
    finally:
        con.close()
    (output / "marts" / "alunos.parquet.partial").replace(output / "marts" / "alunos.parquet")
    coverage_path = output / "coverage.json"
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    output_artifacts = {}
    for path in sorted(output.rglob("*.parquet")):
        output_artifacts[path.relative_to(output).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    receipt = {"schema_version": 1, "snapshot_id": manifest["snapshot_id"],
               "input_manifest_hash": manifest["content_id"], "outputs": output_artifacts,
               "coverage_hash": canonical_hash(coverage), "status": "passed"}
    receipt_temporary = output / "preparation-receipt.json.partial"
    receipt_temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt_temporary.replace(output / "preparation-receipt.json")
    in_progress.unlink()
    return {"snapshot_id": manifest["snapshot_id"], "manifest_hash": manifest["content_id"],
            "manifest_path": str(manifest_path), "marts_dir": str(output / "marts"),
            "alunos": str(output / "marts" / "alunos.parquet"),
            "municipios": str(output / "marts" / "municipios.parquet"),
            "municipios_2026": str(output / "marts" / "municipios_2026.parquet"),
            "gold_dir": str(output / "gold"), "coverage_path": str(coverage_path),
            "receipt_path": str(output / "preparation-receipt.json")}
