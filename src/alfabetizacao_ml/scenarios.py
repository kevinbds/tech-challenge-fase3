"""Deterministic post-processing; never fits a model or changes frozen metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

REGIONS = {
    uf: region
    for region, ufs in {
        "Norte": "AC AP AM PA RO RR TO",
        "Nordeste": "AL BA CE MA PB PE PI RN SE",
        "Centro-Oeste": "DF GO MT MS",
        "Sudeste": "ES MG RJ SP",
        "Sul": "PR RS SC",
    }.items()
    for uf in ufs.split()
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(value):
    if value is None or pd.isna(value):
        return None
    result = Decimal(str(value))
    if not result.is_finite() or not 0 <= result <= 100:
        raise ValueError("Rate/target must be finite in [0,100]")
    return result


def cluster_geography(metrics: dict) -> dict:
    """One municipality -> one UF and one cluster; report missing geography explicitly."""
    clusters = metrics.get("interpretation", {}).get("clusters", {})
    geography = {}
    for row in metrics["municipal"].get("evaluation_rows", []):
        code, uf = str(row["id_municipio"]), row.get("sigla_uf")
        if code in geography:
            raise ValueError("Duplicate municipal evaluation geography")
        geography[code] = uf
    seen, counts, missing = set(), Counter(), 0
    for row in clusters.get("assignments", []):
        code = str(row["id_municipio"])
        if code in seen:
            raise ValueError("Duplicate cluster assignment")
        seen.add(code)
        uf = geography.get(code)
        region = REGIONS.get(uf, "Sem correspondência")
        missing += region == "Sem correspondência"
        counts[(str(row["cluster"]), uf or "ausente", region)] += 1
    rows = [
        {"cluster": key[0], "sigla_uf": key[1], "regiao": key[2], "municipalities": count}
        for key, count in sorted(counts.items())
    ]
    regional = Counter()
    for row in rows:
        regional[(row["cluster"], row["regiao"])] += row["municipalities"]
    return {
        "status": clusters.get("status", "not_available"),
        "assignment_count": len(seen),
        "missing_geography_count": missing,
        "by_uf": rows,
        "by_region": [
            {"cluster": key[0], "regiao": key[1], "municipalities": count}
            for key, count in sorted(regional.items())
        ],
        "profiles": clusters.get("profiles", []),
        "interpretation": "Composição descritiva dos grupos; não demonstra efeito causal nem homogeneidade de toda a região.",
    }


def build_goal_scenarios(
    metrics_path: str | Path, targets_path: str | Path, output_dir: str | Path
) -> dict:
    """Compare the unchanged 2026 forecast to official 2027–2030 targets.

    targets_path accepts a metas parquet or a marts directory containing silver/metas.parquet.
    Only municipal future-target columns/rows are read, never observed 2025 labels.
    """
    source, targets, output = Path(metrics_path), Path(targets_path), Path(output_dir)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("status") not in ("completed", "smoke_completed"):
        raise ValueError("Metrics must be frozen/completed")
    if targets.is_dir():
        if targets.name == "marts":
            targets = targets.parent
        silver = targets / "silver/metas.parquet"
        targets = (
            silver if silver.is_file() else targets / "gold/comparativo_meta_resultado.parquet"
        )
    columns = [
        "ano_meta",
        "nivel_geografico",
        "id_geografia",
        "rede",
        "meta_raw",
        "meta_operador",
        "meta_limite",
        "meta_status",
    ]
    frame = pd.read_parquet(
        targets,
        columns=columns,
        filters=[
            ("ano_meta", "in", [2027, 2028, 2029, 2030]),
            ("nivel_geografico", "==", "municipio"),
            ("rede", "==", "municipal"),
        ],
    )
    if frame.duplicated(["ano_meta", "id_geografia"]).any():
        raise ValueError("Duplicate official municipal target")
    lookup = {
        (int(row["ano_meta"]), str(row["id_geografia"])): row for row in frame.to_dict("records")
    }
    forecasts = data["municipal"].get("forecast_rows", [])
    if not forecasts:
        raise ValueError("No 2026 forecast available for scenario")
    seen, rows = set(), []
    for forecast in forecasts:
        code = str(forecast["id_municipio"])
        if code in seen or forecast.get("ano") != 2026:
            raise ValueError("Expected unique municipality forecasts for 2026")
        seen.add(code)
        rate = _rate(forecast.get("forecast"))
        for year in range(2027, 2031):
            target = lookup.get((year, code), {})
            limit = _rate(target.get("meta_limite"))
            operator = target.get("meta_operador")
            if limit is not None and operator not in (">", ">="):
                raise ValueError("Unknown target operator")
            met = (
                None
                if rate is None or limit is None
                else (rate > limit if operator == ">" else rate >= limit)
            )
            raw = target.get("meta_raw")
            rows.append(
                {
                    "id_municipio": code,
                    "sigla_uf": forecast.get("sigla_uf"),
                    "ano": year,
                    "scenario_rate": float(rate) if rate is not None else None,
                    "meta_raw": None if raw is None or pd.isna(raw) else str(raw),
                    "meta_operador": operator if limit is not None else None,
                    "meta_limite": float(limit) if limit is not None else None,
                    "target_status": "missing_official"
                    if not target
                    else target.get("meta_status"),
                    "goal_met": met,
                    "status": "sem_projecao"
                    if rate is None
                    else "sem_meta"
                    if limit is None
                    else "atinge"
                    if met
                    else "nao_atinge",
                }
            )
    summary = [
        {
            "ano": year,
            **{
                status: sum(row["ano"] == year and row["status"] == status for row in rows)
                for status in ("atinge", "nao_atinge", "sem_meta", "sem_projecao")
            },
        }
        for year in range(2027, 2031)
    ]
    synthetic = data["status"] == "smoke_completed" or data.get("data_kind") == "synthetic_fixture"
    result = {
        "schema_version": 1,
        "run_id": data["run_id"],
        "metrics_sha256": _hash(source),
        "synthetic_fixture": synthetic,
        "scenario": "Manutenção da taxa prevista para 2026 em 2027–2030. Hipótese explícita, sem novo treino, sem previsão validada ou garantia futura.",
        "population": "Municípios com linha na projeção 2026; metas ausentes permanecem ausentes.",
        "summary": summary,
        "municipal_rows": rows,
        "cluster_geography": cluster_geography(data),
    }
    result["future_targets_hash"] = hashlib.sha256(
        json.dumps(
            [
                [r["id_municipio"], r["ano"], r["meta_raw"], r["meta_operador"], r["meta_limite"]]
                for r in rows
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    prefix = "TEST_" if synthetic else ""
    path = output / f"{prefix}scenarios.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    csv_path = output / f"{prefix}scenarios.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"scenarios_json": str(path), "scenarios_csv": str(csv_path)}
