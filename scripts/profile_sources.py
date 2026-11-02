"""Run acquisition/preparation and emit aggregate-only evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from alfabetizacao_ml.data import acquire_sources, prepare_data
from alfabetizacao_ml.data.acquisition import sha256_file


def validate_outputs(result: dict, coverage: dict, receipt: dict) -> dict:
    """Audit integrity only: never compute 2025 target rates or performance."""
    root = Path(result["receipt_path"]).parent
    for relative, expected in receipt["outputs"].items():
        path = root / relative
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("Output digest differs from the preparation receipt")
    checks = {}
    with duckdb.connect() as con:
        for year, counts in coverage["students"].items():
            path = str(root / "marts" / f"alunos_{year}.parquet")
            rows, invalid = con.execute("""SELECT count(*),count(*) FILTER (WHERE
                ano_contexto IS NOT NULL AND ano_contexto<>ano-1
                OR peso_aluno<=0 OR NOT isfinite(peso_aluno)
                OR alfabetizado NOT IN (0,1) OR risco<>1-alfabetizado
                OR serie<>2 OR rede NOT IN ('estadual','municipal')) FROM read_parquet(?)""", [path]).fetchone()
            excluded = sum(value for key, value in counts.items() if key.startswith("excluded_"))
            if invalid or rows != counts["eligible_rows"] or rows + excluded != counts["raw_rows"]:
                raise ValueError("Real student mart integrity or exclusion accounting failed")
            checks[year] = {"rows": rows, "invalid": invalid, "exclusion_accounting": "passed"}
        projection = str(root / "marts" / "municipios_2026.parquet")
        invalid = con.execute("SELECT count(*) FROM read_parquet(?) WHERE ano<>2026 OR ano_contexto<>2025 OR taxa_alfabetizacao IS NOT NULL", [projection]).fetchone()[0]
        if invalid:
            raise ValueError("Projection context is not exactly 2025 or contains observed targets")
    return {"status": "passed", "output_hashes": len(receipt["outputs"]), "students": checks,
            "projection_context": "2025", "target_relationships_2025_examined": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=["update", "reproduce"], default="reproduce")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = acquire_sources(args.cache_dir, args.mode, args.manifest)
    result = prepare_data(args.cache_dir, manifest)
    coverage = json.loads(Path(result["coverage_path"]).read_text(encoding="utf-8"))
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    validation = validate_outputs(result, coverage, receipt)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"coverage": coverage, "receipt": receipt, "validation": validation},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_id": result["snapshot_id"], "status": receipt["status"],
                      "students_by_year": {year: item["eligible_rows"]
                                           for year, item in coverage["students"].items()}}, indent=2))


if __name__ == "__main__":
    main()
