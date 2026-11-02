"""Orquestra o fluxo científico e produz recibos vinculados às entradas."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(root: Path) -> dict:
    """Identifica o código sem coletar identidade da conta ou ambiente secreto."""
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip() if result.returncode == 0 else None
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return {
        "git_sha": revision,
        "tracked_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_tree_sha256": digest.hexdigest(),
    }


def run_workflow(
    *,
    project_root: Path,
    cache_dir: Path,
    output_dir: Path,
    reports_dir: Path,
    protocol_path: Path,
    sources_manifest: Path | None,
    mode: str = "reproduce",
    smoke: bool = False,
    repo_url: str = "https://github.com/kevinbds/tech-challenge-fase3",
) -> dict:
    """Executa aquisição, preparação, modelagem e relatórios."""
    from alfabetizacao_ml.data import acquire_sources, prepare_data
    from alfabetizacao_ml.eda import run_eda
    from alfabetizacao_ml.modeling import run_experiments
    from alfabetizacao_ml.scenarios import build_goal_scenarios
    from alfabetizacao_ml.visualization import build_reports

    if mode not in {"reproduce", "update"}:
        raise ValueError("mode must be reproduce or update")
    if mode == "reproduce" and sources_manifest is None:
        raise ValueError("reproduce requires an expected source manifest")
    public_root = project_root.resolve()
    if cache_dir.resolve().is_relative_to(public_root):
        raise ValueError("Private source cache must be outside the public project")
    if output_dir.resolve().is_relative_to(public_root):
        raise ValueError("Private modeling output must be outside the public project")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {
        "schema_version": 1,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "code": source_identity(project_root),
        "protocol_sha256": file_hash(protocol_path),
        "mode": mode,
        "smoke": smoke,
        "stages": [],
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "pyarrow", "duckdb", "scikit-learn", "shap"]
            },
        },
    }
    receipt_path = output_dir / "execution-receipt.json"

    def checkpoint() -> None:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    checkpoint()
    try:
        manifest = Path(acquire_sources(cache_dir, mode=mode, expected_manifest=sources_manifest))
        receipt["input_manifest_sha256"] = file_hash(manifest)
        receipt["stages"].append({"stage": "acquire", "status": "passed"})
        checkpoint()
        prepared = prepare_data(cache_dir, manifest)
        receipt["stages"].append({"stage": "prepare", "status": "passed"})
        checkpoint()
        run_eda(Path(prepared["marts_dir"]), output_dir / "eda")
        receipt["stages"].append({"stage": "development_eda", "status": "passed"})
        checkpoint()
        metrics = run_experiments(
            Path(prepared["marts_dir"]), output_dir, protocol_path, smoke=smoke
        )
        metrics_path = output_dir / "metrics.json"
        receipt["metrics_sha256"] = file_hash(metrics_path)
        receipt["run_id"] = metrics["run_id"]
        receipt["stages"].append({"stage": "modeling", "status": metrics["status"]})
        checkpoint()
        supplement = build_goal_scenarios(
            metrics_path, Path(prepared["marts_dir"]), output_dir / "scenarios"
        )
        receipt["stages"].append({"stage": "future_goal_scenarios", "status": "passed"})
        checkpoint()
        build_reports(
            metrics_path,
            reports_dir,
            repo_url=repo_url,
            scenarios_path=Path(supplement["scenarios_json"]),
        )
        receipt["stages"].append({"stage": "reports", "status": "passed"})
        checkpoint()
        receipt["status"] = "smoke_completed" if smoke else "completed"
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        checkpoint()
    except Exception as exc:
        # The receipt records the error type, never a source row or credential.
        receipt["status"] = "failed"
        receipt["error_type"] = type(exc).__name__
        checkpoint()
        raise
    return receipt
