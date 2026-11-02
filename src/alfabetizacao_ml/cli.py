"""Comandos para aquisição dos dados, modelagem e análise de resultados."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Annotated

import typer

from alfabetizacao_ml import __version__

app = typer.Typer(
    help="Alfabetização no Brasil: dados, modelos e resultados.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def emit(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))


def private_cache() -> Path:
    return Path(os.environ.get("FASE3_PRIVATE_DIR") or str(Path.home() / ".cache" / "fiap-fase3"))


def checked_private_path(path: Path) -> Path:
    """Mantém fontes individuais e arquivos de execução fora do repositório."""
    resolved, working = path.resolve(), Path.cwd().resolve()
    if (working / "pyproject.toml").is_file() and resolved.is_relative_to(working):
        raise typer.BadParameter("O diretório de dados deve ficar fora do projeto.")
    return resolved


@app.command()
def version() -> None:
    """Exibe a versão do pacote."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Confere o ambiente de execução."""
    versions = {}
    for name in ["duckdb", "pandas", "pyarrow", "scikit-learn", "shap", "playwright"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    emit(
        {
            "project": __version__,
            "python": platform.python_version(),
            "platform": platform.system(),
            "packages": versions,
            "tools": {name: shutil.which(name) is not None for name in ["git", "curl"]},
        }
    )


@app.command()
def acquire(
    cache: Annotated[Path | None, typer.Option(help="Diretório local para as fontes.")] = None,
    mode: Annotated[
        str, typer.Option(help="update coleta; reproduce verifica a edição fixada.")
    ] = "update",
    expected_manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Obtém os arquivos oficiais."""
    from alfabetizacao_ml.data import acquire_sources

    if mode not in {"update", "reproduce"}:
        raise typer.BadParameter("mode deve ser update ou reproduce")
    manifest = acquire_sources(
        checked_private_path(cache or private_cache()),
        mode=mode,
        expected_manifest=expected_manifest,
    )
    emit({"status": "acquired", "manifest": manifest})


@app.command()
def prepare(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    cache: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Valida as fontes e constrói as camadas e os conjuntos de análise."""
    from alfabetizacao_ml.data import prepare_data

    emit(prepare_data(checked_private_path(cache or private_cache()), manifest))


@app.command()
def train(
    marts: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option(help="Diretório dos modelos e resultados.")],
    protocol: Annotated[Path, typer.Option(exists=True)] = Path("config/protocol.json"),
    smoke: Annotated[
        bool, typer.Option(help="Execução reduzida para testar a integração.")
    ] = False,
) -> None:
    """Treina e avalia os modelos com separação temporal."""
    from alfabetizacao_ml.modeling import run_experiments

    result = run_experiments(marts, checked_private_path(output), protocol, smoke=smoke)
    emit(
        {"status": result["status"], "run_id": result["run_id"], "metrics": output / "metrics.json"}
    )


@app.command()
def report(
    metrics: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
    repo_url: Annotated[str, typer.Option()] = "https://github.com/kevinbds/tech-challenge-fase3",
    supplement: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    """Gera o relatório técnico, a apresentação e as tabelas de resultados."""
    from alfabetizacao_ml.visualization import build_reports

    emit(build_reports(metrics, output, repo_url=repo_url, scenarios_path=supplement))


@app.command()
def scenarios(
    metrics: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    targets: Annotated[Path, typer.Option(exists=True)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Compara a manutenção da taxa de 2026 com as metas de 2027–2030."""
    from alfabetizacao_ml.scenarios import build_goal_scenarios

    emit(build_goal_scenarios(metrics, targets, output))


@app.command("run")
def run_pipeline(
    output: Annotated[Path, typer.Option()],
    reports: Annotated[Path, typer.Option()],
    cache: Annotated[Path | None, typer.Option()] = None,
    protocol: Annotated[Path, typer.Option(exists=True)] = Path("config/protocol.json"),
    sources_manifest: Annotated[Path | None, typer.Option()] = Path("config/sources.lock.json"),
    mode: Annotated[str, typer.Option()] = "reproduce",
    smoke: Annotated[bool, typer.Option()] = False,
) -> None:
    """Executa a análise desde a aquisição até os relatórios."""
    from alfabetizacao_ml.workflow import run_workflow

    result = run_workflow(
        project_root=Path.cwd(),
        cache_dir=cache or private_cache(),
        output_dir=output,
        reports_dir=reports,
        protocol_path=protocol,
        sources_manifest=sources_manifest,
        mode=mode,
        smoke=smoke,
    )
    emit({"status": result["status"], "run_id": result.get("run_id")})


if __name__ == "__main__":
    app()
