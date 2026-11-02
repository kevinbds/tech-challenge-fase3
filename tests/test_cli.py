import pytest
import typer
from typer.testing import CliRunner

from alfabetizacao_ml.cli import app, checked_private_path, private_cache

runner = CliRunner()


def test_doctor_does_not_expose_environment_secrets(monkeypatch):
    monkeypatch.setenv("EXAMPLE_PRIVATE_VALUE", "SYNTHETIC_SECRET_MUST_NOT_APPEAR")
    response = runner.invoke(app, ["doctor"])
    assert response.exit_code == 0
    assert "SYNTHETIC_SECRET_MUST_NOT_APPEAR" not in response.stdout


def test_help_documents_analysis_commands():
    response = runner.invoke(app, ["--help"])
    assert response.exit_code == 0
    assert "report" in response.stdout
    assert "acquire" in response.stdout
    assert "train" in response.stdout


def test_empty_private_environment_uses_home_cache(monkeypatch):
    monkeypatch.setenv("FASE3_PRIVATE_DIR", "")
    assert private_cache().name == "fiap-fase3"
    assert private_cache().is_absolute()


def test_private_cache_cannot_be_placed_inside_public_checkout(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic'", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.BadParameter, match="fora do projeto"):
        checked_private_path(tmp_path / "bronze")
    assert (
        checked_private_path(tmp_path.parent / "restricted-cache")
        == tmp_path.parent / "restricted-cache"
    )
