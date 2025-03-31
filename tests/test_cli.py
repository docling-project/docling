from pathlib import Path

from typer.testing import CliRunner

from docling.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0


def test_cli_convert(tmp_path):
    source = "./tests/data/pdf/2305.03393v1-pg9.pdf"
    output = tmp_path / "out"
    output.mkdir()
    result = runner.invoke(app, [source, "--output", str(output)])
    assert result.exit_code == 0
    converted = output / f"{Path(source).stem}.md"
    assert converted.exists()
