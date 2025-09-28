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


def test_cli_stats_feature(tmp_path):
    """Test the new --stats feature displays performance statistics."""
    source = "./tests/data/pdf/2305.03393v1-pg9.pdf"
    output = tmp_path / "out"
    output.mkdir()
    result = runner.invoke(app, [source, "--stats", "--output", str(output)])
    assert result.exit_code == 0

    # Check that the stats output contains expected sections
    output_text = result.stdout
    assert "📊 Performance Statistics" in output_text
    assert "Total Documents" in output_text
    assert "Successful" in output_text
    assert "Total Time" in output_text
    assert "Throughput" in output_text

    # Check that pipeline timings are included
    assert "⚙️  Pipeline Timings" in output_text
    assert "Operation" in output_text

    # Verify the converted file exists
    converted = output / f"{Path(source).stem}.md"
    assert converted.exists()
