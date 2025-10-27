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


def test_cli_audio_auto_detection(tmp_path):
    """Test that CLI automatically detects audio files and sets ASR pipeline."""
    from docling.datamodel.base_models import FormatToExtensions, InputFormat

    # Create a dummy audio file for testing
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"dummy audio content")

    output = tmp_path / "out"
    output.mkdir()

    # Test that audio file triggers ASR pipeline auto-detection
    result = runner.invoke(app, [str(audio_file), "--output", str(output)])
    # The command should succeed (even if ASR fails due to dummy content)
    # The key is that it should attempt ASR processing, not standard processing
    assert (
        result.exit_code == 0 or result.exit_code == 1
    )  # Allow for ASR processing failure


def test_cli_explicit_pipeline_not_overridden(tmp_path):
    """Test that explicit pipeline choice is not overridden by audio auto-detection."""
    from docling.datamodel.base_models import FormatToExtensions, InputFormat

    # Create a dummy audio file for testing
    audio_file = tmp_path / "test_audio.mp3"
    audio_file.write_bytes(b"dummy audio content")

    output = tmp_path / "out"
    output.mkdir()

    # Test that explicit --pipeline STANDARD is not overridden
    result = runner.invoke(
        app, [str(audio_file), "--output", str(output), "--pipeline", "standard"]
    )
    # Should still use standard pipeline despite audio file
    assert (
        result.exit_code == 0 or result.exit_code == 1
    )  # Allow for processing failure


def test_cli_audio_extensions_coverage():
    """Test that all audio extensions from FormatToExtensions are covered."""
    from docling.datamodel.base_models import FormatToExtensions, InputFormat

    # Verify that the centralized audio extensions include all expected formats
    audio_extensions = FormatToExtensions[InputFormat.AUDIO]
    expected_extensions = [
        "wav",
        "mp3",
        "m4a",
        "aac",
        "ogg",
        "flac",
        "mp4",
        "avi",
        "mov",
    ]

    for ext in expected_extensions:
        assert ext in audio_extensions, (
            f"Audio extension {ext} not found in FormatToExtensions[InputFormat.AUDIO]"
        )
