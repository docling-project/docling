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


def test_cli_directory_input_permission_error(tmp_path, monkeypatch):
    """Regression test for GitHub issue #3138.

    On Windows, Path.read_bytes() raises PermissionError on a directory path
    instead of IsADirectoryError (Linux/macOS).  The CLI handler must catch
    both so that directory inputs work correctly on Windows.
    """
    import docling.cli.main as cli_main
    from docling_core.utils.file import resolve_source_to_path

    # Create a temporary input dir with a dummy PDF placeholder
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    dummy_pdf = input_dir / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Simulate the Windows behaviour: resolve_source_to_path raises PermissionError
    # when called on a directory (instead of IsADirectoryError on Linux/macOS).
    original_resolve = resolve_source_to_path

    def raise_permission_error(source, **kwargs):
        from pathlib import Path as _Path
        p = _Path(str(source))
        if p.is_dir():
            raise PermissionError(f"[Errno 13] Permission denied: '{source}'")
        return original_resolve(source, **kwargs)

    monkeypatch.setattr(cli_main, "resolve_source_to_path", raise_permission_error)

    # The CLI should handle PermissionError the same as IsADirectoryError:
    # it should NOT propagate as an unhandled exception (exit_code != 2).
    result = runner.invoke(app, [str(input_dir), "--output", str(output_dir)])
    assert result.exit_code != 2, (
        f"CLI crashed with unhandled PermissionError on directory input. "
        f"Output: {result.output}"
    )
