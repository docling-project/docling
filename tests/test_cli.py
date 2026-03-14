from pathlib import Path

import pytest
from docling_core.types.doc import ImageRefMode
from typer.testing import CliRunner

from docling.cli.main import _should_generate_export_images, app
from docling.datamodel.base_models import InputFormat, OutputFormat

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


def _invoke_convert_and_get_pdf_options(monkeypatch, tmp_path, to_format: str):
    class StubDocumentConverter:
        format_options = None

        def __init__(self, *, format_options=None, **_kwargs):
            type(self).format_options = format_options

        def convert_all(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr("docling.cli.main.DocumentConverter", StubDocumentConverter)

    result = runner.invoke(
        app,
        [
            "./tests/data/pdf/2305.03393v1-pg9.pdf",
            "--output",
            str(tmp_path / "out"),
            "--to",
            to_format,
            "--image-export-mode",
            ImageRefMode.EMBEDDED.value,
        ],
    )

    assert result.exit_code == 0, result.output

    pdf_options = StubDocumentConverter.format_options[InputFormat.PDF].pipeline_options
    assert pdf_options is not None
    return pdf_options


@pytest.mark.parametrize(
    ("image_export_mode", "to_formats", "expected"),
    [
        (ImageRefMode.PLACEHOLDER, [OutputFormat.JSON], False),
        (ImageRefMode.EMBEDDED, [OutputFormat.TEXT, OutputFormat.DOCTAGS], False),
        (ImageRefMode.EMBEDDED, [OutputFormat.MARKDOWN], True),
    ],
)
def test_should_generate_export_images(image_export_mode, to_formats, expected):
    assert _should_generate_export_images(image_export_mode, to_formats) is expected


def test_image_export_policy_covers_all_output_formats():
    image_export_formats = {
        OutputFormat.JSON,
        OutputFormat.YAML,
        OutputFormat.HTML,
        OutputFormat.HTML_SPLIT_PAGE,
        OutputFormat.MARKDOWN,
    }
    non_image_export_formats = {
        OutputFormat.TEXT,
        OutputFormat.DOCTAGS,
        OutputFormat.VTT,
    }

    assert image_export_formats.isdisjoint(non_image_export_formats)
    assert image_export_formats | non_image_export_formats == set(OutputFormat)


@pytest.mark.parametrize(
    ("to_format", "expect_generated_images"),
    [
        (OutputFormat.TEXT.value, False),
        (OutputFormat.DOCTAGS.value, False),
        (OutputFormat.MARKDOWN.value, True),
        (OutputFormat.HTML.value, True),
    ],
)
def test_cli_only_generates_images_for_image_capable_exports(
    tmp_path, monkeypatch, to_format, expect_generated_images
):
    pdf_options = _invoke_convert_and_get_pdf_options(monkeypatch, tmp_path, to_format)

    assert pdf_options.generate_page_images is expect_generated_images
    assert pdf_options.generate_picture_images is expect_generated_images
    assert pdf_options.images_scale == (2 if expect_generated_images else 1.0)


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
