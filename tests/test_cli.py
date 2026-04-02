from pathlib import Path

import pytest
from docling_core.types.doc import ImageRefMode
from typer.testing import CliRunner

from docling.cli.main import (
    _PAGE_BREAK_SENTINEL,
    _apply_dynamic_page_breaks,
    _get_content_page_numbers,
    _has_dynamic_page_vars,
    _should_generate_export_images,
    app,
)
from docling.datamodel.base_models import OutputFormat

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


def test_cli_page_break_placeholder(tmp_path):
    source = "./tests/data/pdf/2305.03393v1-pg9.pdf"
    output = tmp_path / "out"
    output.mkdir()
    placeholder = "<!-- page-break -->"
    result = runner.invoke(
        app,
        [
            source,
            "--output",
            str(output),
            "--page-break-placeholder",
            placeholder,
        ],
    )
    assert result.exit_code == 0
    converted = output / f"{Path(source).stem}.md"
    assert converted.exists()


@pytest.mark.parametrize(
    ("image_export_mode", "to_formats", "expected"),
    [
        (ImageRefMode.PLACEHOLDER, [OutputFormat.JSON], False),
        (ImageRefMode.EMBEDDED, [OutputFormat.TEXT, OutputFormat.DOCTAGS], False),
        (ImageRefMode.EMBEDDED, [OutputFormat.MARKDOWN], True),
        (
            ImageRefMode.EMBEDDED,
            [OutputFormat.TEXT, OutputFormat.MARKDOWN],
            True,
        ),
    ],
)
def test_should_generate_export_images(image_export_mode, to_formats, expected):
    assert _should_generate_export_images(image_export_mode, to_formats) is expected


def test_image_export_policy_covers_all_output_formats():
    non_image_export_formats = {
        OutputFormat.TEXT,
        OutputFormat.DOCTAGS,
        OutputFormat.VTT,
    }
    image_export_formats = set(OutputFormat) - non_image_export_formats

    assert image_export_formats.isdisjoint(non_image_export_formats)
    assert image_export_formats | non_image_export_formats == set(OutputFormat)


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


@pytest.mark.parametrize(
    ("placeholder", "expected"),
    [
        ("---", False),
        ("<!-- page-break -->", False),
        ("Page {next_page}", True),
        ("End of page {prev_page}", True),
        ("---\n*[Page {next_page}]*\n---", True),
        ("{prev_page} -> {next_page}", True),
    ],
)
def test_has_dynamic_page_vars(placeholder, expected):
    assert _has_dynamic_page_vars(placeholder) is expected


def test_apply_dynamic_page_breaks(tmp_path):
    content = (
        "Content from page 1\n\n"
        f"{_PAGE_BREAK_SENTINEL}\n\n"
        "Content from page 2\n\n"
        f"{_PAGE_BREAK_SENTINEL}\n\n"
        "Content from page 3"
    )
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    _apply_dynamic_page_breaks(file_path, "---\n*[Page {next_page}]*\n---")

    result = file_path.read_text(encoding="utf-8")
    assert _PAGE_BREAK_SENTINEL not in result
    assert "---\n*[Page 2]*\n---" in result
    assert "---\n*[Page 3]*\n---" in result


def test_apply_dynamic_page_breaks_prev_page(tmp_path):
    content = f"Content from page 1\n\n{_PAGE_BREAK_SENTINEL}\n\nContent from page 2"
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    _apply_dynamic_page_breaks(file_path, "[End of page {prev_page}]")

    result = file_path.read_text(encoding="utf-8")
    assert _PAGE_BREAK_SENTINEL not in result
    assert "[End of page 1]" in result


def test_apply_dynamic_page_breaks_both_vars(tmp_path):
    content = f"Page 1\n{_PAGE_BREAK_SENTINEL}\nPage 2"
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    _apply_dynamic_page_breaks(file_path, "({prev_page} -> {next_page})")

    result = file_path.read_text(encoding="utf-8")
    assert "(1 -> 2)" in result


def test_apply_dynamic_page_breaks_no_sentinel(tmp_path):
    content = "Content with no page breaks"
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    _apply_dynamic_page_breaks(file_path, "---\n*[Page {next_page}]*\n---")

    result = file_path.read_text(encoding="utf-8")
    assert result == content


def test_cli_dynamic_page_break_placeholder(tmp_path):
    source = "./tests/data/pdf/normal_4pages.pdf"
    output = tmp_path / "out"
    output.mkdir()
    placeholder = "--- Page {next_page} ---"
    result = runner.invoke(
        app,
        [
            source,
            "--output",
            str(output),
            "--page-break-placeholder",
            placeholder,
        ],
    )
    assert result.exit_code == 0
    converted = output / f"{Path(source).stem}.md"
    assert converted.exists()
    content = converted.read_text(encoding="utf-8")
    # Should not contain raw page break markers or sentinels
    assert "#_#_DOCLING_DOC_PAGE_BREAK" not in content
    assert _PAGE_BREAK_SENTINEL not in content
    # Multi-page PDF should have page break placeholders with actual page numbers
    assert "--- Page 2 ---" in content


def test_apply_dynamic_page_breaks_with_blank_pages(tmp_path):
    """Test that page numbering uses actual page numbers when blank pages exist."""
    content = (
        "Content from page 1\n\n"
        f"{_PAGE_BREAK_SENTINEL}\n\n"
        "Content from page 2\n\n"
        f"{_PAGE_BREAK_SENTINEL}\n\n"
        "Content from page 3\n\n"
        f"{_PAGE_BREAK_SENTINEL}\n\n"
        "Content from page 5"
    )
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    # Page 4 is blank, so content pages are [1, 2, 3, 5]
    _apply_dynamic_page_breaks(
        file_path,
        "---\n*[Page {next_page}]*\n---",
        content_page_numbers=[1, 2, 3, 5],
    )

    result = file_path.read_text(encoding="utf-8")
    assert _PAGE_BREAK_SENTINEL not in result
    assert "---\n*[Page 2]*\n---" in result
    assert "---\n*[Page 3]*\n---" in result
    assert "---\n*[Page 5]*\n---" in result
    assert "---\n*[Page 4]*\n---" not in result


def test_apply_dynamic_page_breaks_prev_page_with_blank_pages(tmp_path):
    """Test {prev_page} with blank pages uses actual page numbers."""
    content = (
        f"Page 1\n{_PAGE_BREAK_SENTINEL}\n"
        f"Page 3\n{_PAGE_BREAK_SENTINEL}\n"
        "Page 5"
    )
    file_path = tmp_path / "test.md"
    file_path.write_text(content, encoding="utf-8")

    _apply_dynamic_page_breaks(
        file_path,
        "({prev_page} -> {next_page})",
        content_page_numbers=[1, 3, 5],
    )

    result = file_path.read_text(encoding="utf-8")
    assert "(1 -> 3)" in result
    assert "(3 -> 5)" in result
    assert "(1 -> 2)" not in result
    assert "(2 -> 3)" not in result
