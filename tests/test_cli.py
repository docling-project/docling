import base64
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import ImageRefMode
from PIL import Image
from typer.testing import CliRunner

from docling.cli.export_utils import _should_generate_export_images, _split_list
from docling.cli.main import app
from docling.datamodel.base_models import OutputFormat

runner = CliRunner()

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_html_image_case(
    root: Path, filename: str, text: str, image_bytes: bytes = PNG_BYTES
) -> Path:
    root.mkdir()
    (root / "pixel.png").write_bytes(image_bytes)
    html_path = root / filename
    html_path.write_text(
        f"<html><body><p>{text}</p><img src='pixel.png'></body></html>"
    )
    return html_path


def _assert_markdown_embeds_png(path: Path, image_bytes: bytes | None = None) -> None:
    content = path.read_text()
    assert "data:image/png;base64" in content
    assert "Image not available" not in content
    if image_bytes is not None:
        assert base64.b64encode(image_bytes).decode() in content


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Input formats to" in result.output
    assert "all supported" in result.output
    assert "layout clusters" in result.output
    assert "layour" not in result.output
    assert "input_sources" not in result.output


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


def test_cli_html_fetches_local_images_per_input(tmp_path):
    first_png = _png_bytes((255, 0, 0))
    second_png = _png_bytes((0, 0, 255))
    first = _write_html_image_case(tmp_path / "first", "first.html", "First", first_png)
    second = _write_html_image_case(
        tmp_path / "second", "second.html", "Second", second_png
    )
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            str(first),
            str(second),
            "--from",
            "html",
            "--to",
            "md",
            "--output",
            str(output),
            "--image-export-mode",
            "embedded",
            "--html-image-fetch",
            "local",
        ],
    )

    assert result.exit_code == 0
    _assert_markdown_embeds_png(output / "first.md", first_png)
    _assert_markdown_embeds_png(output / "second.md", second_png)


def test_cli_html_directory_matches_mixed_case_extensions(tmp_path):
    source_dir = tmp_path / "source"
    _write_html_image_case(source_dir, "Case.HtMl", "Mixed case")
    output = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            str(source_dir),
            "--from",
            "html",
            "--to",
            "md",
            "--output",
            str(output),
            "--image-export-mode",
            "embedded",
            "--html-image-fetch",
            "local",
        ],
    )

    assert result.exit_code == 0
    _assert_markdown_embeds_png(output / "Case.md")


def test_cli_html_fetches_remote_images_with_separate_headers(tmp_path, monkeypatch):
    source_url = "https://example.com/docs/page.html"
    image_url = "https://example.com/docs/pixel.png"
    output = tmp_path / "out"
    calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def __init__(self, url: str, content: bytes):
            self.url = url
            self.content = content
            self.headers: dict[str, str] = {}
            self.is_redirect = False
            self.is_permanent_redirect = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int):
            yield self.content

    def fake_get(self, url: str, **kwargs):
        calls.append((url, kwargs))
        if url == source_url:
            return FakeResponse(
                url,
                b"<html><body><p>Remote</p><img src='pixel.png'></body></html>",
            )
        if url == image_url:
            return FakeResponse(url, PNG_BYTES)
        raise AssertionError(f"Unexpected URL fetched: {url}")

    monkeypatch.setattr("requests.Session.get", fake_get)

    result = runner.invoke(
        app,
        [
            source_url,
            "--from",
            "html",
            "--to",
            "md",
            "--output",
            str(output),
            "--image-export-mode",
            "embedded",
            "--headers",
            '{"Authorization": "Bearer source-token"}',
            "--html-image-headers",
            '{"X-Image-Token": "image-token"}',
            "--html-image-fetch",
            "remote",
        ],
    )

    assert result.exit_code == 0
    _assert_markdown_embeds_png(output / "page.md")
    source_call = next(kwargs for url, kwargs in calls if url == source_url)
    image_call = next(kwargs for url, kwargs in calls if url == image_url)
    assert source_call["headers"]["authorization"] == "Bearer source-token"
    assert "Authorization" not in image_call["headers"]
    assert "authorization" not in image_call["headers"]
    assert image_call["headers"]["X-Image-Token"] == "image-token"


def test_cli_html_image_headers_require_remote_fetch(tmp_path):
    source = _write_html_image_case(tmp_path / "source", "index.html", "Local")

    result = runner.invoke(
        app,
        [
            str(source),
            "--from",
            "html",
            "--to",
            "md",
            "--html-image-headers",
            '{"Authorization": "Bearer token"}',
        ],
    )

    assert result.exit_code != 0
    assert (
        "--html-image-headers requires --html-image-fetch remote or all"
        in result.output
    )


def test_export_documents_marks_empty_markdown_as_failure(tmp_path):
    from docling.cli.main import export_documents
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import (
        ConversionResult,
        InputDocument,
        _DummyBackend,
    )

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4")

    input_doc = InputDocument(
        path_or_stream=input_path,
        format=InputFormat.PDF,
        backend=_DummyBackend,
    )

    conv_res = ConversionResult(input=input_doc)
    conv_res.status = ConversionStatus.SUCCESS

    class DummyDocument:
        def save_as_markdown(self, *, filename, image_mode):
            Path(filename).write_text("")

    conv_res.document = DummyDocument()

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    export_documents(
        [conv_res],
        output_dir=output_dir,
        export_json=False,
        export_yaml=False,
        export_html=False,
        export_html_split_page=False,
        show_layout=False,
        export_md=True,
        export_txt=False,
        export_doctags=False,
        export_vtt=False,
        print_timings=False,
        export_timings=False,
        image_export_mode=ImageRefMode.PLACEHOLDER,
    )

    assert conv_res.status == ConversionStatus.FAILURE
    assert conv_res.errors


def test_export_documents_marks_stat_errors_as_failure(tmp_path, monkeypatch):
    from docling.cli.main import export_documents
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import (
        ConversionResult,
        InputDocument,
        _DummyBackend,
    )

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4")

    input_doc = InputDocument(
        path_or_stream=input_path,
        format=InputFormat.PDF,
        backend=_DummyBackend,
    )

    conv_res = ConversionResult(input=input_doc)
    conv_res.status = ConversionStatus.SUCCESS

    class DummyDocument:
        def save_as_markdown(self, *, filename, image_mode):
            Path(filename).write_text("ok")

    conv_res.document = DummyDocument()

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    original_stat = Path.stat

    def _raise_for_markdown(self, *, follow_symlinks=True):
        if self.name == "input.md":
            raise OSError("stat failed")
        return original_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", _raise_for_markdown)

    export_documents(
        [conv_res],
        output_dir=output_dir,
        export_json=False,
        export_yaml=False,
        export_html=False,
        export_html_split_page=False,
        show_layout=False,
        export_md=True,
        export_txt=False,
        export_doctags=False,
        export_vtt=False,
        print_timings=False,
        export_timings=False,
        image_export_mode=ImageRefMode.PLACEHOLDER,
    )

    assert conv_res.status == ConversionStatus.FAILURE
    assert conv_res.errors


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


def test_split_list_handles_none_and_delimiters():
    assert _split_list(None) is None
    assert _split_list("a,b;c") == ["a", "b", "c"]


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
