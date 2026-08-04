"""Tests for the Apple Pages (``.pages``) document backend.

Test Data Attribution
---------------------
The ``.pages`` fixtures are synthetic. Apple Pages runs only on macOS and iOS, so
the containers are built by ``scripts/make_iwork_pages_fixtures.py``.

``pages_modern.pages`` mirrors, member for member, the layout of a real Pages 5+
document reported by a maintainer on the pull request: ``Index/*.iwa`` content,
``Metadata/*`` plists, root-level ``preview.jpg`` / ``preview-micro.jpg`` /
``preview-web.jpg``, and **no** ``QuickLook/Preview.pdf``. The IWA chunk framing
and Snappy payload are real; only the protobuf inside is a stand-in, and nothing
here decodes it.

The two ``pages_legacy09*`` fixtures reproduce the iWork '09 layout, which is the
only generation that embedded a convertible PDF render.
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.iwork_backend import IWorkPagesDocumentBackend
from docling.backend.noop_backend import NoOpBackend
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

SOURCES = Path("./tests/data/pages/sources")
MODERN = SOURCES / "pages_modern.pages"
LEGACY = SOURCES / "pages_legacy09.pages"
LEGACY_NO_PREVIEW = SOURCES / "pages_legacy09_no_preview.pages"


def _backend(
    path: Path, options: IWorkBackendOptions | None = None
) -> IWorkPagesDocumentBackend:
    """Instantiate the backend directly.

    ``InputDocument`` converts a ``DocumentLoadError`` raised during backend init
    into a rejection, so going through it would hide the failure modes these
    tests are about. ``NoOpBackend`` is only used to build a well-formed
    ``InputDocument`` (hash, limits, filename) for the backend under test.
    """
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.IWORK_PAGES,
        backend=NoOpBackend,
    )
    return IWorkPagesDocumentBackend(in_doc, path, options)


def test_detects_pages_from_path_and_named_stream():
    """`.pages` is a ZIP, so detection must not stop at ``application/zip``."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])

    assert conv_input._guess_format(MODERN) == InputFormat.IWORK_PAGES

    stream = DocumentStream(name="report.pages", stream=BytesIO(MODERN.read_bytes()))
    assert conv_input._guess_format(stream) == InputFormat.IWORK_PAGES


def test_extensionless_pages_stream_is_not_claimed():
    """Without the extension a Pages container is indistinguishable from Keynote
    and Numbers, so the backend must not claim it rather than guess wrong."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="blob", stream=BytesIO(MODERN.read_bytes()))

    assert conv_input._guess_format(stream) is None


def test_legacy_pages_are_read_through_the_embedded_preview():
    """iWork '09 containers embed a full PDF render, which converts normally."""
    backend = _backend(LEGACY)
    try:
        assert backend.is_valid()
        assert backend.page_count() == 2

        page = backend.load_page(0)
        text = " ".join(cell.text for cell in page.get_text_cells())
        assert "Docling Pages fixture" in text
    finally:
        backend.unload()


def test_legacy_pages_backend_accepts_a_stream():
    stream = BytesIO(LEGACY.read_bytes())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.IWORK_PAGES,
        backend=NoOpBackend,
        filename="report.pages",
    )
    backend = IWorkPagesDocumentBackend(in_doc, stream)
    try:
        assert backend.is_valid()
        assert backend.page_count() == 2
    finally:
        backend.unload()


def test_modern_pages_are_rejected_without_impossible_advice():
    """Pages 5+ (2013 onwards) embeds no PDF render at all — the QuickLook PDF was
    an iWork '08/'09 feature. Telling such a user to enable "Include preview in
    document" would be advice they cannot act on, so the error must instead name
    the real cause and point at export."""
    with pytest.raises(DocumentLoadError) as exc_info:
        _backend(MODERN)

    message = str(exc_info.value)
    assert "Pages 5 or later" in message
    assert "Index/*.iwa" in message
    assert "Export the document to PDF" in message
    assert "Include preview in document" not in message


def test_legacy_missing_preview_reports_the_save_setting():
    """For iWork '09 the save-setting advice is genuinely actionable, so it stays."""
    with pytest.raises(DocumentLoadError, match="Include preview in document"):
        _backend(LEGACY_NO_PREVIEW)


def test_zip_without_pages_index_is_rejected(tmp_path: Path):
    other_zip = tmp_path / "not_really.pages"
    with zipfile.ZipFile(other_zip, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")

    with pytest.raises(DocumentLoadError, match="does not look like a Pages document"):
        _backend(other_zip)


def test_non_zip_input_is_rejected(tmp_path: Path):
    broken = tmp_path / "broken.pages"
    broken.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentLoadError, match="not a readable ZIP container"):
        _backend(broken)


def test_archive_limits_are_enforced():
    """The container is attacker-controlled, so the limits must bite before the
    preview is read into memory."""
    with pytest.raises(DocumentLoadError, match="max_file_bytes"):
        _backend(LEGACY, IWorkBackendOptions(max_file_bytes=64))

    with pytest.raises(DocumentLoadError, match="max_member_count"):
        _backend(LEGACY, IWorkBackendOptions(max_member_count=1))

    with pytest.raises(DocumentLoadError, match="max_total_bytes"):
        _backend(LEGACY, IWorkBackendOptions(max_total_bytes=32))


def test_pdf_backend_options_are_widened():
    """PdfFormatOption (used by the CLI) passes plain PDF backend options; the
    backend must still come up with usable archive limits."""
    from docling.datamodel.backend_options import PdfBackendOptions

    in_doc = InputDocument(
        path_or_stream=LEGACY,
        format=InputFormat.IWORK_PAGES,
        backend=NoOpBackend,
    )
    backend = IWorkPagesDocumentBackend(
        in_doc, LEGACY, PdfBackendOptions(enforce_same_font=False)
    )
    try:
        assert backend.is_valid()
        assert backend.options.enforce_same_font is False
        assert backend.options.max_member_count > 0
    finally:
        backend.unload()


def test_unloaded_backend_refuses_page_access():
    backend = _backend(LEGACY)
    backend.unload()

    assert not backend.is_valid()
    with pytest.raises(RuntimeError, match="already been unloaded"):
        backend.page_count()


@pytest.mark.ml_pdf_model
def test_end_to_end_conversion():
    converter = DocumentConverter(allowed_formats=[InputFormat.IWORK_PAGES])
    result = converter.convert(LEGACY)

    markdown = result.document.export_to_markdown()
    assert "Docling Pages fixture" in markdown
    assert "Second page body text." in markdown

    # StandardPdfPipeline stamps every document it produces with the PDF mimetype
    # (image and METS-GBS inputs behave the same way), so only the filename
    # identifies the document as Pages.
    assert result.document.origin is not None
    assert result.document.origin.filename == "pages_legacy09.pages"
