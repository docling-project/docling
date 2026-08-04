"""Tests for the Apple Pages (``.pages``) document backend.

Test Data Attribution
---------------------
``pages_2013.pages`` and ``pages_iwork09.pages`` are ``testPages2013.pages`` and
``testPages.pages`` from the Apache Tika test corpus, licensed under the Apache
License 2.0. They are genuine Apple Pages output, and between them cover both
container generations: ``pages_2013.pages`` stores its content as ``Index/*.iwa``
with no PDF render, while ``pages_iwork09.pages`` uses the iWork '09 ``index.xml``
layout. Conveniently, both hold the same source document, so the two code paths
can be checked against each other.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).
"""

import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.iwork_backend import IWorkPagesDocumentBackend
from docling.backend.iwork_iwa import iter_objects, read_fields
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

SOURCES = Path("./tests/data/pages/sources")
PAGES_2013 = SOURCES / "pages_2013.pages"
PAGES_IWORK09 = SOURCES / "pages_iwork09.pages"

# Present in the body of both fixtures.
_BODY_SENTENCE = "Some plain text to parse."


def _backend(
    path: Path, options: IWorkBackendOptions | None = None
) -> IWorkPagesDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.IWORK_PAGES,
        backend=IWorkPagesDocumentBackend,
        backend_options=options,
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkPagesDocumentBackend)
    return backend


def test_detects_pages_from_path_and_named_stream():
    """`.pages` is a ZIP, so detection must not stop at ``application/zip``."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])

    assert conv_input._guess_format(PAGES_2013) == InputFormat.IWORK_PAGES

    stream = DocumentStream(
        name="report.pages", stream=BytesIO(PAGES_2013.read_bytes())
    )
    assert conv_input._guess_format(stream) == InputFormat.IWORK_PAGES


def test_extensionless_pages_stream_is_not_claimed():
    """Without the extension a Pages container is indistinguishable from Keynote
    and Numbers, so the backend must not claim it rather than guess wrong."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="blob", stream=BytesIO(PAGES_2013.read_bytes()))

    assert conv_input._guess_format(stream) is None


def test_modern_pages_body_text_is_extracted():
    """Pages 5+ keeps its body in Index/*.iwa with no PDF render, so this is the
    path that matters for essentially every Pages document in circulation."""
    doc = _backend(PAGES_2013).convert()

    text = doc.export_to_markdown()
    assert "Sample pages document" in text
    assert _BODY_SENTENCE in text
    assert "Both Pages 1.x and Keynote 2.x" in text


def test_legacy_pages_body_text_is_extracted():
    doc = _backend(PAGES_IWORK09).convert()

    assert _BODY_SENTENCE in doc.export_to_markdown()


def test_both_generations_agree_on_the_shared_body_text():
    """The two fixtures are the same source document saved by different Pages
    releases, so the independent IWA and XML readers must agree on its text."""
    modern = _backend(PAGES_2013).convert().export_to_markdown()
    legacy = _backend(PAGES_IWORK09).convert().export_to_markdown()

    for sentence in ("Sample pages document", _BODY_SENTENCE):
        assert sentence in modern
        assert sentence in legacy


def test_legacy_placeholder_text_is_not_emitted():
    """iWork '09 templates carry sf:ghost-text placeholders. That is what the
    template displays before the author types, not document content."""
    raw = zipfile.ZipFile(PAGES_IWORK09).read("index.xml").decode("utf-8", "replace")
    assert "ghost-text" in raw, "fixture no longer exercises the placeholder path"

    text = _backend(PAGES_IWORK09).convert().export_to_markdown()
    assert "Lorem ipsum dolor sit amet" not in text


def test_pages_backend_accepts_a_stream():
    stream = BytesIO(PAGES_2013.read_bytes())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.IWORK_PAGES,
        backend=IWorkPagesDocumentBackend,
        filename="report.pages",
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkPagesDocumentBackend)

    assert _BODY_SENTENCE in backend.convert().export_to_markdown()


def test_object_replacement_characters_are_dropped():
    """Apple marks inline attachments with U+FFFC inside the text run; it carries
    no text and must not leak into the output."""
    doc = _backend(PAGES_2013).convert()

    assert "￼" not in doc.export_to_markdown()


def test_iwa_reader_walks_the_real_object_graph():
    """Guards the container layer itself: chunk framing, raw Snappy and the
    TSP.ArchiveInfo walk, against genuine Apple output."""
    archive = zipfile.ZipFile(PAGES_2013)
    objects = {}
    for name in archive.namelist():
        if name.endswith(".iwa"):
            for obj in iter_objects(archive.read(name)):
                objects[obj.identifier] = obj

    assert len(objects) > 100

    # TP.DocumentArchive must be present and reference a TSWP.StorageArchive.
    document = next(o for o in objects.values() if o.message_type == 10000)
    body_ref = read_fields(document.payload)[4][0]
    assert isinstance(body_ref, bytes)


def test_zip_without_pages_index_is_rejected(tmp_path: Path):
    other_zip = tmp_path / "not_really.pages"
    with zipfile.ZipFile(other_zip, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")

    with pytest.raises(DocumentLoadError, match="does not look like a Pages document"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=other_zip,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            other_zip,
        )


def test_non_zip_input_is_rejected(tmp_path: Path):
    broken = tmp_path / "broken.pages"
    broken.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentLoadError, match="not a readable ZIP container"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=broken,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            broken,
        )


def test_archive_limits_are_enforced():
    """The container is attacker-controlled, so limits must bite before the IWA
    archives are decompressed."""
    with pytest.raises(DocumentLoadError, match="max_member_count"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=PAGES_2013,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            PAGES_2013,
            IWorkBackendOptions(max_member_count=1),
        )

    with pytest.raises(DocumentLoadError, match="max_total_bytes"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=PAGES_2013,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            PAGES_2013,
            IWorkBackendOptions(max_total_bytes=1024),
        )


def test_end_to_end_conversion():
    """No models involved: the backend is declarative, so this runs in CI without
    the PDF pipeline."""
    converter = DocumentConverter(allowed_formats=[InputFormat.IWORK_PAGES])
    result = converter.convert(PAGES_2013)

    assert _BODY_SENTENCE in result.document.export_to_markdown()
    assert result.document.origin is not None
    assert result.document.origin.mimetype == "application/vnd.apple.pages"
    assert result.document.origin.filename == "pages_2013.pages"
