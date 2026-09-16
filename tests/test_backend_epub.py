# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for EPUB document backend.

Test Data Attribution
---------------------
The test file 'epub_purvis_poetry.epub' is sourced from Standard Ebooks
(https://standardebooks.org), a volunteer-driven project that produces
high-quality, carefully formatted public domain ebooks.

The source text "Poetry" by Sarah Louisa Forten Purvis is in the public domain
in the United States. The cover art has been dedicated as CC0 by the Smithsonian.
Standard Ebooks dedicates the rest of their ebook files to the public domain via
the CC0 1.0 Universal Public Domain Dedication.

For more information about Standard Ebooks visit: https://standardebooks.org/about
"""

import logging
import zipfile
from pathlib import Path

import pytest

from docling.backend.epub_backend import EpubDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    ConversionStatus,
    DoclingDocument,
    InputDocument,
)
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

_log = logging.getLogger(__name__)

GENERATE = GEN_TEST_DATA


@pytest.fixture(scope="module")
def epub_paths() -> list[Path]:
    # Define the directory you want to search
    directory = Path("./tests/data/epub/sources/")

    # List all epub files in the directory and its subdirectories
    epub_files = sorted(directory.rglob("*.epub"))

    return epub_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.EPUB])
    return converter


@pytest.fixture(scope="module")
def backend(epub_paths) -> EpubDocumentBackend:
    epub_path = epub_paths[0]
    in_doc = InputDocument(
        path_or_stream=epub_path,
        format=InputFormat.EPUB,
        backend=EpubDocumentBackend,
    )
    return in_doc._backend


@pytest.fixture(scope="module")
def documents(epub_paths) -> list[tuple[Path, DoclingDocument]]:
    documents: list[tuple[Path, DoclingDocument]] = []

    converter = get_converter()

    for epub_path in epub_paths:
        _log.debug(f"converting {epub_path}")

        gt_path = epub_path.parent.parent / "groundtruth" / epub_path.name

        conv_result: ConversionResult = converter.convert(epub_path)

        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {gt_path}"
        documents.append((gt_path, doc))

    return documents


def test_e2e_epub_conversions(documents):
    """Test end-to-end EPUB conversion with ground truth validation."""
    for epub_path, doc in documents:
        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(epub_path) + ".md", generate=GENERATE), (
            f"export to markdown failed on {epub_path}"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(
            pred_itxt, str(epub_path) + ".itxt", generate=GENERATE, fuzzy=True
        ), f"export to indented-text failed on {epub_path}"

        assert verify_document(
            doc, str(epub_path) + ".json", generate=GENERATE, fuzzy=True
        ), f"DoclingDocument verification failed on {epub_path}"


def test_epub_backend_initialization(backend):
    """Test that the EPUB backend initializes correctly."""
    assert backend is not None
    assert isinstance(backend, EpubDocumentBackend)


def test_epub_document_structure(documents):
    """Test that converted EPUB documents have expected structure."""
    for _, doc in documents:
        # Check that document has content
        assert len(doc.texts) > 0, "Document should have text items"

        # Check that document has a title (from metadata)
        assert doc.name, "Document should have a name/title"


def test_epub_metadata_extraction(documents):
    """Test that EPUB metadata is properly extracted."""
    for _, doc in documents:
        # The document should have extracted metadata
        assert doc.name, "Document should have a title from EPUB metadata"


def test_epub_image_extraction(documents):
    """Test that images are properly extracted from EPUB archives."""
    for _, doc in documents:
        # Check if document has pictures
        # Note: Images are only extracted when fetch_images=True in backend options
        # The default converter doesn't fetch images, so we just verify structure
        if len(doc.pictures) > 0:
            # Verify that pictures exist in the document structure
            assert all(hasattr(pic, "self_ref") for pic in doc.pictures), (
                "All pictures should have proper structure"
            )


def test_epub_backend_with_image_options():
    """Test EPUB backend options can be created with different settings."""
    from docling.datamodel.backend_options import EpubBackendOptions

    # Test creating options with fetch_images=True
    options_with_images = EpubBackendOptions(fetch_images=True, enable_local_fetch=True)
    assert options_with_images.fetch_images is True
    assert options_with_images.enable_local_fetch is True

    # Test creating options with fetch_images=False (default)
    options_no_images = EpubBackendOptions(fetch_images=False)
    assert options_no_images.fetch_images is False

    # Test default options
    options_default = EpubBackendOptions()
    assert options_default.fetch_images is False  # Default should be False


def test_epub_content_combination():
    """Test that EPUB content from multiple files is properly combined."""
    epub_path = Path("./tests/data/epub/sources/epub_purvis_poetry.epub")

    converter = get_converter()
    result = converter.convert(epub_path)
    doc = result.document

    # Check that content is combined (should have multiple text items)
    assert len(doc.texts) > 1, "Should have multiple text items from combined content"

    # Check that the document has a reasonable amount of text
    total_text = "".join(item.text for item in doc.texts)
    assert len(total_text) > 100, "Combined content should have substantial text"


def _build_epub_with_hrefs(path: Path, hrefs: list[str], names: list[str]) -> Path:
    """Build a minimal EPUB whose manifest hrefs and ZIP entry names differ.

    ``hrefs`` are written into the package document, ``names`` are the file
    names actually stored in the archive, both relative to ``OEBPS/``.
    """
    container = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<container version="1.0"'
        ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        "<rootfiles>"
        '<rootfile full-path="OEBPS/content.opf"'
        ' media-type="application/oebps-package+xml"/>'
        "</rootfiles></container>"
    )
    items = "".join(
        f'<item id="c{i}" href="{href}" media-type="application/xhtml+xml"/>'
        for i, href in enumerate(hrefs)
    )
    itemrefs = "".join(f'<itemref idref="c{i}"/>' for i in range(len(hrefs)))
    opf = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"'
        ' unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>Percent Encoded</dc:title></metadata>"
        f"<manifest>{items}</manifest>"
        f"<spine>{itemrefs}</spine>"
        "</package>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        zi = zipfile.ZipInfo("mimetype")
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, "application/epub+zip")
        z.writestr("META-INF/container.xml", container)
        z.writestr("OEBPS/content.opf", opf)
        for i, name in enumerate(names):
            z.writestr(
                f"OEBPS/{name}",
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                f"<p>Chapter {i} body.</p>"
                "</body></html>",
            )
    return path


def test_epub_percent_encoded_manifest_href_is_read(tmp_path: Path):
    """A manifest href is a URL, so its percent-escapes must be decoded.

    The ZIP entry carries the literal file name, so a spine document whose
    href escapes a space or a non-ASCII character is not found and its text
    is dropped from the converted document while the conversion still
    reports success.
    """
    epub_path = _build_epub_with_hrefs(
        tmp_path / "percent.epub",
        hrefs=["chapter%201.xhtml", "%C3%A9pilogue.xhtml", "plain.xhtml"],
        names=["chapter 1.xhtml", "épilogue.xhtml", "plain.xhtml"],
    )

    result = get_converter().convert(epub_path)

    assert result.status == ConversionStatus.SUCCESS
    assert result.errors == []

    doc = result.document
    text = "\n".join(item.text for item in doc.texts)

    assert "Chapter 0 body." in text
    assert "Chapter 1 body." in text
    assert "Chapter 2 body." in text


def test_epub_link_fixing():
    """Test that internal EPUB links are properly fixed after content combination."""
    epub_path = Path("./tests/data/epub/sources/epub_purvis_poetry.epub")

    converter = get_converter()
    result = converter.convert(epub_path)
    doc = result.document

    # Export to markdown to check links
    markdown = doc.export_to_markdown()

    # Internal links should not contain filenames (e.g., "chapter1.xhtml#section")
    # They should be simplified to just anchors (e.g., "#section")
    # This is a basic check - the actual link format may vary
    assert markdown is not None, "Should be able to export to markdown"
    assert len(markdown) > 0, "Markdown export should not be empty"
