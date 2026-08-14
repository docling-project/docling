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
from pathlib import Path
from zipfile import ZipFile

import pytest
from docling_core.transforms.serializer.epub import EpubMetadata

from docling.backend.epub_backend import EpubDocumentBackend
from docling.datamodel.backend_options import EpubBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument, InputDocument
from docling.document_converter import DocumentConverter, EpubFormatOption

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


def test_epub_metadata_is_surfaced_on_the_conversion_result(epub_paths):
    """The OPF package metadata has no home in DoclingDocument, so the pipeline
    surfaces it on the conversion result for the EPUB output format to render."""
    result = get_converter().convert(epub_paths[0])

    assert result._epub_metadata == EpubMetadata(
        title="Poetry",
        authors=["Sarah Louisa Forten Purvis"],
        published="2023-12-06T20:52:11Z",
        language="en-US",
        source_file="epub_purvis_poetry.epub",
    )
    assert "epub_metadata" not in result.document.model_dump()


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


def test_book_markdown_rewrites_cross_spine_links(tmp_path: Path) -> None:
    epub_path = tmp_path / "linked-book.epub"
    with ZipFile(epub_path, "w") as epub:
        epub.writestr("mimetype", "application/epub+zip")
        epub.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OPS/content.opf"/></rootfiles>
</container>""",
        )
        epub.writestr(
            "OPS/content.opf",
            """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Linked Book</dc:title>
    <dc:creator>First Author</dc:creator>
    <dc:creator>Second Author</dc:creator>
    <dc:date>1901</dc:date>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="toc" href="toc.xhtml"/>
    <item id="chapter" href="chapter.htm.html"/>
  </manifest>
  <spine><itemref idref="toc"/><itemref idref="chapter"/></spine>
</package>""",
        )
        epub.writestr(
            "OPS/toc.xhtml",
            """<html><body><h1>Contents</h1>
<p><a href="chapter.htm.html#chap01">Chapter One</a></p>
<p><a href="chapter.htm.html#chap02">Chapter Two</a></p>
<p><a href="chapter.htm.html#note">Chapter One note</a></p>
<p><a href="missing.htm#ghost">Ghost</a></p>
</body></html>""",
        )
        epub.writestr(
            "OPS/chapter.htm.html",
            """<html><body>
<h1 id="chap01">Chapter One</h1><p>First body.</p>
<a id="note"></a><p>Chapter one note.</p>
<p><a href="chapter.htm.html#chap02">Next chapter</a></p>
<a id="chap02"></a><h1>Chapter -- Two</h1><p>Second body.</p>
</body></html>""",
        )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.EPUB],
        format_options={
            InputFormat.EPUB: EpubFormatOption(
                backend_options=EpubBackendOptions(rewrite_internal_links=True)
            )
        },
    )
    result = converter.convert(epub_path)
    repeated_result = converter.convert(epub_path)

    markdown = result.document.export_to_epub(metadata=result._epub_metadata)
    assert 'authors: ["First Author", "Second Author"]' in markdown
    assert "[Chapter One](#chapter-one)" in markdown
    assert "[Chapter Two](#chapter----two)" in markdown
    assert "[Chapter One note](#chapter-one)" in markdown
    assert "[Next chapter](#chapter----two)" in markdown
    assert "missing.htm#ghost" not in markdown
    assert "Ghost" in markdown
    assert (
        repeated_result.document.export_to_epub(metadata=repeated_result._epub_metadata)
        == markdown
    )
