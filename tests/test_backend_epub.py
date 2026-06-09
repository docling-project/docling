from pathlib import Path

import pytest

from docling.backend.epub_backend import EpubDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document

pytestmark = pytest.mark.cross_platform


def test_epub_backend_basic():
    """Test basic EPUB backend functionality."""
    fmt = InputFormat.EPUB
    cls = EpubDocumentBackend

    epub_path = Path("tests/data/epub/sarah-louisa-forten-purvis_poetry.epub")

    if not epub_path.exists():
        pytest.skip(f"Test EPUB file not found: {epub_path}")

    in_doc = InputDocument(
        path_or_stream=epub_path,
        format=fmt,
        backend=cls,
    )
    backend = cls(
        in_doc=in_doc,
        path_or_stream=epub_path,
    )

    assert backend.is_valid()
    assert backend.supported_formats() == {InputFormat.EPUB}
    assert not backend.supports_pagination()


def test_epub_conversion():
    """Test EPUB to DoclingDocument conversion."""
    epub_path = Path("tests/data/epub/sarah-louisa-forten-purvis_poetry.epub")

    if not epub_path.exists():
        pytest.skip(f"Test EPUB file not found: {epub_path}")

    converter = DocumentConverter(allowed_formats=[InputFormat.EPUB])
    result = converter.convert(epub_path)

    assert result.status.name == "SUCCESS"
    assert result.document is not None

    doc = result.document
    assert doc.name == "sarah-louisa-forten-purvis_poetry"
    assert doc.origin.mimetype == "application/epub+zip"

    # Check that content was extracted
    md_output = doc.export_to_markdown()
    assert len(md_output) > 1000  # Should have substantial content
    assert "Poetry" in md_output
    assert "Sarah Louisa Forten Purvis" in md_output

    # Verify some poem titles are present
    assert "The Grave of the Slave" in md_output
    assert "The Slave Girl's Address to Her Mother" in md_output


def test_epub_metadata_extraction():
    """Test that EPUB metadata is properly extracted."""
    epub_path = Path("tests/data/epub/sarah-louisa-forten-purvis_poetry.epub")

    if not epub_path.exists():
        pytest.skip(f"Test EPUB file not found: {epub_path}")

    in_doc = InputDocument(
        path_or_stream=epub_path,
        format=InputFormat.EPUB,
        backend=EpubDocumentBackend,
    )
    backend = EpubDocumentBackend(
        in_doc=in_doc,
        path_or_stream=epub_path,
    )

    # Check metadata was extracted
    assert "title" in backend.metadata
    assert backend.metadata["title"] == "Poetry"

    # Check content files were found
    assert len(backend.content_files) > 0
    assert any("poetry.xhtml" in f for f in backend.content_files)


def test_epub_with_document_converter():
    """Test EPUB conversion using DocumentConverter with groundtruth comparison."""
    epub_path = Path("tests/data/epub/sarah-louisa-forten-purvis_poetry.epub")

    if not epub_path.exists():
        pytest.skip(f"Test EPUB file not found: {epub_path}")

    gt_path = Path(
        "tests/data/groundtruth/docling_v2/sarah-louisa-forten-purvis_poetry.epub.json"
    )

    converter = DocumentConverter(allowed_formats=[InputFormat.EPUB])
    result = converter.convert(epub_path)

    assert result.status.name == "SUCCESS"
    assert result.document is not None

    # Verify against groundtruth if it exists or generate it
    if GEN_TEST_DATA:
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        result.document.save_as_json(gt_path)
    elif gt_path.exists():
        assert verify_document(result.document, gt_path, generate=False)
