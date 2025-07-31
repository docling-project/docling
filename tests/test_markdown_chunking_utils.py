"""Tests for markdown text chunking utilities."""

import pytest

from docling.chunking import HybridChunker
from docling.chunking.utils import (
    chunk_text_with_hybrid_chunker,
    markdown_to_docling_document,
    text_to_docling_document,
)
from docling.datamodel.base_models import InputFormat


def test_markdown_to_docling_document():
    """Test conversion of markdown text to DoclingDocument."""
    markdown_text = """# Test Title

This is a test paragraph.

## Subsection

- Item 1
- Item 2
- Item 3

Another paragraph here.
"""

    # Convert markdown to DoclingDocument
    doc = markdown_to_docling_document(markdown_text)

    # Verify the document was created
    assert doc is not None
    assert doc.name == "markdown_input"
    assert doc.origin.filename == "markdown_input.md"
    assert doc.origin.mimetype == "text/markdown"

    # Verify document structure
    assert len(doc.texts) > 0  # Should have text items


def test_markdown_to_docling_document_with_custom_filename():
    """Test conversion with custom filename."""
    markdown_text = "# Test\n\nContent here."

    doc = markdown_to_docling_document(markdown_text, "custom_file.md")

    assert doc.origin.filename == "custom_file.md"


def test_text_to_docling_document():
    """Test the general text conversion function."""
    text = "# Title\n\nContent"

    # Test with markdown format
    doc = text_to_docling_document(text, InputFormat.MD)
    assert doc is not None

    # Test with unsupported format
    with pytest.raises(ValueError, match="Format .* is not currently supported"):
        text_to_docling_document(text, InputFormat.PDF)


def test_chunk_text_with_hybrid_chunker():
    """Test direct text chunking with HybridChunker."""
    markdown_text = """# Main Title

This is the first section with some content that should be chunked appropriately.

## Section 1

Content for section 1 goes here. This should be in a separate chunk.

## Section 2

Content for section 2. This is another chunk.

- List item 1
- List item 2
- List item 3

Final paragraph.
"""

    # Test basic chunking
    chunks = list(chunk_text_with_hybrid_chunker(markdown_text))

    assert len(chunks) > 0
    assert all(hasattr(chunk, "text") for chunk in chunks)
    assert all(len(chunk.text.strip()) > 0 for chunk in chunks)


def test_chunk_text_with_custom_chunker_kwargs():
    """Test text chunking with custom chunker configuration."""
    markdown_text = "# Title\n\nSome content here."

    # Test with custom chunker kwargs
    chunks = list(
        chunk_text_with_hybrid_chunker(
            markdown_text, chunker_kwargs={"merge_peers": False}
        )
    )

    assert len(chunks) > 0


def test_integration_with_existing_hybrid_chunker():
    """Test that our utility works with existing HybridChunker workflow."""
    markdown_text = """# Test Document

This is a test document for verifying integration.

## Section A

Content in section A.

## Section B

Content in section B.
"""

    # Method 1: Using our utility
    doc = markdown_to_docling_document(markdown_text)
    chunker = HybridChunker()
    chunks1 = list(chunker.chunk(dl_doc=doc))

    # Method 2: Using direct text chunking
    chunks2 = list(chunk_text_with_hybrid_chunker(markdown_text))

    # Both methods should produce the same number of chunks
    assert len(chunks1) == len(chunks2)

    # Chunk content should be the same
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.text == c2.text


def test_empty_text():
    """Test handling of empty text input."""
    empty_text = ""

    doc = markdown_to_docling_document(empty_text)
    assert doc is not None

    chunks = list(chunk_text_with_hybrid_chunker(empty_text))
    # Empty text might produce no chunks or empty chunks
    assert isinstance(chunks, list)


def test_simple_text():
    """Test with very simple text input."""
    simple_text = "Just a simple line of text."

    doc = markdown_to_docling_document(simple_text)
    assert doc is not None

    chunks = list(chunk_text_with_hybrid_chunker(simple_text))
    assert len(chunks) >= 1
    assert any(simple_text in chunk.text for chunk in chunks)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")

    try:
        test_markdown_to_docling_document()
        print("✓ test_markdown_to_docling_document passed")

        test_markdown_to_docling_document_with_custom_filename()
        print("✓ test_markdown_to_docling_document_with_custom_filename passed")

        test_text_to_docling_document()
        print("✓ test_text_to_docling_document passed")

        test_chunk_text_with_hybrid_chunker()
        print("✓ test_chunk_text_with_hybrid_chunker passed")

        test_chunk_text_with_custom_chunker_kwargs()
        print("✓ test_chunk_text_with_custom_chunker_kwargs passed")

        test_integration_with_existing_hybrid_chunker()
        print("✓ test_integration_with_existing_hybrid_chunker passed")

        test_simple_text()
        print("✓ test_simple_text passed")

        print("\nAll tests passed! ✓")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
