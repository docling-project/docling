from pathlib import Path

import pytest

from docling_core.transforms.chunker.language_code_chunkers import JavaFunctionChunker
from docling.document_converter import DocumentConverter


def test_java_function_chunking():
    """Test Java function chunking with DocumentConverter."""

    source = Path(__file__).parent / "data" / "java" / "FlightLoader.java"
    
    if not source.exists():
        pytest.skip(f"Test file not found at {source}")
    
    converter = DocumentConverter()
    result = converter.convert(source)
    
    assert result.status.value == "success", f"Conversion failed: {result.errors}"
    assert result.document is not None, "Document should not be None"
    
    doc = result.document
    
    assert doc.texts, "Document should have text content"
    code_texts = [text for text in doc.texts if text.label.value == "code"]
    assert code_texts, "Document should have code content"
    
    chunker = JavaFunctionChunker(max_tokens=5000)
    chunk_iter = chunker.chunk(dl_doc=doc)
    
    chunks = list(chunk_iter)
    
    assert chunks, "Should produce at least one chunk"
    
    for i, chunk in enumerate(chunks):
        assert chunk.text, f"Chunk {i} should have text content"
        assert isinstance(chunk.text, str), f"Chunk {i} text should be a string"
        assert len(chunk.text) > 0, f"Chunk {i} should have non-empty text"
        
        assert chunk.meta is not None, f"Chunk {i} should have metadata"
        assert chunk.meta.part_name, f"Chunk {i} should have a part_name"
        assert chunk.meta.start_line is not None, f"Chunk {i} should have start_line"
        assert chunk.meta.end_line is not None, f"Chunk {i} should have end_line"
        assert chunk.meta.sha256 is not None, f"Chunk {i} should have sha256 hash"
        
        assert chunk.meta.chunk_type in ["function", "class", "preamble"], \
            f"Chunk {i} should have a valid chunk_type"
    
    function_chunks = [chunk for chunk in chunks if chunk.meta.chunk_type == "function"]
    assert function_chunks, "Should have at least one function chunk"
    
    java_keywords = ["public", "private", "class", "void", "return", "if", "for", "while"]
    for chunk in chunks:
        chunk_text_lower = chunk.text.lower()
        assert any(keyword in chunk_text_lower for keyword in java_keywords), \
            f"Chunk should contain Java code: {chunk.text[:100]}..."


def test_java_function_chunking_deterministic():
    """Test that Java function chunking produces deterministic results."""
    source = Path(__file__).parent / "data" / "java" / "FlightLoader.java"
    
    if not source.exists():
        pytest.skip(f"Test file not found at {source}")
    
    converter = DocumentConverter()
    chunker = JavaFunctionChunker(max_tokens=5000)
    
    results = []
    for _ in range(3):
        result = converter.convert(source)
        doc = result.document
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)
        results.append(chunks)
    
    chunk_counts = [len(chunks) for chunks in results]
    assert all(count == chunk_counts[0] for count in chunk_counts), \
        f"Chunk counts should be identical: {chunk_counts}"
    
    for i, chunks in enumerate(results[1:], 1):
        assert len(chunks) == len(results[0]), f"Run {i} should have same number of chunks"
        
        for j, (chunk1, chunk2) in enumerate(zip(results[0], chunks)):
            assert chunk1.text == chunk2.text, \
                f"Chunk {j} text should be identical between runs"
            assert chunk1.meta.sha256 == chunk2.meta.sha256, \
                f"Chunk {j} sha256 should be identical between runs"
            assert chunk1.meta.part_name == chunk2.meta.part_name, \
                f"Chunk {j} part_name should be identical between runs"


def test_java_function_chunking_with_different_max_tokens():
    """Test Java function chunking with different max_tokens settings."""
    source = Path(__file__).parent / "data" / "java" / "FlightLoader.java"
    
    if not source.exists():
        pytest.skip(f"Test file not found at {source}")
    
    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document
    
    max_tokens_values = [1000, 5000, 10000]
    chunk_counts = []
    
    for max_tokens in max_tokens_values:
        chunker = JavaFunctionChunker(max_tokens=max_tokens)
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)
        chunk_counts.append(len(chunks))
    

    assert all(count > 0 for count in chunk_counts), \
        f"All max_tokens values should produce chunks: {chunk_counts}"
    
    for max_tokens, chunks in zip(max_tokens_values, [list(chunker.chunk(dl_doc=doc)) for chunker in [JavaFunctionChunker(max_tokens=mt) for mt in max_tokens_values]]):
        for chunk in chunks:

            assert len(chunk.text) < max_tokens * 10, \
                f"Chunk text length should be reasonable for max_tokens={max_tokens}"


if __name__ == "__main__":
    test_java_function_chunking()
    test_java_function_chunking_deterministic()
    test_java_function_chunking_with_different_max_tokens()
    print("All tests passed!")
