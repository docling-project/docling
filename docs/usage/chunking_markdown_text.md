# Chunking Markdown Text Directly with HybridChunker

This document explains how to use `HybridChunker` directly with markdown text without first converting it to a `DoclingDocument`.

## Problem

Previously, users who wanted to chunk markdown text needed to:

1. Convert their markdown text to a `DoclingDocument` using `DocumentConverter`
2. Extract the document from the conversion result
3. Pass the `DoclingDocument` to `HybridChunker`

This was cumbersome for users who already had markdown text and just wanted to chunk it.

## Solution

We've added utility functions that allow you to use `HybridChunker` directly with markdown text:

### Method 1: Two-step process

```python
from docling.chunking.utils import markdown_to_docling_document
from docling.chunking import HybridChunker

# Convert markdown to DoclingDocument
markdown_text = """# My Document

This is some content that I want to chunk.

## Section 1

More content here.
"""

doc = markdown_to_docling_document(markdown_text)

# Create chunker and chunk the document
chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))

# Process chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.text[:100]}...")
```

### Method 2: One-step process

```python
from docling.chunking.utils import chunk_text_with_hybrid_chunker

markdown_text = """# My Document

This is some content that I want to chunk.

## Section 1

More content here.
"""

# Chunk text directly in one step
chunks = list(chunk_text_with_hybrid_chunker(markdown_text))

# Process chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.text[:100]}...")
```

### Method 3: With custom chunker configuration

```python
from docling.chunking.utils import chunk_text_with_hybrid_chunker

markdown_text = """# My Document

This is some content that I want to chunk.
"""

# Chunk with custom HybridChunker settings
chunks = list(chunk_text_with_hybrid_chunker(
    markdown_text,
    chunker_kwargs={
        "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
        "merge_peers": True,
    }
))
```

## API Reference

### `markdown_to_docling_document(markdown_text, filename="markdown_input.md")`

Converts markdown text to a `DoclingDocument`.

**Parameters:**
- `markdown_text` (str): The markdown text to convert
- `filename` (str, optional): Filename to use for the document metadata

**Returns:**
- `DoclingDocument`: The converted document

### `text_to_docling_document(text, format_type=InputFormat.MD, filename="text_input.md")`

Converts text to a `DoclingDocument` using the specified format.

**Parameters:**
- `text` (str): The text content to convert
- `format_type` (InputFormat, optional): The format of the input text (currently only `InputFormat.MD` is supported)
- `filename` (str, optional): Filename to use for the document metadata

**Returns:**
- `DoclingDocument`: The converted document

**Raises:**
- `ValueError`: If format_type is not supported

### `chunk_text_with_hybrid_chunker(text, chunker_kwargs=None, filename="markdown_input.md")`

Convenience function to chunk markdown text directly using `HybridChunker`.

**Parameters:**
- `text` (str): The markdown text to chunk
- `chunker_kwargs` (dict, optional): Dictionary of arguments to pass to `HybridChunker` constructor
- `filename` (str, optional): Filename to use for the document metadata

**Returns:**
- `Iterator`: An iterator of chunks

## Benefits

1. **Simplified workflow**: No need to manually create `InputDocument` or use `DocumentConverter`
2. **Direct text processing**: Skip the intermediate steps when you already have text
3. **Maintains full functionality**: All `HybridChunker` features and configuration options are preserved
4. **Backward compatible**: Existing code using `DoclingDocument` continues to work unchanged

## Use Cases

This feature is particularly useful for:

1. **RAG applications**: When you have markdown content from various sources and want to chunk it for embedding
2. **Content processing pipelines**: When processing markdown from CMSs, wikis, or documentation systems
3. **Text analysis workflows**: When you need to analyze structure-aware chunks of markdown content
4. **Rapid prototyping**: When you want to quickly test chunking strategies on markdown text

## Example: Complete RAG Workflow

```python
from docling.chunking.utils import chunk_text_with_hybrid_chunker

# Markdown content from your source
markdown_content = """
# Product Documentation

## Overview
Our product is a revolutionary AI assistant.

## Features
- Natural language processing
- Multi-modal understanding
- Real-time responses

## Installation
To install the product, run:
```bash
pip install our-product
```

## Usage
Here's how to get started...
"""

# Chunk the content directly
chunks = list(chunk_text_with_hybrid_chunker(
    markdown_content,
    chunker_kwargs={
        "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
    }
))

# Now you can embed and store the chunks
for chunk in chunks:
    # Your embedding and storage logic here
    embedding = embed_text(chunk.text)
    store_in_vector_db(chunk.text, embedding, chunk.meta)
```

## Migration Guide

If you were previously doing this:

```python
# Old approach
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from io import BytesIO

markdown_text = "# Title\n\nContent"
converter = DocumentConverter()
stream = BytesIO(markdown_text.encode())
result = converter.convert(stream)
doc = result.document
chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=doc))
```

You can now simply do:

```python
# New approach
from docling.chunking.utils import chunk_text_with_hybrid_chunker

markdown_text = "# Title\n\nContent"
chunks = list(chunk_text_with_hybrid_chunker(markdown_text))
```

## Troubleshooting

### ImportError
If you get import errors, ensure you have the latest version of docling and docling-core installed:

```bash
pip install --upgrade docling docling-core
```

### Empty chunks
If you're getting empty chunks, check that your markdown text is properly formatted and not empty.

### Performance considerations
For large amounts of text, consider chunking in batches or using streaming approaches to manage memory usage.
