#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Utility functions for chunking operations."""

from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional, Union

from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def markdown_to_docling_document(
    markdown_text: str, filename: str = "markdown_input.md"
):
    """
    Convert markdown text to a DoclingDocument.

    This utility function allows converting raw markdown text into a DoclingDocument
    that can be used with Docling chunkers like HybridChunker.

    Args:
        markdown_text: The markdown text to convert
        filename: Optional filename to use for the document (used in metadata)

    Returns:
        DoclingDocument: The converted document

    Example:
        >>> from docling.chunking.utils import markdown_to_docling_document
        >>> from docling.chunking import HybridChunker
        >>>
        >>> markdown = "# Title\\n\\nSome content here."
        >>> doc = markdown_to_docling_document(markdown)
        >>> chunker = HybridChunker()
        >>> chunks = list(chunker.chunk(doc))
    """
    # Create a BytesIO stream from the markdown text
    markdown_stream = BytesIO(markdown_text.encode("utf-8"))

    # Create an InputDocument for the markdown backend
    input_doc = InputDocument(
        path_or_stream=markdown_stream,
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename=filename,
    )

    # Use the MarkdownDocumentBackend to convert the text to DoclingDocument
    backend = MarkdownDocumentBackend(in_doc=input_doc, path_or_stream=markdown_stream)

    try:
        docling_doc = backend.convert()
        return docling_doc
    finally:
        # Clean up the backend
        backend.unload()


def text_to_docling_document(
    text: str,
    format_type: InputFormat = InputFormat.MD,
    filename: str = "text_input.md",
):
    """
    Convert text to a DoclingDocument using the specified format.

    This is a more general utility that can handle different text formats.
    Currently supports markdown format.

    Args:
        text: The text content to convert
        format_type: The format of the input text (currently only InputFormat.MD is supported)
        filename: Optional filename to use for the document (used in metadata)

    Returns:
        DoclingDocument: The converted document

    Raises:
        ValueError: If format_type is not supported

    Example:
        >>> from docling.chunking.utils import text_to_docling_document
        >>> from docling.chunking import HybridChunker
        >>> from docling.datamodel.pipeline_options import InputFormat
        >>>
        >>> text = "# Title\\n\\nSome content here."
        >>> doc = text_to_docling_document(text, InputFormat.MD)
        >>> chunker = HybridChunker()
        >>> chunks = list(chunker.chunk(doc))
    """
    if format_type == InputFormat.MD:
        return markdown_to_docling_document(text, filename)
    else:
        raise ValueError(
            f"Format {format_type} is not currently supported. Only InputFormat.MD is supported."
        )


def chunk_text_with_hybrid_chunker(
    text: str,
    chunker_kwargs: Optional[dict] = None,
    filename: str = "markdown_input.md",
) -> Iterator:
    """
    Convenience function to chunk markdown text directly using HybridChunker.

    This function combines text-to-document conversion and chunking in one step.

    Args:
        text: The markdown text to chunk
        chunker_kwargs: Optional dictionary of arguments to pass to HybridChunker constructor
        filename: Optional filename to use for the document (used in metadata)

    Returns:
        Iterator: An iterator of chunks

    Example:
        >>> from docling.chunking.utils import chunk_text_with_hybrid_chunker
        >>>
        >>> markdown = "# Title\\n\\nSome content here."
        >>> chunks = list(chunk_text_with_hybrid_chunker(markdown))
        >>>
        >>> # With custom chunker settings
        >>> chunks = list(chunk_text_with_hybrid_chunker(
        ...     markdown,
        ...     chunker_kwargs={"tokenizer": "sentence-transformers/all-MiniLM-L6-v2"}
        ... ))
    """
    # Import here to avoid circular imports and ensure the module is available
    from docling.chunking import HybridChunker

    # Convert text to DoclingDocument
    doc = markdown_to_docling_document(text, filename)

    # Create chunker with provided arguments
    if chunker_kwargs is None:
        chunker_kwargs = {}

    chunker = HybridChunker(**chunker_kwargs)

    # Return the chunks
    return chunker.chunk(dl_doc=doc)
