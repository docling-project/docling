import enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field


class ChunkingExportOptions(BaseModel):
    """Export options for the response of the chunking task."""

    include_converted_doc: bool = False


class ChunkerType(str, enum.Enum):
    """Choice of the chunkers available in Docling."""

    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


class BaseChunkerOptions(BaseModel):
    """Configuration options for document chunking using Docling chunkers."""

    chunker: ChunkerType

    use_markdown_tables: Annotated[
        bool,
        Field(
            description="Use markdown table format instead of triplets for table serialization.",
        ),
    ] = False

    use_markdown_images: Annotated[
        bool,
        Field(
            description=(
                "Enable image serialization and image references inside chunks. "
                "Also adds a `has_image` field to chunk metadata to make image-containing "
                "chunks easier to identify."
            ),
        ),
    ] = False

    image_placeholder: Annotated[
        str,
        Field(
            description=(
                "Placeholder text used inside chunks to reference an image when "
                "markdown image serialization is disabled."
            ),
        ),
    ] = "![IMAGE]"

    include_raw_text: Annotated[
        bool,
        Field(
            description="Include both raw_text and text (contextualized) in response. If False, only text is included.",
        ),
    ] = False


class HierarchicalChunkerOptions(BaseChunkerOptions):
    """Configuration options for the HierarchicalChunker."""

    chunker: Literal[ChunkerType.HIERARCHICAL] = ChunkerType.HIERARCHICAL


class HybridChunkerOptions(BaseChunkerOptions):
    """Configuration options for the HybridChunker."""

    chunker: Literal[ChunkerType.HYBRID] = ChunkerType.HYBRID

    max_tokens: Annotated[
        Optional[int],
        Field(
            description="Maximum number of tokens per chunk. When left to none, the value is automatically extracted from the tokenizer.",
        ),
    ] = None

    tokenizer: Annotated[
        str,
        Field(
            description="HuggingFace model name for custom tokenization. If not specified, uses 'sentence-transformers/all-MiniLM-L6-v2' as default.",
            examples=[
                "Qwen/Qwen3-Embedding-0.6B",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        ),
    ] = "sentence-transformers/all-MiniLM-L6-v2"

    merge_peers: Annotated[
        bool,
        Field(
            description="Merge undersized successive chunks with same headings.",
        ),
    ] = True


# Union of all chunking options in the factory
ChunkingOptionType = HybridChunkerOptions | HierarchicalChunkerOptions
