## Introduction

!!! note "Chunking approaches"

    Starting from a `DoclingDocument`, there are in principle two possible chunking
    approaches:

    1. exporting the `DoclingDocument` to Markdown (or similar format) and then
      performing user-defined chunking as a post-processing step, or
    2. using native Docling chunkers, i.e. operating directly on the `DoclingDocument`

    This page is about the latter, i.e. using native Docling chunkers.
    For an example of using approach (1) check out e.g.
    [this recipe](../examples/rag_langchain.ipynb) looking at the Markdown export mode.

A *chunker* is a Docling abstraction that, given a
[`DoclingDocument`](./docling_document.md), returns a stream of chunks, each of which
captures some part of the document as a string accompanied by respective metadata.

To enable both flexibility for downstream applications and out-of-the-box utility,
Docling defines a chunker class hierarchy, providing a base type, `BaseChunker`, as well
as specific subclasses.

Docling integration with gen AI frameworks like LlamaIndex is done using the
`BaseChunker` interface, so users can easily plug in any built-in, self-defined, or
third-party `BaseChunker` implementation.

## Base Chunker

The `BaseChunker` base class API defines that any chunker should provide the following:

- `def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]`:
  Returning the chunks for the provided document.
- `def contextualize(self, chunk: BaseChunk) -> str`:
  Returning the potentially metadata-enriched serialization of the chunk, typically
  used to feed an embedding model (or generation model).

## Hybrid Chunker

!!! note "To access `HybridChunker`"

    - If you are using the `docling` package, you can import as follows:
        ```python
        from docling.chunking import HybridChunker
        ```
    - If you are only using the `docling-core` package, you must ensure to install
        the `chunking` extra if you want to use HuggingFace tokenizers, e.g.
        ```shell
        pip install 'docling-core[chunking]'
        ```
        or the `chunking-openai` extra if you prefer Open AI tokenizers (tiktoken), e.g.
        ```shell
        pip install 'docling-core[chunking-openai]'
        ```
        and then you
        can import as follows:
        ```python
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        ```

The `HybridChunker` implementation uses a hybrid approach, applying tokenization-aware
refinements on top of document-based [hierarchical](#hierarchical-chunker) chunking.

More precisely:

- it starts from the result of the hierarchical chunker and, based on the user-provided
  tokenizer (typically to be aligned to the embedding model tokenizer), it:
- does one pass where it splits chunks only when needed (i.e. oversized w.r.t.
tokens), &
- another pass where it merges chunks only when possible (i.e. undersized successive
chunks with same headings & captions) — users can opt out of this step via param
`merge_peers` (by default `True`)

### Table Chunking with Repeated Headers

When chunking tables with [`HybridChunker`](#hybrid-chunker), you can control how table headers are handled:

- **`repeat_table_header`** (default: `True`): When enabled, table headers are repeated at the beginning of each chunk when a table spans multiple chunks. This ensures each chunk maintains context about the table structure.

- **`omit_header_on_overflow`** (default: `False`): When enabled along with `repeat_table_header=True`, this parameter provides flexibility for handling wide tables where rows might not fit with the header included:
    - If a table row fits within the token limit **without** the header but would overflow **with** the header, the header is omitted for that specific row
    - This helps maximize token efficiency while preserving line integrity for structured content
    - Particularly useful for tables with very wide headers or when working with strict token limits

## Line-Based Token Chunker

!!! note "To access `LineBasedTokenChunker`"

    - If you are using the `docling` package, you can import as follows:
        ```python
        from docling.chunking import LineBasedTokenChunker
        ```
    - If you are only using the `docling-core` package, you must ensure to install
        the `chunking` extra, then import as follows:
        ```python
        from docling_core.transforms.chunker.line_chunker import LineBasedTokenChunker
        ```

The `LineBasedTokenChunker` is a tokenization-aware chunker that preserves line boundaries, particularly useful for structured content like tables, code, logs, and lists. It attempts to keep lines intact within chunks, only splitting a line if it exceeds the maximum token limit on its own.

Key capabilities:

- Prioritizes keeping entire lines within a single chunk
- Supports adding a repeated prefix to each chunk (e.g., table headers for context)
- Offers overflow handling via `omit_prefix_on_overflow` parameter: when `True`, omits the prefix for lines that would overflow with it but fit without it

## Hierarchical Chunker

The `HierarchicalChunker` implementation uses the document structure information from
the [`DoclingDocument`](./docling_document.md) to create one chunk for each individual
detected document element, by default only merging together list items (can be opted out
via param `merge_list_items`). It also takes care of attaching all relevant document
metadata, including headers and captions.


## Usage Examples

- [Hybrid chunking](../examples/hybrid_chunking.ipynb)
- [Line-based chunking](../examples/line_based_chunking.ipynb)
- [Advanced chunking & serialization](../examples/advanced_chunking_and_serialization.ipynb)
