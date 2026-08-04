# %% [markdown]
# A trivial (pass-through) chunker: one chunk per document item.
#
# Docling's built-in chunkers are retrieval-oriented: `HierarchicalChunker`
# GROUPS items by section, and `HybridChunker` additionally MERGES undersized
# chunks and SPLITS oversized ones to fit a token window. Sometimes the opposite
# is wanted — one chunk per document element, verbatim, with no grouping and no
# splitting — so that each chunk maps back to exactly one item. That is handy for
# element-level extraction, indexing, or row-per-element storage.
#
# What this example does
# - Implements `TrivialChunker` by subclassing `BaseChunker` and providing its
#   single abstract method, `chunk()`.
# - Emits one `DocChunk` per `DocItem` (heading, text, table, picture, ...), in
#   reading order — no grouping, no splitting.
# - Reuses the same `ChunkingSerializerProvider` the built-in chunkers use, so the
#   chunk text is formatted identically; only the granularity differs.
#
# When to use this
# - Downstream systems that need INDIVIDUAL document elements (one per chunk)
#   rather than the retrieval-oriented, token-bounded chunks the built-in chunkers
#   produce.
#
# Prerequisites
# - Install Docling.
#
# How to run
# - From the repo root: `python docs/examples/trivial_chunking.py`.
#
# Input document
# - Defaults to a bundled sample (a 4-page PDF already converted to a
#   DoclingDocument JSON). Any `DoclingDocument` works, including one produced by
#   `DocumentConverter().convert(source).document`.

# %%

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docling_core.transforms.chunker.base import BaseChunk, BaseChunker
from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import BaseSerializerProvider
from docling_core.types.doc import DocItem, DoclingDocument
from pydantic import ConfigDict


class TrivialChunker(BaseChunker):
    """A pass-through chunker that emits one chunk per document item.

    Unlike `HierarchicalChunker` (which groups items by section) and
    `HybridChunker` (which additionally merges and splits to fit a token window),
    this chunker applies no grouping and no splitting: every `DocItem` in the
    document becomes exactly one `DocChunk`, in reading order. Chunk text is
    produced with the same serializer the built-in chunkers use, so the output
    formatting is consistent — only the granularity differs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """Chunk the document into one chunk per item.

        Args:
            dl_doc: the document to chunk.

        Yields:
            One `DocChunk` per `DocItem`, in reading order. Items that serialize
            to empty text (e.g. a picture with no caption) are skipped.
        """
        doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        excluded_refs = doc_ser.get_excluded_refs(**kwargs)
        for item, _level in dl_doc.iterate_items():
            if not isinstance(item, DocItem) or item.self_ref in excluded_refs:
                continue
            ser_res = doc_ser.serialize(item=item)
            if not ser_res.text:
                continue
            yield DocChunk(
                text=ser_res.text,
                meta=DocMeta(doc_items=[item], origin=dl_doc.origin),
            )


def main() -> None:
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/groundtruth/normal_4pages.json"

    doc = DoclingDocument.load_from_json(input_doc_path)
    chunker = TrivialChunker()

    chunks = list(chunker.chunk(doc))
    print(f"{len(chunks)} chunks (one per item)\n")
    for i, chunk in enumerate(chunks[:8]):
        item = chunk.meta.doc_items[0]
        print(f"{i}: [{item.label}] {chunk.text[:70]!r}")


if __name__ == "__main__":
    main()
