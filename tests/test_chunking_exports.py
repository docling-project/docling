# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from docling_core.transforms.chunker.line_chunker import (
    LineBasedTokenChunker as CoreLineBasedTokenChunker,
)


def test_line_based_token_chunker_import_from_docling_chunking():
    # docs/concepts/chunking.md documents this import path
    from docling.chunking import LineBasedTokenChunker

    assert LineBasedTokenChunker is CoreLineBasedTokenChunker
