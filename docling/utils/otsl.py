# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Utilities for parsing OTSL table output."""

import re
from itertools import groupby

from docling_core.types.doc import TableCell

_CONTENT_TOKENS = {"fcel", "ecel", "ched", "rhed", "srow"}
_TAG_PATTERN = re.compile(
    r"<(?P<tag>[a-z]+)>(?P<text>.*?)</(?P=tag)>"
    r"|<(?P<stag>[a-z]+)\s*/>"
    r"|<(?P<otag>[a-z]+)>(?P<otext>[^<]*)",
    re.DOTALL,
)


def parse_otsl_output(text: str) -> tuple[list[str], list[TableCell], int, int]:
    """Parse VLM OTSL output into its token sequence and table grid."""
    if not text or not text.strip():
        return [], [], 0, 0

    otsl_match = re.search(r"<otsl>(.*)</otsl>", text, re.DOTALL)
    if otsl_match:
        text = otsl_match.group(1)

    token_pairs: list[tuple[str, str]] = []
    for match in _TAG_PATTERN.finditer(text):
        if match.group("tag"):
            token_pairs.append((match.group("tag"), match.group("text") or ""))
        elif match.group("stag"):
            token_pairs.append((match.group("stag"), ""))
        elif match.group("otag"):
            token_pairs.append((match.group("otag"), match.group("otext") or ""))

    if not token_pairs:
        return [], [], 0, 0

    otsl_seq = [tag for tag, _text in token_pairs]
    rows = [
        list(group)
        for is_newline, group in groupby(token_pairs, lambda token: token[0] == "nl")
        if not is_newline
    ]
    if not rows:
        return otsl_seq, [], 0, 0

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)
    grid = [row + [("", "")] * (num_cols - len(row)) for row in rows]
    table_cells: list[TableCell] = []
    for row_index, row in enumerate(grid):
        for col_index, (tag, inner_text) in enumerate(row):
            if tag not in _CONTENT_TOKENS:
                continue

            col_span = 1
            for following_col in range(col_index + 1, num_cols):
                if grid[row_index][following_col][0] not in {"lcel", "xcel"}:
                    break
                col_span += 1

            row_span = 1
            for following_row in range(row_index + 1, num_rows):
                if grid[following_row][col_index][0] not in {"ucel", "xcel"}:
                    break
                row_span += 1

            table_cells.append(
                TableCell(
                    text=inner_text,
                    bbox=None,
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=row_index,
                    end_row_offset_idx=row_index + row_span,
                    start_col_offset_idx=col_index,
                    end_col_offset_idx=col_index + col_span,
                    column_header=tag == "ched",
                    row_header=tag == "rhed",
                    row_section=tag == "srow",
                )
            )

    return otsl_seq, table_cells, num_rows, num_cols
