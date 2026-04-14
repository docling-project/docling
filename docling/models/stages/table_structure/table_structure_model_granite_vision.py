import logging
import re
from itertools import groupby

from docling_core.types.doc import TableCell

_log = logging.getLogger(__name__)

# OTSL tokens that represent content-bearing cells (produce a TableCell)
_CONTENT_TOKENS = {"fcel", "ecel", "ched", "rhed", "srow"}
# OTSL tokens that are span-extensions (no separate TableCell, affect span of predecessor)
_SPAN_TOKENS = {"lcel", "ucel", "xcel"}

# Regex to extract (tag_name, inner_text) from VLM OTSL output.
# Matches: <tag>text</tag>  OR  <tag/>  OR  <tag>  (self-closing / bare nl)
_TAG_RE = re.compile(
    r"<(?P<tag>[a-z]+)>(?P<text>.*?)</(?P=tag)>"  # <tag>text</tag>
    r"|<(?P<stag>[a-z]+)\s*/>"                     # <tag/>
    r"|<(?P<btag>[a-z]+)>",                        # <tag> (bare, e.g. <nl>, <lcel>)
    re.DOTALL,
)


def _parse_otsl_output(
    text: str,
) -> tuple[list[str], list[TableCell], int, int]:
    """Parse VLM OTSL text output into structured table data.

    Parameters
    ----------
    text:
        Raw VLM output string, e.g.
        ``"<ched>Name</ched><ched>Val</ched><nl><fcel>Foo</fcel><fcel>42</fcel><nl>"``

    Returns
    -------
    tuple of (otsl_seq, table_cells, num_rows, num_cols)
        otsl_seq: list of bare tag names, e.g. ["ched", "ched", "nl", "fcel", "fcel", "nl"]
        table_cells: list of TableCell (bbox always None)
        num_rows: int
        num_cols: int
    """
    if not text or not text.strip():
        return [], [], 0, 0

    # Extract (tag, inner_text) pairs
    token_pairs: list[tuple[str, str]] = []
    for m in _TAG_RE.finditer(text):
        if m.group("tag"):
            token_pairs.append((m.group("tag"), m.group("text") or ""))
        elif m.group("stag"):
            token_pairs.append((m.group("stag"), ""))
        elif m.group("btag"):
            token_pairs.append((m.group("btag"), ""))

    if not token_pairs:
        return [], [], 0, 0

    otsl_seq = [tag for tag, _ in token_pairs]

    # Split into rows on "nl" tokens
    rows: list[list[tuple[str, str]]] = [
        list(group)
        for k, group in groupby(token_pairs, lambda x: x[0] == "nl")
        if not k
    ]

    if not rows:
        return otsl_seq, [], 0, 0

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    # Pad rows to equal width
    grid: list[list[tuple[str, str]]] = [
        row + [("", "")] * (num_cols - len(row)) for row in rows
    ]

    table_cells: list[TableCell] = []
    for row_idx, row in enumerate(grid):
        for col_idx, (tag, inner_text) in enumerate(row):
            if tag not in _CONTENT_TOKENS:
                continue

            # Detect colspan: count consecutive span-extension tokens to the right
            colspan = 1
            for c in range(col_idx + 1, num_cols):
                if grid[row_idx][c][0] in _SPAN_TOKENS:
                    colspan += 1
                else:
                    break

            # Detect rowspan: count consecutive span-extension tokens below
            rowspan = 1
            for r in range(row_idx + 1, num_rows):
                if grid[r][col_idx][0] in _SPAN_TOKENS:
                    rowspan += 1
                else:
                    break

            cell = TableCell(
                text=inner_text,
                bbox=None,
                row_span=rowspan,
                col_span=colspan,
                start_row_offset_idx=row_idx,
                end_row_offset_idx=row_idx + rowspan,
                start_col_offset_idx=col_idx,
                end_col_offset_idx=col_idx + colspan,
                column_header=(tag == "ched"),
                row_header=(tag == "rhed"),
                row_section=(tag == "srow"),
            )
            table_cells.append(cell)

    return otsl_seq, table_cells, num_rows, num_cols
