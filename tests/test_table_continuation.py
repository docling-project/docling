# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for merging tables that continue across a page break."""

from pathlib import Path

from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
    TableItem,
)

from docling.datamodel.pipeline_options import TableContinuationOptions
from docling.models.stages.table_continuation.table_continuation_model import (
    TableContinuationModel,
)

_PAGE = Size(width=600, height=800)
_COLUMN_LEFT = [50.0, 250.0, 400.0]
_ROW_HEIGHT = 20.0


def _add_table(
    doc: DoclingDocument,
    page_no: int,
    rows: list[list[str]],
    *,
    header_rows: int = 0,
    top: float = 50.0,
    column_left: list[float] | None = None,
) -> TableItem:
    """Add a table fragment whose cells sit on a regular grid, top-left origin."""
    lefts = column_left or _COLUMN_LEFT
    cells = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            y = top + r * _ROW_HEIGHT
            cells.append(
                TableCell(
                    text=text,
                    bbox=BoundingBox(
                        l=lefts[c],
                        t=y,
                        r=lefts[c] + 10 + 5 * len(text),
                        b=y + _ROW_HEIGHT - 4,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=r < header_rows,
                )
            )
    if page_no not in doc.pages:
        doc.add_page(page_no=page_no, size=_PAGE)
    prov = ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(
            l=lefts[0],
            t=top,
            r=550,
            b=top + len(rows) * _ROW_HEIGHT,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
        charspan=(0, 0),
    )
    data = TableData(table_cells=cells, num_rows=len(rows), num_cols=len(rows[0]))
    return doc.add_table(data=data, prov=prov)


def _attach(
    doc: DoclingDocument, table: TableItem, label: DocItemLabel, text: str
) -> None:
    item = doc.add_text(label=label, text=text, parent=table)
    if label == DocItemLabel.CAPTION:
        table.captions.append(item.get_ref())
    else:
        table.footnotes.append(item.get_ref())


def _merge(doc: DoclingDocument, **opts):
    model = TableContinuationModel(TableContinuationOptions(enabled=True, **opts))
    return model.merge_continued_tables(doc)


def _rows(table: TableItem) -> list[list[str]]:
    return [[cell.text for cell in row] for row in table.data.grid]


_HEADER = ["No.", "Question", "Answer"]


def test_repeated_header_is_dropped():
    doc = DoclingDocument(name="repeated-header")
    _add_table(doc, 1, [_HEADER, ["1", "Scope?", "Yes"]], header_rows=1, top=600)
    _add_table(doc, 2, [_HEADER, ["2", "Deadline?", "May"]], header_rows=1)

    decisions = _merge(doc)

    assert [d.merged for d in decisions] == [True]
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert _rows(table) == [_HEADER, ["1", "Scope?", "Yes"], ["2", "Deadline?", "May"]]
    assert [p.page_no for p in table.prov] == [1, 2]
    assert [c.column_header for c in table.data.table_cells].count(True) == 3


def test_split_record_is_joined_and_false_header_cleared():
    # The continuation starts with the tail of record 1, which table structure flags as a
    # header because it is the first row on the page (the TenderNed case from #2976).
    doc = DoclingDocument(name="split-record")
    _add_table(
        doc,
        1,
        [_HEADER, ["1", "What is the", "It is"]],
        header_rows=1,
        top=700,
    )
    _add_table(
        doc,
        2,
        [["", "scope of work?", "described in annex A."], ["2", "Deadline?", "May"]],
        header_rows=1,
    )

    _merge(doc)

    table = doc.tables[0]
    assert _rows(table) == [
        _HEADER,
        ["1", "What is the scope of work?", "It is described in annex A."],
        ["2", "Deadline?", "May"],
    ]
    header_rows = {
        c.start_row_offset_idx for c in table.data.table_cells if c.column_header
    }
    assert header_rows == {0}


def test_split_row_repair_can_be_disabled():
    doc = DoclingDocument(name="split-record-kept")
    _add_table(doc, 1, [_HEADER, ["1", "What is the", "It is"]], header_rows=1)
    _add_table(doc, 2, [["", "scope of work?", "described."], ["2", "When?", "May"]])

    _merge(doc, repair_split_rows=False)

    assert _rows(doc.tables[0])[1:] == [
        ["1", "What is the", "It is"],
        ["", "scope of work?", "described."],
        ["2", "When?", "May"],
    ]


def test_three_fragments_become_one_table():
    doc = DoclingDocument(name="three-pages")
    _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    _add_table(doc, 2, [["2", "c", "d"]])
    _add_table(doc, 3, [["3", "e", "f"]])

    decisions = _merge(doc)

    assert [d.merged for d in decisions] == [True, True]
    assert len(doc.tables) == 1
    assert [row[0] for row in _rows(doc.tables[0])] == ["No.", "1", "2", "3"]
    assert [p.page_no for p in doc.tables[0].prov] == [1, 2, 3]


def test_different_tables_on_consecutive_pages_are_kept_apart():
    doc = DoclingDocument(name="different-tables")
    _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    _add_table(
        doc,
        2,
        [["Name", "Role", "Since"], ["Ann", "Lead", "2020"]],
        header_rows=1,
        column_left=[50.0, 180.0, 450.0],
    )

    decisions = _merge(doc)

    assert [d.merged for d in decisions] == [False]
    assert "misaligned" in decisions[0].reason
    assert len(doc.tables) == 2


def test_tables_are_kept_apart_when_not_on_consecutive_pages():
    doc = DoclingDocument(name="gap")
    _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    _add_table(doc, 3, [["2", "c", "d"]])

    assert [d.merged for d in _merge(doc)] == [False]
    assert len(doc.tables) == 2


def test_text_between_tables_prevents_merge():
    doc = DoclingDocument(name="text-between")
    _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    doc.add_text(label=DocItemLabel.TEXT, text="The next table lists the deadlines.")
    _add_table(doc, 2, [["2", "c", "d"]])

    assert _merge(doc) == []
    assert len(doc.tables) == 2


def test_continuation_with_own_caption_is_kept_apart():
    doc = DoclingDocument(name="own-caption")
    _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    second = _add_table(doc, 2, [["2", "c", "d"]])
    _attach(doc, second, DocItemLabel.CAPTION, "Table 2: Deadlines")

    assert [d.merged for d in _merge(doc)] == [False]
    assert len(doc.tables) == 2


def test_continued_caption_is_dropped_and_footnote_moved(tmp_path: Path):
    doc = DoclingDocument(name="continued-caption")
    first = _add_table(doc, 1, [_HEADER, ["1", "a", "b"]], header_rows=1)
    _attach(doc, first, DocItemLabel.CAPTION, "Table 1: Questions")
    second = _add_table(doc, 2, [["2", "c", "d"]])
    _attach(doc, second, DocItemLabel.CAPTION, "Table 1 (continued)")
    _attach(doc, second, DocItemLabel.FOOTNOTE, "* Answers are binding.")

    _merge(doc)

    table = doc.tables[0]
    assert [ref.resolve(doc).text for ref in table.captions] == ["Table 1: Questions"]
    assert [ref.resolve(doc).text for ref in table.footnotes] == [
        "* Answers are binding."
    ]
    assert [t.text for t in doc.texts] == [
        "Table 1: Questions",
        "* Answers are binding.",
    ]

    # All references stay valid: the merged document saves, reloads and exports.
    path = tmp_path / "merged.json"
    doc.save_as_json(path)
    reloaded = DoclingDocument.load_from_json(path)
    footnote = reloaded.tables[0].footnotes[0].resolve(reloaded)
    assert footnote.parent is not None
    assert footnote.parent.cref == reloaded.tables[0].self_ref
    assert "continued" not in reloaded.export_to_markdown()
