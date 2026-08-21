from types import SimpleNamespace

from docling_core.types.doc import TableCell

from docling.models.stages.table_structure.table_topology import (
    cells_to_otsl,
    infer_table_from_text_geometry,
    looks_undersegmented,
    repair_overlapping_cells,
    validate_table_topology,
)


def make_cell(
    *,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    text: str = "",
    column_header: bool = False,
    row_header: bool = False,
    row_section: bool = False,
):
    return TableCell.model_validate(
        {
            "row_span": end_row - start_row,
            "col_span": end_col - start_col,
            "start_row_offset_idx": start_row,
            "end_row_offset_idx": end_row,
            "start_col_offset_idx": start_col,
            "end_col_offset_idx": end_col,
            "text": text,
            "column_header": column_header,
            "row_header": row_header,
            "row_section": row_section,
            "bbox": None,
        }
    )


def test_validate_table_topology_detects_overlapping_cells():
    cells = [
        make_cell(start_row=4, end_row=7, start_col=3, end_col=4, text="Broad topic"),
        make_cell(
            start_row=6, end_row=7, start_col=3, end_col=4, text="Specific topic"
        ),
    ]

    diag = validate_table_topology(cells, num_rows=8, num_cols=4)

    assert not diag.valid
    assert (6, 3) in diag.overlapping_slots


def test_repair_overlapping_cells_shrinks_trailing_rowspan():
    cells = [
        make_cell(start_row=4, end_row=7, start_col=3, end_col=4, text="Broad topic"),
        make_cell(
            start_row=6, end_row=7, start_col=3, end_col=4, text="Specific topic"
        ),
    ]

    repaired, diag = repair_overlapping_cells(cells, num_rows=8, num_cols=4)

    assert diag.valid
    assert repaired[0].start_row_offset_idx == 4
    assert repaired[0].end_row_offset_idx == 6
    assert repaired[0].row_span == 2
    assert repaired[1].start_row_offset_idx == 6
    assert repaired[1].end_row_offset_idx == 7


def test_cells_to_otsl_represents_row_and_column_spans():
    cells = [
        make_cell(
            start_row=0,
            end_row=2,
            start_col=0,
            end_col=1,
            text="Row span",
            column_header=True,
        ),
        make_cell(
            start_row=0,
            end_row=1,
            start_col=1,
            end_col=3,
            text="Column span",
            column_header=True,
        ),
        make_cell(start_row=1, end_row=2, start_col=1, end_col=2, text="A"),
        make_cell(start_row=1, end_row=2, start_col=2, end_col=3, text="B"),
    ]

    otsl = cells_to_otsl(cells, num_rows=2, num_cols=3)

    assert "ucel" in otsl
    assert "lcel" in otsl


def test_looks_undersegmented_uses_geometry_not_fixture_text():
    cells = [
        make_cell(start_row=0, end_row=1, start_col=0, end_col=1, text="Header"),
        make_cell(start_row=1, end_row=2, start_col=0, end_col=1, text="A"),
        make_cell(start_row=2, end_row=3, start_col=0, end_col=1, text="B"),
    ]

    text_cells = [
        SimpleNamespace(text="row 1", bbox=SimpleNamespace(t=10, b=20)),
        SimpleNamespace(text="row 2", bbox=SimpleNamespace(t=30, b=40)),
        SimpleNamespace(text="row 3", bbox=SimpleNamespace(t=50, b=60)),
        SimpleNamespace(text="row 4", bbox=SimpleNamespace(t=70, b=80)),
        SimpleNamespace(text="row 5", bbox=SimpleNamespace(t=90, b=100)),
        SimpleNamespace(text="row 6", bbox=SimpleNamespace(t=110, b=120)),
    ]

    assert looks_undersegmented(
        cells,
        num_rows=3,
        otsl_seq=["fcel", "nl", "fcel", "nl", "fcel", "nl"],
        text_cells=text_cells,
    )


def make_bbox(*, left: float, top: float, right: float, bottom: float):
    return SimpleNamespace(l=left, t=top, r=right, b=bottom)


def attach_bbox(cell, *, left: float, top: float, right: float, bottom: float):
    object.__setattr__(
        cell,
        "bbox",
        make_bbox(left=left, top=top, right=right, bottom=bottom),
    )
    return cell


def make_text_cell(text: str, *, row: int, col: int):
    center_x = 50 + col * 100
    center_y = 10 + row * 20
    return SimpleNamespace(
        text=text,
        bbox=make_bbox(
            left=center_x - 20,
            top=center_y - 5,
            right=center_x + 20,
            bottom=center_y + 5,
        ),
    )


def test_infer_table_from_text_geometry_recovers_undersegmented_v2_table():
    # This models the V2 failure mode from issue #3698: the decoded OTSL has
    # too few rows, so text from multiple visual rows gets collapsed into one
    # logical row/cell. The fallback should use text geometry to recover the
    # richer grid without relying on any fixture-specific strings.
    cells = [
        attach_bbox(
            make_cell(
                start_row=0,
                end_row=1,
                start_col=1,
                end_col=4,
                text="Seminar",
                column_header=True,
            ),
            left=100,
            top=0,
            right=400,
            bottom=20,
        ),
        attach_bbox(
            make_cell(
                start_row=1,
                end_row=2,
                start_col=1,
                end_col=3,
                text="Schedule",
                column_header=True,
            ),
            left=100,
            top=20,
            right=300,
            bottom=40,
        ),
        attach_bbox(
            make_cell(
                start_row=1,
                end_row=3,
                start_col=3,
                end_col=4,
                text="Topic",
                column_header=True,
            ),
            left=300,
            top=20,
            right=400,
            bottom=60,
        ),
        make_cell(
            start_row=2,
            end_row=3,
            start_col=3,
            end_col=4,
            text="Collapsed topic text",
        ),
        make_cell(
            start_row=3,
            end_row=4,
            start_col=3,
            end_col=4,
            text="Collapsed later topic text",
        ),
    ]

    text_cells = [
        make_text_cell("Seminar", row=0, col=2),
        make_text_cell("Day", row=1, col=0),
        make_text_cell("Schedule", row=1, col=2),
        make_text_cell("Topic", row=1, col=3),
        make_text_cell("Begin", row=2, col=1),
        make_text_cell("End", row=2, col=2),
        make_text_cell("Monday", row=3, col=0),
        make_text_cell("8:00 a.m.", row=3, col=1),
        make_text_cell("5.00 p.m.", row=3, col=2),
        make_text_cell("Introduction", row=3, col=3),
        make_text_cell("8:00 a.m.", row=4, col=1),
        make_text_cell("11:00 a.m.", row=4, col=2),
        make_text_cell("XPath", row=4, col=3),
        make_text_cell("Tuesday", row=4, col=0),
        make_text_cell("11:00 a.m.", row=5, col=1),
        make_text_cell("2:00 p.m.", row=5, col=2),
        make_text_cell("2:00 p.m.", row=6, col=1),
        make_text_cell("5:00 p.m.", row=6, col=2),
        make_text_cell("XSL Transformations", row=6, col=3),
        make_text_cell("Wednesday", row=7, col=0),
        make_text_cell("8:00 a.m.", row=7, col=1),
        make_text_cell("12:00 p.m.", row=7, col=2),
        make_text_cell("XSL Formatting Object", row=7, col=3),
    ]

    fallback_cells, fallback_rows, fallback_cols, fallback_otsl, diag = (
        infer_table_from_text_geometry(
            cells,
            num_rows=5,
            num_cols=4,
            otsl_seq=["fcel", "fcel", "fcel", "fcel", "nl"] * 5,
            text_cells=text_cells,
        )
    )

    by_text = {cell.text: cell for cell in fallback_cells}

    assert diag.valid
    assert fallback_rows == 8
    assert fallback_cols == 4
    assert "ucel" in fallback_otsl
    assert "lcel" in fallback_otsl

    assert by_text["Seminar"].start_col_offset_idx == 1
    assert by_text["Seminar"].end_col_offset_idx == 4

    assert by_text["Schedule"].start_col_offset_idx == 1
    assert by_text["Schedule"].end_col_offset_idx == 3

    assert by_text["Tuesday"].start_row_offset_idx == 4
    assert by_text["Tuesday"].end_row_offset_idx == 7

    assert by_text["XPath"].start_row_offset_idx == 4
    assert by_text["XPath"].end_row_offset_idx == 6

    assert by_text["XSL Transformations"].start_row_offset_idx == 6
    assert by_text["XSL Transformations"].end_row_offset_idx == 7


def test_infer_table_from_text_geometry_repairs_same_row_count_overspan():
    # This models the V1 do_cell_matching=False failure mode from issue #3698:
    # the table already has the correct number of rows, but one topic cell
    # overspans existing rows and hides a later topic cell. The fallback must
    # still run when allow_same_row_count=True.
    cells = [
        make_cell(
            start_row=0,
            end_row=1,
            start_col=1,
            end_col=4,
            text="Seminar",
            column_header=True,
        ),
        make_cell(
            start_row=1,
            end_row=2,
            start_col=1,
            end_col=3,
            text="Schedule",
            column_header=True,
        ),
        make_cell(
            start_row=1,
            end_row=3,
            start_col=3,
            end_col=4,
            text="Topic",
            column_header=True,
        ),
        make_cell(
            start_row=4,
            end_row=7,
            start_col=0,
            end_col=1,
            text="Tuesday",
        ),
        make_cell(
            start_row=4,
            end_row=7,
            start_col=3,
            end_col=4,
            text="XPath",
        ),
    ]

    text_cells = [
        make_text_cell("Seminar", row=0, col=2),
        make_text_cell("Day", row=1, col=0),
        make_text_cell("Schedule", row=1, col=2),
        make_text_cell("Topic", row=1, col=3),
        make_text_cell("Begin", row=2, col=1),
        make_text_cell("End", row=2, col=2),
        make_text_cell("Monday", row=3, col=0),
        make_text_cell("8:00 a.m.", row=3, col=1),
        make_text_cell("5.00 p.m.", row=3, col=2),
        make_text_cell("Introduction", row=3, col=3),
        make_text_cell("Tuesday", row=4, col=0),
        make_text_cell("8:00 a.m.", row=4, col=1),
        make_text_cell("11:00 a.m.", row=4, col=2),
        make_text_cell("XPath", row=4, col=3),
        make_text_cell("11:00 a.m.", row=5, col=1),
        make_text_cell("2:00 p.m.", row=5, col=2),
        make_text_cell("2:00 p.m.", row=6, col=1),
        make_text_cell("5:00 p.m.", row=6, col=2),
        make_text_cell("XSL Transformations", row=6, col=3),
        make_text_cell("Wednesday", row=7, col=0),
        make_text_cell("8:00 a.m.", row=7, col=1),
        make_text_cell("12:00 p.m.", row=7, col=2),
        make_text_cell("XSL Formatting Object", row=7, col=3),
    ]

    fallback_cells, fallback_rows, fallback_cols, _, diag = (
        infer_table_from_text_geometry(
            cells,
            num_rows=8,
            num_cols=4,
            otsl_seq=["fcel", "fcel", "fcel", "fcel", "nl"] * 8,
            text_cells=text_cells,
            allow_same_row_count=True,
        )
    )

    by_text = {cell.text: cell for cell in fallback_cells}

    assert diag.valid
    assert fallback_rows == 8
    assert fallback_cols == 4

    assert by_text["Tuesday"].start_row_offset_idx == 4
    assert by_text["Tuesday"].end_row_offset_idx == 7

    assert by_text["XPath"].start_row_offset_idx == 4
    assert by_text["XPath"].end_row_offset_idx == 6

    assert by_text["XSL Transformations"].start_row_offset_idx == 6
    assert by_text["XSL Transformations"].end_row_offset_idx == 7


def test_looks_undersegmented_detects_column_collapse_from_geometry():
    cells = [
        make_cell(start_row=0, end_row=1, start_col=0, end_col=1, text="A"),
        make_cell(start_row=0, end_row=1, start_col=1, end_col=2, text="B"),
        make_cell(start_row=0, end_row=1, start_col=2, end_col=3, text="C"),
        make_cell(start_row=0, end_row=1, start_col=3, end_col=4, text="D"),
    ]

    text_cells = [
        make_text_cell("col 0", row=0, col=0),
        make_text_cell("col 1", row=0, col=1),
        make_text_cell("col 2", row=0, col=2),
        make_text_cell("col 3", row=0, col=3),
        make_text_cell("col 4", row=0, col=4),
    ]

    assert looks_undersegmented(
        cells,
        num_rows=1,
        num_cols=4,
        otsl_seq=["ucel", "fcel", "fcel", "fcel", "nl"],
        text_cells=text_cells,
    )


def test_infer_table_from_text_geometry_allows_column_count_growth_when_enabled():
    from types import SimpleNamespace

    from docling.models.stages.table_structure.table_topology import (
        infer_table_from_text_geometry,
    )

    def text_cell(text: str, left: float, top: float) -> SimpleNamespace:
        return SimpleNamespace(
            text=text,
            bbox=SimpleNamespace(l=left, t=top, r=left + 20, b=top + 10),
        )

    text_cells = [
        text_cell("A", 0, 0),
        text_cell("B", 50, 0),
        text_cell("C", 100, 0),
        text_cell("D", 150, 0),
        text_cell("E", 200, 0),
    ]

    base_cells, base_rows, base_cols, base_otsl, base_diag = (
        infer_table_from_text_geometry(
            [],
            1,
            4,
            [],
            text_cells=text_cells,
            allow_same_row_count=True,
            allow_column_count_growth=False,
        )
    )

    grown_cells, grown_rows, grown_cols, grown_otsl, grown_diag = (
        infer_table_from_text_geometry(
            [],
            1,
            4,
            [],
            text_cells=text_cells,
            allow_same_row_count=True,
            allow_column_count_growth=True,
        )
    )

    assert base_diag.valid
    assert grown_diag.valid
    assert base_rows == 1
    assert base_cols == 4
    assert grown_rows == 1
    assert grown_cols == 5
    assert len(base_cells) <= len(grown_cells)
    assert base_otsl != grown_otsl
