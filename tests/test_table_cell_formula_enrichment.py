# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Table-cell formula enrichment: rich-cell construction, gating and failure modes.

Driven by a stub engine so the file runs on CPU. The stub appends ``</formula>`` and
``<end_of_utterance>``, which only disappear if the stage really post-processes through the
shared ``CodeFormulaVlmModel``.
"""

from collections.abc import Iterable
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    GroupItem,
    GroupLabel,
    ProvenanceItem,
    RichTableCell,
    Size,
    TableCell,
    TableItem,
)
from docling_core.types.doc.items.table.table_data import TableData
from PIL import Image

from docling.datamodel.base_models import Page
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions, PdfPipelineOptions
from docling.models.inference_engines.vlm import VlmEngineInput, VlmEngineOutput
from docling.models.stages.code_formula.code_formula_vlm_model import (
    CodeFormulaVlmModel,
)
from docling.models.stages.code_formula.table_cell_formula_vlm_model import (
    TableCellFormulaVlmModel,
)

PAGE_SIZE = Size(width=612.0, height=792.0)
# The stub's raw output, and what it must look like after post-processing.
RAW_OUTPUT = "a ^ { 2 } + 8 = 1 2</formula><end_of_utterance>"
CLEAN_OUTPUT = "a ^ { 2 } + 8 = 1 2"


class _DummyEngine:
    def __init__(self, text: str = RAW_OUTPUT, raises: bool = False):
        self.text = text
        self.raises = raises
        self.received_inputs: list[VlmEngineInput] = []
        self.batches: list[int] = []

    def predict_batch(self, inputs: Iterable[VlmEngineInput]):
        batch = list(inputs)
        if self.raises:
            raise RuntimeError("engine exploded")
        self.received_inputs.extend(batch)
        self.batches.append(len(batch))
        return [
            VlmEngineOutput(text=self.text, stop_reason="end_of_sequence")
            for _ in batch
        ]

    def cleanup(self):  # pragma: no cover - lifecycle only
        pass


class _FakePageBackend:
    """Just enough backend for Page.get_image(cropbox=...) to return a crop."""

    def __init__(self):
        self.cropboxes: list[BoundingBox] = []

    def get_page_image(self, scale: float = 1.0, cropbox=None):
        self.cropboxes.append(cropbox)
        return Image.new("RGB", (16, 8), "white")


def _make_code_formula_model(
    *, extract_formulas: bool = True, engine: _DummyEngine | None = None
) -> CodeFormulaVlmModel:
    model = CodeFormulaVlmModel.__new__(CodeFormulaVlmModel)
    model.enabled = extract_formulas
    model.options = CodeFormulaVlmOptions.from_preset("codeformulav2").model_copy(
        update={"extract_formulas": extract_formulas, "extract_code": False}
    )
    # engine is None exactly when the formula stage is disabled, which is what the table-cell
    # stage keys its own gate on.
    model.engine = (engine or _DummyEngine()) if extract_formulas else None
    return model


def _make_stage(
    *,
    enabled: bool = True,
    extract_formulas: bool = True,
    engine: _DummyEngine | None = None,
) -> TableCellFormulaVlmModel:
    return TableCellFormulaVlmModel(
        enabled=enabled,
        code_formula_model=_make_code_formula_model(
            extract_formulas=extract_formulas, engine=engine
        ),
    )


def _cell(text: str, row: int, col: int, *, row_span: int = 1, col_span: int = 1):
    # Well inside the page, so clamping never changes them.
    return TableCell(
        text=text,
        bbox=BoundingBox(
            l=50.0 + col * 100.0,
            r=50.0 + (col + col_span) * 100.0,
            t=700.0 - row * 50.0,
            b=700.0 - (row + row_span) * 50.0,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        ),
        row_span=row_span,
        col_span=col_span,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=col,
        end_col_offset_idx=col + col_span,
    )


def _doc_with_tables(*tables: list[TableCell]) -> DoclingDocument:
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=PAGE_SIZE)
    for cells in tables:
        num_rows = max((c.end_row_offset_idx for c in cells), default=0)
        num_cols = max((c.end_col_offset_idx for c in cells), default=0)
        doc.add_table(
            data=TableData(table_cells=cells, num_rows=num_rows, num_cols=num_cols),
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(
                    l=40.0,
                    r=560.0,
                    t=720.0,
                    b=100.0,
                    coord_origin=CoordOrigin.BOTTOMLEFT,
                ),
                charspan=(0, 0),
            ),
        )
    return doc


def _conv_res(doc: DoclingDocument, *, pages: list[Page] | None = None, backend=True):
    if pages is None:
        page = Page(page_no=1, size=PAGE_SIZE)
        if backend:
            page._backend = _FakePageBackend()
        pages = [page]
    conv_res = MagicMock()
    conv_res.document = doc
    conv_res.pages = pages
    return conv_res


def _enrich(stage: TableCellFormulaVlmModel, doc: DoclingDocument, conv_res) -> None:
    """Run the stage the way BasePipeline._enrich_document does, one table at a time."""
    prepared = [
        p
        for item, _ in doc.iterate_items()
        if (p := stage.prepare_element(conv_res=conv_res, element=item)) is not None
    ]
    for element in prepared:
        list(stage(doc=doc, element_batch=[element]))  # must exhaust


def _rich_cells(table: TableItem) -> list[RichTableCell]:
    return [c for c in table.data.table_cells if isinstance(c, RichTableCell)]


# --------------------------------------------------------------------------------- gating


def test_option_is_off_by_default() -> None:
    assert PdfPipelineOptions().do_table_cell_formula_enrichment is False


def test_disabled_stage_touches_nothing() -> None:
    engine = _DummyEngine()
    stage = _make_stage(enabled=False, engine=engine)
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    table = doc.tables[0]
    conv_res = _conv_res(doc)

    assert stage.is_processable(doc=doc, element=table) is False
    assert stage.prepare_element(conv_res=conv_res, element=table) is None
    _enrich(stage, doc, conv_res)

    assert _rich_cells(table) == []
    assert engine.received_inputs == []
    assert doc.groups == []


def test_requires_formula_enrichment_and_says_so(caplog) -> None:
    with caplog.at_level("WARNING"):
        stage = _make_stage(enabled=True, extract_formulas=False)
    assert stage.enabled is False
    assert "do_formula_enrichment" in caplog.text

    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    _enrich(stage, doc, _conv_res(doc))
    assert _rich_cells(doc.tables[0]) == []


# ---------------------------------------------------------------------- rich-cell output


def test_matching_cell_becomes_a_rich_cell_pointing_at_a_group() -> None:
    doc = _doc_with_tables([_cell("Pathloss", 0, 0), _cell("a^2 + 8 = 12", 0, 1)])
    table = doc.tables[0]
    stage = _make_stage()
    _enrich(stage, doc, _conv_res(doc))

    rich = _rich_cells(table)
    assert len(rich) == 1, "only the formula-looking cell should be enriched"
    cell = rich[0]
    assert cell.start_col_offset_idx == 1
    # The plain fields survive the replacement.
    assert cell.text == "a^2 + 8 = 12"
    assert cell.bbox is not None

    group = cell.ref.resolve(doc)
    assert isinstance(group, GroupItem)
    assert group.label is GroupLabel.UNSPECIFIED
    assert len(group.children) == 1

    formula = group.children[0].resolve(doc)
    assert isinstance(formula, FormulaItem)
    assert formula.label is DocItemLabel.FORMULA
    # Post-processed through the shared model: the stub's special tokens are gone.
    assert formula.text == CLEAN_OUTPUT
    # The garbled source text is kept, so the transcription stays auditable.
    assert formula.orig == "a^2 + 8 = 12"
    assert len(formula.prov) == 1
    assert formula.prov[0].page_no == 1

    # Both halves of what validate_tree demands of a RichTableCell ref.
    assert group.parent.resolve(doc) is table
    assert cell.ref.cref in {child.cref for child in table.children}


def test_enriched_document_round_trips() -> None:
    """DoclingDocument.validate_tree runs on model_validate, and it checks rich cells."""
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    _enrich(_make_stage(), doc, _conv_res(doc))

    reloaded = DoclingDocument.model_validate(doc.model_dump())

    rich = _rich_cells(reloaded.tables[0])
    assert len(rich) == 1
    group = rich[0].ref.resolve(reloaded)
    assert isinstance(group, GroupItem)
    assert isinstance(group.children[0].resolve(reloaded), FormulaItem)


def test_latex_reaches_the_exported_dataframe() -> None:
    doc = _doc_with_tables([_cell("Scenario", 0, 0), _cell("a^2 + 8 = 12", 0, 1)])
    _enrich(_make_stage(), doc, _conv_res(doc))

    rendered = doc.tables[0].export_to_dataframe(doc=doc).to_string()
    # Asserted with `in` rather than equality: the delimiter the markdown serializer wraps a
    # formula in is its business, not this stage's contract.
    assert CLEAN_OUTPUT in rendered
    assert "<!-- rich cell -->" not in rendered


def test_two_tables_in_one_document_both_enrich() -> None:
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)], [_cell("P = ∑ x_i", 0, 0)])
    stage = _make_stage()
    _enrich(stage, doc, _conv_res(doc))

    assert [len(_rich_cells(t)) for t in doc.tables] == [1, 1]
    # Distinct group names, so nothing collides in a document with many tables.
    names = {g.name for g in doc.groups}
    assert len(names) == 2, names


# ------------------------------------------------------------------ selection and batching


def test_prefilter_decides_which_cells_are_cropped() -> None:
    cells = [
        _cell("Scenario", 0, 0),
        _cell("a^2 + 8 = 12", 0, 1),
        _cell("20 dB", 1, 0),
        _cell("d_{3D} ≤ 20 m", 1, 1),
    ]
    doc = _doc_with_tables(cells)
    conv_res = _conv_res(doc)
    stage = _make_stage()

    prepared = stage.prepare_element(conv_res=conv_res, element=doc.tables[0])
    assert prepared is not None
    assert [idx for idx, _ in prepared.cell_crops] == [1, 3]
    # One render per candidate cell, not per cell.
    assert len(conv_res.pages[0]._backend.cropboxes) == 2


def test_prepared_element_wraps_the_live_table() -> None:
    """If pydantic ever copied `item`, every mutation would be silently lost."""
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    prepared = _make_stage().prepare_element(
        conv_res=_conv_res(doc), element=doc.tables[0]
    )
    assert prepared is not None
    assert prepared.item is doc.tables[0]


def test_spanned_cell_is_transcribed_once() -> None:
    cells = [_cell("a^2 + 8 = 12", 0, 0, row_span=2, col_span=2)]
    doc = _doc_with_tables(cells)
    engine = _DummyEngine()
    _enrich(_make_stage(engine=engine), doc, _conv_res(doc))

    assert len(engine.received_inputs) == 1
    rich = _rich_cells(doc.tables[0])
    assert len(rich) == 1
    grid = doc.tables[0].data.grid
    # The one object appears at every position it covers.
    assert all(grid[r][c] is rich[0] for r in range(2) for c in range(2))


def test_crops_are_batched_for_the_engine_not_per_table() -> None:
    # Seven candidates in one table, laid out down a column so they all stay on the page --
    # _cell()'s 100pt columns would push the seventh past the 612pt page width, where
    # _clamp_to_page would rightly reject it and this test would be measuring clamping.
    cells = [_cell(f"x_{i} = {i} + a^2", i, 0) for i in range(7)]
    doc = _doc_with_tables(cells)
    engine = _DummyEngine()
    stage = _make_stage(engine=engine)
    _enrich(stage, doc, _conv_res(doc))

    assert len(engine.received_inputs) == 7
    assert engine.batches == [stage.vlm_batch_size, 7 - stage.vlm_batch_size]


# ----------------------------------------------------------------------------- edge cases


def test_already_rich_cell_is_left_alone() -> None:
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    table = doc.tables[0]
    # Pre-seed the cell as rich, the way the reading-order stage does for a picture cell.
    group = doc.add_group(label=GroupLabel.UNSPECIFIED, name="pre", parent=table)
    doc.add_text(label=DocItemLabel.TEXT, text="pre-existing", parent=group)
    original = table.data.table_cells[0]
    table.data.table_cells[0] = RichTableCell(
        **original.model_dump(exclude={"ref"}), ref=group.get_ref()
    )

    engine = _DummyEngine()
    conv_res = _conv_res(doc)
    stage = _make_stage(engine=engine)
    assert stage.prepare_element(conv_res=conv_res, element=table) is None
    _enrich(stage, doc, conv_res)

    assert table.data.table_cells[0].ref.cref == group.self_ref
    assert engine.received_inputs == []


def test_engine_failure_leaves_the_table_intact() -> None:
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    _enrich(_make_stage(engine=_DummyEngine(raises=True)), doc, _conv_res(doc))

    assert _rich_cells(doc.tables[0]) == []
    assert doc.groups == []


def test_empty_transcription_adds_no_group() -> None:
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    _enrich(_make_stage(engine=_DummyEngine(text="")), doc, _conv_res(doc))

    assert _rich_cells(doc.tables[0]) == []
    assert doc.groups == []


@pytest.mark.parametrize("case", ["no_pages", "no_backend", "no_prov"])
def test_prepare_element_declines_without_page_geometry(case: str) -> None:
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    table = doc.tables[0]
    stage = _make_stage()

    if case == "no_pages":
        # The arithmetic this replaced (page_no - pages[0].page_no) raised IndexError here.
        conv_res = _conv_res(doc, pages=[])
    elif case == "no_backend":
        conv_res = _conv_res(doc, backend=False)
    else:
        table.prov = []
        conv_res = _conv_res(doc)

    assert stage.prepare_element(conv_res=conv_res, element=table) is None


def test_cell_without_bbox_is_skipped() -> None:
    cell = _cell("a^2 + 8 = 12", 0, 0)
    cell.bbox = None
    doc = _doc_with_tables([cell])
    assert (
        _make_stage().prepare_element(conv_res=_conv_res(doc), element=doc.tables[0])
        is None
    )


def test_degenerate_cell_bbox_is_rejected() -> None:
    cell = _cell("a^2 + 8 = 12", 0, 0)
    # Zero-width cell: expanding it by a scale factor keeps it zero-width.
    cell.bbox = BoundingBox(
        l=100.0, r=100.0, t=700.0, b=650.0, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    doc = _doc_with_tables([cell])
    assert (
        _make_stage().prepare_element(conv_res=_conv_res(doc), element=doc.tables[0])
        is None
    )


def test_cell_bbox_is_converted_to_the_table_prov_origin() -> None:
    """The two table-structure paths disagree about cell origin, so it must be derived."""
    cell = _cell("a^2 + 8 = 12", 0, 0)
    cell.bbox = BoundingBox(
        l=50.0, r=150.0, t=92.0, b=142.0, coord_origin=CoordOrigin.TOPLEFT
    )
    doc = _doc_with_tables([cell])
    _enrich(_make_stage(), doc, _conv_res(doc))

    formula = _rich_cells(doc.tables[0])[0].ref.resolve(doc).children[0].resolve(doc)
    prov_bbox = formula.prov[0].bbox
    assert prov_bbox.coord_origin is CoordOrigin.BOTTOMLEFT
    # 792 - 92 = 700, 792 - 142 = 650
    assert (prov_bbox.t, prov_bbox.b) == (700.0, 650.0)


def test_mutating_during_iteration_terminates_and_does_not_re_enrich() -> None:
    """The stage appends to table.children while iterate_items() is mid-walk."""
    doc = _doc_with_tables([_cell("a^2 + 8 = 12", 0, 0)])
    stage = _make_stage()
    conv_res = _conv_res(doc)

    visited: list[str] = []
    prepared_count = 0
    # One live pass, mutating as it goes -- the shape _enrich_document uses.
    for item, _ in doc.iterate_items():
        visited.append(type(item).__name__)
        assert len(visited) < 50, "iteration did not terminate"
        element = stage.prepare_element(conv_res=conv_res, element=item)
        if element is not None:
            prepared_count += 1
            list(stage(doc=doc, element_batch=[element]))

    # The table was enriched exactly once, and the FormulaItem the stage added was visited
    # (proving the mutation is seen) but declined by is_processable.
    assert prepared_count == 1
    assert visited.count("FormulaItem") == 1
    assert len(_rich_cells(doc.tables[0])) == 1
    assert len(doc.tables[0].data.table_cells) == 1
