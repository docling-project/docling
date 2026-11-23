# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from pathlib import Path

from docling_core.types.doc import BoundingBox, DocItemLabel, Size
from docling_core.types.doc.page import (
    BoundingRectangle,
    PdfPageBoundaryType,
    PdfPageGeometry,
    SegmentedPdfPage,
    TextCell,
)
from PIL import Image

from docling.datamodel.base_models import (
    Cluster,
    InputFormat,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult, InputDocument, _DummyBackend
from docling.models.stages.table_structure.table_structure_model import (
    TableStructureModel,
    _words_and_ocr_cells,
)


def _cell(bbox: BoundingBox, text: str, from_ocr: bool) -> TextCell:
    return TextCell(
        index=0,
        rect=BoundingRectangle.from_bounding_box(bbox),
        text=text,
        orig=text,
        from_ocr=from_ocr,
    )


HEADER_BOX = BoundingBox(l=0, t=0, r=50, b=10)
VALUE_BOX = BoundingBox(l=60, t=20, r=90, b=30)


def test_table_tokens_keep_ocr_lines_next_to_native_words() -> None:
    # A scanned form with one filled-in native value: the OCR header must still
    # reach the table, and a native word under OCR text must not be doubled.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    filled_value = _cell(VALUE_BOX, "5000", from_ocr=False)
    hidden_word = _cell(BoundingBox(l=5, t=2, r=20, b=8), "Week", from_ocr=False)

    tokens = _words_and_ocr_cells(
        word_cells=[filled_value, hidden_word],
        cluster_cells=[header, native_line],
    )

    assert [c.text for c in tokens] == ["5000", "Week Begin"]


def test_table_tokens_prefer_ocr_words_over_their_line() -> None:
    # An OCR engine with word output (e.g. a third-party plugin) also writes
    # OCR word cells. They replace their OCR line. The OCR word on the native
    # value is dropped: the OCR merge kept the native text there.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    words = [
        _cell(BoundingBox(l=0, t=0, r=22, b=10), "Week", from_ocr=True),
        _cell(BoundingBox(l=26, t=0, r=50, b=10), "Begin", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=False),
    ]

    tokens = _words_and_ocr_cells(word_cells=words, cluster_cells=[header, native_line])

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week", True),
        ("Begin", True),
    ]


def test_table_tokens_keep_native_lines_without_word_cells() -> None:
    # Backends such as PyPdfiumDocumentBackend emit textline_cells only
    # (word_cells is empty). When OCR keeps even one cell in the table
    # cluster, all surviving native lines must still reach TableFormer.
    col_header = _cell(BoundingBox(l=10, t=10, r=80, b=22), "Function", from_ocr=False)
    row_value = _cell(BoundingBox(l=10, t=30, r=80, b=42), "Govern", from_ocr=False)
    ocr_noise = _cell(BoundingBox(l=120, t=30, r=150, b=42), "| =|", from_ocr=True)

    tokens = _words_and_ocr_cells(
        word_cells=[],
        cluster_cells=[col_header, row_value, ocr_noise],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("Function", False),
        ("Govern", False),
        ("| =|", True),
    ]


def test_table_tokens_drop_ocr_words_on_native_lines_without_native_words() -> None:
    # When a line-only PDF backend is paired with a word-level OCR plugin,
    # OCR words overlapping a kept native line must be dropped, while OCR
    # words on an OCR line replace that OCR line.
    header = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    ocr_words = [
        _cell(BoundingBox(l=0, t=0, r=22, b=10), "Week", from_ocr=True),
        _cell(BoundingBox(l=26, t=0, r=50, b=10), "Begin", from_ocr=True),
        _cell(VALUE_BOX, "5000", from_ocr=True),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=ocr_words,
        cluster_cells=[header, native_line],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week", True),
        ("Begin", True),
    ]


def test_table_tokens_keep_uncovered_native_lines_alongside_native_words() -> None:
    # Native lines covered by finer native words use those words, while any
    # surviving native line without covering native words is retained.
    row1 = _cell(BoundingBox(l=10, t=10, r=90, b=22), "Row 1", from_ocr=False)
    row2 = _cell(BoundingBox(l=10, t=30, r=90, b=42), "Row 2", from_ocr=False)
    ocr_note = _cell(BoundingBox(l=10, t=50, r=90, b=62), "Note", from_ocr=True)
    words = [
        _cell(BoundingBox(l=10, t=10, r=45, b=22), "Row", from_ocr=False),
        _cell(BoundingBox(l=55, t=10, r=90, b=22), "1", from_ocr=False),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=words,
        cluster_cells=[row1, row2, ocr_note],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("Row", False),
        ("1", False),
        ("Row 2", False),
        ("Note", True),
    ]


def test_table_tokens_ignore_blank_word_cells() -> None:
    # A whitespace-only word cell must not mask a non-empty line cell.
    native_line = _cell(VALUE_BOX, "5000", from_ocr=False)
    ocr_line = _cell(HEADER_BOX, "Week Begin", from_ocr=True)
    blank_words = [
        _cell(VALUE_BOX, "   ", from_ocr=False),
        _cell(HEADER_BOX, " ", from_ocr=True),
    ]

    tokens = _words_and_ocr_cells(
        word_cells=blank_words,
        cluster_cells=[native_line, ocr_line],
    )

    assert [(c.text, c.from_ocr) for c in tokens] == [
        ("5000", False),
        ("Week Begin", True),
    ]


class _FakePageBackend:
    def is_valid(self) -> bool:
        return True

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        return ""


class _RecordingPredictor:
    def __init__(self) -> None:
        self.received_tokens: list[list[dict]] = []

    def multi_table_predict(self, page_input, table_bboxes, do_matching=True):
        self.received_tokens.append(list(page_input["tokens"]))
        return [
            {
                "tf_responses": [],
                "predict_details": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "prediction": {
                        "rs_seq": ["fcel", "fcel", "nl", "fcel", "fcel", "nl"]
                    },
                },
            }
            for _ in table_bboxes
        ]


def test_predict_tables_keeps_native_lines_when_page_has_no_word_cells(
    tmp_path: Path,
) -> None:
    # Regression test for #5237: PyPdfiumDocumentBackend sets word_cells=[]
    # and has_words=False. When OCR leaves even one cell inside a table
    # cluster, predict_tables must pass all native table cells alongside the
    # OCR cell instead of dropping every native cell.
    page_w, page_h = 400.0, 300.0
    table_bbox = BoundingBox(l=20, t=20, r=360, b=200)
    full = BoundingBox(l=0, t=0, r=page_w, b=page_h)
    geometry = PdfPageGeometry(
        angle=0,
        boundary_type=PdfPageBoundaryType.CROP_BOX,
        rect=BoundingRectangle.from_bounding_box(full),
        art_bbox=full,
        bleed_bbox=full,
        crop_bbox=full,
        media_bbox=full,
        trim_bbox=full,
    )
    cluster_cells = [
        _cell(BoundingBox(l=30, t=30, r=120, b=45), "Function", from_ocr=False),
        _cell(BoundingBox(l=150, t=30, r=260, b=45), "Category", from_ocr=False),
        _cell(BoundingBox(l=30, t=60, r=120, b=75), "Govern", from_ocr=False),
        _cell(BoundingBox(l=150, t=60, r=260, b=75), "GV.OC", from_ocr=False),
        _cell(BoundingBox(l=290, t=60, r=330, b=75), "| =|", from_ocr=True),
    ]
    table_cluster = Cluster(
        id=1,
        label=DocItemLabel.TABLE,
        bbox=table_bbox,
        confidence=0.95,
        cells=cluster_cells,
    )
    page = Page(page_no=1, size=Size(width=page_w, height=page_h))
    page.parsed_page = SegmentedPdfPage(
        dimension=geometry,
        textline_cells=cluster_cells,
        char_cells=[],
        word_cells=[],
        has_chars=False,
        has_words=False,
        has_lines=True,
    )
    page.predictions.layout = LayoutPrediction(clusters=[table_cluster])
    page._backend = _FakePageBackend()  # type: ignore[assignment]
    page._image_cache[2.0] = Image.new(
        "RGB", (int(page_w * 2), int(page_h * 2)), "white"
    )

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    conv_res = ConversionResult(
        input=InputDocument(
            path_or_stream=pdf_path, format=InputFormat.PDF, backend=_DummyBackend
        )
    )

    predictor = _RecordingPredictor()
    model = object.__new__(TableStructureModel)
    model.enabled = True
    model.do_cell_matching = True
    model.scale = 2.0
    model.tm_config = {"predict": {"max_steps": 1024}}
    model.tf_predictor = predictor  # type: ignore[assignment]

    model.predict_tables(conv_res, [page])

    assert len(predictor.received_tokens) == 1
    assert [tok["text"] for tok in predictor.received_tokens[0]] == [
        "Function",
        "Category",
        "Govern",
        "GV.OC",
        "| =|",
    ]
    assert [tok["id"] for tok in predictor.received_tokens[0]] == [0, 1, 2, 3, 4]
