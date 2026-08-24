from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin

from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf/sources/2206.01062.pdf")


def _get_backend(pdf_doc):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
    )

    doc_backend = in_doc._backend
    return doc_backend


def test_get_text_from_rect_rotated():
    pdf_doc = Path("./tests/data/ocr/sources/sample_with_rotation_mismatch.pdf")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )
    conv_res = doc_converter.convert(pdf_doc)

    assert "1972" in conv_res.document.export_to_markdown()


def test_text_cell_counts():
    pdf_doc = Path("./tests/data/pdf/sources/redp5110_sampled.pdf")

    doc_backend = _get_backend(pdf_doc)

    for page_index in range(doc_backend.page_count()):
        last_cell_count = None
        for i in range(10):
            page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)
            cells = list(page_backend.get_text_cells())

            if last_cell_count is None:
                last_cell_count = len(cells)

            if len(cells) != last_cell_count:
                assert False, (
                    "Loading page multiple times yielded non-identical text cell counts"
                )
            last_cell_count = len(cells)


def test_get_text_from_rect(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

    # Get the title text of the DocLayNet paper
    textpiece = page_backend.get_text_in_rect(
        bbox=BoundingBox(l=102, t=77, r=511, b=124)
    )
    ref = "DocLayNet: A Large Human-Annotated Dataset for\r\nDocument-Layout Analysis"

    assert textpiece.strip() == ref


def test_crop_page_image(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

    # Crop out "Figure 1" from the DocLayNet paper
    page_backend.get_page_image(
        scale=2, cropbox=BoundingBox(l=317, t=246, r=574, b=527)
    )
    # im.show()


def test_num_pages(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    assert doc_backend.page_count() == 9


def test_merge_row():
    pdf_doc = Path("./tests/data/pdf/sources/multi_page.pdf")

    doc_backend = _get_backend(pdf_doc)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(4)
    cell = page_backend.get_text_cells()[0]

    assert (
        cell.text
        == "The journey of the word processor—from clunky typewriters to AI-powered platforms—"
    )


def test_pdfium_shape_regions_approximate_docling_parse():
    """The bounds-based approximation must agree with the docling-parse decoder."""
    pdf_doc = Path("./tests/data/pdf/sources/2305.03393v1-pg9.pdf")

    pdfium_backend = _get_backend(pdf_doc)
    parse_in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=ThreadedDoclingParseDocumentBackend,
    )
    parse_backend = parse_in_doc._backend

    try:
        pdfium_regions = pdfium_backend.load_page(
            0
        ).get_connected_shape_bounding_boxes()
        parse_regions = next(
            iter(parse_backend.iter_pages())
        ).get_connected_shape_bounding_boxes()

        assert len(pdfium_regions) == len(parse_regions) == 1
        for pdfium_side, parse_side in zip(
            pdfium_regions[0].as_tuple(), parse_regions[0].as_tuple()
        ):
            # pypdfium2 reports painted bounds, docling-parse the geometric path, so the
            # two differ by roughly the stroke width.
            assert pdfium_side == pytest.approx(parse_side, abs=1.0)
    finally:
        pdfium_backend.unload()
        parse_backend.unload()


def test_pdfium_intersects_only_where_content_is():
    """`has_content_in` must discriminate between the ruled table and a blank margin."""
    pdf_doc = Path("./tests/data/pdf/sources/2305.03393v1-pg9.pdf")

    doc_backend = _get_backend(pdf_doc)
    try:
        page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

        blank_margin = BoundingBox(
            l=0, t=0, r=20, b=20, coord_origin=CoordOrigin.TOPLEFT
        )
        table = BoundingBox(
            l=150, t=350, r=460, b=460, coord_origin=CoordOrigin.TOPLEFT
        )

        assert page_backend.has_content_in(bbox=table) is True
        assert page_backend.has_content_in(bbox=blank_margin) is False
    finally:
        doc_backend.unload()


def test_pdfium_intersects_ignores_invisible_text():
    """Text drawn with rendering mode 3 paints nothing, so it must not count as content."""
    doc_backend = _get_backend(Path("./tests/data/pdf/invisible_text_layer.pdf"))
    try:
        page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

        visible_line = BoundingBox(
            l=60, t=70, r=400, b=110, coord_origin=CoordOrigin.TOPLEFT
        )
        invisible_line = BoundingBox(
            l=60, t=470, r=400, b=510, coord_origin=CoordOrigin.TOPLEFT
        )
        text_only = {"chars": True, "shapes": False, "bitmaps": False}

        assert page_backend.has_content_in(bbox=visible_line, **text_only) is True
        assert page_backend.has_content_in(bbox=invisible_line, **text_only) is False

        # The cell itself is still extracted; only the visibility query ignores it.
        assert "Invisible OCR text layer" in {
            cell.text for cell in page_backend.get_text_cells()
        }
    finally:
        doc_backend.unload()
