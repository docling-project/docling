# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

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


def _build_multi_object_pdf(n_paths: int = 40) -> bytes:
    """A minimal single-page PDF whose content stream draws ``n_paths`` stroked
    rectangles, i.e. ``n_paths`` separate PATH page objects, laid out in a grid
    inside the region (10, 10)-(250, 250) of a 300x300 MediaBox."""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R >>",
    ]
    parts = [b"1 0 0 RG 2 w"]
    for i in range(n_paths):
        x = 10 + (i % 8) * 30
        y = 10 + (i // 8) * 30
        parts.append(f"{x} {y} 20 20 re S".encode())
    stream = b"\n".join(parts)
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = b"%PDF-1.7\n"
    offsets = []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, o)
    xref_off = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objs) + 1,
        xref_off,
    )
    return out


def test_pdfium_object_index_built_once(tmp_path, monkeypatch):
    """The page-object walk must run once per page even when ``has_content_in`` is
    called repeatedly (once per layout cluster) -- otherwise the backend is
    O(clusters x objects). Count the underlying pypdfium2 object enumeration."""
    import pypdfium2 as pdfium

    pdf_path = tmp_path / "multi_object.pdf"
    pdf_path.write_bytes(_build_multi_object_pdf(40))

    original_get_objects = pdfium.PdfPage.get_objects
    calls = {"n": 0}

    def counting_get_objects(self, *args, **kwargs):
        calls["n"] += 1
        return original_get_objects(self, *args, **kwargs)

    monkeypatch.setattr(pdfium.PdfPage, "get_objects", counting_get_objects)

    doc_backend = _get_backend(pdf_path)
    try:
        page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)

        content = BoundingBox(
            l=10, t=50, r=250, b=250, coord_origin=CoordOrigin.TOPLEFT
        )
        blank = BoundingBox(
            l=260, t=260, r=299, b=299, coord_origin=CoordOrigin.TOPLEFT
        )

        # Emulate the per-cluster query pattern (2-3 calls per cluster, many clusters).
        for _ in range(5):
            assert page_backend.has_content_in(bbox=content) is True
            assert page_backend.has_content_in(bbox=blank) is False
        # These reuse the same index too.
        list(page_backend.get_bitmap_rects())
        page_backend.get_connected_shape_bounding_boxes()

        assert calls["n"] == 1, (
            f"page objects were enumerated {calls['n']} times; expected exactly 1"
        )
    finally:
        doc_backend.unload()
