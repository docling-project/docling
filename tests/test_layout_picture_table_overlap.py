"""Integration test for the PICTURE/TABLE duplicate-emission fix (issue #3522).

The layout model can emit the same region as both a PICTURE and a TABLE cluster.
`LayoutPostprocessor._handle_cross_type_overlaps` drops a PICTURE that nearly
coincides with a TABLE, keeping the structured TABLE.

Fixture: a single page (from a public UNODC "Global Study on Legal Aid" document)
where a bordered table is mislabeled as a picture covering the same region. On this
page the layout model emits one PictureItem coinciding with the TableItem (IoU 1.0);
after the fix no PictureItem coincides with a TableItem.

We assert the invariant (no picture coincides with a table) rather than a raw
PictureItem count: removing a coinciding picture can change how many other picture
clusters survive de-overlap, so a count assertion would be brittle.
"""

from pathlib import Path

import pytest
from docling_core.types.doc.document import PictureItem, TableItem

from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

pytestmark = pytest.mark.ml_pdf_model

PDF_PATH = Path("tests/data/pdf/sources/table_mislabeled_as_picture.pdf")
IOU_THRESHOLD = 0.9


def _bbox(item):
    return item.prov[0].bbox if item.prov else None


def _iou(a, b) -> float:
    """IoU of two BoundingBoxes, coordinate-origin agnostic."""
    a_lo, a_hi = sorted((a.t, a.b))
    b_lo, b_hi = sorted((b.t, b.b))
    inter_w = max(0.0, min(a.r, b.r) - max(a.l, b.l))
    inter_h = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.r - a.l) * max(0.0, a_hi - a_lo)
    area_b = max(0.0, b.r - b.l) * max(0.0, b_hi - b_lo)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _get_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def test_picture_not_emitted_over_table():
    """No PictureItem should coincide with a TableItem (issue #3522)."""
    doc = _get_converter().convert(PDF_PATH).document

    pictures = [
        it for it, _ in doc.iterate_items() if isinstance(it, PictureItem) and it.prov
    ]
    tables = [
        it for it, _ in doc.iterate_items() if isinstance(it, TableItem) and it.prov
    ]

    assert tables, "fixture should contain at least one table"

    overlaps = [
        (p, t, _iou(_bbox(p), _bbox(t)))
        for p in pictures
        for t in tables
        if _iou(_bbox(p), _bbox(t)) >= IOU_THRESHOLD
    ]

    assert not overlaps, (
        "a PictureItem coincides with a TableItem (table emitted as both); "
        f"offending IoUs: {[round(o[2], 3) for o in overlaps]}"
    )
