from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc import BoundingBox
from PIL import Image

from docling.backend.pypdfium2_backend import (
    PyPdfiumDocumentBackend,
    PyPdfiumPageBackend,
)
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pytestmark = pytest.mark.ml_pdf_model


@pytest.fixture
def test_doc_path():
    return Path("./tests/data/pdf/2206.01062.pdf")


def _get_backend(pdf_doc, backend_options=None):
    in_doc = InputDocument(
        path_or_stream=pdf_doc,
        format=InputFormat.PDF,
        backend=PyPdfiumDocumentBackend,
        backend_options=backend_options,
        filename="test.pdf" if isinstance(pdf_doc, BytesIO) else None,
    )

    doc_backend = in_doc._backend
    return doc_backend


def _make_image_only_pdf(page_width=612, page_height=792) -> bytes:
    """Build a PDF whose single page is one full-bleed image, like a scan."""
    img_w, img_h = 306, 396
    pixels = bytes(
        (x * 7 + y * 13 + channel * 41) % 256
        for y in range(img_h)
        for x in range(img_w)
        for channel in range(3)
    )
    pil_image = Image.frombytes("RGB", (img_w, img_h), pixels)

    pdf = pdfium.PdfDocument.new()
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(pil_image))
    image.set_matrix(pdfium.PdfMatrix().scale(page_width, page_height))
    page = pdf.new_page(page_width, page_height)
    page.insert_obj(image)
    page.gen_content()
    buf = BytesIO()
    pdf.save(buf)
    pdf.close()
    return buf.getvalue()


def _direct_render(pdf_bytes_or_path, scale: float) -> Image.Image:
    """Reference render of page 0 at the given scale, without supersampling."""
    pdf = pdfium.PdfDocument(pdf_bytes_or_path)
    image = pdf[0].render(scale=scale, rotation=0, crop=(0, 0, 0, 0)).to_pil()
    pdf.close()
    return image


def test_get_text_from_rect_rotated():
    pdf_doc = Path("./tests/data_scanned/sample_with_rotation_mismatch.pdf")
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
    pdf_doc = Path("./tests/data/pdf/redp5110_sampled.pdf")

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


def test_raster_only_page_skips_supersample():
    # For image-only (scanned) pages, get_page_image must return the direct
    # render at the requested scale: the 1.5x supersample-downsample round-trip
    # resamples already-rasterized pixels and degrades OCR quality (issue #3587).
    pdf_bytes = _make_image_only_pdf()

    doc_backend = _get_backend(BytesIO(pdf_bytes))
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)
    assert page_backend._is_raster_only()

    image = page_backend.get_page_image(scale=2)
    reference = _direct_render(pdf_bytes, scale=2)

    assert image.size == reference.size
    assert image.mode == reference.mode
    assert image.tobytes() == reference.tobytes()


def test_vector_page_keeps_supersample(test_doc_path):
    # Pages with text/vector content keep the default 1.5x supersampling,
    # so the rendered image differs from a direct render at the same scale.
    doc_backend = _get_backend(test_doc_path)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)
    assert not page_backend._is_raster_only()

    image = page_backend.get_page_image(scale=2)
    reference = _direct_render(test_doc_path, scale=2)

    assert image.size == reference.size
    assert image.tobytes() != reference.tobytes()


def test_supersample_factor_option(test_doc_path):
    # supersample_factor=1.0 renders any page directly at the requested scale.
    doc_backend = _get_backend(
        test_doc_path, backend_options=PdfBackendOptions(supersample_factor=1.0)
    )
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(0)
    assert not page_backend._is_raster_only()

    image = page_backend.get_page_image(scale=2)
    reference = _direct_render(test_doc_path, scale=2)

    assert image.size == reference.size
    assert image.tobytes() == reference.tobytes()


def test_num_pages(test_doc_path):
    doc_backend = _get_backend(test_doc_path)
    assert doc_backend.page_count() == 9


def test_merge_row():
    pdf_doc = Path("./tests/data/pdf/multi_page.pdf")

    doc_backend = _get_backend(pdf_doc)
    page_backend: PyPdfiumPageBackend = doc_backend.load_page(4)
    cell = page_backend.get_text_cells()[0]

    assert (
        cell.text
        == "The journey of the word processor—from clunky typewriters to AI-powered platforms—"
    )
