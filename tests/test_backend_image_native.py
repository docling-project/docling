from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend, _ImagePageBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.document_extractor import DocumentExtractor


def _make_png_stream(
    width: int = 64, height: int = 48, color=(123, 45, 67)
) -> DocumentStream:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return DocumentStream(name="test.png", stream=buf)


def _make_multipage_tiff_stream(num_pages: int = 3, size=(32, 32)) -> DocumentStream:
    frames = [
        Image.new("RGB", size, (i * 10 % 255, i * 20 % 255, i * 30 % 255))
        for i in range(num_pages)
    ]
    buf = BytesIO()
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    buf.seek(0)
    return DocumentStream(name="test.tiff", stream=buf)


def test_docs_builder_uses_image_backend_for_image_stream():
    stream = _make_png_stream()
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[stream])
    # Provide format options mapping that includes IMAGE -> ImageFormatOption (which carries ImageDocumentBackend)
    format_options = {InputFormat.IMAGE: ImageFormatOption()}

    docs = list(conv_input.docs(format_options))
    assert len(docs) == 1
    in_doc = docs[0]
    assert in_doc.format == InputFormat.IMAGE
    assert isinstance(in_doc._backend, ImageDocumentBackend)
    assert in_doc.page_count == 1


def test_docs_builder_multipage_tiff_counts_frames():
    stream = _make_multipage_tiff_stream(num_pages=4)
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[stream])
    format_options = {InputFormat.IMAGE: ImageFormatOption()}

    in_doc = next(conv_input.docs(format_options))
    assert isinstance(in_doc._backend, ImageDocumentBackend)
    assert in_doc.page_count == 4


def test_converter_default_maps_image_to_image_backend():
    converter = DocumentConverter(allowed_formats=[InputFormat.IMAGE])
    backend_cls = converter.format_to_options[InputFormat.IMAGE].backend
    assert backend_cls is ImageDocumentBackend


def test_extractor_default_maps_image_to_image_backend():
    extractor = DocumentExtractor(allowed_formats=[InputFormat.IMAGE])
    backend_cls = extractor.extraction_format_to_options[InputFormat.IMAGE].backend
    assert backend_cls is ImageDocumentBackend


def _get_backend_from_stream(stream: DocumentStream):
    """Helper to create InputDocument with ImageDocumentBackend from a stream."""
    in_doc = InputDocument(
        path_or_stream=stream.stream,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename=stream.name,
    )
    return in_doc._backend


def test_num_pages_single():
    """Test page count for single-page image."""
    stream = _make_png_stream(width=100, height=80)
    doc_backend = _get_backend_from_stream(stream)
    assert doc_backend.page_count() == 1


def test_num_pages_multipage():
    """Test page count for multi-page TIFF."""
    stream = _make_multipage_tiff_stream(num_pages=5, size=(64, 64))
    doc_backend = _get_backend_from_stream(stream)
    assert doc_backend.page_count() == 5


def test_get_size():
    """Test getting page size."""
    width, height = 120, 90
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)
    size = page_backend.get_size()
    assert size.width == width
    assert size.height == height


def test_get_page_image_full():
    """Test getting full page image."""
    width, height = 100, 80
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)
    img = page_backend.get_page_image()
    assert img.width == width
    assert img.height == height


def test_get_page_image_scaled():
    """Test getting scaled page image."""
    width, height = 100, 80
    scale = 2.0
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)
    img = page_backend.get_page_image(scale=scale)
    assert img.width == round(width * scale)
    assert img.height == round(height * scale)


def test_crop_page_image():
    """Test cropping page image."""
    width, height = 200, 150
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)

    # Crop a region from the center
    cropbox = BoundingBox(l=50, t=30, r=150, b=120, coord_origin=CoordOrigin.TOPLEFT)
    img = page_backend.get_page_image(cropbox=cropbox)
    assert img.width == 100  # 150 - 50
    assert img.height == 90  # 120 - 30


def test_crop_page_image_scaled():
    """Test cropping and scaling page image."""
    width, height = 200, 150
    scale = 0.5
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)

    cropbox = BoundingBox(l=50, t=30, r=150, b=120, coord_origin=CoordOrigin.TOPLEFT)
    img = page_backend.get_page_image(scale=scale, cropbox=cropbox)
    assert img.width == round(100 * scale)  # cropped width * scale
    assert img.height == round(90 * scale)  # cropped height * scale


def test_get_bitmap_rects():
    """Test getting bitmap rects - should return full page rectangle."""
    width, height = 100, 80
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)

    rects = list(page_backend.get_bitmap_rects())
    assert len(rects) == 1
    bbox = rects[0]
    assert bbox.l == 0.0
    assert bbox.t == 0.0
    assert bbox.r == float(width)
    assert bbox.b == float(height)
    assert bbox.coord_origin == CoordOrigin.TOPLEFT


def test_get_bitmap_rects_scaled():
    """Test getting bitmap rects with scaling."""
    width, height = 100, 80
    scale = 2.0
    stream = _make_png_stream(width=width, height=height)
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)

    rects = list(page_backend.get_bitmap_rects(scale=scale))
    assert len(rects) == 1
    bbox = rects[0]
    assert bbox.l == 0.0
    assert bbox.t == 0.0
    assert bbox.r == float(width * scale)
    assert bbox.b == float(height * scale)
    assert bbox.coord_origin == CoordOrigin.TOPLEFT


def test_get_text_in_rect():
    """Test that get_text_in_rect returns empty string for images (no OCR)."""
    stream = _make_png_stream()
    doc_backend = _get_backend_from_stream(stream)
    page_backend: _ImagePageBackend = doc_backend.load_page(0)

    bbox = BoundingBox(l=10, t=10, r=50, b=50, coord_origin=CoordOrigin.TOPLEFT)
    text = page_backend.get_text_in_rect(bbox)
    assert text == ""


def test_multipage_access():
    """Test accessing different pages in multi-page image."""
    num_pages = 4
    stream = _make_multipage_tiff_stream(num_pages=num_pages, size=(64, 64))
    doc_backend = _get_backend_from_stream(stream)
    assert doc_backend.page_count() == num_pages

    # Access each page
    for i in range(num_pages):
        page_backend = doc_backend.load_page(i)
        assert page_backend.is_valid()
        size = page_backend.get_size()
        assert size.width == 64
        assert size.height == 64
