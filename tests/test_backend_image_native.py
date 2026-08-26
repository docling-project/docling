# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, TiffImagePlugin

from docling.backend.image_backend import ImageDocumentBackend, _ImagePageBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import (
    InputDocument,
    _DocumentConversionInput,
    _DummyBackend,
    get_input_rejection_cause,
)
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.document_extractor import DocumentExtractor
from docling.exceptions import DocumentLoadError


def _make_png_stream(
    width: int = 64,
    height: int = 48,
    color=(123, 45, 67),
    dpi: tuple[float, float] | None = None,
) -> DocumentStream:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG", **({} if dpi is None else {"dpi": dpi}))
    buf.seek(0)
    return DocumentStream(name="test.png", stream=buf)


def _make_multipage_tiff_stream(
    num_pages: int = 3,
    size=(32, 32),
    dpi: tuple[float, float] | None = (72, 72),
) -> DocumentStream:
    frames = [
        Image.new("RGB", size, (i * 10 % 255, i * 20 % 255, i * 30 % 255))
        for i in range(num_pages)
    ]
    buf = BytesIO()
    frames[0].save(
        buf,
        format="TIFF",
        save_all=True,
        append_images=frames[1:],
        **({} if dpi is None else {"dpi": dpi}),
    )
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


def test_one_dpi_defaults_to_72_dpi():
    stream = _make_multipage_tiff_stream(num_pages=1, size=(64, 48), dpi=None)
    page_backend = _get_backend_from_stream(stream).load_page(0)

    assert page_backend.get_size().as_tuple() == (64, 48)


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


def test_300_dpi_tiff_uses_72_dpi_document_geometry():
    stream = _make_multipage_tiff_stream(
        num_pages=1,
        size=(2550, 3300),
        dpi=(300, 300),
    )
    page_backend = _get_backend_from_stream(stream).load_page(0)

    assert page_backend.get_size().as_tuple() == pytest.approx((612, 792))
    assert page_backend.get_page_image().size == (612, 792)
    assert page_backend.get_page_image(scale=3).size == (1836, 2376)
    assert page_backend.get_page_image(scale=300 / 72).size == (2550, 3300)
    assert next(page_backend.get_bitmap_rects()).as_tuple() == pytest.approx(
        (0, 0, 612, 792)
    )
    assert next(page_backend.get_bitmap_rects(scale=3)).as_tuple() == pytest.approx(
        (0, 0, 1836, 2376)
    )


def test_tiff_centimeter_resolution_is_converted_to_dpi():
    tiff_info = TiffImagePlugin.ImageFileDirectory_v2()
    tiff_info[TiffImagePlugin.X_RESOLUTION] = 100
    tiff_info[TiffImagePlugin.Y_RESOLUTION] = 50
    tiff_info[TiffImagePlugin.RESOLUTION_UNIT] = "cm"
    buf = BytesIO()
    Image.new("RGB", (254, 127)).save(buf, format="TIFF", tiffinfo=tiff_info)
    buf.seek(0)

    page_backend = _get_backend_from_stream(
        DocumentStream(name="test.tiff", stream=buf)
    ).load_page(0)

    assert page_backend.get_size().as_tuple() == pytest.approx((72, 72))


@pytest.mark.parametrize(
    ("image_format", "suffix"),
    [("PNG", "png"), ("JPEG", "jpg"), ("BMP", "bmp")],
)
def test_image_dpi_is_applied_independently_per_axis(image_format, suffix):
    buf = BytesIO()
    Image.new("RGB", (300, 150)).save(buf, format=image_format, dpi=(300, 150))
    buf.seek(0)
    page_backend = _get_backend_from_stream(
        DocumentStream(name=f"test.{suffix}", stream=buf)
    ).load_page(0)

    assert page_backend.get_size().as_tuple() == pytest.approx((72, 72), abs=0.01)
    assert page_backend.get_page_image(scale=2).size == (144, 144)


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


def test_crop_uses_logical_coordinates_for_high_dpi_image():
    image = Image.new("RGB", (300, 300), "red")
    image.paste("blue", (75, 75, 225, 225))
    buf = BytesIO()
    image.save(buf, format="PNG", dpi=(300, 300))
    buf.seek(0)
    page_backend = _get_backend_from_stream(
        DocumentStream(name="test.png", stream=buf)
    ).load_page(0)

    crop = page_backend.get_page_image(
        scale=2,
        cropbox=BoundingBox(
            l=18,
            t=18,
            r=54,
            b=54,
            coord_origin=CoordOrigin.TOPLEFT,
        ),
    )

    assert crop.size == (72, 72)
    assert crop.getpixel((36, 36)) == (0, 0, 255)


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


def test_invalid_explicit_dpi_rejects_document():
    stream = _make_png_stream(dpi=(0, 300))
    in_doc = InputDocument(
        path_or_stream=stream.stream,
        format=InputFormat.IMAGE,
        backend=ImageDocumentBackend,
        filename=stream.name,
    )

    assert in_doc.valid is False
    assert isinstance(get_input_rejection_cause(in_doc), DocumentLoadError)


def test_source_image_is_closed_after_backend_init(tmp_path, monkeypatch):
    image_path = tmp_path / "test.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(image_path)

    opened_images = []
    original_open = Image.open

    class TrackingImage:
        def __init__(self, image):
            self._image = image
            self.closed = False

        def __getattr__(self, attr):
            return getattr(self._image, attr)

        def close(self):
            self.closed = True
            return self._image.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def tracking_open(*args, **kwargs):
        tracked_image = TrackingImage(original_open(*args, **kwargs))
        opened_images.append(tracked_image)
        return tracked_image

    input_doc = InputDocument(
        path_or_stream=image_path,
        format=InputFormat.IMAGE,
        backend=_DummyBackend,
        filename=image_path.name,
    )

    monkeypatch.setattr("docling.backend.image_backend.Image.open", tracking_open)
    backend = ImageDocumentBackend(
        in_doc=input_doc,
        path_or_stream=image_path,
    )

    assert len(opened_images) == 1
    assert opened_images[0].closed is True
    backend.unload()


def test_unload_closes_cached_frames():
    stream = _make_multipage_tiff_stream(num_pages=3, size=(32, 32))
    doc_backend = _get_backend_from_stream(stream)

    tracked_closers = []
    for frame in doc_backend._frames:
        closer = MagicMock(wraps=frame.close)
        frame.close = closer
        tracked_closers.append(closer)

    doc_backend.unload()

    assert doc_backend._frames == []
    assert doc_backend._frame_dpi == []
    for closer in tracked_closers:
        closer.assert_called_once()
