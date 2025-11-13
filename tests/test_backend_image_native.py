from io import BytesIO

from PIL import Image

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
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
