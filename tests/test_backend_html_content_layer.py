from io import BytesIO

from docling_core.types.doc.document import ContentLayer

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument, InputDocument


def _convert_html(
    raw_html: bytes, options: HTMLBackendOptions | None = None
) -> DoclingDocument:
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw_html),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        filename="test.html",
    )
    backend = HTMLDocumentBackend(
        in_doc=in_doc,
        path_or_stream=BytesIO(raw_html),
        options=options,
    )
    return backend.convert()


def test_html_pre_heading_content_layer_defaults_to_furniture():
    raw_html = (
        b"<html><body><p>Intro before heading</p>"
        b"<h1>Main Heading</h1>"
        b"<p>Body content</p></body></html>"
    )

    doc = _convert_html(raw_html)

    assert doc.export_to_markdown() == "# Main Heading\n\nBody content"
    assert (
        doc.export_to_markdown(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        )
        == "Intro before heading\n\n# Main Heading\n\nBody content"
    )


def test_html_pre_heading_content_layer_can_be_body():
    raw_html = (
        b"<html><body><p>Intro before heading</p>"
        b"<h1>Main Heading</h1>"
        b"<p>Body content</p>"
        b"<footer><p>Footer content</p></footer></body></html>"
    )

    doc = _convert_html(
        raw_html,
        options=HTMLBackendOptions(pre_heading_content_layer=ContentLayer.BODY),
    )

    assert doc.export_to_markdown() == (
        "Intro before heading\n\n# Main Heading\n\nBody content"
    )
