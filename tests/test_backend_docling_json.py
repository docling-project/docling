# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test methods in module docling.backend.json.docling_json_backend.py."""

import base64
import re
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import ImageRef, ImageRefMode, Size
from PIL import Image
from pydantic import ValidationError

from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.datamodel.backend_options import DeclarativeBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument, InputDocument
from docling.document_converter import DoclingJSONFormatOption, DocumentConverter

GT_PATH: Path = Path("./tests/data/pdf/groundtruth/2206.01062.json")
pytestmark = pytest.mark.cross_platform


def test_convert_valid_docling_json():
    """Test ingestion of valid Docling JSON."""
    cls = DoclingJSONBackend
    path_or_stream = GT_PATH
    in_doc = InputDocument(
        path_or_stream=path_or_stream,
        format=InputFormat.JSON_DOCLING,
        backend=cls,
    )
    backend = cls(
        in_doc=in_doc,
        path_or_stream=path_or_stream,
    )
    assert backend.is_valid()

    act_doc = backend.convert()
    act_data = act_doc.export_to_dict()

    exp_doc = DoclingDocument.load_from_json(GT_PATH)
    exp_data = exp_doc.export_to_dict()

    assert act_data == exp_data


def test_invalid_docling_json():
    """Test ingestion of invalid Docling JSON."""
    cls = DoclingJSONBackend
    path_or_stream = BytesIO(b"{}")
    in_doc = InputDocument(
        path_or_stream=path_or_stream,
        format=InputFormat.JSON_DOCLING,
        backend=cls,
        filename="foo",
    )
    backend = cls(
        in_doc=in_doc,
        path_or_stream=path_or_stream,
    )

    assert not backend.is_valid()

    with pytest.raises(ValidationError):
        backend.convert()


def test_utf8_bom_does_not_fail_the_load(tmp_path):
    """A leading UTF-8 BOM must not reach model_validate_json.

    It is rejected as an unexpected character, so the document failed to load
    outright. The path branch decodes and the stream branch hands raw bytes
    over, so both are covered.
    """
    json_bytes = b"\xef\xbb\xbf" + GT_PATH.read_bytes()
    exp_data = DoclingDocument.load_from_json(GT_PATH).export_to_dict()

    json_file = tmp_path / "bom.json"
    json_file.write_bytes(json_bytes)

    for path_or_stream in (json_file, BytesIO(json_bytes)):
        in_doc = InputDocument(
            path_or_stream=path_or_stream,
            format=InputFormat.JSON_DOCLING,
            backend=DoclingJSONBackend,
            filename="bom.json",
        )
        backend = DoclingJSONBackend(in_doc=in_doc, path_or_stream=path_or_stream)

        assert backend.is_valid()
        assert backend.convert().export_to_dict() == exp_data


def _write_image(path: Path) -> bytes:
    """Write a small PNG with distinctive pixels and return those pixels."""
    img = Image.new("RGB", (7, 5))
    img.putdata([(i * 37 % 256, i * 91 % 256, i * 13 % 256) for i in range(35)])
    img.save(path)
    return img.tobytes()


def _embedded_pixels(exported: str) -> list[bytes]:
    """Decode every base64 PNG embedded in an export to its RGB pixels."""
    return [
        Image.open(BytesIO(base64.b64decode(data))).convert("RGB").tobytes()
        for data in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", exported)
    ]


def _image_ref(uri: str | Path) -> ImageRef:
    return ImageRef(mimetype="image/png", dpi=72, size=Size(width=7, height=5), uri=uri)


def _convert(
    tmp_path: Path, doc: DoclingDocument, converter: DocumentConverter
) -> DoclingDocument:
    json_file = tmp_path / "input.json"
    json_file.write_text(doc.model_dump_json(), encoding="utf-8")
    return converter.convert(json_file).document


def _embedded_exports(tmp_path: Path, doc: DoclingDocument) -> list[str]:
    out_json = tmp_path / "out.json"
    doc.save_as_json(out_json, image_mode=ImageRefMode.EMBEDDED)
    return [
        doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED),
        doc.export_to_html(image_mode=ImageRefMode.EMBEDDED),
        out_json.read_text(encoding="utf-8"),
    ]


@pytest.mark.parametrize("uri_kind", ["absolute", "relative", "file_uri"])
def test_local_picture_uri_is_not_embedded_by_default(tmp_path, monkeypatch, uri_kind):
    image_file = tmp_path / "local.png"
    pixels = _write_image(image_file)
    monkeypatch.chdir(tmp_path)
    uri: str | Path = {
        "absolute": image_file,
        "relative": Path("local.png"),
        "file_uri": image_file.as_uri(),
    }[uri_kind]
    doc = DoclingDocument(name="pic")
    doc.add_picture(image=_image_ref(uri))

    result = _convert(tmp_path, doc, DocumentConverter())

    assert result.pictures[0].image is None
    for exported in _embedded_exports(tmp_path, result):
        assert pixels not in _embedded_pixels(exported)


def test_local_page_image_is_not_embedded_by_default(tmp_path):
    image_file = tmp_path / "page.png"
    pixels = _write_image(image_file)
    doc = DoclingDocument(name="page")
    doc.add_page(page_no=1, size=Size(width=7, height=5), image=_image_ref(image_file))

    result = _convert(tmp_path, doc, DocumentConverter())

    assert result.pages[1].image is None
    for exported in _embedded_exports(tmp_path, result):
        assert pixels not in _embedded_pixels(exported)


def test_local_image_uri_is_embedded_with_enable_local_fetch(tmp_path):
    image_file = tmp_path / "local.png"
    pixels = _write_image(image_file)
    doc = DoclingDocument(name="pic")
    doc.add_picture(image=_image_ref(image_file))
    doc.add_page(page_no=1, size=Size(width=7, height=5), image=_image_ref(image_file))
    converter = DocumentConverter(
        format_options={
            InputFormat.JSON_DOCLING: DoclingJSONFormatOption(
                backend_options=DeclarativeBackendOptions(enable_local_fetch=True)
            )
        }
    )

    result = _convert(tmp_path, doc, converter)

    assert result.pictures[0].image is not None
    assert result.pages[1].image is not None
    assert pixels in _embedded_pixels(
        result.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    )


def test_data_and_remote_image_uris_are_kept(tmp_path):
    image_file = tmp_path / "local.png"
    pixels = _write_image(image_file)
    with Image.open(image_file) as img:
        embedded = ImageRef.from_pil(img, dpi=72)
    remote = "https://example.com/figure.png"
    doc = DoclingDocument(name="pic")
    doc.add_picture(image=embedded)
    doc.add_picture(image=_image_ref(remote))

    result = _convert(tmp_path, doc, DocumentConverter())

    assert result.pictures[0].image == embedded
    assert result.pictures[1].image is not None
    assert str(result.pictures[1].image.uri) == remote
    assert pixels in _embedded_pixels(
        result.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    )
