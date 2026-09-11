# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test methods in module docling.backend.json.docling_json_backend.py."""

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


def _doc_with_picture_uri(uri) -> bytes:
    doc = DoclingDocument(name="pic")
    doc.add_picture(
        image=ImageRef(
            mimetype="image/png", dpi=72, size=Size(width=1, height=1), uri=uri
        )
    )
    return doc.model_dump_json().encode()


def _load(json_bytes: bytes, options=None) -> DoclingDocument:
    stream = BytesIO(json_bytes)
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.JSON_DOCLING,
        backend=DoclingJSONBackend,
        filename="pic.json",
    )
    backend = DoclingJSONBackend(in_doc=in_doc, path_or_stream=stream, options=options)
    assert backend.is_valid()
    return backend.convert()


def test_local_image_refs_are_dropped_by_default(tmp_path):
    """A JSON document must not be able to make docling read a host file.

    ``ImageRef.uri`` accepts a bare path, and an embedded-image export or the
    picture enrichment stages would open it. Without ``enable_local_fetch``
    the backend strips such references; ``data:`` URIs are untouched.
    """
    secret = tmp_path / "secret.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(secret)

    for uri in (secret, secret.as_uri()):
        doc = _load(_doc_with_picture_uri(uri))
        assert doc.pictures[0].image is None
        md = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        assert "base64" not in md

    embedded = ImageRef.from_pil(Image.new("RGB", (4, 4)), dpi=72)
    doc = _load(_doc_with_picture_uri(embedded.uri))
    assert doc.pictures[0].image is not None
    assert str(doc.pictures[0].image.uri).startswith("data:")


def test_local_image_refs_kept_with_enable_local_fetch(tmp_path):
    """Opting in keeps the reference for trusted round-trips."""
    img = tmp_path / "img.png"
    Image.new("RGB", (4, 4)).save(img)

    doc = _load(
        _doc_with_picture_uri(img),
        options=DeclarativeBackendOptions(enable_local_fetch=True),
    )
    assert doc.pictures[0].image is not None
    assert doc.pictures[0].image.uri == img
