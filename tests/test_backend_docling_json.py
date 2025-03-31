"""Test methods in module docling.backend.json.docling_json_backend.py."""

from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError

from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument, InputDocument

GT_PATH: Path = Path("./tests/data/groundtruth/docling_v2/2206.01062.json")


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
