"""
Test unit for document extraction functionality.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


class ExampleTemplate(BaseModel):
    bill_no: str = Field(
        examples=["A123", "5414"]
    )  # provide some examples, but not the actual value of the test sample
    total: int = Field(
        default=10, examples=[20]
    )  # provide a default value and some examples


@pytest.fixture
def converter() -> DocumentConverter:
    """Create a document converter instance for testing."""
    return DocumentConverter(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])


@pytest.fixture
def test_file_path() -> Path:
    """Get the path to the test QR bill image."""
    return Path(__file__).parent / "data_scanned" / "qr_bill_example.jpg"


def test_extraction_with_string_template(
    converter: DocumentConverter, test_file_path: Path
) -> None:
    """Test extraction using string template."""
    str_templ = '{"bill_no": "string", "total": "integer"}'

    result = converter.extract(test_file_path, template=str_templ)

    assert result.status is not None
    assert result.data["bill_no"] == "3139"
    assert result.data["total"] == 3949.75


def test_extraction_with_dict_template(
    converter: DocumentConverter, test_file_path: Path
) -> None:
    """Test extraction using dictionary template."""
    dict_templ = {
        "bill_no": "string",
        "total": "integer",
    }

    result = converter.extract(test_file_path, template=dict_templ)

    assert result.data["bill_no"] == "3139"
    assert result.data["total"] == 3949.75


def test_extraction_with_pydantic_instance_template(
    converter: DocumentConverter, test_file_path: Path
) -> None:
    """Test extraction using pydantic instance template."""
    pydantic_instance_templ = ExampleTemplate(bill_no="4321")

    result = converter.extract(test_file_path, template=pydantic_instance_templ)

    assert result.data["bill_no"] == "3139"
    assert result.data["total"] == 3949.75


def test_extraction_with_pydantic_class_template(
    converter: DocumentConverter, test_file_path: Path
) -> None:
    """Test extraction using pydantic class template."""
    pydantic_class_templ = ExampleTemplate

    result = converter.extract(test_file_path, template=pydantic_class_templ)

    assert result.data["bill_no"] == "3139"
    assert result.data["total"] == 3949.75
