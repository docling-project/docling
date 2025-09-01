"""
Test unit for document extraction functionality.
"""

import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.document_extractor import DocumentExtractor

IS_CI = bool(os.getenv("CI"))


class ExampleTemplate(BaseModel):
    bill_no: str = Field(
        examples=["A123", "5414"]
    )  # provide some examples, but not the actual value of the test sample
    total: float = Field(
        default=10.0, examples=[20.0]
    )  # provide a default value and some examples


@pytest.fixture
def extractor() -> DocumentExtractor:
    """Create a document converter instance for testing."""

    return DocumentExtractor(allowed_formats=[InputFormat.IMAGE, InputFormat.PDF])


@pytest.fixture
def test_file_path() -> Path:
    """Get the path to the test QR bill image."""
    return Path(__file__).parent / "data_scanned" / "qr_bill_example.jpg"
    # return Path("tests/data/pdf/code_and_formula.pdf")


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_string_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using string template."""
    str_templ = '{"bill_no": "string", "total": "number"}'

    result = extractor.extract(test_file_path, template=str_templ)

    print(result.pages)

    assert result.status is not None
    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_dict_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using dictionary template."""
    dict_templ = {
        "bill_no": "string",
        "total": "number",
    }

    result = extractor.extract(test_file_path, template=dict_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_pydantic_instance_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using pydantic instance template."""
    pydantic_instance_templ = ExampleTemplate(bill_no="4321")

    result = extractor.extract(test_file_path, template=pydantic_instance_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75


@pytest.mark.skipif(
    IS_CI, reason="Skipping test in CI because the dataset is too heavy."
)
def test_extraction_with_pydantic_class_template(
    extractor: DocumentExtractor, test_file_path: Path
) -> None:
    """Test extraction using pydantic class template."""
    pydantic_class_templ = ExampleTemplate

    result = extractor.extract(test_file_path, template=pydantic_class_templ)

    assert len(result.pages) == 1
    assert result.pages[0].extracted_data["bill_no"] == "3139"
    assert result.pages[0].extracted_data["total"] == 3949.75
