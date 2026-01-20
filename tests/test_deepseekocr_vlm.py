"""Test DeepSeek OCR markdown parsing in VLM pipeline."""

import json
import os
import sys
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument, Size
from PIL import Image as PILImage

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import (
    InputFormat,
    Page,
    PagePredictions,
    VlmPrediction,
)
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_md_deepseek_paths():
    """Get all DeepSeek markdown test files."""
    directory = Path("./tests/data/md_deepseek/")
    md_files = sorted(directory.glob("*.md"))
    return md_files


def mock_parsing(content: str, filename: str) -> DoclingDocument:
    """Create a mock conversion result with the DeepSeek OCR markdown as VLM response."""

    # Create a page with the DeepSeek OCR markdown as VLM response
    page = Page(page_no=1)
    page._image_cache[1.0] = PILImage.new("RGB", (612, 792), color="white")
    page.predictions = PagePredictions()
    page.predictions.vlm_response = VlmPrediction(text=content)

    # Parse the DeepSeek OCR markdown using the utility function
    doc = parse_deepseekocr_markdown(
        content=content,
        original_page_size=Size(width=612, height=792),
        page_image=page.image,
        page_no=1,
        filename=filename,
    )

    return doc


def test_e2e_deepseekocr_parsing():
    """Test DeepSeek OCR markdown parsing for all test files."""
    md_paths = get_md_deepseek_paths()

    for md_path in md_paths:
        # Read the annotated markdown content
        with open(md_path, encoding="utf-8") as f:
            annotated_content = f.read()

        # Define groundtruth path
        gt_path = md_path.parent.parent / "groundtruth" / "docling_v2" / md_path.name

        # Parse the markdown using mock_parsing
        doc: DoclingDocument = mock_parsing(annotated_content, md_path.name)

        # Export to markdown
        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        # Export to indented text
        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        # Verify document structure
        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_e2e_deepseekocr_conversion():
    """Test DeepSeek OCR VLM conversion on a PDF file."""

    # Skip in CI or if ollama is not available
    if os.getenv("CI"):
        pytest.skip("Skipping in CI environment")

    # Check if ollama is available
    try:
        import requests

        response = requests.get("http://localhost:11434/v1/models", timeout=2)
        if response.status_code != 200:
            pytest.skip("Ollama is not available")
    except Exception:
        pytest.skip("Ollama is not available")

    # Setup the converter with DeepSeek OCR VLM
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.DEEPSEEKOCR_OLLAMA,
        enable_remote_services=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )

    # Convert the PDF
    pdf_path = Path("./tests/data/pdf/2206.01062.pdf")
    conv_result = converter.convert(pdf_path)

    # Load reference document
    ref_path = Path("./tests/data/groundtruth/docling_v2/deepseek_title.md.json")
    ref_doc = DoclingDocument.load_from_json(ref_path)

    # Validate conversion result
    doc = conv_result.document

    # Check number of pages
    assert len(doc.pages) == 9, f"Number of pages mismatch: {len(doc.pages)}"

    # Compare features of the first page (excluding bbox which can vary)
    # Check that we have similar structure
    assert len(doc.texts) > 0, "Document should have text elements"
    assert len(doc.pictures) > 0, "Document should have picture elements"

    # Check that the title is present
    title_texts = [t for t in doc.texts if t.label == "title"]
    assert len(title_texts) > 0, "Document should have a title"

    # Check that we have section headers
    section_headers = [t for t in doc.texts if t.label == "section_header"]
    assert len(section_headers) > 0, "Document should have section headers"

    # Compare with reference document structure (not exact bbox)
    ref_title_texts = [t for t in ref_doc.texts if t.label == "title"]
    assert len(title_texts) == len(ref_title_texts), (
        f"Title count mismatch: {len(title_texts)} vs {len(ref_title_texts)}"
    )

    print(
        f"✓ Conversion successful with {len(doc.texts)} text elements and {len(doc.pictures)} pictures"
    )


if __name__ == "__main__":
    test_e2e_deepseekocr_parsing()
    test_e2e_deepseekocr_conversion()
