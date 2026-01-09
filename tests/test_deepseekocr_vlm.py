"""Test DeepSeek OCR markdown parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DoclingDocument, Size
from PIL import Image as PILImage

from docling.datamodel.base_models import (
    InputFormat,
    Page,
    PagePredictions,
    VlmPrediction,
)
from docling.datamodel.document import ConversionResult, InputDocument
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


if __name__ == "__main__":
    test_e2e_deepseekocr_parsing()
