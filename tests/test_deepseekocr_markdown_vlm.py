"""Test DeepSeek OCR markdown parsing in VLM pipeline."""

from pathlib import Path

from docling_core.types.doc import DocItem, DocItemLabel, DoclingDocument

from docling.datamodel.base_models import InputFormat


def mock_parsing(content: str) -> DoclingDocument:
    # Create a mock conversion result with the DeepSeek OCR markdown as VLM response
    from PIL import Image as PILImage

    from docling.datamodel.base_models import Page, PagePredictions, VlmPrediction
    from docling.datamodel.document import ConversionResult, InputDocument
    from docling.pipeline.vlm_pipeline import VlmPipeline

    # Create a mock InputDocument class that bypasses custom __init__
    class MockInputDocument(InputDocument):
        def __init__(self, **data):
            super(InputDocument, self).__init__(**data)

    in_doc = MockInputDocument(
        file=Path("test.md"), format=InputFormat.MD, document_hash="0"
    )

    conv_res = ConversionResult(
        input=in_doc,
    )

    # Create a page with the DeepSeek OCR markdown as VLM response
    page = Page(page_no=1)
    page._image_cache[1.0] = PILImage.new("RGB", (800, 1000), color="white")
    page.predictions = PagePredictions()
    page.predictions.vlm_response = VlmPrediction(text=content)

    conv_res.pages = [page]

    # Call the utility function directly to parse the content
    from docling.utils.deepseekocr_utils import parse_deepseekocr_markdown

    # Parse the DeepSeek OCR markdown using the utility function
    doc = parse_deepseekocr_markdown(
        content=content,
        page_image=page.image,
        page_no=1,
        filename="test.md",
    )

    return doc


def test_parse_deepseekocr_markdown_example():
    # Test with first file
    test_dir = Path("tests/data/md_deepseek")
    for test_file in test_dir.glob("*.md"):
        # Read the annotated markdown content
        with open(test_file, encoding="utf-8") as f:
            annotated_content = f.read()

        doc = mock_parsing(annotated_content)

        doc.save_as_html(Path(f"test_{test_file.stem}.html"))
        doc.save_as_markdown(Path(f"test_{test_file.stem}.md"))
        doc.save_as_json(Path(f"test_{test_file.stem}.json"))


if __name__ == "__main__":
    test_parse_deepseekocr_markdown_example()
