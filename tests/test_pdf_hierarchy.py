from io import BytesIO
from pathlib import Path

from docling.document_converter import DocumentConverter

results_path = Path(__file__).parent / "data" / "groundtruth" / "docling_v2"
sample_path = Path(__file__).parent / "data" / "pdf"


def compare(res_text, fn):
    p = results_path / fn
    if p.exists():
        assert res_text.strip() == p.read_text().strip()
    else:
        p.write_text(res_text)


def test_result_postprocessor_textpdf_no_bookmarks():
    source = (
        sample_path / "sample_document_hierarchical_no_bookmarks.pdf"
    )  # document per local path or URL
    converter = DocumentConverter()
    result = converter.convert(source)

    compare(
        result.document.export_to_markdown(),
        "sample_document_hierarchical_no_bookmarks.md",
    )

    allowed_headers = [
        "Some kind of text document",
        "1. Introduction",
        "1.1 Background",
        "1.2 Purpose",
        "2. Main Content",
        "2.1 Section One",
        "2.1.1 Subsection",
        "2.1.2 Another Subsection",
        "2.2 Section Two",
        "3. Conclusion",
    ]

    for item_ref in result.document.body.children:
        item = item_ref.resolve(result.document)
        assert item.text in allowed_headers
