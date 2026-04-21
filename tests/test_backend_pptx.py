from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_pptx_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/pptx/")

    # List all PPTX files in the directory and its subdirectories
    pptx_files = sorted(directory.rglob("*.pptx"))
    return pptx_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.PPTX])

    return converter


def test_e2e_pptx_conversions():
    pptx_paths = get_pptx_paths()
    converter = get_converter()

    for pptx_path in pptx_paths:
        # print(f"converting {pptx_path}")

        gt_path = (
            pptx_path.parent.parent / "groundtruth" / "docling_v2" / pptx_path.name
        )

        conv_result: ConversionResult = converter.convert(pptx_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", GENERATE), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )


def test_pptx_unrecognized_shape_type():
    """PPTX with a <p:sp> that has no geometry should not crash.

    python-pptx raises NotImplementedError from Shape.shape_type for shapes
    that aren't placeholders, autoshapes, textboxes, or freeforms. The
    backend should skip the unrecognized shape gracefully and still extract
    text from the rest of the presentation.

    Ref: https://github.com/docling-project/docling/issues/3308
    """
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/powerpoint_unrecognized_shape.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path)
    doc: DoclingDocument = conv_result.document

    pred_md = doc.export_to_markdown()

    # Normal slide content should still be extracted
    assert "Q3 Revenue Summary" in pred_md
    assert "Enterprise segment" in pred_md
    assert "Key Metrics" in pred_md
    assert "Next Steps" in pred_md


def test_pptx_page_range():
    converter = get_converter()
    pptx_path = Path("./tests/data/pptx/powerpoint_sample.pptx")

    conv_result: ConversionResult = converter.convert(pptx_path, page_range=(2, 2))

    assert conv_result.input.page_count == 3
    assert conv_result.document.num_pages() == 1
    assert list(conv_result.document.pages.keys()) == [2]

    pred_md = conv_result.document.export_to_markdown()
    assert "Second slide title" in pred_md
    assert "Test Table Slide" not in pred_md
    assert "List item4" not in pred_md
