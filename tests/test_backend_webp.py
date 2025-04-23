from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.document_converter import DocumentConverter, ImageFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_webp_paths():
    # Define the directory you want to search
    directory = Path("./tests/data/webp/")

    # List all WEBP files in the directory and its subdirectories
    webp_files = sorted(directory.rglob("*.webp"))
    return webp_files


def get_converter():
    image_format_option = ImageFormatOption()
    image_format_option.pipeline_options.ocr_options = RapidOcrOptions()
    converter = DocumentConverter(
        format_options={InputFormat.IMAGE: image_format_option},
        allowed_formats=[InputFormat.IMAGE],
    )

    return converter


def test_e2e_webp_conversions():
    webp_paths = get_webp_paths()
    converter = get_converter()

    for webp_path in webp_paths:
        print(f"converting {webp_path}")

        gt_path = (
            webp_path.parent.parent / "groundtruth" / "docling_v2" / webp_path.name
        )

        conv_result: ConversionResult = converter.convert(webp_path)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt"), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            "document document"
        )
