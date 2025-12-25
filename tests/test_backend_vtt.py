from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def test_e2e_vtt_conversions():
    directory = Path("./tests/data/webvtt/")
    vtt_paths = sorted(directory.rglob("*.vtt"))
    converter = DocumentConverter(allowed_formats=[InputFormat.VTT])

    for vtt in vtt_paths:
        gt_path = vtt.parent.parent / "groundtruth" / "docling_v2" / vtt.name

        conv_result: ConversionResult = converter.convert(vtt)

        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(escape_html=False)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE)
