import os
from io import BytesIO
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_pubmed_paths():
    directory = Path(os.path.dirname(__file__) + "/data/pubmed/")
    xml_files = sorted(directory.rglob("*.xml"))
    return xml_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_JATS])
    return converter


def test_e2e_pubmed_conversions(use_stream=False):
    pubmed_paths = get_pubmed_paths()
    converter = get_converter()

    for pubmed_path in pubmed_paths:
        gt_path = (
            pubmed_path.parent.parent / "groundtruth" / "docling_v2" / pubmed_path.name
        )
        if use_stream:
            buf = BytesIO(pubmed_path.open("rb").read())
            stream = DocumentStream(name=pubmed_path.name, stream=buf)
            conv_result: ConversionResult = converter.convert(stream)
        else:
            conv_result: ConversionResult = converter.convert(pubmed_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md"), "export to md"

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt"), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_e2e_pubmed_conversions_stream():
    test_e2e_pubmed_conversions(use_stream=True)


def test_e2e_pubmed_conversions_no_stream():
    test_e2e_pubmed_conversions(use_stream=False)
