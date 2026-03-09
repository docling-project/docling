"""Test module for the XBRL backend parser.

The data used in this test is in the public domain. It has been downloaded from the
U.S. Securities and Exchange Commission (SEC)'s Electronic Data Gathering, Analysis,
and Retrieval (EDGAR) system.
"""

import os
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, XBRLFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


@pytest.fixture(scope="module")
def xbrl_paths() -> list[tuple[Path, Path]]:
    directory = Path(os.path.dirname(__file__) + "/data/xbrl/")
    xml_files = sorted(
        [
            item
            for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in {".xml", ".xbrl"}
        ],
        key=lambda p: p.name.lower(),
    )
    taxonomy_dir = sorted(
        [
            item
            for item in directory.iterdir()
            if item.is_dir() and str(item).endswith("-taxonomy")
        ],
        key=lambda p: p.name.lower(),
    )
    assert len(xml_files) == len(taxonomy_dir), (
        "Mismatch in XBRL instance reports and taxonomy directories"
    )

    return zip(xml_files, taxonomy_dir)


def test_e2e_xbrl_conversions(xbrl_paths, use_stream=False):
    for report, taxonomy in xbrl_paths:
        gt_path = report.parent.parent / "groundtruth" / "docling_v2" / report.name

        backend_options = XBRLBackendOptions(enable_local_fetch=True, taxonomy=taxonomy)
        # set enable_remote_fetch to download the necessary external taxonomy files in web cache
        # backend_options = XBRLBackendOptions(enable_local_fetch=True, enable_remote_fetch=True, taxonomy=taxonomy)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(backend_options=backend_options)
            },
        )

        if use_stream:
            buf = BytesIO(report.open("rb").read())
            stream = DocumentStream(name=report.name, stream=buf)
            conv_result: ConversionResult = converter.convert(stream)
        else:
            conv_result: ConversionResult = converter.convert(report)
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
