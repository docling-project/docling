# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test module for the XBRL backend parser.

The data used in this test is in the public domain. It has been downloaded from the
U.S. Securities and Exchange Commission (SEC)'s Electronic Data Gathering, Analysis,
and Retrieval (EDGAR) system.
"""

from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, XBRLFormatOption

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

_SOURCES_DIR = Path(__file__).parent / "data" / "xbrl" / "sources"
_GT_DIR = Path(__file__).parent / "data" / "xbrl" / "groundtruth"


@pytest.fixture(scope="module")
def xbrl_paths() -> list[tuple[Path, Path]]:
    xml_files = sorted(
        [
            item
            for item in _SOURCES_DIR.iterdir()
            if item.is_file() and item.suffix.lower() in {".xml", ".xbrl"}
        ],
        key=lambda p: p.name.lower(),
    )
    taxonomy_dirs = sorted(
        [
            item
            for item in _SOURCES_DIR.iterdir()
            if item.is_dir() and item.name.endswith("-taxonomy")
        ],
        key=lambda p: p.name.lower(),
    )
    assert len(xml_files) == len(taxonomy_dirs), (
        "Mismatch in XBRL instance reports and taxonomy directories"
    )
    return list(zip(xml_files, taxonomy_dirs))


@pytest.fixture(scope="module")
def documents(
    xbrl_paths: list[tuple[Path, Path]],
) -> list[tuple[Path, DoclingDocument]]:
    results: list[tuple[Path, DoclingDocument]] = []
    for report, taxonomy in xbrl_paths:
        gt_path = _GT_DIR / report.name
        # To download external taxonomy files into the web cache, replace with:
        # XBRLBackendOptions(enable_local_fetch=True, enable_remote_fetch=True, taxonomy=taxonomy)
        backend_options = XBRLBackendOptions(enable_local_fetch=True, taxonomy=taxonomy)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(backend_options=backend_options)
            },
        )
        conv_result: ConversionResult = converter.convert(report)
        doc: DoclingDocument = conv_result.document
        assert doc, f"Failed to convert document from file {report}"
        results.append((gt_path, doc))
    return results


def test_e2e_xbrl_conversions(documents):
    for gt_path, doc in documents:
        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_xbrl_divide_unit_keeps_denominator(documents):
    """A fact measured in a divide unit must report both of its measures.

    XBRL 2.1 (sections 4.8.3-4.8.4) defines a ``<divide>`` unit as the ratio of its
    numerator and denominator measures. In ``grve_10q_htm.xml`` the per-share facts
    use the unit ``USDPShares`` (``iso4217:USD`` divided by ``shares``), which must
    not be reported as a plain ``USD`` amount.
    """
    name = "grve_10q_htm.xml"
    doc = next(item[1] for item in documents if item[0].name == name)

    unit_texts: dict[str, set[str]] = {}
    for kv_item in doc.key_value_items:
        cells = {cell.cell_id: cell for cell in kv_item.graph.cells}
        for link in kv_item.graph.links:
            target = cells[link.target_cell_id]
            if target.orig == "unit":
                concept = cells[link.source_cell_id].orig
                unit_texts.setdefault(concept, set()).add(target.text)

    for concept in (
        "us-gaap:EarningsPerShareDiluted",
        "us-gaap:CommonStockParOrStatedValuePerShare",
        "us-gaap:PreferredStockParOrStatedValuePerShare",
    ):
        assert unit_texts.get(concept) == {"currency: USD / shares"}, concept
    assert unit_texts.get("us-gaap:Assets") == {"currency: USD"}
