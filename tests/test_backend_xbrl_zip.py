# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test module for the XBRL ZIP backend parser.

The data used in this test is in the public domain. It has been downloaded from the
U.S. Securities and Exchange Commission (SEC)'s Electronic Data Gathering, Analysis,
and Retrieval (EDGAR) system.

Two ZIP formats are exercised:

- ``grve_10q_htm.zip``: constructed from the files already committed under
  ``tests/data/xbrl/sources/`` (``grve_10q_htm.xml`` plus the ``grve-taxonomy/``
  files laid flat at the ZIP root). It exercises the traditional XBRL instance
  (``.xml``) path through the ZIP backend.

- ``ibm-20260630.zip``: the exact XBRL ZIP for IBM's Q2 2026 10-Q filing as
  distributed by SEC EDGAR (accession 0000051143-26-000078). It contains an
  inline XBRL (iXBRL) instance (``.htm``) together with the company taxonomy
  extension files. The standard us-gaap/dei taxonomy is not bundled, so concept
  relationships that require it are gracefully skipped in offline mode; numeric
  facts and the document title are still extracted.

  Source URL (public domain):
  https://www.sec.gov/Archives/edgar/data/51143/000005114326000078/0000051143-26-000078-xbrl.zip
"""

import tempfile
import zipfile
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
)
from docling.document_converter import (
    DocumentConverter,
    XBRLFormatOption,
    XBRLZipFormatOption,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

_SOURCES_DIR = Path(__file__).parent / "data" / "xbrl_zip" / "sources"
_GT_DIR = Path(__file__).parent / "data" / "xbrl_zip" / "groundtruth"


@pytest.fixture(scope="module")
def documents() -> list[tuple[Path, DoclingDocument]]:
    # XBRLZipFormatOption sets enable_local_fetch=True by default (the user's
    # explicit consent is expressed by choosing this format option).
    converter = DocumentConverter(
        allowed_formats=[InputFormat.ZIP_XBRL],
        format_options={InputFormat.ZIP_XBRL: XBRLZipFormatOption()},
    )
    results: list[tuple[Path, DoclingDocument]] = []
    for zip_file in sorted(_SOURCES_DIR.glob("*.zip")):
        conv: ConversionResult = converter.convert(zip_file)
        assert conv.document, f"Failed to convert {zip_file}"
        gt_path = _GT_DIR / zip_file.name
        results.append((gt_path, conv.document))
    return results


def test_e2e_xbrl_zip_conversions(
    documents: list[tuple[Path, DoclingDocument]],
) -> None:
    for gt_path, doc in documents:
        pred_md = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            f"export to md failed for {gt_path.name}"
        )

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            f"export to indented-text failed for {gt_path.name}"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            f"export to json failed for {gt_path.name}"
        )


def test_format_autodetection() -> None:
    """A ZIP containing XBRL data is detected as ZIP_XBRL without an explicit format."""
    conv = DocumentConverter()
    for zip_file in sorted(_SOURCES_DIR.glob("*.zip")):
        result = conv.convert(zip_file, raises_on_error=False)
        assert result.input.format == InputFormat.ZIP_XBRL, (
            f"Expected ZIP_XBRL, got {result.input.format} for {zip_file.name}"
        )
        assert result.document is not None, (
            f"Conversion failed for {zip_file.name}: {result.errors}"
        )


def test_grve_zip_equals_plain_xml() -> None:
    """ZIP packaging of a traditional XBRL instance yields the same document as
    loading the plain XML instance directly."""
    xml_conv = DocumentConverter(
        allowed_formats=[InputFormat.XML_XBRL],
        format_options={
            InputFormat.XML_XBRL: XBRLFormatOption(
                backend_options=XBRLBackendOptions(
                    enable_local_fetch=True,
                    taxonomy=Path(__file__).parent
                    / "data"
                    / "xbrl"
                    / "sources"
                    / "grve-taxonomy",
                )
            )
        },
    )
    r_xml = xml_conv.convert(
        Path(__file__).parent / "data" / "xbrl" / "sources" / "grve_10q_htm.xml"
    )

    r_zip = DocumentConverter(
        allowed_formats=[InputFormat.ZIP_XBRL],
        format_options={InputFormat.ZIP_XBRL: XBRLZipFormatOption()},
    ).convert(_SOURCES_DIR / "grve_10q_htm.zip")

    assert r_xml.document.export_to_markdown(compact_tables=True) == (
        r_zip.document.export_to_markdown(compact_tables=True)
    ), "ZIP and plain-XML paths produced different Markdown output"


def test_ibm_ixbrl_title_and_facts(
    documents: list[tuple[Path, DoclingDocument]],
) -> None:
    """The IBM iXBRL ZIP is detected as an inline XBRL document and the
    document title is always extracted from the DEI facts.

    Numeric fact extraction depends on whether Arelle has already loaded the
    standard us-gaap / dei taxonomies in the current process (e.g. via another
    test's taxonomy package).  The KV item count is therefore not asserted
    here; the e2e test covers the full output against stable groundtruth.
    """
    doc = next(d for gt, d in documents if gt.name == "ibm-20260630.zip")
    md = doc.export_to_markdown(compact_tables=True)
    assert "INTERNATIONAL BUSINESS MACHINES" in md
    assert "10-Q" in md


def test_ixbrl_htm_with_taxonomy_dir() -> None:
    """Scenario 4: a plain iXBRL ``.htm`` file loaded with a separate taxonomy dir.

    The IBM SEC XBRL ZIP contains ``ibm-20260630.htm`` (an iXBRL document) plus
    several company-taxonomy extension files.  This test extracts those files to a
    temporary directory and converts the ``.htm`` directly, providing the taxonomy
    directory via ``XBRLBackendOptions.taxonomy``.  It verifies that:

    * The ``.htm`` file is auto-detected as ``InputFormat.XML_XBRL``.
    * The conversion succeeds and the document title is extracted from DEI facts.
    """
    zip_path = _SOURCES_DIR / "ibm-20260630.zip"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        htm_path = tmp / "ibm-20260630.htm"

        # --- conversion with taxonomy dir + auto-detection check ---
        opts = XBRLBackendOptions(enable_local_fetch=True, taxonomy=tmp)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(backend_options=opts)
            },
        )
        result = converter.convert(htm_path)
        assert result.input.format == InputFormat.XML_XBRL, (
            f"Expected XML_XBRL auto-detection for iXBRL .htm, got {result.input.format!r}"
        )
        assert result.document is not None, f"Conversion failed: {result.errors}"
        md = result.document.export_to_markdown(compact_tables=True)
        assert "INTERNATIONAL BUSINESS MACHINES" in md
        assert "10-Q" in md
