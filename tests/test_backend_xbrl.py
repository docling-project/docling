# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test module for the XBRL backend parser.

The data used in this test is in the public domain. It has been downloaded from the
U.S. Securities and Exchange Commission (SEC)'s Electronic Data Gathering, Analysis,
and Retrieval (EDGAR) system.
"""

import os
import shutil
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument

from docling.backend.xml.xbrl_backend import XBRLDocumentBackend
from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.document_converter import DocumentConverter, XBRLFormatOption
from docling.exceptions import DocumentLoadError, OperationNotAllowed

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


@pytest.fixture(scope="module")
def xbrl_paths() -> list[tuple[Path, Path]]:
    directory = Path(os.path.dirname(__file__) + "/data/xbrl/sources/")
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

    return list(zip(xml_files, taxonomy_dir))


@pytest.fixture(scope="module")
def documents(xbrl_paths) -> list[tuple[Path, DoclingDocument]]:
    documents: list[tuple[Path, DoclingDocument]] = []
    for report, taxonomy in xbrl_paths:
        gt_path = report.parent.parent / "groundtruth" / report.name

        backend_options = XBRLBackendOptions(enable_local_fetch=True, taxonomy=taxonomy)
        # set enable_remote_fetch to download the necessary external taxonomy files in web cache
        # backend_options = XBRLBackendOptions(enable_local_fetch=True, enable_remote_fetch=True, taxonomy=taxonomy)
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(backend_options=backend_options)
            },
        )
        conv_result: ConversionResult = converter.convert(report)
        doc: DoclingDocument = conv_result.document

        assert doc, f"Failed to convert document from file {report}"
        documents.append((gt_path, doc))

    return documents


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


def test_ixbrl_report_converts_offline():
    """An iXBRL report converts offline with a taxonomy package.

    IBM's 10-Q has facts with typed dimensions, which have no member QName.
    """
    ixbrl_dir = Path(__file__).parent / "data" / "xbrl" / "ixbrl"
    converter = DocumentConverter(
        allowed_formats=[InputFormat.XML_XBRL],
        format_options={
            InputFormat.XML_XBRL: XBRLFormatOption(
                backend_options=XBRLBackendOptions(
                    enable_local_fetch=True, taxonomy=ixbrl_dir / "ibm-taxonomy"
                )
            )
        },
    )
    doc = converter.convert(ixbrl_dir / "ibm-20260930.htm").document

    assert (
        doc.texts[0].text
        == "10-Q INTERNATIONAL BUSINESS MACHINES CORPORATION 2026-09-30"
    )
    assert len(doc.tables) == 166

    graph = doc.key_value_items[0].graph
    cells = {cell.cell_id: cell for cell in graph.cells}
    facts: dict[str, list[list[str]]] = {}
    for cell in graph.cells:
        if cell.text == "nonFraction":
            targets = [
                cells[link.target_cell_id].text
                for link in graph.links
                if link.source_cell_id == cell.cell_id
            ]
            facts.setdefault(cell.orig, []).append(targets)
    assert facts["us-gaap:RevenueRemainingPerformanceObligation"] == [
        [
            "value: 69000000000",
            "period: 2026-09-30",
            "currency: USD",
            "decimals: -9",
        ]
    ]


def test_ixbrl_warns_when_taxonomy_is_not_found(tmp_path, caplog, no_taxonomy_packages):
    """A package without the extension taxonomy leaves the facts unresolved.

    The package maps every ``https://`` URL to its own files. The extension
    taxonomy of the report is not among them, and it is not fetched remotely.
    """
    ixbrl_dir = Path(__file__).parent / "data" / "xbrl" / "ixbrl"
    taxonomy = tmp_path / "taxonomy"
    taxonomy.mkdir()
    with (
        zipfile.ZipFile(ixbrl_dir / "ibm-taxonomy" / "taxonomy_package.zip") as src,
        zipfile.ZipFile(taxonomy / "taxonomy_package.zip", "w") as dst,
    ):
        for item in src.infolist():
            if "/www.sec.gov/" not in item.filename:
                dst.writestr(item, src.read(item))
    converter = DocumentConverter(
        allowed_formats=[InputFormat.XML_XBRL],
        format_options={
            InputFormat.XML_XBRL: XBRLFormatOption(
                backend_options=XBRLBackendOptions(
                    enable_local_fetch=True, taxonomy=taxonomy
                )
            )
        },
    )

    with caplog.at_level("WARNING", logger="docling.backend.xml.xbrl_backend"):
        doc = converter.convert(ixbrl_dir / "ibm-20260930.htm").document

    assert len(doc.tables) == 0
    assert "could not be linked to a concept" in caplog.text


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


def test_xbrl_source_uri_rejects_credentials():
    with pytest.raises(ValueError):
        XBRLBackendOptions(source_uri="https://user:pass@www.sec.gov/report.htm")


def test_xbrl_format_option_does_not_keep_url_with_credentials():
    option = XBRLFormatOption(
        backend_options=XBRLBackendOptions(enable_remote_fetch=True)
    )

    options = option.backend_options_for_input(
        "https://user:pass@www.sec.gov/Archives/report.htm"
    )

    assert options is not None
    assert options.source_uri is None


def test_xbrl_source_uri_does_not_bypass_fetch_policy(
    no_taxonomy_packages, local_server
):
    """A `source_uri` on a non-public host must not be fetched from."""
    base_url, requests = local_server
    stream = _instance_with_schema_ref("relative.xsd")

    message = _load_refused(
        stream,
        XBRLBackendOptions(
            enable_remote_fetch=True, source_uri=f"{base_url}/dir/report.xbrl"
        ),
    )

    assert f"{base_url}/dir/relative.xsd" in message
    assert "restricted IP address" in message
    assert requests == []


def test_xbrl_source_uri_resolves_relative_references(
    no_taxonomy_packages, local_server
):
    """Missing relative references stay on the origin of `source_uri`."""
    base_url, requests = local_server

    message = _load_refused(
        _instance_with_schema_ref(
            "relative.xsd",
            "/absolute.xsd",
            "../parent.xsd",
            "a&amp;b.xsd",
            "a&quot; onload=&quot;x.xsd",
        ),
        XBRLBackendOptions(
            enable_remote_fetch=True, source_uri=f"{base_url}/dir/report.xbrl"
        ),
    )

    for url in (
        f"{base_url}/dir/relative.xsd",
        f"{base_url}/absolute.xsd",
        f"{base_url}/parent.xsd",
        f"{base_url}/dir/a&b.xsd",
        f"{base_url}/dir/a%22%20onload=%22x.xsd",
    ):
        assert f"{url} (Access to restricted IP address" in message
    assert requests == []


def test_xbrl_source_uri_does_not_rewrite_other_origins(
    no_taxonomy_packages, local_server
):
    base_url, _ = local_server

    message = _load_refused(
        _instance_with_schema_ref("//other.example/x.xsd"),
        XBRLBackendOptions(
            enable_remote_fetch=True, source_uri=f"{base_url}/dir/report.xbrl"
        ),
    )

    assert "fetch options: //other.example/x.xsd (" in message


def test_xbrl_source_uri_keeps_local_taxonomy_files(
    no_taxonomy_packages, local_server, tmp_path
):
    """A reference found in the `taxonomy` folder is loaded from there."""
    base_url, requests = local_server
    (tmp_path / "local.xsd").write_text(
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>'
    )
    stream = _instance_with_schema_ref("local.xsd")
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.XML_XBRL,
        backend=XBRLDocumentBackend,
        filename="instance.xbrl",
    )
    options = XBRLBackendOptions(
        enable_local_fetch=True,
        enable_remote_fetch=True,
        source_uri=f"{base_url}/dir/report.xbrl",
        taxonomy=tmp_path,
    )

    backend = XBRLDocumentBackend(in_doc, stream, options)
    backend.unload()

    assert requests == []


def test_xbrl_conversion_does_not_keep_taxonomy_packages(
    no_taxonomy_packages, tmp_path
):
    """The taxonomy package of one conversion does not affect the next one."""
    sources = Path(__file__).parent / "data" / "xbrl" / "sources"
    extension_only = tmp_path / "mlac-taxonomy"
    shutil.copytree(
        sources / "mlac-taxonomy",
        extension_only,
        ignore=shutil.ignore_patterns("*.zip"),
    )

    def count_cells(report: str, taxonomy: Path) -> int:
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(
                    backend_options=XBRLBackendOptions(
                        enable_local_fetch=True, taxonomy=taxonomy
                    )
                )
            },
        )
        doc = converter.convert(sources / report).document
        return len(doc.key_value_items[0].graph.cells)

    expected = count_cells("mlac-20251231.xml", extension_only)
    count_cells("grve_10q_htm.xml", sources / "grve-taxonomy")

    assert expected > 0
    assert count_cells("mlac-20251231.xml", extension_only) == expected


_SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/1/2/report-20260930.htm"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_SOURCE_URL, _SOURCE_URL),
        (Path("report.htm"), None),
        ("local/report.htm", None),
        (DocumentStream(name="report.htm", stream=BytesIO(b"")), None),
    ],
)
def test_xbrl_format_option_keeps_remote_source_uri(source, expected):
    option = XBRLFormatOption(
        backend_options=XBRLBackendOptions(enable_remote_fetch=True)
    )

    options = option.backend_options_for_input(source)

    assert options is not None
    assert (str(options.source_uri) if options.source_uri else None) == expected
    assert options.enable_remote_fetch


def _instance_with_schema_ref(*hrefs: str) -> BytesIO:
    refs = "".join(
        f'<link:schemaRef xlink:type="simple" xlink:href="{href}"/>' for href in hrefs
    )
    return BytesIO(
        (
            '<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"'
            ' xmlns:link="http://www.xbrl.org/2003/linkbase"'
            ' xmlns:xlink="http://www.w3.org/1999/xlink">'
            f"{refs}</xbrli:xbrl>"
        ).encode()
    )


def _load_refused(stream: BytesIO, options: XBRLBackendOptions) -> str:
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.XML_XBRL,
        backend=XBRLDocumentBackend,
        filename="instance.xbrl",
    )
    with pytest.raises(DocumentLoadError) as exc_info:
        XBRLDocumentBackend(in_doc, stream, options)
    cause = exc_info.value.__cause__
    assert isinstance(cause, OperationNotAllowed)
    return str(cause)


@pytest.fixture
def no_taxonomy_packages() -> None:
    """Drop the taxonomy packages that earlier loads registered.

    Arelle keeps the packages and their URL mappings in a process-wide registry.
    The packages of the test data map every http(s) URL to their own files.
    """
    from arelle import PackageManager  # type: ignore

    PackageManager.reset()


@pytest.fixture
def local_server() -> Iterator[tuple[str, list[str]]]:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.path)
            body = b'<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>'
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()


def test_xbrl_remote_fetch_refuses_non_public_address(
    no_taxonomy_packages, local_server
):
    base_url, requests = local_server
    url = f"{base_url}/x.xsd"

    message = _load_refused(
        _instance_with_schema_ref(url),
        XBRLBackendOptions(enable_remote_fetch=True),
    )

    assert url in message
    assert "restricted IP address" in message
    assert requests == []


def test_xbrl_file_reference_needs_local_fetch(no_taxonomy_packages, tmp_path):
    schema = tmp_path / "local.xsd"
    schema.write_text('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"/>')

    message = _load_refused(
        _instance_with_schema_ref(schema.as_uri()),
        XBRLBackendOptions(enable_remote_fetch=True, enable_local_fetch=False),
    )

    assert "enable_local_fetch=True" in message
