# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test module for the XBRL backend parser.

The data used in this test is in the public domain. It has been downloaded from the
U.S. Securities and Exchange Commission (SEC)'s Electronic Data Gathering, Analysis,
and Retrieval (EDGAR) system.
"""

import os
import threading
import time
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DoclingDocument
from lxml import etree

from docling.backend.xml.xbrl_backend import (
    XBRLDocumentBackend,
    _absolutize_missing_refs,
)
from docling.datamodel.backend_options import XBRLBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling.document_converter import DocumentConverter, XBRLFormatOption
from docling.exceptions import DocumentLoadError, OperationNotAllowed

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


def test_ixbrl_report_converts_offline():
    """An iXBRL report converts offline with a taxonomy package.

    IBM's 10-Q has facts with typed dimensions, which have no member QName.
    """
    ixbrl_dir = _SOURCES_DIR.parent / "ixbrl"
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
    assert len(doc.tables) > 0
    assert len(doc.key_value_items[0].graph.links) > 0


def test_ixbrl_warns_when_taxonomy_is_not_found(tmp_path, caplog, no_taxonomy_packages):
    """A package without the extension taxonomy leaves the facts unresolved.

    The package maps every ``https://`` URL to its own files. The extension
    taxonomy of the report is not among them, and it is not fetched remotely.
    """
    ixbrl_dir = _SOURCES_DIR.parent / "ixbrl"
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


_SOURCE_URL = "https://www.sec.gov/Archives/edgar/data/1/2/report-20260930.htm"


def test_absolutize_missing_refs_resolves_only_missing_relative_refs(tmp_path):
    (tmp_path / "local.xsd").write_text("")
    content = (
        b'<ix:references xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"'
        b' xmlns:link="http://www.xbrl.org/2003/linkbase"'
        b' xmlns:xlink="http://www.w3.org/1999/xlink">'
        b'<link:schemaRef xlink:type="simple" xlink:href="report-20260930.xsd"/>'
        b'<link:schemaRef xlink:href="local.xsd"/>'
        b'<link:linkbaseRef xlink:href="sub/report_pre.xml#id"/>'
        b'<link:schemaRef xlink:href="https://example.org/base.xsd"/>'
        b"</ix:references>"
    )

    result = _absolutize_missing_refs(content, tmp_path, _SOURCE_URL)

    base = b"https://www.sec.gov/Archives/edgar/data/1/2/"
    assert b'xlink:href="' + base + b'report-20260930.xsd"' in result
    assert b'xlink:href="' + base + b'sub/report_pre.xml#id"' in result
    assert b'xlink:href="local.xsd"' in result
    assert b'xlink:href="https://example.org/base.xsd"' in result


def test_absolutize_missing_refs_is_safe_with_hostile_references(tmp_path):
    (tmp_path / "secret.xsd").write_text("")
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    origin = "https://www.sec.gov/"
    hrefs = [
        "../secret.xsd",
        "/etc/passwd",
        "a&quot; onload=&quot;x.xsd",
        "a&amp;b.xsd",
        "x.xsd#a&apos;b",
        "//evil.example/x.xsd",
        "https://evil.example/x.xsd",
        "file:///etc/passwd",
    ]
    content = (
        b"".join(
            b'<link:schemaRef xlink:href="' + href.encode() + b'"/>' for href in hrefs
        )
        + b"<link:schemaRef xlink:href='say \"hi\".xsd'/>"
    )

    result = _absolutize_missing_refs(content, base_dir, _SOURCE_URL)

    # The result is well-formed XML: no value can close its attribute early.
    root = etree.fromstring(b'<r xmlns:link="l" xmlns:xlink="x">' + result + b"</r>")
    resolved = [ref.get("{x}href") for ref in root]
    # A reference that leaves the work folder never counts as a local file: it is
    # resolved against the source URL instead of probing the file system.
    assert resolved[0] == origin + "Archives/edgar/data/1/secret.xsd"
    assert resolved[1] == origin + "etc/passwd"
    assert resolved[2] == origin + 'Archives/edgar/data/1/2/a" onload="x.xsd'
    assert resolved[3] == origin + "Archives/edgar/data/1/2/a&b.xsd"
    assert resolved[4] == origin + "Archives/edgar/data/1/2/x.xsd#a'b"
    # References to other origins are not rewritten.
    assert resolved[5] == "//evil.example/x.xsd"
    assert resolved[6] == "https://evil.example/x.xsd"
    assert resolved[7] == "file:///etc/passwd"
    assert resolved[8] == origin + 'Archives/edgar/data/1/2/say "hi".xsd'


def test_absolutize_missing_refs_is_linear_on_malformed_input(tmp_path):
    start = time.perf_counter()
    _absolutize_missing_refs(b"<schemaRef " * 50_000, tmp_path, _SOURCE_URL)
    _absolutize_missing_refs(b"<schemaRef href='" * 50_000, tmp_path, _SOURCE_URL)

    assert time.perf_counter() - start < 5


@pytest.mark.parametrize(
    "value",
    [
        "https://user:pass@www.sec.gov/Archives/report.htm",
        "ftp://www.sec.gov/report.htm",
        "file:///etc/passwd",
        "report.htm",
    ],
)
def test_xbrl_source_uri_rejects_credentials_and_other_schemes(value):
    with pytest.raises(ValueError):
        XBRLBackendOptions(source_uri=value)


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


def test_xbrl_source_uri_is_ignored_without_remote_fetch(
    no_taxonomy_packages, local_server
):
    base_url, requests = local_server
    stream = _instance_with_schema_ref("relative.xsd")
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.XML_XBRL,
        backend=XBRLDocumentBackend,
        filename="instance.xbrl",
    )
    options = XBRLBackendOptions(
        enable_local_fetch=True, source_uri=f"{base_url}/dir/report.xbrl"
    )

    backend = XBRLDocumentBackend(in_doc, stream, options)
    backend.unload()

    assert requests == []


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


def _instance_with_schema_ref(href: str) -> BytesIO:
    return BytesIO(
        b'<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"'
        b' xmlns:link="http://www.xbrl.org/2003/linkbase"'
        b' xmlns:xlink="http://www.w3.org/1999/xlink">'
        b'<link:schemaRef xlink:type="simple" xlink:href="'
        + href.encode()
        + b'"/></xbrli:xbrl>'
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
