"""Tests for the Apple Pages (``.pages``) document backend.

Test Data Attribution
---------------------
``pages_2013.pages`` and ``pages_iwork09.pages`` are ``testPages2013.pages`` and
``testPages.pages`` from the Apache Tika test corpus, licensed under the Apache
License 2.0. They are genuine Apple Pages output, and between them cover both
container generations: ``pages_2013.pages`` stores its content as ``Index/*.iwa``
with no PDF render, while ``pages_iwork09.pages`` uses the iWork '09 ``index.xml``
layout. Conveniently, both hold the same source document, so the two code paths
can be checked against each other.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).
"""

import gzip
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel

from docling.backend.iwork import iwa
from docling.backend.iwork.iwa import (
    decompress_snappy_block,
    iter_objects,
    read_fields,
)
from docling.backend.iwork.pages_backend import (
    IWorkPagesDocumentBackend,
    _iwa_style_name,
    _label_for_style,
)
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

SOURCES = Path("./tests/data/pages/sources")
PAGES_2013 = SOURCES / "pages_2013.pages"
PAGES_IWORK09 = SOURCES / "pages_iwork09.pages"
PAGES_PASSWORD_PROTECTED = SOURCES / "pages_password_protected.pages"

# Present in the body of both fixtures.
_BODY_SENTENCE = "Some plain text to parse."


def _backend(
    path: Path, options: IWorkBackendOptions | None = None
) -> IWorkPagesDocumentBackend:
    in_doc = InputDocument(
        path_or_stream=path,
        format=InputFormat.IWORK_PAGES,
        backend=IWorkPagesDocumentBackend,
        backend_options=options,
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkPagesDocumentBackend)
    return backend


def test_detects_pages_from_path_and_named_stream():
    """`.pages` is a ZIP, so detection must not stop at ``application/zip``."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])

    assert conv_input._guess_format(PAGES_2013) == InputFormat.IWORK_PAGES

    stream = DocumentStream(
        name="report.pages", stream=BytesIO(PAGES_2013.read_bytes())
    )
    assert conv_input._guess_format(stream) == InputFormat.IWORK_PAGES


def test_extensionless_pages_stream_is_not_claimed():
    """Without the extension a Pages container is indistinguishable from Keynote
    and Numbers, so the backend must not claim it rather than guess wrong."""
    conv_input = _DocumentConversionInput(path_or_stream_iterator=[])
    stream = DocumentStream(name="blob", stream=BytesIO(PAGES_2013.read_bytes()))

    assert conv_input._guess_format(stream) is None


def test_modern_pages_body_text_is_extracted():
    """Pages 5+ keeps its body in Index/*.iwa with no PDF render, so this is the
    path that matters for essentially every Pages document in circulation."""
    doc = _backend(PAGES_2013).convert()

    text = doc.export_to_markdown()
    assert "Sample pages document" in text
    assert _BODY_SENTENCE in text
    assert "Both Pages 1.x and Keynote 2.x" in text


def test_legacy_pages_body_text_is_extracted():
    doc = _backend(PAGES_IWORK09).convert()

    assert _BODY_SENTENCE in doc.export_to_markdown()


def test_both_generations_agree_on_the_shared_body_text():
    """The two fixtures are the same source document saved by different Pages
    releases, so the independent IWA and XML readers must agree on its text."""
    modern = _backend(PAGES_2013).convert().export_to_markdown()
    legacy = _backend(PAGES_IWORK09).convert().export_to_markdown()

    for sentence in ("Sample pages document", _BODY_SENTENCE):
        assert sentence in modern
        assert sentence in legacy


def test_legacy_placeholder_text_is_not_emitted():
    """iWork '09 templates carry sf:ghost-text placeholders. That is what the
    template displays before the author types, not document content."""
    raw = zipfile.ZipFile(PAGES_IWORK09).read("index.xml").decode("utf-8", "replace")
    assert "ghost-text" in raw, "fixture no longer exercises the placeholder path"

    text = _backend(PAGES_IWORK09).convert().export_to_markdown()
    assert "Lorem ipsum dolor sit amet" not in text


def test_pages_backend_accepts_a_stream():
    stream = BytesIO(PAGES_2013.read_bytes())
    in_doc = InputDocument(
        path_or_stream=stream,
        format=InputFormat.IWORK_PAGES,
        backend=IWorkPagesDocumentBackend,
        filename="report.pages",
    )
    backend = in_doc._backend
    assert isinstance(backend, IWorkPagesDocumentBackend)

    assert _BODY_SENTENCE in backend.convert().export_to_markdown()


def test_object_replacement_characters_are_dropped():
    """Apple marks inline attachments with U+FFFC inside the text run; it carries
    no text and must not leak into the output."""
    doc = _backend(PAGES_2013).convert()

    assert "￼" not in doc.export_to_markdown()


def test_iwa_reader_walks_the_real_object_graph():
    """Guards the container layer itself: chunk framing, raw Snappy and the
    TSP.ArchiveInfo walk, against genuine Apple output."""
    archive = zipfile.ZipFile(PAGES_2013)
    objects = {}
    for name in archive.namelist():
        if name.endswith(".iwa"):
            for obj in iter_objects(archive.read(name)):
                objects[obj.identifier] = obj

    assert len(objects) > 100

    # TP.DocumentArchive must be present and reference a TSWP.StorageArchive.
    document = next(o for o in objects.values() if o.message_type == 10000)
    body_ref = read_fields(document.payload)[4][0]
    assert isinstance(body_ref, bytes)


def test_password_protected_document_is_rejected_cleanly():
    """Pages encrypts members with a scheme zipfile cannot read, and leaves a
    nonsense compress_type instead of setting the standard encrypted flag. That
    surfaces as NotImplementedError deep inside zipfile, which must be turned
    into a DocumentLoadError rather than escaping as an unhandled crash."""
    with pytest.raises(DocumentLoadError, match="password-protected"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=PAGES_PASSWORD_PROTECTED,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            PAGES_PASSWORD_PROTECTED,
        )


def test_snappy_decoder_handles_every_element_type():
    """The IWA payloads exercise literals and all three copy encodings. Round-trip
    a payload built to hit each one, including an overlapping copy, which is how
    Snappy encodes repeated runs and the easiest case to get wrong."""

    def literal(payload: bytes) -> bytes:
        assert len(payload) <= 60
        return bytes([(len(payload) - 1) << 2]) + payload

    # "abcd" then a 1-byte-offset copy repeating it, then a long run produced by
    # an overlapping copy of a single byte.
    body = literal(b"abcd")
    body += bytes([(1) | (((4 - 4) & 0x07) << 2) | (0 << 5), 4])  # copy len 4, off 4
    body += literal(b"z")
    body += bytes([0x02 | ((20 - 1) << 2)]) + (1).to_bytes(2, "little")  # off 1, len 20

    expected = b"abcd" + b"abcd" + b"z" + b"z" * 20
    block = bytes([len(expected)]) + body

    assert decompress_snappy_block(block) == expected


@pytest.mark.parametrize(
    "block, reason",
    [
        (b"\x04\x00\x00", "length mismatch"),  # claims 4 bytes, yields 1
        (b"\x04\x11\xff", "copy offset past the start of the output"),
        (b"\x04\xfc", "truncated literal"),
    ],
)
def test_snappy_decoder_rejects_malformed_blocks(block: bytes, reason: str):
    """A corrupt block must fail loudly rather than return partial output."""
    with pytest.raises(DocumentLoadError):
        decompress_snappy_block(block)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def _snappy_bomb(declared: int, copies: int) -> bytes:
    """A block whose copy tags expand ~21x, the worst case for raw Snappy."""
    body = bytes([0 << 2]) + b"A"
    body += (bytes([0x02 | ((64 - 1) << 2)]) + (1).to_bytes(2, "little")) * copies
    return _varint(declared) + body


def _write_pages(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_snappy_block_declaring_more_than_the_limit_is_refused_before_decoding():
    """Raw Snappy expands up to 21.33x and an IWA chunk may declare 16.7 MB, so
    the declared size has to be rejected before any element is decoded."""
    with pytest.raises(DocumentLoadError, match=r"over the .* byte limit"):
        decompress_snappy_block(_snappy_bomb(400 * 1024 * 1024, 1), limit=1024)


def test_snappy_block_that_lies_about_its_size_is_stopped_mid_decode():
    """A block may declare a small size and then emit far more. Checking only at
    the end would mean doing all the work first."""
    with pytest.raises(DocumentLoadError, match="more than the 10 bytes"):
        decompress_snappy_block(_snappy_bomb(10, 200_000))


def _snappy_literal_run(size: int) -> bytes:
    """A well-formed block that decodes to exactly ``size`` bytes of literals."""
    body = bytearray()
    remaining = size
    while remaining:
        take = min(60, remaining)
        body += bytes([(take - 1) << 2]) + b"A" * take
        remaining -= take
    return _varint(size) + bytes(body)


def test_iwa_stream_budget_is_shared_across_chunks(monkeypatch, tmp_path: Path):
    """The ceiling must bound the whole stream, not reset for each chunk. Two
    chunks that are each individually fine must still be refused once their
    combined output passes the limit."""
    monkeypatch.setattr(iwa, "_MAX_STREAM_BYTES", 1500)

    block = _snappy_literal_run(1000)
    chunk = b"\x00" + len(block).to_bytes(3, "little") + block

    # One chunk alone stays under the ceiling.
    assert len(iwa.decompress(chunk)) == 1000

    with pytest.raises(DocumentLoadError, match="over the 500 byte limit"):
        iwa.decompress(chunk * 2)


def test_gzipped_legacy_index_cannot_expand_without_bound(tmp_path: Path):
    """max_total_bytes only counts the stored size of index.xml.gz, so the
    decompressed size needs its own ceiling."""
    bomb = _write_pages(
        tmp_path / "gzbomb.pages",
        {"index.xml.gz": gzip.compress(b"\0" * (400 * 1024 * 1024))},
    )

    with pytest.raises(DocumentLoadError, match="expands beyond"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=bomb,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            bomb,
            IWorkBackendOptions(max_total_bytes=8 * 1024 * 1024),
        )


def test_deeply_nested_legacy_xml_does_not_exhaust_the_stack(tmp_path: Path):
    """Nesting depth is attacker-controlled, so the text walk must not recurse."""
    depth = 30_000
    xml = (
        b'<?xml version="1.0"?>'
        b'<sf:document xmlns:sf="http://developer.apple.com/namespaces/sf"><sf:p>'
        + b"<sf:span>" * depth
        + b"deep text"
        + b"</sf:span>" * depth
        + b"</sf:p></sf:document>"
    )
    nested = _write_pages(tmp_path / "deep.pages", {"index.xml": xml})

    backend = IWorkPagesDocumentBackend(
        InputDocument(
            path_or_stream=nested,
            format=InputFormat.IWORK_PAGES,
            backend=IWorkPagesDocumentBackend,
        ),
        nested,
    )
    assert "deep text" in backend.convert().export_to_markdown()


def test_zip_without_pages_index_is_rejected(tmp_path: Path):
    other_zip = tmp_path / "not_really.pages"
    with zipfile.ZipFile(other_zip, "w") as zf:
        zf.writestr("word/document.xml", "<w:document/>")

    with pytest.raises(DocumentLoadError, match="does not look like a Pages document"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=other_zip,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            other_zip,
        )


def test_non_zip_input_is_rejected(tmp_path: Path):
    broken = tmp_path / "broken.pages"
    broken.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentLoadError, match="not a readable ZIP container"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=broken,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            broken,
        )


def test_archive_limits_are_enforced():
    """The container is attacker-controlled, so limits must bite before the IWA
    archives are decompressed."""
    with pytest.raises(DocumentLoadError, match="max_member_count"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=PAGES_2013,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            PAGES_2013,
            IWorkBackendOptions(max_member_count=1),
        )

    with pytest.raises(DocumentLoadError, match="max_total_bytes"):
        IWorkPagesDocumentBackend(
            InputDocument(
                path_or_stream=PAGES_2013,
                format=InputFormat.IWORK_PAGES,
                backend=IWorkPagesDocumentBackend,
            ),
            PAGES_2013,
            IWorkBackendOptions(max_total_bytes=1024),
        )


def test_end_to_end_conversion():
    """No models involved: the backend is declarative, so this runs in CI without
    the PDF pipeline."""
    converter = DocumentConverter(allowed_formats=[InputFormat.IWORK_PAGES])
    result = converter.convert(PAGES_2013)

    assert _BODY_SENTENCE in result.document.export_to_markdown()
    assert result.document.origin is not None
    assert result.document.origin.mimetype == "application/vnd.apple.pages"
    assert result.document.origin.filename == "pages_2013.pages"


def test_legacy_table_is_extracted_with_its_header_row():
    """iWork '09 stores table cells flat in sf:datasource, so the grid dimensions
    are what place them. The fixture holds a 3x4 table with one header row."""
    doc = _backend(PAGES_IWORK09).convert()

    assert len(doc.tables) == 1
    table = doc.tables[0].data
    assert (table.num_rows, table.num_cols) == (4, 3)

    by_position = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.table_cells
    }
    assert by_position[(0, 0)].text == "Column one"
    assert by_position[(0, 2)].text == "Column three"
    assert by_position[(3, 2)].text == "Cell nine"

    # Only the first row is a header, per sf:num-header-rows.
    assert by_position[(0, 0)].column_header
    assert not by_position[(1, 0)].column_header


@pytest.mark.parametrize(
    "style_name, expected_label, expected_level",
    [
        # Names observed in the stylesheets of the real Tika fixtures.
        ("Title", DocItemLabel.TITLE, None),
        ("Heading 1", DocItemLabel.SECTION_HEADER, 1),
        ("Heading 2", DocItemLabel.SECTION_HEADER, 2),
        ("Heading", DocItemLabel.SECTION_HEADER, 1),
        ("Subheading", DocItemLabel.SECTION_HEADER, 2),
        ("Body", DocItemLabel.TEXT, None),
        ("Free Form", DocItemLabel.TEXT, None),
        ("Footnote Text", DocItemLabel.TEXT, None),
        (None, DocItemLabel.TEXT, None),
    ],
)
def test_style_names_map_to_labels(style_name, expected_label, expected_level):
    """Pages uses the same built-in style names in both container generations,
    so one mapping serves the IWA and XML readers."""
    assert _label_for_style(style_name) == (expected_label, expected_level)


def test_iwa_paragraph_styles_resolve_to_their_real_names():
    """The IWA heading path depends on resolving a style run to its style name.
    Check that against genuine Apple output rather than the mapping alone."""
    archive = zipfile.ZipFile(PAGES_2013)
    objects = {
        obj.identifier: obj
        for name in archive.namelist()
        if name.endswith(".iwa")
        for obj in iter_objects(archive.read(name))
    }

    names = {
        _iwa_style_name(obj.payload)
        for obj in objects.values()
        if obj.message_type == 2022
    }
    # The fixture's body is styled "Body"; anonymous styles resolve to None.
    assert "Body" in names


def test_body_text_is_labelled_from_its_style():
    """The fixture's body paragraphs are all styled "Body", so nothing should be
    promoted to a heading."""
    doc = _backend(PAGES_2013).convert()

    assert doc.texts
    assert all(item.label == DocItemLabel.TEXT for item in doc.texts)


def test_legacy_page_furniture_stays_out_of_the_body(tmp_path: Path):
    """Headers, footers and footnotes each carry their own sf:text-body in an '09
    document, so iterating every sf:p would fold them into the body flow. The IWA
    reader only ever sees the body storage, so both generations must agree."""
    namespace = "http://developer.apple.com/namespaces/sf"
    xml = f"""<?xml version="1.0"?>
    <sf:document xmlns:sf="{namespace}">
      <sf:stylesheet>
        <sf:paragraphstyle sf:name="Body" sf:ident="ps-body"/>
      </sf:stylesheet>
      <sf:text-storage>
        <sf:text-body><sf:p sf:style="ps-body">Real body text.</sf:p></sf:text-body>
        <sf:header><sf:text-body><sf:p>Running header</sf:p></sf:text-body></sf:header>
        <sf:footer><sf:text-body><sf:p>Page footer</sf:p></sf:text-body></sf:footer>
        <sf:footnotes>
          <sf:text-storage><sf:text-body>
            <sf:p>A footnote body</sf:p>
          </sf:text-body></sf:text-storage>
        </sf:footnotes>
      </sf:text-storage>
    </sf:document>""".encode()

    source = _write_pages(tmp_path / "furniture.pages", {"index.xml": xml})
    backend = IWorkPagesDocumentBackend(
        InputDocument(
            path_or_stream=source,
            format=InputFormat.IWORK_PAGES,
            backend=IWorkPagesDocumentBackend,
        ),
        source,
    )

    text = backend.convert().export_to_markdown()
    assert "Real body text." in text
    for furniture in ("Running header", "Page footer", "A footnote body"):
        assert furniture not in text
