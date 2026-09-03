# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the Apple Pages (``.pages``) document backend.

Test Data Attribution
---------------------
``pages_2013.pages``, ``pages_iwork09.pages``, ``pages_iwork09_formatted.pages``
and ``pages_iwork09_comments.pages`` are ``testPages2013.pages``,
``testPages.pages``, ``testPagesHeadersFootersFootnotes.pages`` and
``testPagesComments.pages`` from the Apache Tika test corpus, licensed under the
Apache License 2.0. They are genuine Apple Pages output, and between them cover
both container generations: ``pages_2013.pages`` stores its content as
``Index/*.iwa`` with no PDF render, while the rest use the iWork '09
``index.xml`` layout. Conveniently, the first two hold the same source document,
so the two code paths can be checked against each other.

See https://github.com/apache/tika (``tika-parser-apple-module`` test resources).
"""

import gzip
import zipfile
from io import BytesIO
from pathlib import Path

import defusedxml.ElementTree as ET
import pytest
from docling_core.types.doc import ContentLayer, DocItemLabel, Script
from docling_core.types.doc.items.text import ListItem
from PIL import Image as PILImage

from docling.backend.iwork import iwa
from docling.backend.iwork.content import label_for_style
from docling.backend.iwork.iwa import (
    IWAObject,
    decompress_snappy_block,
    iter_objects,
    read_fields,
)
from docling.backend.iwork.pages_iwa import (
    iwa_formatting,
    iwa_list_style,
    iwa_style_name,
)
from docling.backend.iwork.pages_xml import legacy_formatting
from docling.backend.iwork_backend import IWorkPagesDocumentBackend
from docling.datamodel.backend_options import IWorkBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import InputDocument, _DocumentConversionInput
from docling.document_converter import DocumentConverter
from docling.exceptions import DocumentLoadError

SOURCES = Path("./tests/data/pages/sources")
PAGES_2013 = SOURCES / "pages_2013.pages"
PAGES_IWORK09 = SOURCES / "pages_iwork09.pages"
PAGES_PASSWORD_PROTECTED = SOURCES / "pages_password_protected.pages"
PAGES_IWORK09_FORMATTED = SOURCES / "pages_iwork09_formatted.pages"
PAGES_IWORK09_COMMENTS = SOURCES / "pages_iwork09_comments.pages"

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
    assert label_for_style(style_name) == (expected_label, expected_level)


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
        iwa_style_name(obj.payload)
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
    document, so iterating every sf:p would fold them into the body flow. They
    are recovered as furniture instead, which is what keeps them out of the
    reading order while still making them available."""
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

    doc = backend.convert()
    text = doc.export_to_markdown()
    assert "Real body text." in text
    for furniture in ("Running header", "Page footer", "A footnote body"):
        assert furniture not in text

    recovered = {
        item.text: item.label
        for item in doc.texts
        if item.content_layer == ContentLayer.FURNITURE
    }
    assert recovered["Running header"] == DocItemLabel.PAGE_HEADER
    assert recovered["Page footer"] == DocItemLabel.PAGE_FOOTER
    assert recovered["A footnote body"] == DocItemLabel.FOOTNOTE


def test_modern_table_is_extracted_from_the_tile_storage():
    """A Pages 5+ table places its cells through per-row offsets into a packed
    buffer, each referencing a shared string by key. Reading the tile is what
    makes repeated values safe: they share one entry in the value list."""
    doc = _backend(PAGES_2013).convert()

    assert len(doc.tables) == 1
    table = doc.tables[0].data
    assert (table.num_rows, table.num_cols) == (4, 3)

    by_position = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell
        for cell in table.table_cells
    }
    assert by_position[(0, 0)].text == "Column one"
    assert by_position[(3, 2)].text == "Cell nine"
    assert by_position[(0, 1)].column_header
    assert not by_position[(1, 1)].column_header


def test_both_generations_agree_on_the_table():
    """The two fixtures are the same document saved by different Pages releases,
    so the IWA tile reader and the '09 XML reader must produce the same grid."""

    def grid(path: Path) -> dict[tuple[int, int], str]:
        table = _backend(path).convert().tables[0].data
        return {
            (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
            for cell in table.table_cells
        }

    assert grid(PAGES_2013) == grid(PAGES_IWORK09)


def test_modern_text_box_content_is_extracted():
    """Text boxes are floating drawables, reached from TP.DocumentArchive rather
    than from the body storage."""
    doc = _backend(PAGES_2013).convert()

    assert "A text box with text." in doc.export_to_markdown()


def test_text_boxes_are_reached_by_ownership_not_by_scanning():
    """Scanning every TSWP.StorageArchive would be simpler but would also pick up
    headers, footers and footnotes, which are deliberately excluded. Following
    the drawables field keeps that distinction, so the storage holding the text
    box must not be the body storage it is read alongside."""
    archive = zipfile.ZipFile(PAGES_2013)
    objects = {
        obj.identifier: obj
        for name in archive.namelist()
        if name.endswith(".iwa")
        for obj in iter_objects(archive.read(name))
    }

    document = next(o for o in objects.values() if o.message_type == 10000)
    storages_with_text = [
        obj.identifier
        for obj in objects.values()
        if obj.message_type == 2001
        and any(
            isinstance(v, bytes) and v.strip()
            for v in read_fields(obj.payload).get(3, [])
        )
    ]

    # The fixture holds exactly two: the body, and the text box.
    assert len(storages_with_text) == 2
    assert (
        "A text box with text." in _backend(PAGES_2013).convert().export_to_markdown()
    )
    assert document.message_type == 10000


def test_legacy_character_formatting_is_recovered():
    """iWork '09 applies character styles through sf:span. This fixture underlines
    part of a paragraph, so the run has to keep its formatting while the rest of
    the paragraph does not."""
    doc = _backend(PAGES_IWORK09_FORMATTED).convert()

    underlined = [
        item.text
        for item in doc.texts
        if item.formatting is not None and item.formatting.underline
    ]
    assert underlined, "fixture no longer carries an underlined run"
    assert any("Both Pages 1.x and Keynote 2.x" in text for text in underlined)

    # The formatting must not bleed onto text outside the span.
    assert not all(
        item.formatting is not None and item.formatting.underline for item in doc.texts
    )


def test_iwa_character_styles_map_onto_formatting():
    """The property fields of a character style were established by correlating
    style names across real Apple documents. Check that mapping against the
    styles the fixture actually defines."""
    archive = zipfile.ZipFile(PAGES_2013)
    objects = {
        obj.identifier: obj
        for name in archive.namelist()
        if name.endswith(".iwa")
        for obj in iter_objects(archive.read(name))
    }

    by_name = {
        iwa_style_name(obj.payload): iwa_formatting(obj.payload)
        for obj in objects.values()
        if obj.message_type == 2021
    }

    assert by_name["Emphasis"] is not None and by_name["Emphasis"].bold
    assert by_name["Underline"] is not None and by_name["Underline"].underline
    assert (
        by_name["Strikethrough"] is not None and by_name["Strikethrough"].strikethrough
    )


def test_a_run_boundary_does_not_eat_the_space_around_it():
    """Pages ends a character style mid-sentence, so the space on either side of
    a formatted phrase belongs to the paragraph rather than to the run. Trimming
    every run would silently glue the words on both sides together.

    The '09 fixture splits "APXL file" and "the <key:slide-list> element" across
    spans; the IWA fixture holds the same sentences in one piece, so the two
    generations pin the expected text between them."""
    legacy = " ".join(
        item.text for item in _backend(PAGES_IWORK09_FORMATTED).convert().texts
    )
    modern = " ".join(item.text for item in _backend(PAGES_2013).convert().texts)

    for sentence in (
        "Keynote APXL file is the engine",
        "slide-list> element in a text",
    ):
        assert sentence in modern
        assert sentence in legacy


def _iwa_objects(path: Path) -> dict[int, IWAObject]:
    """Every archived object of a Pages 5+ fixture, keyed by identifier."""
    archive = zipfile.ZipFile(path)
    return {
        obj.identifier: obj
        for name in archive.namelist()
        if name.endswith(".iwa")
        for obj in iter_objects(archive.read(name))
    }


def test_iwa_list_styles_decode_to_their_real_labels():
    """Whether a paragraph is a list item is decided by the list style in force,
    not by its nesting depth: Pages leaves a style in force over plain paragraphs
    too and marks them with the "None" style. Check that against the styles the
    fixture's template actually defines."""
    by_name = {
        iwa_style_name(obj.payload): iwa_list_style(obj.payload)
        for obj in _iwa_objects(PAGES_2013).values()
        if obj.message_type == 2023
    }

    bullet = by_name["Bullet"].label(0)
    assert bullet is not None
    assert not bullet.enumerated
    assert bullet.marker == "\u2022"

    numbered = by_name["Numbered List"].label(0)
    assert numbered is not None and numbered.enumerated

    # The style Pages applies to ordinary body text labels no level at all.
    assert by_name["None"].label(0) is None


def _snappy_literals(payload: bytes) -> bytes:
    """Encode ``payload`` as one raw Snappy block, using literals only."""
    out = bytearray(_varint(len(payload)))
    for start in range(0, len(payload), 60):
        piece = payload[start : start + 60]
        out.append((len(piece) - 1) << 2)
        out += piece
    return bytes(out)


def _with_bullets_applied(target: Path) -> Path:
    """Copy the Pages 5+ fixture with its body text styled as a bulleted list.

    The fixture is a real Apple document that defines a "Bullet" list style but
    never applies it, so the only change made here is which style the list-style
    run table points at: object 2708 ("None") becomes object 4078 ("Bullet"),
    both of them written into this file by Pages. The two identifiers encode to
    varints of the same width, so nothing else in the archive moves.
    """
    none_style, bullet_style = _varint(2708), _varint(4078)
    entry = bytes([0x08, 0x00, 0x12, 0x03, 0x08]) + none_style
    # Field 7 of TSWP.StorageArchive, holding a one-entry run table.
    table = bytes([0x3A, len(entry) + 2, 0x0A, len(entry)]) + entry

    archive = zipfile.ZipFile(PAGES_2013)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        for info in archive.infolist():
            data = archive.read(info)
            if info.filename == "Index/Document.iwa":
                stream = iwa.decompress(data)
                assert table in stream, "fixture no longer carries a list style run"
                stream = stream.replace(table, table.replace(none_style, bullet_style))
                block = _snappy_literals(stream)
                data = bytes([0x00]) + len(block).to_bytes(3, "little") + block
            out.writestr(info.filename, data)
    return target


def test_bulleted_list_is_recovered_from_the_list_style_run(tmp_path: Path):
    """The end of the list path: a run table pointing at a labelling style turns
    the paragraphs it covers into list items collected under one group."""
    doc = _backend(_with_bullets_applied(tmp_path / "bullets.pages")).convert()

    items = [item for item in doc.texts if isinstance(item, ListItem)]
    assert any(_BODY_SENTENCE in item.text for item in items)
    assert all(not item.enumerated for item in items)

    # Consecutive items share one group rather than each opening their own.
    groups = {item.parent.cref for item in items if item.parent is not None}
    assert 0 < len(groups) < len(items)


def test_body_text_is_not_a_list_when_the_style_labels_nothing():
    """The untouched fixture carries the same run table, pointing at the "None"
    style. Reading the nesting depth alone would turn every paragraph in the
    document into a list item."""
    doc = _backend(PAGES_2013).convert()

    assert not [item for item in doc.texts if item.label == DocItemLabel.LIST_ITEM]


def _len_field(number: int, payload: bytes) -> bytes:
    """Encode one length-delimited protobuf field."""
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _varint_field(number: int, value: int) -> bytes:
    """Encode one varint protobuf field."""
    return _varint(number << 3) + _varint(value)


def _reference_field(number: int, identifier: int) -> bytes:
    """Encode a TSP.Reference field, whose only field is the target identifier."""
    return _len_field(number, _varint_field(1, identifier))


def _iwa_member(objects: list[tuple[int, int, bytes]]) -> bytes:
    """Encode objects as one Index/*.iwa member.

    An .iwa member is a Snappy-compressed stream of archives, each a
    TSP.ArchiveInfo naming an object identifier and a message type, followed by
    that message's bytes. Members are read in archive order and later objects
    replace earlier ones with the same identifier, which is how a test can
    redefine one object of a real document and leave the rest untouched.
    """
    stream = bytearray()
    for identifier, message_type, payload in objects:
        info = _varint_field(1, identifier) + _len_field(
            2, _varint_field(1, message_type) + _varint_field(3, len(payload))
        )
        stream += _varint(len(info)) + info + payload
    block = _snappy_literals(bytes(stream))
    return bytes([0x00]) + len(block).to_bytes(3, "little") + block


def _tiny_png() -> bytes:
    """A one-pixel PNG, so a picture has real bytes to carry."""
    buffer = BytesIO()
    PILImage.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _with_image_anchored(target: Path, png: bytes) -> Path:
    """Copy the Pages 5+ fixture with an image where it anchors its table.

    The fixture already anchors a drawable inline, at the U+FFFC its body text
    carries: object 4106 is the attachment and 4107 the drawable it holds.
    Redefining 4107 as a TSD.ImageArchive, and naming a Data/ member for it in
    the package metadata, exercises the path from the anchor to the bytes
    without disturbing the text, the styles or the attachment table.
    """
    data_id = 990
    member = "Data/anchored.png"
    image = _reference_field(11, data_id)  # TSD.ImageArchive.data
    metadata = _len_field(
        4,  # TSP.PackageMetadata.datas
        _varint_field(1, data_id)
        + _len_field(3, member.removeprefix("Data/").encode()),
    )

    archive = zipfile.ZipFile(PAGES_2013)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        for info in archive.infolist():
            out.writestr(info.filename, archive.read(info))
        out.writestr(member, png)
        out.writestr(
            "Index/Redefined.iwa",
            _iwa_member([(4107, 3005, image), (2, 11006, metadata)]),
        )
    return target


def test_anchored_image_is_placed_where_the_text_anchors_it(tmp_path: Path):
    """Apple marks an inline attachment with U+FFFC and says in the storage's
    attachment table which drawable it is. An image resolves through the package
    metadata to a Data/ member, and belongs at the anchor rather than at the end
    of the document."""
    png = _tiny_png()
    doc = _backend(_with_image_anchored(tmp_path / "image.pages", png)).convert()

    assert len(doc.pictures) == 1
    picture = doc.pictures[0]
    assert picture.image is not None
    assert picture.image.size.width == 1

    # The anchor sits early in the body text, so the picture must stay there
    # rather than being pushed past everything that follows it.
    order = [child.cref for child in doc.body.children]
    assert 0 < order.index(picture.self_ref) < len(order) - 1


def test_image_without_stored_bytes_still_keeps_its_place(tmp_path: Path):
    """Pages names every rendition of a placed image, including ones it never
    wrote into the container. A picture whose bytes are missing is still part of
    the document."""
    target = _with_image_anchored(tmp_path / "missing.pages", _tiny_png())
    stripped = tmp_path / "stripped.pages"
    source = zipfile.ZipFile(target)
    with zipfile.ZipFile(stripped, "w", zipfile.ZIP_DEFLATED) as out:
        for info in source.infolist():
            if not info.filename.startswith("Data/"):
                out.writestr(info.filename, source.read(info))

    doc = _backend(stripped).convert()

    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is None


def test_legacy_image_is_read_from_the_container_member(tmp_path: Path):
    """An iWork '09 image names its bytes by container path rather than through
    an object graph, so the whole path is the sf:data element."""
    png = _tiny_png()
    namespace = "http://developer.apple.com/namespaces/sf"
    xml = f"""<?xml version="1.0"?>
    <sf:document xmlns:sf="{namespace}">
      <sf:text-storage>
        <sf:text-body>
          <sf:p>Before the image.</sf:p>
          <sf:media>
            <sf:content><sf:image-media><sf:filtered-image><sf:unfiltered>
              <sf:data sf:path="pasted-image.png"/>
            </sf:unfiltered></sf:filtered-image></sf:image-media></sf:content>
          </sf:media>
          <sf:p>After the image.</sf:p>
        </sf:text-body>
      </sf:text-storage>
    </sf:document>""".encode()

    source = _write_pages(
        tmp_path / "media.pages", {"index.xml": xml, "pasted-image.png": png}
    )
    doc = _backend(source).convert()

    assert len(doc.pictures) == 1
    assert doc.pictures[0].image is not None

    order = [item.cref for item in doc.body.children]
    assert order.index(doc.pictures[0].self_ref) == 1


def test_table_is_placed_where_the_document_anchors_it():
    """The fixture anchors its table inline, at a U+FFFC early in the body text.
    Reading the attachment table is what puts it there instead of after
    everything else."""
    doc = _backend(PAGES_2013).convert()

    order = [child.cref for child in doc.body.children]
    assert 0 < order.index(doc.tables[0].self_ref) < len(order) - 1


def _redefined(target: Path, objects: list[tuple[int, int, bytes]]) -> Path:
    """Copy the Pages 5+ fixture with some of its objects redefined."""
    archive = zipfile.ZipFile(PAGES_2013)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out:
        for info in archive.infolist():
            out.writestr(info.filename, archive.read(info))
        out.writestr("Index/Redefined.iwa", _iwa_member(objects))
    return target


def _storage(text: str) -> bytes:
    """A TSWP.StorageArchive holding nothing but text."""
    return _len_field(3, text.encode())


def test_modern_headers_and_footers_are_read_from_the_page_master(tmp_path: Path):
    """Headers and footers belong to the page master a stretch of the document
    runs under, not to the text, so they are reached through the body storage's
    page master table. The fixture already names six such storages and leaves
    them all empty, so two of them are given text."""
    source = _redefined(
        tmp_path / "furniture.pages",
        [(4117, 2001, _storage("Running header")), (4120, 2001, _storage("Page 1"))],
    )
    doc = _backend(source).convert()

    by_label = {
        item.label: item.text
        for item in doc.texts
        if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER)
    }
    assert by_label[DocItemLabel.PAGE_HEADER] == "Running header"
    assert by_label[DocItemLabel.PAGE_FOOTER] == "Page 1"

    # Furniture stays out of the reading order.
    assert "Running header" not in doc.export_to_markdown()


def test_modern_footnotes_are_read_through_their_anchors(tmp_path: Path):
    """A footnote is anchored at one of the U+FFFC placeholders in the text, and
    the note it names holds a storage of its own. The fixture anchors a drawable
    at character 52; a footnote table is appended to the same body storage so the
    note path is exercised on real surrounding data."""
    body = next(
        obj
        for obj in _iwa_objects(PAGES_2013).values()
        if obj.identifier == 4038 and obj.message_type == 2001
    )
    note_table = _len_field(
        16, _len_field(1, _varint_field(1, 52) + _reference_field(2, 901))
    )
    source = _redefined(
        tmp_path / "footnote.pages",
        [
            (900, 2001, _storage("Do a lot of people really use iWork?")),
            (901, 2008, _reference_field(2, 900)),
            (4038, 2001, body.payload + note_table),
        ],
    )
    doc = _backend(source).convert()

    footnotes = [item.text for item in doc.texts if item.label == DocItemLabel.FOOTNOTE]
    assert footnotes == ["Do a lot of people really use iWork?"]
    assert _BODY_SENTENCE in doc.export_to_markdown()


def test_legacy_page_furniture_is_recovered_as_furniture():
    """The '09 fixture carries a real header, a real footer and three footnotes.
    They belong to the page rather than to the body flow, so they are labelled
    and kept in the furniture layer."""
    doc = _backend(PAGES_IWORK09_FORMATTED).convert()

    by_label: dict[DocItemLabel, list[str]] = {}
    for item in doc.texts:
        by_label.setdefault(item.label, []).append(item.text)

    assert by_label[DocItemLabel.PAGE_HEADER] == ["THIS IS SOME HEADER TEXT"]
    assert by_label[DocItemLabel.PAGE_FOOTER] == ["THIS IS SOME FOOTER TEXT"]

    footnotes = by_label[DocItemLabel.FOOTNOTE]
    assert "Footnote: What does APXL stand for?!?!?" in footnotes
    # The third footnote links to a URL partway through, so its runs differ and
    # it becomes an inline group rather than a single item.
    assert "www.oasis-open.org" in footnotes

    # Pages writes a first-page, an even-page and an odd-page variant of every
    # header and footer, so the same text must not be emitted three times.
    assert len(by_label[DocItemLabel.PAGE_HEADER]) == 1


def test_legacy_comments_are_recovered_and_linked_to_what_they_annotate():
    """An '09 comment lives in sf:annotations, outside the body, and names the
    sf:annotation-field in the text that it targets. Walking every sf:p would
    fold the comment text into the body flow instead."""
    doc = _backend(PAGES_IWORK09_COMMENTS).convert()

    notes = [item for item in doc.texts if item.content_layer == ContentLayer.NOTES]
    assert "comment about the APXL file here!!" in [item.text for item in notes]
    assert len(notes) == 3

    # The comment is attached to the paragraph holding the annotated text.
    annotated = {
        item.text: {ref.cref for ref in item.comments}
        for item in doc.texts
        if item.comments
    }
    target = next(text for text in annotated if "Keynote APXL file" in text)
    assert annotated[target]

    # And none of it leaks into the reading order.
    assert "comment about the APXL file" not in doc.export_to_markdown()


def test_modern_comments_are_read_through_the_highlight_they_cover(tmp_path: Path):
    """Pages 5 records a comment as a highlight over the words it is about, so
    the comment run table gives the stretch rather than a character in the text.
    Replies are comments in their own right and are followed as a chain."""
    comment_table = _len_field(
        23, _len_field(1, _varint_field(1, 5) + _reference_field(2, 913))
    )
    body = next(
        obj for obj in _iwa_objects(PAGES_2013).values() if obj.identifier == 4038
    )
    source = _redefined(
        tmp_path / "comments.pages",
        [
            (912, 212, _len_field(1, b"Ada Lovelace")),
            (911, 3056, _len_field(1, b"And a reply.")),
            (
                910,
                3056,
                _len_field(1, b"Is this the right title?")
                + _reference_field(3, 912)
                + _reference_field(4, 911),
            ),
            (913, 2013, _reference_field(1, 910)),
            (4038, 2001, body.payload + comment_table),
        ],
    )
    doc = _backend(source).convert()

    notes = [
        item.text for item in doc.texts if item.content_layer == ContentLayer.NOTES
    ]
    assert notes == [
        "[author: Ada Lovelace]: Is this the right title?",
        "And a reply.",
    ]

    # The highlight starts inside the first paragraph, so that is what the
    # comment is attached to.
    annotated = [item for item in doc.texts if item.comments]
    assert len(annotated) == 1
    assert annotated[0].text == "Sample pages document"


def test_legacy_hyperlink_survives_as_a_run_of_its_own():
    """An '09 link wraps the spans it covers, and href is one of the attributes
    iWork writes without its namespace. The link covers part of a footnote in
    this fixture, so the runs around it must not take the address with them."""
    doc = _backend(PAGES_IWORK09_FORMATTED).convert()

    linked = [item for item in doc.texts if item.hyperlink is not None]
    assert len(linked) == 1
    assert linked[0].text == "www.oasis-open.org"
    assert str(linked[0].hyperlink).startswith("http://www.oasis-open.org")


def test_modern_hyperlink_is_read_from_the_smart_field_table(tmp_path: Path):
    """Pages calls a hyperlink a smart field and keeps it in a run table of its
    own, so the link and the character styling are separate boundaries that both
    have to cut the paragraph."""
    link_table = _len_field(
        11, _len_field(1, _varint_field(1, 0) + _reference_field(2, 920))
    ) + _len_field(11, b"")
    body = next(
        obj for obj in _iwa_objects(PAGES_2013).values() if obj.identifier == 4038
    )
    source = _redefined(
        tmp_path / "link.pages",
        [
            (920, 2032, _len_field(2, b"https://www.apple.com/pages/")),
            (4038, 2001, body.payload + link_table),
        ],
    )
    doc = _backend(source).convert()

    linked = [item for item in doc.texts if item.hyperlink is not None]
    assert linked
    assert linked[0].text == "Sample pages document"
    assert str(linked[0].hyperlink) == "https://www.apple.com/pages/"


def test_superscript_and_subscript_are_read_from_the_character_style():
    """The script setting is one field of a character style in the modern
    container and one property-map entry in '09, and they use the same numbering:
    one raises the text, two lowers it. Check both against the styles the
    fixtures actually define."""
    namespace = "http://developer.apple.com/namespaces/sf"
    numbers = "http://developer.apple.com/namespaces/sfa"
    legacy = ET.fromstring(
        f"""<sf:characterstyle xmlns:sf="{namespace}" xmlns:sfa="{numbers}">
              <sf:property-map>
                <sf:superscript><sf:number sfa:number="2" sfa:type="i"/></sf:superscript>
              </sf:property-map>
            </sf:characterstyle>""".encode()
    )
    formatting = legacy_formatting(legacy)
    assert formatting is not None and formatting.script == Script.SUB

    # The '09 fixture defines a real superscript style; it must decode the same
    # way, and the modern container must read the same numbering from field 10.
    raw = zipfile.ZipFile(PAGES_IWORK09_FORMATTED).read("index.xml")
    styles = [
        legacy_formatting(element)
        for element in ET.fromstring(raw).iter(f"{{{namespace}}}characterstyle")
    ]
    assert any(
        style is not None and style.script == Script.SUPER for style in styles
    ), "fixture no longer defines a superscript character style"

    modern = iwa_formatting(_len_field(11, _varint_field(10, 1)))
    assert modern is not None and modern.script == Script.SUPER


def _current_cell(string_key: int) -> bytes:
    """One cell in the storage layout Pages 5.2 and later write.

    Byte 0 is the version and byte 1 the value type; the flags at byte 8 say
    which values follow from byte 12, and only the string key is set here.
    """
    cell = bytearray(16)
    cell[0] = 5
    cell[1] = 3  # textCellType
    cell[8:12] = (0x8).to_bytes(4, "little")  # a string key follows
    cell[12:16] = string_key.to_bytes(4, "little")
    return bytes(cell)


def _current_tile(rows: list[list[int]], wide: bool) -> bytes:
    """A TST.Tile whose rows use the current storage layout.

    Pages keeps the older buffer and offsets in place alongside the new ones, so
    a reader that took the first pair it found would still see the old cells.
    Only the new pair is written here, which is what makes the test meaningful.
    """
    payload = b""
    for index, keys in enumerate(rows):
        buffer = b"".join(_current_cell(key) for key in keys)
        offsets = b"".join(
            ((position * 16) // (4 if wide else 1)).to_bytes(2, "little", signed=True)
            for position in range(len(keys))
        )
        payload += _len_field(
            5,
            _varint_field(1, index)
            + _len_field(6, buffer)
            + _len_field(7, offsets)
            + (_varint_field(8, 1) if wide else b""),
        )
    return payload


@pytest.mark.parametrize("wide", [False, True])
def test_table_saved_by_a_recent_pages_release_is_read(tmp_path: Path, wide: bool):
    """Pages 5.2 moved a row's cell buffer and offsets to new fields, changed the
    cell layout, and began scaling the offsets by four. A reader that knows only
    the older layout finds no cells at all and drops the table silently.

    The fixture's own table is rewritten into the new layout, keeping its model,
    its geometry and the string list its cells reference by key."""
    source = _redefined(
        tmp_path / "recent.pages",
        [(4027, 6002, _current_tile([[1, 2, 3], [4, 5, 6]], wide))],
    )
    doc = _backend(source).convert()

    assert len(doc.tables) == 1
    by_position = {
        (cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
        for cell in doc.tables[0].data.table_cells
    }
    assert by_position[(0, 0)] == "Column one"
    assert by_position[(0, 2)] == "Column three"
    assert by_position[(1, 1)] == "Cell two"
