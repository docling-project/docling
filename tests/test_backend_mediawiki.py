# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import gzip
import tarfile
import tracemalloc
import zipfile
from io import BytesIO
from pathlib import Path
from random import Random
from xml.sax.saxutils import escape

import pytest
from docling_core.types.doc import CodeItem, DocItemLabel, FormulaItem, RichTableCell
from docling_core.types.doc.document import Script
from docling_core.types.doc.labels import CodeLanguageLabel
from typer.testing import CliRunner

from docling.backend.xml.mediawiki._tokenizer import Tokenizer
from docling.backend.xml.mediawiki._tokenizer_state import ParserError
from docling.backend.xml.mediawiki.export import read_wiki_pages
from docling.cli.main import _iter_input_paths_from_directory, app
from docling.datamodel.backend_options import MediaWikiBackendOptions
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import _DocumentConversionInput
from docling.document_converter import DocumentConverter, MediaWikiFormatOption
from docling.exceptions import DocumentLoadError
from docling.utils.mediawiki_input import probe_gzip

WIKITEXT = """A line with ''emphasis'', '''bold''', and [[Target page|a link]].

== Topic ==

# First
#* Nested
# Second

{| class="wikitable"
|+ Values
! Left !! Right
|-
| One || Two
|}

{{Box|text={{Nested|x}}}}<nowiki>[[literal]]</nowiki>
"""


def _export() -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <siteinfo><base>https://wiki.example/wiki/Main_Page</base></siteinfo>
  <page>
    <title>First page</title>
    <revision>
      <id>2</id><timestamp>2025-01-02T00:00:00Z</timestamp>
      <model>wikitext</model><text>{escape(WIKITEXT)}</text>
    </revision>
    <revision>
      <id>1</id><timestamp>2025-01-01T00:00:00Z</timestamp><text>Old content</text>
    </revision>
  </page>
  <page>
    <title>Second page</title>
    <revision>
      <content><role>other</role><text>Other slot</text></content>
      <content><role>main</role><model>wikitext</model><text>Page content</text></content>
    </revision>
  </page>
  <page>
    <title>Deleted page</title>
    <revision><id>2</id><text deleted="deleted">Deleted content</text></revision>
    <revision><id>1</id><text>Superseded content</text></revision>
  </page>
</mediawiki>""".encode()


def _archive(mode: str, members: dict[str, bytes]) -> bytes:
    with BytesIO() as payload:
        with tarfile.open(fileobj=payload, mode=mode) as archive:
            for name, data in members.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, BytesIO(data))
            link = tarfile.TarInfo("ignored.wiki")
            link.type = tarfile.SYMTYPE
            link.linkname = "/outside.wiki"
            archive.addfile(link)
        return payload.getvalue()


@pytest.mark.parametrize("as_stream", [False, True], ids=["path", "stream"])
@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("page.wiki", WIKITEXT.encode()),
        ("page.WIKITEXT", WIKITEXT.encode("utf-16")),
        ("page.wiki.gz", gzip.compress(WIKITEXT.encode())),
        ("export.xml", _export()),
        ("export.xml.gz", gzip.compress(_export())),
    ],
)
def test_mediawiki_preserves_structure(
    tmp_path: Path, as_stream: bool, name: str, contents: bytes
) -> None:
    source: Path | DocumentStream
    if as_stream:
        source = DocumentStream(name=name, stream=BytesIO(contents))
    else:
        source = tmp_path / name
        source.write_bytes(contents)

    result = DocumentConverter(allowed_formats=[InputFormat.MEDIAWIKI]).convert(source)
    assert result.input.format is InputFormat.MEDIAWIKI
    assert result.document.origin is not None
    assert result.document.origin.filename == name

    markdown = result.document.export_to_markdown()
    assert "*emphasis*" in markdown
    assert "**bold**" in markdown
    assert "Topic" in markdown
    assert "1. First\n    - Nested\n2. Second" in markdown
    assert "[[literal]]" in markdown
    assert "Box" not in markdown

    table = result.document.tables[0]
    assert table.caption_text(result.document) == "Values"
    assert [[cell.text for cell in row] for row in table.data.grid] == [
        ["Left", "Right"],
        ["One", "Two"],
    ]
    if name.startswith("export"):
        assert "https://wiki.example/wiki/Target_page" in markdown
        assert "First page" in markdown
        assert "Second page" in markdown
        assert markdown.index("First page") < markdown.index("Second page")
        assert "Page content" in markdown
        assert "Other slot" not in markdown
        assert "Old content" not in markdown
        assert "Deleted page" not in markdown
        assert "Superseded content" not in markdown


@pytest.mark.parametrize("as_stream", [False, True], ids=["path", "stream"])
@pytest.mark.parametrize(
    "name", ["article.wiki", "article.WIKITEXT", "article.wiki.gz"]
)
def test_wikitext_with_tar_signature(
    tmp_path: Path, as_stream: bool, name: str
) -> None:
    prefix = b"Article text.<!--".ljust(257, b" ")
    contents = prefix + b"ustar-->"
    if name.endswith(".gz"):
        contents = gzip.compress(contents)

    source: Path | DocumentStream
    if as_stream:
        source = DocumentStream(name=name, stream=BytesIO(contents))
    else:
        source = tmp_path / name
        source.write_bytes(contents)

    result = DocumentConverter().convert(source)
    assert result.status is ConversionStatus.SUCCESS
    assert result.input.format is InputFormat.MEDIAWIKI
    assert "Article text." in result.document.export_to_markdown()


def _converter(**options) -> DocumentConverter:
    return DocumentConverter(
        allowed_formats=[InputFormat.MEDIAWIKI],
        format_options={
            InputFormat.MEDIAWIKI: MediaWikiFormatOption(
                backend_options=MediaWikiBackendOptions(**options)
            )
        },
    )


@pytest.mark.parametrize(
    ("name", "mode"),
    [("batch.tar", "w"), ("batch.tar.gz", "w:gz"), ("batch.tgz", "w:gz")],
)
def test_mediawiki_tar_is_unsupported(name: str, mode: str) -> None:
    stream = DocumentStream(
        name=name, stream=BytesIO(_archive(mode, {"article.wiki": b"Article"}))
    )
    result = _converter().convert(stream, raises_on_error=False)
    assert result.status in {ConversionStatus.FAILURE, ConversionStatus.SKIPPED}
    assert result.input.format is not InputFormat.MEDIAWIKI


def test_mets_gbs_takes_precedence_over_wikitext_members() -> None:
    stream = DocumentStream(
        name="book.tar.gz",
        stream=BytesIO(
            _archive(
                "w:gz",
                {
                    "README.wiki": b"Archive information",
                    "mets.xml": b'<mets xmlns="http://www.loc.gov/METS/"/>',
                },
            )
        ),
    )
    conversion_input = _DocumentConversionInput(path_or_stream_iterator=[stream])
    assert conversion_input._guess_format(stream) is InputFormat.METS_GBS


def test_mediawiki_zip_members_produce_separate_documents() -> None:
    contents = BytesIO()
    with zipfile.ZipFile(contents, "w") as archive:
        archive.writestr("first.wiki", "First article")
        archive.writestr("second.xml", _export())
    stream = DocumentStream(name="batch.zip", stream=BytesIO(contents.getvalue()))
    first, second = _converter().convert_all([stream])

    assert first.document.export_to_markdown() == "First article"
    assert "First page" in second.document.export_to_markdown()
    assert "Second page" in second.document.export_to_markdown()


def test_mediawiki_detection_skips_gzip_probe_if_disallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(*args, **kwargs):
        pytest.fail("PDF-only format detection must skip the gzip prefix probe")

    monkeypatch.setattr("docling.utils.mediawiki_input.probe_gzip", unexpected_probe)
    source = DocumentStream(name="page.wiki.gz", stream=BytesIO(gzip.compress(b"text")))
    result = DocumentConverter(allowed_formats=[InputFormat.PDF]).convert(
        source, raises_on_error=False
    )
    assert result.status in {ConversionStatus.FAILURE, ConversionStatus.SKIPPED}


def test_mediawiki_directory_detects_xml_by_content(tmp_path: Path) -> None:
    export = tmp_path / "page.xml"
    export.write_bytes(_export())
    (tmp_path / "other.xml").write_text("<root>Other XML</root>", encoding="utf-8")
    assert list(
        _iter_input_paths_from_directory(tmp_path, [InputFormat.MEDIAWIKI])
    ) == [export]


def test_gzip_probe_preserves_position_and_leaves_the_rest_unread() -> None:
    class MeasuredStream(BytesIO):
        bytes_read = 0

        def read(self, size: int = -1, /) -> bytes:
            chunk = super().read(size)
            self.bytes_read += len(chunk)
            return chunk

    stream = MeasuredStream(gzip.compress(Random(0).randbytes(128 * 1024)))
    stream.seek(10)
    probe = probe_gzip(stream, "unknown.gz", mediawiki_allowed=True)

    assert probe.mediawiki is None
    assert not probe.is_tar
    assert stream.tell() == 10
    assert 0 < stream.bytes_read <= 64 * 1024


def test_mediawiki_cli_can_convert_a_directory(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "example.wiki.gz").write_bytes(gzip.compress(b"== Topic ==\nText"))
    output = tmp_path / "output"

    result = CliRunner().invoke(
        app, [str(source), "--from", "mediawiki", "--to", "md", "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    exported = list(output.glob("*.md"))
    assert len(exported) == 1
    assert "## Topic" in exported[0].read_text(encoding="utf-8")


def test_mediawiki_rejects_a_dtd() -> None:
    source = DocumentStream(
        name="unsafe.xml",
        stream=BytesIO(
            b'<!DOCTYPE mediawiki [<!ENTITY example "expanded">]>'
            b"<mediawiki><page><title>&example;</title></page></mediawiki>"
        ),
    )

    result = DocumentConverter().convert(source, raises_on_error=False)

    assert result.input.format is InputFormat.MEDIAWIKI
    assert result.status is ConversionStatus.FAILURE
    assert any("cannot contain a DTD" in error.error_message for error in result.errors)


@pytest.mark.parametrize(
    ("name", "contents", "option", "limit"),
    [
        ("large.wiki.gz", gzip.compress(b"A" * 48), "max_total_bytes", 32),
        ("large.wiki", b"A" * 48, "max_file_bytes", 32),
        ("large.wiki", ("é" * 48).encode(), "max_page_chars", 32),
        ("large.xml.gz", gzip.compress(_export()), "max_total_bytes", 32),
        ("large.xml", _export(), "max_page_chars", 32),
    ],
)
def test_mediawiki_size_limits(
    name: str, contents: bytes, option: str, limit: int
) -> None:
    source = DocumentStream(name=name, stream=BytesIO(contents))
    result = _converter(**{option: limit}).convert(source, raises_on_error=False)
    assert result.status is ConversionStatus.FAILURE
    assert any(option in error.error_message for error in result.errors)


class _UnfinishedXml(BytesIO):
    def read(self, size: int = -1, /) -> bytes:
        assert self.tell() == 0, "Reader should stop before looking for the closing tag"
        return super().read(size)


def test_xml_reader_checks_revision_text_before_reading_its_end() -> None:
    stream = _UnfinishedXml(b"<mediawiki><page><revision><text>0123456789")
    with pytest.raises(DocumentLoadError, match="max_page_chars"):
        list(read_wiki_pages(stream, max_page_chars=8))


@pytest.mark.parametrize(
    ("prefix", "field"),
    [
        (b"<siteinfo><base>", "base"),
        (b"<page><title>", "title"),
        (b"<page><revision><timestamp>", "timestamp"),
        (b"<page><revision><id>", "id"),
        (b"<page><revision><model>", "model"),
        (b"<page><revision><content><role>", "role"),
        (b"<page><revision><content><model>", "model"),
    ],
)
def test_xml_reader_checks_metadata_before_reading_its_end(
    prefix: bytes, field: str
) -> None:
    stream = _UnfinishedXml(b"<mediawiki>" + prefix + b"x" * 8193)
    with pytest.raises(DocumentLoadError, match=rf"'{field}'.*metadata limit"):
        list(read_wiki_pages(stream, max_page_chars=16))


@pytest.mark.parametrize("section", ["revision", "siteinfo"])
def test_xml_reader_discards_unknown_elements(section: str) -> None:
    ignored = b"<unknown/>" * 20_000
    siteinfo = b"<siteinfo><base>https://wiki.example/wiki/Main_Page</base>"
    siteinfo += (ignored if section == "siteinfo" else b"") + b"</siteinfo>"
    revision = b"<revision><text>Article text.</text>"
    revision += (ignored if section == "revision" else b"") + b"</revision>"
    stream = BytesIO(
        b"<mediawiki>"
        + siteinfo
        + b"<page><title>Article</title>"
        + revision
        + b"</page></mediawiki>"
    )

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        initial, _ = tracemalloc.get_traced_memory()
        (page,) = list(read_wiki_pages(stream, max_page_chars=16))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert peak - initial < 512 * 1024
    assert (page.title, page.wikitext, page.wiki_base) == (
        "Article",
        "Article text.",
        "https://wiki.example/wiki/",
    )


def test_infobox_and_common_templates_preserve_content() -> None:
    source = """{{Infobox person
| name = Alice Example
| birth_date = {{birth date|1900|2|3}}
| occupation = [[Writer]]
| known_for = {{ubl|First work|Second work}}
| image = portrait.jpg
}}
Lead {{lang|fr|bonjour}} and {{convert|12|km|mi}}.
<ref>{{cite web|title=Source title|date=2024}}</ref>
"""
    result = DocumentConverter().convert_string(source, format=InputFormat.MEDIAWIKI)
    markdown = result.document.export_to_markdown()
    assert "Alice Example" in markdown
    assert "1900-2-3" in markdown
    assert "Writer" in markdown
    assert "First work" in markdown and "Second work" in markdown
    assert "portrait.jpg" not in markdown
    assert "bonjour" in markdown and "12 km" in markdown
    assert "Source title" not in markdown
    assert result.document.tables


def test_reference_modes_and_empty_sections() -> None:
    from docling.datamodel.backend_options import MediaWikiBackendOptions
    from docling.document_converter import MediaWikiFormatOption

    source = """Lead<ref name="source"/> and more<ref name="source">A ''book''</ref>.
Math <sup>2</sup>.

== Empty ==
== Parent ==
=== Child ===
Text.
== References ==
{{reflist}}
"""
    for mode in ("omit", "inline", "list"):
        converter = DocumentConverter(
            format_options={
                InputFormat.MEDIAWIKI: MediaWikiFormatOption(
                    backend_options=MediaWikiBackendOptions(reference_mode=mode)
                )
            }
        )
        doc = converter.convert_string(source, format=InputFormat.MEDIAWIKI).document
        text = doc.export_to_markdown()
        assert "Empty" not in text
        assert "Parent" in text and "Child" in text
        if mode == "omit":
            assert "A *book*" not in text and "References" not in text
        elif mode == "inline":
            assert text.count("A *book*") == 2
            assert "References" not in text
        else:
            assert text.count("[1]") == 2
            assert text.count("A *book*") == 1
            assert "References" in text


def test_wiki_quote_state_does_not_consume_following_sections() -> None:
    source = "''[[The Daily Telegraph]]'''s text.\n\n== New section ==\n\nNormal text and ''real italics''."
    result = DocumentConverter().convert_string(source, format=InputFormat.MEDIAWIKI)
    markdown = result.document.export_to_markdown()
    assert "[*The Daily Telegraph*](The_Daily_Telegraph)'s text." in markdown
    assert "## New section" in markdown
    assert "Normal text and *real italics*." in markdown


def test_reference_groups_and_forward_definitions() -> None:
    from docling.backend.xml.mediawiki.render import render_wikitext

    source = """One<ref name="later"/> two<ref group="note">A note</ref>
three<ref name="later"/> unresolved<ref name="missing"/>.
== Notes ==
{{reflist|group=note}}
== References ==
<references><ref name="later">A source</ref></references>"""
    html = render_wikitext(source, reference_mode="list")
    assert html.count("<sup>[1]</sup>") == 2
    assert "<sup>[note 1]</sup>" in html
    assert "missing" not in html
    assert html.count("<li>A source</li>") == 1
    assert html.count("<li>A note</li>") == 1
    assert html.index("<h2>Notes</h2>") < html.index("<li>A note</li>")
    assert html.index("<h2>References</h2>") < html.index("<li>A source</li>")


def test_headings_before_lists_and_empty_descendants() -> None:
    from docling.backend.xml.mediawiki.render import render_wikitext

    html = render_wikitext(
        "== Parent ==\n=== Empty ===\n=== Populated ===\n* Item\n== Final =="
    )
    assert "Empty" not in html and "Final" not in html
    assert html.index("<h2>Parent</h2>") < html.index("<h3>Populated</h3>")
    assert html.index("<h3>Populated</h3>") < html.index("<ul>")


@pytest.mark.parametrize(
    ("preserve_bold", "preserve_italic", "expected"),
    [
        (True, True, "**onetwo three** and *four five* and ***six seven***"),
        (False, True, "onetwo three and *four five* and *six seven*"),
        (True, False, "**onetwo three** and four five and **six seven**"),
        (False, False, "onetwo three and four five and six seven"),
    ],
)
def test_adjacent_emphasis_is_merged(
    preserve_bold: bool, preserve_italic: bool, expected: str
) -> None:
    doc = (
        _converter(preserve_bold=preserve_bold, preserve_italic=preserve_italic)
        .convert_string(
            "<b>one</b><strong>two</strong> <b>three</b> and <i>four</i>\n"
            "<em>five</em> and <i><b>six</b></i> <b><em>seven</em></b>",
            format=InputFormat.MEDIAWIKI,
        )
        .document
    )
    assert doc.export_to_markdown() == expected
    assert sum(
        bool(item.formatting and item.formatting.bold) for item in doc.texts
    ) == (2 if preserve_bold else 0)
    assert sum(
        bool(item.formatting and item.formatting.italic) for item in doc.texts
    ) == (2 if preserve_italic else 0)


def test_emphasis_keeps_inline_boundaries_and_other_styles() -> None:
    from docling.backend.xml.mediawiki.render import render_wikitext

    source = (
        "<b>one</b> <span><b>two</b></span> <a href='/x'><b>link</b></a><br><b>next</b>"
    )
    assert render_wikitext(source) == (
        '<p><b>one two</b> <a href="/x"><b>link</b></a><br><b>next</b></p>'
    )
    doc = (
        _converter(preserve_bold=False, preserve_italic=False)
        .convert_string(
            ";Term\n:Definition\n\n<b><i><sup>raised</sup></i></b>",
            format=InputFormat.MEDIAWIKI,
        )
        .document
    )
    assert "Term" in doc.export_to_markdown()
    assert all(not item.formatting or not item.formatting.bold for item in doc.texts)
    raised = next(item for item in doc.texts if item.text == "raised")
    assert raised.formatting is not None
    assert raised.formatting.script is Script.SUPER


def test_mediawiki_formula_code_and_picture_caption() -> None:
    doc = (
        _converter()
        .convert_string(
            "Before <math>x^2</math> after.\n\n"
            "* <math>1+2</math>\n\n"
            '<syntaxhighlight lang="python">print(1)</syntaxhighlight>\n\n'
            "[[File:Sample.png|thumb|200px|A [[Topic|link]] <nowiki>|</nowiki> a caption]]",
            format=InputFormat.MEDIAWIKI,
        )
        .document
    )
    assert [item.text for item in doc.texts if isinstance(item, FormulaItem)] == [
        "x^2",
        "1+2",
    ]
    (code,) = [item for item in doc.texts if isinstance(item, CodeItem)]
    assert code.code_language is CodeLanguageLabel.PYTHON
    (picture,) = doc.pictures
    assert picture.image is None
    caption = picture.captions[0].resolve(doc)
    assert caption.label is DocItemLabel.CAPTION
    assert picture.caption_text(doc) == "A link | a caption"
    assert "Before $x^2$ after." in doc.export_to_markdown()


def test_mediawiki_keeps_formula_and_caption_owners_in_nested_content() -> None:
    doc = (
        _converter()
        .convert_string(
            "* Illustration [[File:Sample.png|thumb|A caption]]\n\n"
            "{|\n| <math>y^2</math>\n|}",
            format=InputFormat.MEDIAWIKI,
        )
        .document
    )

    (picture,) = doc.pictures
    assert picture.parent is not None
    assert picture.parent.resolve(doc).label is DocItemLabel.LIST_ITEM
    assert picture.caption_text(doc) == "A caption"
    assert doc.export_to_markdown().count("A caption") == 1

    (formula,) = [item for item in doc.texts if isinstance(item, FormulaItem)]
    assert formula.text == "y^2"
    cell = doc.tables[0].data.grid[0][0]
    assert isinstance(cell, RichTableCell)
    assert cell.ref.resolve(doc).children == [formula.get_ref()]


def test_inline_reference_budget_counts_each_reuse() -> None:
    one = '<ref name="a">A named reference</ref>'
    repeat = '<ref name="a"/>'
    converter = _converter(reference_mode="inline", max_inline_reference_bytes=60)
    result = converter.convert_string(one + repeat, format=InputFormat.MEDIAWIKI)
    assert result.document.export_to_markdown().count("A named reference") == 2

    result = converter.convert(
        DocumentStream(name="cited.wiki", stream=BytesIO((one + repeat * 2).encode())),
        raises_on_error=False,
    )
    assert result.status is ConversionStatus.FAILURE
    assert any(
        "max_inline_reference_bytes" in error.error_message for error in result.errors
    )


def test_inline_reference_budget_applies_to_the_whole_export() -> None:
    pages = "".join(
        f"<page><title>{title}</title><revision><text>"
        "&lt;ref&gt;Source for this page&lt;/ref&gt;</text></revision></page>"
        for title in ("First", "Second")
    )
    result = _converter(reference_mode="inline", max_inline_reference_bytes=40).convert(
        DocumentStream(
            name="references.xml",
            stream=BytesIO(f"<mediawiki>{pages}</mediawiki>".encode()),
        ),
        raises_on_error=False,
    )
    assert result.status is ConversionStatus.FAILURE
    assert any(
        "max_inline_reference_bytes" in error.error_message for error in result.errors
    )


def test_mediawiki_preserves_entity_text() -> None:
    doc = (
        _converter()
        .convert_string(
            "A &amp; B &#00065; &#x42; &unknown;", format=InputFormat.MEDIAWIKI
        )
        .document
    )
    assert [item.text for item in doc.texts] == ["A & B A B &unknown;"]


@pytest.mark.parametrize(
    "opener",
    [
        "<p>text\n",
        "<small>text\n",
        "<a ",
        "[https://example.com ",
        "{{{key|",
        "{|\n",
        "{{name|[[",
        "<!--",
    ],
)
def test_tokenizer_incomplete_markup_work_grows_linearly(opener: str) -> None:
    tokenizer = Tokenizer()
    tokenizer.tokenize(opener * 1000)
    work = tokenizer.work
    tokenizer.tokenize(opener * 2000)
    assert tokenizer.work <= 2 * work + 16


def test_tokenizer_bounds_ambiguous_markup_work() -> None:
    with pytest.raises(ParserError, match="work limit"):
        Tokenizer().tokenize("<p>text\n" * 1000 + "</p>")


def test_tokenizer_skips_unrelated_tag_closers() -> None:
    opener = "<p>text\n"
    closer = "</different>"
    tokenizer = Tokenizer()
    tokenizer.tokenize(opener * 1000 + closer)
    work = tokenizer.work
    tokenizer.tokenize(opener * 2000 + closer)
    assert tokenizer.work <= 2 * work


def test_mediawiki_preserves_pitch_class_vectors_in_tables() -> None:
    vector = "<0,0,0,0,0,0>"
    row = "|-\n| {{Small|{{Mono|" + vector + "}}}} <ref>Source</REF >\n"
    source = (
        "{|\n! Interval vector\n" + row * 64 + "|}\n\n== After the table ==\n\nTail."
    )
    doc = _converter().convert_string(source, format=InputFormat.MEDIAWIKI).document

    (table,) = doc.tables
    cell_texts = [cells[0].text for cells in table.data.grid]
    assert cell_texts == ["Interval vector"] + [vector] * 64
    assert any(item.text == "After the table" for item in doc.texts)
    assert any(item.text == "Tail." for item in doc.texts)


@pytest.mark.parametrize("marker", [":", ";", "*", "#"])
def test_mediawiki_parser_limits_recover_per_page(
    caplog: pytest.LogCaptureFixture, marker: str
) -> None:
    ambiguous = "<p>text\n" * 1000 + "</p>"
    pages = (
        ("First", "First paragraph."),
        ("Deep list", marker * 34 + " list content."),
        ("Ambiguous markup", ambiguous),
        ("Last", "Last paragraph with ''emphasis''."),
    )
    export = (
        "<mediawiki>"
        + "".join(
            f"<page><title>{title}</title><revision><text>{escape(body)}</text></revision></page>"
            for title, body in pages
        )
        + "</mediawiki>"
    )
    source = DocumentStream(
        name="pages.xml.gz", stream=BytesIO(gzip.compress(export.encode()))
    )
    result = _converter().convert(source)

    assert result.status is ConversionStatus.SUCCESS
    texts = [item.text for item in result.document.texts]
    assert (
        texts.index("First")
        < texts.index("Deep list")
        < texts.index("Ambiguous markup")
        < texts.index("Last")
    )
    assert "First paragraph." in texts
    assert marker * 2 + " list content." in texts
    assert ambiguous in texts
    assert "Last paragraph with *emphasis*." in result.document.export_to_markdown()
    assert "Ambiguous markup' exceeded the tokenizer work limit" in caplog.text
