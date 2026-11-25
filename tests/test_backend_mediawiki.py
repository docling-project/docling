# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import gzip
import tarfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from typer.testing import CliRunner

from docling.cli.main import app
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.document_converter import DocumentConverter
from docling.utils import mediawiki_input

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
        assert "Page content" in markdown
        assert "Other slot" not in markdown
        assert "Old content" not in markdown
        assert "Deleted page" not in markdown
        assert "Superseded content" not in markdown


@pytest.mark.parametrize(
    ("name", "mode"),
    [("mixed.tar", "w"), ("mixed.tar.gz", "w:gz"), ("mixed.tgz", "w:gz")],
)
def test_mediawiki_archive_preserves_member_order(name: str, mode: str) -> None:
    payload = _archive(
        mode,
        {
            "other.xml": b'<?xml version="1.0"?><other/>',
            "notes/Start.wiki": b"Before the dump",
            "dump.xml": _export(),
            "notes/End.wikitext": b"After the dump",
        },
    )
    stream = DocumentStream(name=name, stream=BytesIO(payload))
    result = DocumentConverter().convert(stream)
    assert result.input.format is InputFormat.MEDIAWIKI

    markdown = result.document.export_to_markdown()
    assert markdown.index("Before the dump") < markdown.index("First page")
    assert markdown.index("Second page") < markdown.index("After the dump")
    assert "1. First\n    - Nested\n2. Second" in markdown
    assert "ignored" not in markdown
    assert result.document.tables[0].caption_text(result.document) == "Values"


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


def test_mediawiki_bounds_decompression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mediawiki_input, "_MAX_ARCHIVE_BYTES", 12_000)
    source = DocumentStream(
        name="large.wiki.gz",
        stream=BytesIO(gzip.compress(b"A" * 20_000)),
    )

    result = DocumentConverter().convert(source, raises_on_error=False)

    assert result.input.format is InputFormat.MEDIAWIKI
    assert result.status is ConversionStatus.FAILURE
    assert any("12000-byte limit" in error.error_message for error in result.errors)


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


def test_adjacent_emphasis_is_merged() -> None:
    from docling.backend.xml.mediawiki.render import render_wikitext

    html = render_wikitext("<i>one</i><i>two</i> and <b>three</b><b>four</b>")
    assert "<i>onetwo</i>" in html
    assert "<b>threefour</b>" in html
