# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Block quotes in the HTML and Markdown backends: BlockQuoteGroup, cite and attribution."""

from io import BytesIO

from docling_core.types.doc import (
    BlockQuoteGroup,
    DocItemLabel,
    DoclingDocument,
    ListItem,
    SectionHeaderItem,
    TitleItem,
)
from docling_core.types.doc.document import ContentLayer
from pydantic import AnyUrl

from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def _convert_html(
    raw: bytes, options: HTMLBackendOptions | None = None
) -> DoclingDocument:
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw),
        format=InputFormat.HTML,
        backend=HTMLDocumentBackend,
        backend_options=options,
        filename="test.html",
    )
    return HTMLDocumentBackend(
        in_doc=in_doc, path_or_stream=BytesIO(raw), options=options
    ).convert()


def _convert_md(raw: bytes) -> DoclingDocument:
    in_doc = InputDocument(
        path_or_stream=BytesIO(raw),
        format=InputFormat.MD,
        backend=MarkdownDocumentBackend,
        filename="test.md",
    )
    return MarkdownDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(raw)).convert()


def _quotes(doc: DoclingDocument) -> list[BlockQuoteGroup]:
    return [group for group in doc.groups if isinstance(group, BlockQuoteGroup)]


def _text(doc: DoclingDocument, text: str):
    return next(item for item in doc.texts if item.text == text)


def _ancestors(doc: DoclingDocument, item) -> list:
    chain = []
    while item.parent is not None:
        item = item.parent.resolve(doc)
        chain.append(item)
    return chain


def test_html_blockquote_keeps_cite_and_footer_attribution():
    doc = _convert_html(
        b"""<html><body>
        <blockquote cite="https://example.org/source#p1">
          <p>Quoted.</p>
          <footer>Author, <cite>Work</cite></footer>
        </blockquote>
        <footer>Page footer</footer>
        </body></html>"""
    )
    (quote,) = _quotes(doc)
    assert str(quote.cite) == "https://example.org/source#p1"
    assert quote.caption_text(doc) == "Author, Work"

    attribution = quote.captions[0].resolve(doc)
    assert attribution.label == DocItemLabel.CAPTION
    assert attribution.content_layer == ContentLayer.BODY
    assert attribution.parent == quote.get_ref()
    assert quote.children[-1] == quote.captions[0]

    # A footer outside a quote is page furniture, as before.
    page_footer = next(text for text in doc.texts if text.text == "Page footer")
    assert page_footer.content_layer == ContentLayer.FURNITURE

    html = doc.export_to_html()
    assert '<blockquote cite="https://example.org/source#p1">' in html
    assert "<figcaption>" in html


def test_html_figure_blockquote_figcaption_is_the_attribution():
    doc = _convert_html(
        b"<html><body><figure><blockquote><p>Quoted.</p></blockquote>"
        b"<figcaption>Author</figcaption></figure></body></html>"
    )
    (quote,) = _quotes(doc)
    assert quote.caption_text(doc) == "Author"
    assert quote.captions[0].resolve(doc).parent == quote.get_ref()
    assert [text.text for text in doc.texts] == ["Quoted.", "Author"]


def test_html_relative_cite_resolves_against_source_uri_or_is_dropped():
    raw = b'<html><body><blockquote cite="/acts/1"><p>Quoted.</p></blockquote></body></html>'
    (quote,) = _quotes(_convert_html(raw))
    assert quote.cite is None

    options = HTMLBackendOptions(source_uri=AnyUrl("https://example.org/judgments/5"))
    (quote,) = _quotes(_convert_html(raw, options))
    assert str(quote.cite) == "https://example.org/acts/1"


def test_html_blockquote_inside_list_item_is_a_child_of_the_item():
    doc = _convert_html(
        b"<html><body><ul><li>Submission:<blockquote><p>Quoted.</p></blockquote></li>"
        b"<li>Second.</li></ul></body></html>"
    )
    (quote,) = _quotes(doc)
    item = quote.parent.resolve(doc)
    assert isinstance(item, ListItem)
    assert item.text == "Submission:"
    assert quote.children[0].resolve(doc).text == "Quoted."


def test_html_heading_inside_blockquote_stays_in_the_quote():
    doc = _convert_html(
        b"<html><body><h1>Doc</h1><h2>Section</h2><p>Intro.</p>"
        b"<blockquote><h3>Quoted heading</h3><p>Quoted body.</p>"
        b"<h1>Quoted title</h1><p>More.</p></blockquote>"
        b"<p>Author text.</p></body></html>"
    )
    (quote,) = _quotes(doc)
    section = _text(doc, "Section")
    assert quote.parent == section.get_ref()

    # The quoted headings sit inside the quote, below their own section groups
    heading = _text(doc, "Quoted heading")
    assert isinstance(heading, SectionHeaderItem)
    assert heading.level == 2
    assert quote in _ancestors(doc, heading)
    assert heading.children[0].resolve(doc).text == "Quoted body."
    # An <h1> inside a quote is quoted text, not a second document title
    quoted_title = _text(doc, "Quoted title")
    assert isinstance(quoted_title, SectionHeaderItem)
    assert quoted_title.level == 1
    assert quoted_title.parent == quote.get_ref()
    assert [text.text for text in doc.texts if isinstance(text, TitleItem)] == ["Doc"]

    # The quote closes without closing the enclosing section
    assert _text(doc, "Author text.").parent == section.get_ref()
    assert (
        "> ### Quoted heading\n>\n> Quoted body.\n>\n> ## Quoted title\n>\n> More.\n\n"
        "Author text." in doc.export_to_markdown()
    )


def test_html_first_heading_inside_blockquote_keeps_the_quote_in_the_body():
    doc = _convert_html(
        b"<html><body><blockquote><h2>Quoted heading</h2><p>Quoted body.</p>"
        b"</blockquote><p>Author text.</p></body></html>"
    )
    (quote,) = _quotes(doc)
    assert quote.content_layer == ContentLayer.BODY
    assert quote.parent == doc.body.get_ref()
    heading = quote.children[0].resolve(doc)
    assert isinstance(heading, SectionHeaderItem)
    assert heading.level == 1
    assert _text(doc, "Author text.").parent == doc.body.get_ref()
    assert doc.export_to_markdown().startswith("> ## Quoted heading\n>\n> Quoted body.")


def test_html_blockquote_in_table_cell_is_one_markdown_line():
    doc = _convert_html(
        b"<html><body><table><tr><td>(1)</td><td>Text:<blockquote>"
        b"<p>First para.</p><p>Second para.</p></blockquote></td></tr></table>"
        b"</body></html>"
    )
    (quote,) = _quotes(doc)
    assert [child.resolve(doc).text for child in quote.children] == [
        "First para.",
        "Second para.",
    ]
    markdown = doc.export_to_markdown()
    assert "| Text: > First para. Second para. |" in " ".join(markdown.split())
    assert "> >" not in markdown


def test_markdown_heading_inside_blockquote_is_not_a_title():
    doc = _convert_md(b"# Doc\n\n> # Quoted heading\n>\n> Quoted body.\n\nAfter.\n")
    (quote,) = _quotes(doc)
    heading = quote.children[0].resolve(doc)
    assert isinstance(heading, SectionHeaderItem)
    assert (heading.text, heading.level) == ("Quoted heading", 1)
    assert [text.text for text in doc.texts if isinstance(text, TitleItem)] == ["Doc"]
    assert (
        "> ## Quoted heading\n>\n> Quoted body.\n\nAfter." in doc.export_to_markdown()
    )


def test_markdown_blockquote_nests_lists_and_quotes():
    doc = _convert_md(
        b"Intro.\n\n> Quoted one.\n>\n> - item a\n> - item b\n>\n> > Nested.\n\nAfter.\n"
    )
    outer, inner = _quotes(doc)
    assert outer.parent == doc.body.get_ref()
    assert inner.parent == outer.get_ref()
    children = [child.resolve(doc) for child in outer.children]
    assert children[0].text == "Quoted one."
    assert children[-1] is inner

    markdown = doc.export_to_markdown()
    assert "> Quoted one.\n>\n> - item a\n> - item b\n>\n> > Nested." in markdown
    assert markdown.startswith("Intro.")
    assert markdown.endswith("After.")
