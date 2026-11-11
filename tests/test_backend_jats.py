# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Test module for the JATS backend parser.

Third-party test data notice:
The JATS source files in tests/data/jats/sources/ and derived groundtruth files
are based on open-access articles:
1. ptag100.xml - CC BY 4.0 (Choi et al. 2026, https://doi.org/10.1093/ptep/ptag100, SCOAP3)
2. elife-56337.nxml - CC0 1.0 (Wolf et al. 2020, PMC7289599, https://doi.org/10.7554/eLife.56337)
3. pone.0234687.nxml - CC0 1.0 (Ribeiro-Filho et al. 2020, PMC7302504, https://doi.org/10.1371/journal.pone.0234687)
4. pntd.0008301.nxml - CC0 1.0 (Burgert-Brucker et al. 2020, PMC7289444, https://doi.org/10.1371/journal.pntd.0008301)
"""

import re
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import (
    CodeLanguageLabel,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    GroupLabel,
    ListItem,
    PictureItem,
    RichTableCell,
    TextItem,
)
from docling_core.types.doc.document import Script
from PIL import Image

from docling.backend.xml.jats_backend import JatsDocumentBackend
from docling.datamodel.backend_options import JatsBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import (
    DocumentConverter,
    FormatOption,
    XMLJatsFormatOption,
)

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA

F_ITALIC = (False, True, False, False, Script.BASELINE)
F_BOLD = (True, False, False, False, Script.BASELINE)
F_BOLD_ITALIC = (True, True, False, False, Script.BASELINE)
F_UNDERLINE = (False, False, True, False, Script.BASELINE)
F_STRIKE = (False, False, False, True, Script.BASELINE)
F_SUB = (False, False, False, False, Script.SUB)
F_SUPER = (False, False, False, False, Script.SUPER)

JATS_WRAPPER = (
    '<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving'
    ' and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">\n'
    '<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">\n'
    "{content}\n</article>\n"
)


def get_jats_paths():
    directory = Path(__file__).parent / "data" / "jats"
    nxml_files = list(directory.rglob("*.nxml"))
    nxml_stems = {p.stem for p in nxml_files}
    # Avoid running duplicate conversions of the same content.
    xml_files = [p for p in directory.rglob("*.xml") if p.stem not in nxml_stems]
    return sorted(nxml_files + xml_files)


def get_converter(backend_options: JatsBackendOptions | None = None):
    format_options: dict[InputFormat, FormatOption] | None = (
        {
            InputFormat.XML_JATS: XMLJatsFormatOption(
                backend_options=backend_options,
            )
        }
        if backend_options is not None
        else None
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.XML_JATS], format_options=format_options
    )


def _formatting_tuple(item) -> tuple:
    """Compact, comparable view of a text item's formatting."""
    f = item.formatting
    if f is None:
        return (item.label, item.text, None)
    return (
        item.label,
        item.text,
        (f.bold, f.italic, f.underline, f.strikethrough, f.script),
    )


def _inline_group_items(doc: DoclingDocument) -> list[list]:
    """Return the resolved child items of every INLINE group, in document order."""
    return [
        [child.resolve(doc) for child in group.children]
        for group in doc.groups
        if group.label == GroupLabel.INLINE
    ]


def _make_test_image(
    path: Path,
    size: tuple[int, int] = (7, 5),
    color: tuple[int, int, int] = (255, 0, 0),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)
    return path


@pytest.fixture
def local_fetch_converter() -> DocumentConverter:
    return get_converter(JatsBackendOptions(fetch_images=True, enable_local_fetch=True))


def write_jats_body(path: Path, body: str) -> None:
    path.write_text(
        JATS_WRAPPER.format(content=f"<body>{body}</body>"), encoding="utf-8"
    )


def get_pictures(doc: DoclingDocument) -> list[PictureItem]:
    return [item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)]


def convert_jats_article_meta(article_meta: str, body: str = "") -> DoclingDocument:
    xml = JATS_WRAPPER.format(
        content=f"<front><article-meta>{article_meta}</article-meta></front>"
        + (f"<body>{body}</body>" if body else "")
    )
    stream = DocumentStream(name="article-meta-test.nxml", stream=BytesIO(xml.encode()))
    return get_converter().convert(stream).document


def convert_jats_body(
    body_content: str, backend_options: JatsBackendOptions | None = None
) -> DoclingDocument:
    xml = JATS_WRAPPER.format(
        content=f"<front><article-meta><title-group><article-title>Body Test</article-title></title-group></article-meta></front><body>{body_content}</body>"
    )
    stream = DocumentStream(name="body-test.nxml", stream=BytesIO(xml.encode()))
    return get_converter(backend_options).convert(stream).document


def convert_jats_contribs(contribs: str, affiliations: str = "") -> DoclingDocument:
    return convert_jats_article_meta(
        f"""
      <title-group><article-title>Author Variant Test</article-title></title-group>
      <contrib-group>{contribs}</contrib-group>
      {affiliations}
"""
    )


def test_jats_structured_abstract_sections_are_preserved():
    """Each <sec> inside a structured abstract becomes its own heading + paragraph."""
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Structured Abstract Test</article-title></title-group>
      <abstract>
        <sec>
          <title>Background</title>
          <p>Background text.</p>
        </sec>
        <sec>
          <title>Methods</title>
          <p>Methods text.</p>
        </sec>
      </abstract>
"""
    )

    md = doc.export_to_markdown()
    # Top-level abstract heading
    assert "## Abstract" in md
    # Section titles rendered as sub-headings (not inlined into text)
    assert "### Background" in md
    assert "### Methods" in md
    # Section content appears as separate paragraphs, not prefixed with title
    assert "Background text." in md
    assert "Methods text." in md
    # Old flat format must NOT appear
    assert "Background: Background text." not in md
    assert "Methods: Methods text." not in md

    # Verify document structure: two headings parented under the Abstract heading
    headings = [
        item
        for item, _level in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.SECTION_HEADER
    ]
    heading_texts = [h.text for h in headings]
    assert "Background" in heading_texts
    assert "Methods" in heading_texts


def test_jats_plain_abstract_styling_is_preserved():
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Abstract Styling Test</article-title></title-group>
      <abstract><p>We study <italic>Homo sapiens</italic>, <bold>key <italic>results</italic></bold>, H<sub>2</sub>O at 10<sup>3</sup> Pa, <underline>u</underline>, <strike>old</strike>.</p></abstract>
"""
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    # Each styled span remains a separate inline run; runs keep their boundary whitespace.
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (DocItemLabel.TEXT, "We study ", None),
        (
            DocItemLabel.TEXT,
            "Homo sapiens",
            F_ITALIC,
        ),
        (DocItemLabel.TEXT, ", ", None),
        (
            DocItemLabel.TEXT,
            "key ",
            F_BOLD,
        ),
        (
            DocItemLabel.TEXT,
            "results",
            F_BOLD_ITALIC,
        ),
        (DocItemLabel.TEXT, ", H", None),
        (DocItemLabel.TEXT, "2", F_SUB),
        (DocItemLabel.TEXT, "O at 10", None),
        (
            DocItemLabel.TEXT,
            "3",
            F_SUPER,
        ),
        (DocItemLabel.TEXT, " Pa, ", None),
        (
            DocItemLabel.TEXT,
            "u",
            F_UNDERLINE,
        ),
        (DocItemLabel.TEXT, ", ", None),
        (
            DocItemLabel.TEXT,
            "old",
            F_STRIKE,
        ),
        (DocItemLabel.TEXT, ".", None),
    ]


def test_jats_abstract_styling_preserves_inline_runs():
    """Styled spans become their own inline runs without fusion or loss."""

    def abstract_runs(paragraph: str) -> list:
        doc = convert_jats_article_meta(
            f"""
      <title-group><article-title>Spacing Test</article-title></title-group>
      <abstract><p>{paragraph}</p></abstract>
"""
        )
        groups = _inline_group_items(doc)
        assert len(groups) == 1
        return [_formatting_tuple(item) for item in groups[0]]

    T = DocItemLabel.TEXT

    cases = [
        # (source paragraph, expected inline runs)
        (
            "conditions <italic>in vitro</italic>.",
            [(T, "conditions ", None), (T, "in vitro", F_ITALIC), (T, ".", None)],
        ),
        (
            "cultures of <italic>Y. pestis</italic>, and <italic>E. coli</italic>.",
            [
                (T, "cultures of ", None),
                (T, "Y. pestis", F_ITALIC),
                (T, ", and ", None),
                (T, "E. coli", F_ITALIC),
                (T, ".", None),
            ],
        ),
        (
            "CO<sub>2</sub> and (CO<sub>2</sub>e) at 10<sup>3</sup> Pa",
            [
                (T, "CO", None),
                (T, "2", F_SUB),
                (T, " and (CO", None),
                (T, "2", F_SUB),
                (T, "e) at 10", None),
                (T, "3", F_SUPER),
                (T, " Pa", None),
            ],
        ),
        (
            "rate (ECM))<sup>-1</sup>, lactation<sup>-1</sup>.",
            [
                (T, "rate (ECM))", None),
                (T, "-1", F_SUPER),
                (T, ", lactation", None),
                (T, "-1", F_SUPER),
                (T, ".", None),
            ],
        ),
        (
            "(<italic>term</italic>) and [<bold>other</bold>];",
            [
                (T, "(", None),
                (T, "term", F_ITALIC),
                (T, ") and [", None),
                (T, "other", F_BOLD),
                (T, "];", None),
            ],
        ),
        # Newlines in source XML become spaces; other whitespace is kept
        # as authored (no collapsing, no stripping of interior blanks).
        (
            "a  <italic>b\nc</italic>\n   d.",
            [(T, "a  ", None), (T, "b c", F_ITALIC), (T, " d.", None)],
        ),
    ]
    for paragraph, expected in cases:
        assert abstract_runs(paragraph) == expected, f"{paragraph!r}"


def test_jats_inline_runs_drop_xml_indentation_but_keep_same_line_whitespace():
    paragraph = """<p>Alpha  beta <italic>Yersinia pestis</italic> is
              critical  here.</p>"""
    documents = [
        convert_jats_article_meta(
            f"""
      <title-group><article-title>Abstract Spacing Test</article-title></title-group>
      <abstract>{paragraph}</abstract>
"""
        ),
        convert_jats_body(f"<sec><title>Body Spacing Test</title>{paragraph}</sec>"),
    ]

    expected = [
        (DocItemLabel.TEXT, "Alpha  beta ", None),
        (DocItemLabel.TEXT, "Yersinia pestis", F_ITALIC),
        (DocItemLabel.TEXT, " is critical  here.", None),
    ]
    for doc in documents:
        groups = _inline_group_items(doc)
        assert len(groups) == 1
        assert [_formatting_tuple(item) for item in groups[0]] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" leading  space", " leading  space"),
        ("trailing  space ", "trailing  space "),
        ("a  b", "a  b"),
        ("a \n   b", "a b"),
        ("a\n   \n b", "a b"),
        ("\n", " "),
    ],
)
def test_jats_xml_line_break_normalization_preserves_same_line_whitespace(
    text: str, expected: str
):
    assert JatsDocumentBackend._normalize_xml_line_breaks(text) == expected


def test_jats_structured_abstract_styling_is_preserved():
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Structured Abstract Styling Test</article-title></title-group>
      <abstract>
        <sec>
          <title>Background</title>
          <p><italic>Bg</italic> text.</p>
        </sec>
        <sec>
          <title>Methods</title>
          <p><bold>M</bold> text.</p>
        </sec>
      </abstract>
"""
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 2
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (
            DocItemLabel.TEXT,
            "Bg",
            F_ITALIC,
        ),
        (DocItemLabel.TEXT, " text.", None),
    ]
    assert [_formatting_tuple(item) for item in groups[1]] == [
        (
            DocItemLabel.TEXT,
            "M",
            F_BOLD,
        ),
        (DocItemLabel.TEXT, " text.", None),
    ]


def test_jats_plain_abstract_without_styling_stays_a_single_text_item():
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Plain Abstract Test</article-title></title-group>
      <abstract><p>Plain abstract.</p></abstract>
"""
    )

    # no emphasis anywhere → a single TEXT item, no inline group
    assert _inline_group_items(doc) == []
    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert texts == ["Plain abstract."]


def test_jats_plain_footnote_without_styling_stays_a_single_text_item():
    doc = convert_jats_body(
        """
        <sec>
          <title>Plain Footnote Test</title>
          <fn-group><fn><p>Plain footnote.</p></fn></fn-group>
        </sec>
        """
    )

    assert _inline_group_items(doc) == []
    footnotes = [t.text for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert footnotes == ["Plain footnote."]


@pytest.mark.parametrize(
    "notes",
    [
        (),
        ("Author contribution marker.",),
        ("Author contribution marker.", "Conflict statement marker."),
    ],
    ids=["none", "single", "multiple"],
)
def test_jats_author_footnotes_are_furniture(notes: tuple[str, ...]):
    footnote_xml = "".join(f"<fn><p>{note}</p></fn>" for note in notes)
    author_notes = f"<author-notes>{footnote_xml}</author-notes>" if notes else ""
    doc = convert_jats_article_meta(
        f"""
        <title-group><article-title>Author Footnote Test</article-title></title-group>
        {author_notes}
        """,
        "<p>Body text.</p>",
    )

    footnotes = [t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert [t.text for t in footnotes] == list(notes)
    assert all(t.content_layer == ContentLayer.FURNITURE for t in footnotes)
    assert _furniture_statements(doc) == [[(note, None)] for note in notes]
    body = ["# Author Footnote Test", "Body text."]
    assert doc.export_to_markdown() == "\n\n".join(body)
    assert doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ) == "\n\n".join([*body, *notes])


def test_jats_author_footnote_preserves_formatting_link_formula_and_paragraphs():
    doc = convert_jats_article_meta(
        """
        <title-group><article-title>Author Footnote Test</article-title></title-group>
        <author-notes>
          <fn><label>1</label><p>See <italic><ext-link
            xlink:href="https://example.org/statement">disclosure</ext-link></italic>
            and <inline-formula><tex-math>x+1</tex-math></inline-formula>.</p>
            <p>Further details.</p></fn>
        </author-notes>
        """
    )

    footnotes = [t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    disclosures = [t for t in footnotes if t.text == "disclosure"]
    assert len(disclosures) == 1
    disclosure = disclosures[0]
    assert disclosure.formatting is not None and disclosure.formatting.italic
    assert str(disclosure.hyperlink) == "https://example.org/statement"
    assert all(t.content_layer == ContentLayer.FURNITURE for t in footnotes)
    formula = next(t for t in doc.texts if t.label == DocItemLabel.FORMULA)
    assert formula.text == "x+1"
    assert formula.content_layer == ContentLayer.FURNITURE
    assert disclosure.parent is not None and disclosure.parent == formula.parent
    group = disclosure.parent.resolve(doc)
    assert group.label == GroupLabel.INLINE
    assert group.content_layer == ContentLayer.FURNITURE
    assert doc.export_to_markdown() == "# Author Footnote Test"
    markdown = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
    assert "Author Notes" not in markdown
    assert "1 See" in markdown
    assert "https://example.org/statement" in markdown
    assert "$x+1$. Further details." in markdown


def test_jats_author_footnotes_share_furniture_with_article_statements():
    doc = convert_jats_article_meta(
        """
        <title-group><article-title>Article metadata</article-title></title-group>
        <author-notes>
          <fn><p>First author note.</p></fn>
          <fn><p>Second author note.</p></fn>
        </author-notes>
        <permissions><license><license-p>License statement.</license-p></license></permissions>
        <funding-group><funding-statement>Funding statement.</funding-statement></funding-group>
        """,
        "<p>Body text.</p><fn-group><fn><p>Body footnote.</p></fn></fn-group>",
    )

    assert _furniture_statements(doc) == [
        [("First author note.", None)],
        [("Second author note.", None)],
        [("License statement.", None)],
        [("Funding statement.", None)],
    ]
    body_footnote = next(t for t in doc.texts if t.text == "Body footnote.")
    assert body_footnote.label == DocItemLabel.FOOTNOTE
    assert body_footnote.content_layer == ContentLayer.BODY


_FOOTNOTE = (
    "<label>1</label>{sep}<p>See <italic>ibid</italic> for <bold>details</bold>.</p>"
)


@pytest.mark.parametrize("sep", ["\n              ", ""], ids=["indented", "compact"])
def test_jats_footnote_styling_is_preserved(sep: str):
    fn = _FOOTNOTE.format(sep=sep)
    doc = convert_jats_body(
        f"<sec><title>Footnote Styling Test</title><fn-group><fn>{fn}</fn></fn-group></sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    # The punctuation after the bold run remains separate and unformatted. The label
    # and the paragraph are separate blocks: a " " run keeps them apart.
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (DocItemLabel.FOOTNOTE, "1", None),
        (DocItemLabel.FOOTNOTE, " ", None),
        (DocItemLabel.FOOTNOTE, "See ", None),
        (
            DocItemLabel.FOOTNOTE,
            "ibid",
            F_ITALIC,
        ),
        (DocItemLabel.FOOTNOTE, " for ", None),
        (
            DocItemLabel.FOOTNOTE,
            "details",
            F_BOLD,
        ),
        (DocItemLabel.FOOTNOTE, ".", None),
    ]
    assert doc.export_to_markdown().endswith("- 1 See *ibid* for **details**.")


def test_jats_footnote_group_keeps_external_link_target():
    xml = JATS_WRAPPER.format(
        content=(
            "<front><article-meta><title-group><article-title>Back Matter Test"
            "</article-title></title-group></article-meta></front>"
            "<body><p>Body text.</p></body>"
            "<back><fn-group><fn><p>See "
            '<ext-link ext-link-type="uri" xlink:href="https://example.org/data">'
            "the data</ext-link> now.</p></fn></fn-group></back>"
        )
    )
    stream = DocumentStream(name="back-fn-test.nxml", stream=BytesIO(xml.encode()))
    doc = get_converter().convert(stream).document

    links = [t for t in doc.texts if t.text == "the data"]
    assert len(links) == 1
    assert links[0].label == DocItemLabel.FOOTNOTE
    assert str(links[0].hyperlink) == "https://example.org/data"
    for text in ("See ", " now."):
        run = next(t for t in doc.texts if t.text == text)
        assert run.hyperlink is None
    assert "https://example.org/data" in doc.export_to_markdown()


def test_jats_nested_lists_are_preserved():
    doc = convert_jats_body(
        """
        <sec>
          <title>List Test</title>
          <list>
            <list-item>
              <p>Item 1</p>
              <list>
                <list-item>
                  <p>Subitem A</p>
                </list-item>
              </list>
            </list-item>
          </list>
        </sec>
        """
    )

    # Both items must appear in the rendered output.
    md = doc.export_to_markdown()
    assert "- Item 1" in md
    assert "Subitem A" in md

    # Verify document structure
    list_items = [
        item
        for item, _level in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM
    ]
    assert len(list_items) == 2

    outer_item = next(item for item in list_items if item.text == "Item 1")
    sub_item = next(item for item in list_items if item.text == "Subitem A")

    sub_item_parent = sub_item.parent.resolve(doc)
    assert sub_item_parent.label == GroupLabel.LIST
    assert sub_item_parent.parent == outer_item.get_ref()


def _list_items(doc: DoclingDocument) -> list[ListItem]:
    return [item for item, _ in doc.iterate_items() if isinstance(item, ListItem)]


@pytest.mark.parametrize(
    ("list_type", "enumerated", "markdown"),
    [
        ("order", True, "1. first\n2. second"),
        ("alpha-lower", True, "1. first\n2. second"),
        ("alpha-upper", True, "1. first\n2. second"),
        ("roman-lower", True, "1. first\n2. second"),
        ("roman-upper", True, "1. first\n2. second"),
        ("bullet", False, "- first\n- second"),
        ("simple", False, "- first\n- second"),
        (None, False, "- first\n- second"),
    ],
)
def test_jats_list_type_sets_enumerated(list_type, enumerated, markdown):
    attr = f' list-type="{list_type}"' if list_type else ""
    doc = convert_jats_body(
        f"""
        <sec>
          <title>Steps</title>
          <list{attr}>
            <list-item><p>first</p></list-item>
            <list-item><p>second</p></list-item>
          </list>
        </sec>
        """
    )

    items = _list_items(doc)
    assert [item.text for item in items] == ["first", "second"]
    assert [item.enumerated for item in items] == [enumerated, enumerated]
    assert [item.marker for item in items] == ["", ""]
    assert markdown in doc.export_to_markdown()


def test_jats_nested_list_takes_its_own_list_type():
    doc = convert_jats_body(
        """
        <sec>
          <title>Steps</title>
          <list list-type="bullet">
            <list-item>
              <p>outer</p>
              <list list-type="order">
                <list-item><p>inner one</p></list-item>
                <list-item><p>inner two</p></list-item>
              </list>
            </list-item>
          </list>
        </sec>
        """
    )

    enumerated = {item.text: item.enumerated for item in _list_items(doc)}
    assert enumerated == {"outer": False, "inner one": True, "inner two": True}


def test_jats_list_item_label_becomes_marker():
    doc = convert_jats_body(
        """
        <sec>
          <title>Conditions</title>
          <list list-type="simple">
            <list-item><label>(a)</label><p>first</p></list-item>
            <list-item><label>(b)</label><p>second</p></list-item>
          </list>
        </sec>
        """
    )

    items = _list_items(doc)
    assert [(item.marker, item.text) for item in items] == [
        ("(a)", "first"),
        ("(b)", "second"),
    ]
    assert [item.enumerated for item in items] == [False, False]
    assert "(a) first" in doc.export_to_markdown()


def _convert_list(items: str, list_type: str = "bullet") -> DoclingDocument:
    return convert_jats_body(
        f"""
        <sec>
          <title>Steps</title>
          <list list-type="{list_type}">{items}</list>
        </sec>
        """
    )


def test_jats_list_item_keeps_inline_formatting_and_links():
    doc = _convert_list(
        "<list-item><p>Use <bold>bold</bold>, <italic>italic</italic> and"
        ' <ext-link ext-link-type="uri" xlink:href="https://example.org">a link'
        "</ext-link>.</p></list-item>"
    )

    [item] = _list_items(doc)
    assert item.text == ""
    [group] = [child.resolve(doc) for child in item.children]
    assert group.label == GroupLabel.INLINE
    runs = [child.resolve(doc) for child in group.children]
    assert [_formatting_tuple(run) for run in runs] == [
        (DocItemLabel.TEXT, "Use ", None),
        (DocItemLabel.TEXT, "bold", F_BOLD),
        (DocItemLabel.TEXT, ", ", None),
        (DocItemLabel.TEXT, "italic", F_ITALIC),
        (DocItemLabel.TEXT, " and ", None),
        (DocItemLabel.TEXT, "a link", None),
        (DocItemLabel.TEXT, ".", None),
    ]
    assert str(runs[5].hyperlink) == "https://example.org/"
    assert (
        "- Use **bold**, *italic* and [a link](https://example.org/)."
        in doc.export_to_markdown()
    )


def test_jats_list_item_with_one_styled_run_stays_a_single_item():
    doc = _convert_list("<list-item><p><bold>All bold</bold></p></list-item>")

    [item] = _list_items(doc)
    assert _formatting_tuple(item) == (DocItemLabel.LIST_ITEM, "All bold", F_BOLD)
    assert item.children == []
    assert "- **All bold**" in doc.export_to_markdown()


def test_jats_list_item_keeps_inline_formula():
    doc = _convert_list(
        "<list-item><p>The relation <inline-formula><tex-math>$$E=mc^2$$"
        "</tex-math></inline-formula> holds.</p></list-item>"
    )

    [item] = _list_items(doc)
    [group] = [child.resolve(doc) for child in item.children]
    assert group.label == GroupLabel.INLINE
    assert [_formatting_tuple(child.resolve(doc)) for child in group.children] == [
        (DocItemLabel.TEXT, "The relation ", None),
        (DocItemLabel.FORMULA, "E=mc^2", None),
        (DocItemLabel.TEXT, " holds.", None),
    ]
    assert "- The relation $E=mc^2$ holds." in doc.export_to_markdown()


def test_jats_list_item_keeps_display_formula():
    doc = _convert_list(
        "<list-item><p>Minimize</p>"
        "<disp-formula><tex-math>$$x^2$$</tex-math></disp-formula></list-item>"
        "<list-item><p>next</p></list-item>"
    )

    items = _list_items(doc)
    assert [item.text for item in items] == ["Minimize", "next"]
    [formula] = [item for item in doc.texts if item.label == DocItemLabel.FORMULA]
    assert formula.text == "x^2"
    assert formula.parent == items[0].get_ref()
    assert "- Minimize\n$$x^2$$\n- next" in doc.export_to_markdown()


def test_jats_list_item_keeps_display_formula_inside_paragraph():
    doc = _convert_list(
        "<list-item><p>Minimize"
        "<disp-formula><tex-math>$$x^2$$</tex-math></disp-formula>"
        " subject to y.</p></list-item>"
        "<list-item><p>next</p></list-item>"
    )

    items = _list_items(doc)
    # The text before the block is the text of the item, the rest goes under it.
    assert [item.text for item in items] == ["Minimize", "next"]
    assert [_formatting_tuple(child.resolve(doc)) for child in items[0].children] == [
        (DocItemLabel.FORMULA, "x^2", None),
        (DocItemLabel.TEXT, "subject to y.", None),
    ]
    markdown_lines = doc.export_to_markdown().splitlines()
    assert "- Minimize" in markdown_lines
    assert "- next" in markdown_lines
    assert markdown_lines.index("- Minimize") < markdown_lines.index("- next")


def test_jats_list_item_keeps_mathml_display_formula_inside_paragraph():
    doc = _convert_list(
        "<list-item><p>Minimize<disp-formula>"
        '<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML">'
        "<mml:mi>x</mml:mi></mml:math></disp-formula> subject to y.</p></list-item>"
        "<list-item><p>next</p></list-item>"
    )

    assert [item.text for item in _list_items(doc)] == ["Minimize", "next"]
    markdown_lines = doc.export_to_markdown().splitlines()
    assert "- Minimize" in markdown_lines
    assert "- next" in markdown_lines


def test_jats_list_item_keeps_text_before_list_inside_paragraph():
    doc = _convert_list(
        "<list-item><p>Intro text"
        "<list><list-item><p>sub a</p></list-item>"
        "<list-item><p>sub b</p></list-item></list></p></list-item>"
        "<list-item><p>next</p></list-item>"
    )

    outer, sub_a, sub_b, nxt = _list_items(doc)
    assert outer.text == "Intro text"
    assert sub_a.parent.resolve(doc).parent == outer.get_ref()
    assert (sub_a.text, sub_b.text, nxt.text) == ("sub a", "sub b", "next")
    markdown_lines = doc.export_to_markdown().splitlines()
    assert markdown_lines[-4:] == [
        "- Intro text",
        "    - sub a",
        "    - sub b",
        "- next",
    ]


def test_jats_list_item_keeps_text_after_list_inside_paragraph():
    doc = _convert_list(
        "<list-item><p>Intro"
        "<list><list-item><p>sub</p></list-item></list>"
        " trailing</p></list-item>"
        "<list-item><p>next</p></list-item>"
    )

    markdown_lines = doc.export_to_markdown().splitlines()
    assert markdown_lines[-3:] == ["- Intro trailing", "    - sub", "- next"]


_MATHML = (
    '<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML">'
    "<mml:mi>x</mml:mi><mml:mo>=</mml:mo><mml:mn>1</mml:mn></mml:math>"
)
_TEX_AND_MATHML_FORMULA = (
    "<inline-formula><alternatives><tex-math>$x=1$</tex-math>"
    f"{_MATHML}</alternatives></inline-formula>"
)
_IMAGE_AND_MATHML_FORMULA = (
    '<inline-formula><alternatives><graphic xmlns:xlink="http://www.w3.org/1999/xlink"'
    f' xlink:href="e.gif"/>{_MATHML}</alternatives></inline-formula>'
)


@pytest.mark.parametrize(
    ("wrap", "line"),
    [
        pytest.param("<p>{}</p>", "Val $x=1$ here", id="body"),
        pytest.param(
            "<list><list-item><p>{}</p></list-item></list>",
            "- Val $x=1$ here",
            id="list-item",
        ),
        pytest.param(
            "<fn-group><fn><p>{}</p></fn></fn-group>",
            "- Val $x=1$ here",
            id="footnote",
        ),
    ],
)
def test_jats_formula_mathml_alternative_is_not_repeated(wrap, line):
    doc = convert_jats_body(
        "<sec><title>S</title>"
        + wrap.format(f"Val {_TEX_AND_MATHML_FORMULA} here")
        + "</sec>"
    )

    assert line in doc.export_to_markdown().splitlines()


@pytest.mark.parametrize(
    ("wrap", "line"),
    [
        ("<p>{}</p>", "Val x=1 here"),
        ("<list><list-item><p>{}</p></list-item></list>", "- Val x=1 here"),
        ("<fn-group><fn><p>{}</p></fn></fn-group>", "- Val x=1 here"),
    ],
)
def test_jats_mathml_alternative_to_an_image_reads_as_its_text(wrap, line):
    # the image of the formula is not added, and MathML without TeX is text
    doc = convert_jats_body(
        "<sec><title>S</title>"
        + wrap.format(f"Val {_IMAGE_AND_MATHML_FORMULA} here")
        + "</sec>"
    )

    assert line in doc.export_to_markdown().splitlines()


def test_jats_footnote_bare_mathml_reads_as_its_text():
    doc = convert_jats_body(
        f"<sec><title>S</title><fn-group><fn><p>Note {_MATHML} z</p></fn></fn-group></sec>"
    )

    assert "- Note x=1 z" in doc.export_to_markdown().splitlines()


def test_jats_list_item_bare_mathml_reads_as_its_text():
    doc = _convert_list(
        "<list-item><p>Value "
        '<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML">'
        "<mml:mi>x</mml:mi></mml:math> here</p></list-item>"
    )

    assert "- Value x here" in doc.export_to_markdown().splitlines()


def test_jats_list_item_formatting_with_nested_list():
    doc = _convert_list(
        """
        <list-item>
          <p>outer <italic>styled</italic></p>
          <list list-type="order">
            <list-item><p>inner <bold>one</bold></p></list-item>
            <list-item><p>inner two</p></list-item>
          </list>
        </list-item>
        """
    )

    outer, inner_one, inner_two = _list_items(doc)
    assert [outer.enumerated, inner_one.enumerated, inner_two.enumerated] == [
        False,
        True,
        True,
    ]
    assert inner_two.text == "inner two"
    nested = inner_two.parent.resolve(doc)
    assert nested.label == GroupLabel.LIST
    assert nested.parent == outer.get_ref()
    assert inner_one.parent == nested.get_ref()
    md = doc.export_to_markdown()
    assert "- outer *styled*" in md
    assert "1. inner **one**" in md
    assert "2. inner two" in md


def test_jats_ordered_list_item_with_formatting_keeps_marker():
    doc = _convert_list(
        "<list-item><label>(a)</label><p><bold>first</bold> step</p></list-item>"
        "<list-item><label>(b)</label><p>second step</p></list-item>",
        list_type="order",
    )

    first, second = _list_items(doc)
    assert [(first.marker, first.enumerated), (second.marker, second.enumerated)] == [
        ("(a)", True),
        ("(b)", True),
    ]
    assert first.text == ""
    assert second.text == "second step"
    md = doc.export_to_markdown()
    assert "(a) **first** step" in md
    assert "(b) second step" in md


def test_jats_plain_list_item_stays_a_single_text_item():
    doc = _convert_list(
        """
        <list-item>
          <p>first   paragraph </p>
          <p> second paragraph</p>
        </list-item>
        """
    )

    [item] = _list_items(doc)
    assert item.text == "first   paragraph second paragraph"
    assert item.formatting is None
    assert item.hyperlink is None
    assert item.children == []


def test_jats_list_item_drops_xml_indentation_like_a_paragraph():
    doc = _convert_list("<list-item><p>first\n            line</p></list-item>")

    [item] = _list_items(doc)
    assert item.text == "first line"


def test_jats_list_item_paragraph_after_nested_list_stays_item_text():
    doc = _convert_list(
        "<list-item><p>before</p>"
        "<list><list-item><p>nested</p></list-item></list>"
        "<p>after</p></list-item>"
    )

    outer, nested = _list_items(doc)
    assert outer.text == "before after"
    assert nested.text == "nested"


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        pytest.param(
            "The mass energy relation <inline-formula><tex-math>$$E=mc^2$$</tex-math></inline-formula> is famous.",
            [
                (DocItemLabel.TEXT, "The mass energy relation ", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, " is famous.", None),
            ],
            id="text-formula-text",
        ),
        pytest.param(
            "Given <inline-formula><tex-math>$$a^2$$</tex-math></inline-formula> and <inline-formula><tex-math>$$b^2$$</tex-math></inline-formula> we sum.",
            [
                (DocItemLabel.TEXT, "Given ", None),
                (DocItemLabel.FORMULA, "a^2", None),
                (DocItemLabel.TEXT, " and ", None),
                (DocItemLabel.FORMULA, "b^2", None),
                (DocItemLabel.TEXT, " we sum.", None),
            ],
            id="multiple-formulas",
        ),
        pytest.param(
            # loose text inside <inline-formula> around the <tex-math> is preserved
            # (adjacent unstyled runs are coalesced into one segment)
            "The relation <inline-formula>foo <tex-math>$$E=mc^2$$</tex-math> bar</inline-formula> holds.",
            [
                (DocItemLabel.TEXT, "The relation foo ", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, " bar holds.", None),
            ],
            id="text-inside-inline-formula",
        ),
        pytest.param(
            # tex-math is not always wrapped in $$...$$ in real JATS files
            "The relation <inline-formula><tex-math>E=mc^2</tex-math></inline-formula> holds.",
            [
                (DocItemLabel.TEXT, "The relation ", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, " holds.", None),
            ],
            id="bare-tex-math",
        ),
        pytest.param(
            "We use <inline-formula><italic>x</italic> <tex-math>$$a^2$$</tex-math></inline-formula> here.",
            [
                (DocItemLabel.TEXT, "We use ", None),
                (DocItemLabel.TEXT, "x", F_ITALIC),
                (DocItemLabel.TEXT, " ", None),
                (DocItemLabel.FORMULA, "a^2", None),
                (DocItemLabel.TEXT, " here.", None),
            ],
            id="italic-inside-formula",
        ),
        pytest.param(
            "Index <inline-formula>x<sub>i</sub> <tex-math>$$x_i$$</tex-math></inline-formula> shown.",
            [
                (DocItemLabel.TEXT, "Index x", None),
                (DocItemLabel.TEXT, "i", F_SUB),
                (DocItemLabel.TEXT, " ", None),
                (DocItemLabel.FORMULA, "x_i", None),
                (DocItemLabel.TEXT, " shown.", None),
            ],
            id="subscript-inside-formula",
        ),
        pytest.param(
            "Take <inline-formula><bold><italic>v</italic></bold> <tex-math>$$v$$</tex-math></inline-formula> next.",
            [
                (DocItemLabel.TEXT, "Take ", None),
                (DocItemLabel.TEXT, "v", F_BOLD_ITALIC),
                (DocItemLabel.TEXT, " ", None),
                (DocItemLabel.FORMULA, "v", None),
                (DocItemLabel.TEXT, " next.", None),
            ],
            id="nested-emphasis-inside-formula",
        ),
        pytest.param(
            # a tex-math nested inside an emphasis tag is still parsed as a
            # formula (not leaked as raw ``$$...$$`` text)
            "Val <inline-formula><italic><tex-math>$$x^2$$</tex-math></italic></inline-formula> shown.",
            [
                (DocItemLabel.TEXT, "Val ", None),
                (DocItemLabel.FORMULA, "x^2", None),
                (DocItemLabel.TEXT, " shown.", None),
            ],
            id="tex-math-inside-emphasis",
        ),
        pytest.param(
            # emphasis outside the formula is now preserved (general text styling),
            # and adjacent unstyled runs are coalesced
            "Compare <italic>lhs</italic> <inline-formula><tex-math>$$a$$</tex-math> rhs</inline-formula> and <inline-formula><tex-math>$$b$$</tex-math></inline-formula> now.",
            [
                (DocItemLabel.TEXT, "Compare ", None),
                (
                    DocItemLabel.TEXT,
                    "lhs",
                    F_ITALIC,
                ),
                (DocItemLabel.TEXT, " ", None),
                (DocItemLabel.FORMULA, "a", None),
                (DocItemLabel.TEXT, " rhs and ", None),
                (DocItemLabel.FORMULA, "b", None),
                (DocItemLabel.TEXT, " now.", None),
            ],
            id="formula-with-surrounding-elements",
        ),
    ],
)
def test_jats_inline_formula_is_grouped(paragraph, expected):
    doc = convert_jats_body(f"<sec><title>T</title><p>{paragraph}</p></sec>")

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [_formatting_tuple(item) for item in groups[0]] == expected


@pytest.mark.parametrize(
    ("paragraph", "expected_formulas"),
    [
        pytest.param(
            # a single segment is added directly, without an inline group
            "<inline-formula><tex-math>$$E=mc^2$$</tex-math></inline-formula>",
            ["E=mc^2"],
            id="standalone-formula",
        ),
        pytest.param(
            # per the JATS spec tex-math is bare; a stray single-$ pair is stripped
            "<inline-formula><tex-math>$x^2$</tex-math></inline-formula>",
            ["x^2"],
            id="single-dollar-delimiters",
        ),
        pytest.param(
            # an inline-formula with no usable tex-math is dropped
            "Energy <inline-formula><tex-math/></inline-formula> equation.",
            [],
            id="no-usable-tex-math",
        ),
    ],
)
def test_jats_inline_formula_is_not_grouped(paragraph, expected_formulas):
    doc = convert_jats_body(f"<sec><title>T</title><p>{paragraph}</p></sec>")

    assert _inline_group_items(doc) == []
    formulas = [t.text for t in doc.texts if t.label == DocItemLabel.FORMULA]
    assert formulas == expected_formulas


def test_jats_paragraph_emphasis_is_preserved():
    doc = convert_jats_body(
        "<sec><title>T</title>"
        "<p>The species <italic>Homo sapiens</italic> is <bold>common</bold>.</p>"
        "</sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (DocItemLabel.TEXT, "The species ", None),
        (
            DocItemLabel.TEXT,
            "Homo sapiens",
            F_ITALIC,
        ),
        (DocItemLabel.TEXT, " is ", None),
        (DocItemLabel.TEXT, "common", F_BOLD),
        (DocItemLabel.TEXT, ".", None),
    ]


def test_jats_external_link_boundaries_are_preserved():
    doc = convert_jats_body(
        "<sec><title>T</title><p>Before <ext-link "
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="https://example.com/docs">linked text</ext-link> after.'
        "</p></sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [
        (item.text, str(item.hyperlink) if item.hyperlink is not None else None)
        for item in groups[0]
    ] == [
        ("Before ", None),
        ("linked text", "https://example.com/docs"),
        (" after.", None),
    ]


def test_jats_adjacent_external_links_with_different_targets_do_not_merge():
    doc = convert_jats_body(
        '<sec><title>T</title><p xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<ext-link xlink:href="https://example.com/a">first</ext-link><ext-link '
        'xlink:href="https://example.com/b">second</ext-link></p></sec>'
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [
        (item.text, str(item.hyperlink) if item.hyperlink is not None else None)
        for item in groups[0]
    ] == [
        ("first", "https://example.com/a"),
        ("second", "https://example.com/b"),
    ]


def test_jats_external_link_preserves_nested_formatting():
    doc = convert_jats_body(
        "<sec><title>T</title><p>Before <ext-link "
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="https://example.com/docs">linked '
        "<italic>text</italic></ext-link> after.</p></sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [
        (
            item.text.strip(),
            str(item.hyperlink) if item.hyperlink is not None else None,
            item.formatting.italic if item.formatting is not None else False,
        )
        for item in groups[0]
    ] == [
        ("Before", None, False),
        ("linked", "https://example.com/docs", False),
        ("text", "https://example.com/docs", True),
        ("after.", None, False),
    ]


def test_jats_external_link_without_href_is_plain_text():
    doc = convert_jats_body(
        "<sec><title>T</title><p>Before <ext-link>plain text</ext-link> "
        "after.</p></sec>"
    )

    texts = [item for item in doc.texts if item.label == DocItemLabel.TEXT]
    assert [(item.text, item.hyperlink) for item in texts] == [
        ("Before plain text after.", None)
    ]


def test_jats_external_relative_link_falls_back_to_path():
    doc = convert_jats_body(
        "<sec><title>T</title><p><ext-link "
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="../docs">relative</ext-link></p></sec>'
    )

    texts = [item for item in doc.texts if item.label == DocItemLabel.TEXT]
    assert [(item.text, item.hyperlink) for item in texts] == [
        ("relative", Path("../docs"))
    ]


def test_jats_external_link_is_preserved_on_inline_formula():
    doc = convert_jats_body(
        "<sec><title>T</title><p><ext-link "
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="https://example.com/equation">value '
        "<inline-formula><tex-math>$$x$$</tex-math></inline-formula>"
        "</ext-link> after.</p></sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [
        (
            item.label,
            item.text,
            str(item.hyperlink) if item.hyperlink is not None else None,
        )
        for item in groups[0]
    ] == [
        (DocItemLabel.TEXT, "value ", "https://example.com/equation"),
        (DocItemLabel.FORMULA, "x", "https://example.com/equation"),
        (DocItemLabel.TEXT, " after.", None),
    ]


def test_jats_plain_paragraph_stays_a_single_text_item():
    doc = convert_jats_body(
        "<sec><title>T</title><p>Plain text with a <xref>1</xref> citation.</p></sec>"
    )

    # no emphasis (and coalesced xref) → a single TEXT item, no inline group
    assert _inline_group_items(doc) == []
    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert texts == ["Plain text with a 1 citation."]


def test_jats_inline_term_keeps_its_text():
    # <term> marks a defined term inside running text; its words are part of the sentence.
    doc = convert_jats_body(
        "<sec><title>The <term>PCR</term> step</title>"
        "<p>The <term>polymerase chain reaction</term> amplifies DNA.</p></sec>"
    )

    headings = [t.text for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER]
    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert headings == ["The PCR step"]
    assert texts == ["The polymerase chain reaction amplifies DNA."]


def _def_item(term: str, *defs: str) -> str:
    return f"<def-item><term>{term}</term>{''.join(defs)}</def-item>"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            "<p>Terms:<def-list>"
            + _def_item("DNA", "<def><p>Deoxyribonucleic acid</p></def>")
            + _def_item("RNA", "<def><p>Ribonucleic acid</p></def>")
            + "</def-list></p>",
            "Terms:\n\n- **DNA**\n    - Deoxyribonucleic acid\n"
            "- **RNA**\n    - Ribonucleic acid",
            id="in-paragraph",
        ),
        pytest.param(
            "<def-list>"
            + _def_item(
                "T", "<def><p>one</p><p>more</p></def>", "<def><p>two</p></def>"
            )
            + "</def-list>",
            "- **T**\n    - one more\n    - two",
            id="many-defs",
        ),
        pytest.param(
            "<def-list>"
            + _def_item(
                "A",
                "<def><p>alpha</p><def-list>"
                + _def_item("A1", "<def><p>sub</p></def>")
                + "</def-list></def>",
            )
            + "</def-list>",
            "- **A**\n    - alpha\n        - **A1**\n            - sub",
            id="nested-def-list",
        ),
        pytest.param(
            "<def-list>"
            + _def_item(
                "A",
                "<def><p>alpha</p><list><list-item><p>x</p></list-item></list></def>",
            )
            + "</def-list>",
            "- **A**\n    - alpha\n        - x",
            id="nested-list",
        ),
        pytest.param(
            "<def-list>"
            + _def_item(
                "<italic>E. coli</italic>",
                "<def><p>a <bold>bug</bold> with "
                "<inline-formula><tex-math>x</tex-math></inline-formula></p></def>",
            )
            + "</def-list>",
            "- ***E. coli***\n    - a **bug** with $x$",
            id="inline-styling",
        ),
        pytest.param(
            "<def-list>"
            + _def_item(
                "<inline-formula><tex-math>\\alpha</tex-math></inline-formula>",
                "<def><p>an angle</p></def>",
            )
            + "</def-list>",
            "- $\\alpha$\n    - an angle",
            id="formula-term",
        ),
        pytest.param(
            "<def-list><def-item><term/><def><p>no term</p></def></def-item></def-list>",
            "## S\n\n- no term",
            id="empty-term",
        ),
    ],
)
def test_jats_def_list_keeps_terms(body, expected):
    # Like an HTML <dl>, each <term> becomes a bold list item and each <def> an
    # item of a nested list under it. Only the definitions used to come through.
    doc = convert_jats_body(f"<sec><title>S</title>{body}</sec>")

    assert expected in doc.export_to_markdown()


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        pytest.param(
            "<list><list-item><p>x</p></list-item><list-item><p>y</p></list-item>"
            "</list>",
            [(DocItemLabel.LIST_ITEM, "x"), (DocItemLabel.LIST_ITEM, "y")],
            id="list",
        ),
        pytest.param(
            "<disp-formula><tex-math>y=1</tex-math></disp-formula>",
            [(DocItemLabel.FORMULA, "y=1")],
            id="disp-formula",
        ),
        pytest.param(
            "<table-wrap><label>Table 1</label><caption><p>Cap</p></caption>"
            "<table><tr><td>c1</td><td>c2</td></tr></table></table-wrap>",
            [(DocItemLabel.CAPTION, "Table 1 Cap"), (DocItemLabel.TABLE, "")],
            id="table-wrap",
        ),
    ],
)
def test_jats_def_paragraph_keeps_its_blocks(block, expected):
    # A block in a paragraph of a <def> stays a block, as it does in a <p> elsewhere.
    doc = convert_jats_body(
        "<sec><title>S</title><def-list>"
        + _def_item("A", f"<def><p>before{block}after</p></def>")
        + "</def-list></sec>"
    )

    items = {
        (item.label, item.text if isinstance(item, TextItem) else ""): item
        for item, _ in doc.iterate_items()
    }
    assert (DocItemLabel.LIST_ITEM, "A") in items
    for key in expected:
        assert key in items
        assert items[key].parent.resolve(doc).label != GroupLabel.INLINE
    texts = " ".join(text for label, text in items if label == DocItemLabel.TEXT)
    assert "before" in texts
    assert "after" in texts


def test_jats_def_paragraphs_with_edge_whitespace_are_one_text_run():
    doc = convert_jats_body(
        "<sec><title>S</title><def-list>"
        + _def_item(
            "T", "<def>\n  <p>\n    one\n  </p>\n  <p>\n    two\n  </p>\n</def>"
        )
        + "</def-list></sec>"
    )

    assert [item.text for item in _list_items(doc)] == ["T", "one two"]
    assert _inline_group_items(doc) == []


def test_jats_def_outside_def_list_is_unchanged():
    # A <def> in an <abbrev> is not part of a definition list.
    doc = convert_jats_body(
        "<sec><title>S</title><p>The <abbrev>DNA<def><p>deoxyribonucleic acid</p>"
        "</def></abbrev> is long.</p></sec>"
    )

    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert texts == ["deoxyribonucleic acid", "The DNA is long."]


def test_jats_glossary_title_is_a_heading():
    xml = JATS_WRAPPER.format(
        content="<body><p>Text.</p></body><back><glossary><title>Glossary</title>"
        "<def-list><def-item><term>PCR</term><def><p>Polymerase chain reaction</p>"
        "</def></def-item></def-list></glossary></back>"
    )
    stream = DocumentStream(name="glossary.nxml", stream=BytesIO(xml.encode()))
    doc = get_converter().convert(stream).document

    md = doc.export_to_markdown()
    assert "## Glossary\n\n- **PCR**\n    - Polymerase chain reaction" in md


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            "<disp-formula><tex-math>$$E=mc^2$$</tex-math></disp-formula>",
            "E=mc^2",
            id="direct-tex-math",
        ),
        pytest.param(
            "<disp-formula><alternatives><tex-math>$$a+b$$</tex-math>"
            "</alternatives></disp-formula>",
            "a+b",
            id="tex-math-under-alternatives",
        ),
    ],
)
def test_jats_disp_formula_is_block_formula(body, expected):
    doc = convert_jats_body(f"<sec><title>T</title>{body}</sec>")

    formulas = [t.text for t in doc.texts if t.label == DocItemLabel.FORMULA]
    assert formulas == [expected]
    # a block formula is emitted standalone, not inside an inline group
    assert _inline_group_items(doc) == []


def test_jats_empty_display_formula_does_not_drop_following_content():
    # A display equation whose <tex-math> is empty must be skipped, not crash the
    # walk. _add_equation used to call node.text.split("$$") unconditionally, so an
    # empty <tex-math/> raised AttributeError; convert() swallows it and returns a
    # truncated document, silently losing everything after the equation.
    doc = convert_jats_body(
        "<sec><title>T</title>"
        "<p>Before the equation.</p>"
        "<disp-formula><tex-math/></disp-formula>"
        "<p>After the equation.</p>"
        "</sec>"
    )

    md = doc.export_to_markdown()
    assert "Before the equation." in md
    assert "After the equation." in md
    # The empty display formula produced no FORMULA item.
    assert [t.text for t in doc.texts if t.label == DocItemLabel.FORMULA] == []


_MML = '<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML">{}</mml:math>'
# a/b, written as presentation MathML
_MML_FRAC = "<mml:mfrac><mml:mi>a</mml:mi><mml:mi>b</mml:mi></mml:mfrac>"
_MML_FRAC_TEX = _MML.format(
    "<mml:semantics><mml:mrow>"
    + _MML_FRAC
    + '</mml:mrow><mml:annotation encoding="application/x-tex">'
    r"\frac{a}{b}</mml:annotation></mml:semantics>"
)
_MML_SUM = "<mml:mi>x</mml:mi><mml:mo>+</mml:mo><mml:mn>1</mml:mn>"


def _convert_formula_shapes(formula: str) -> DoclingDocument:
    """Convert `formula` as a display formula, and inline in a paragraph and a note."""
    return convert_jats_body(
        f"<sec><title>T</title><p>Before.</p>"
        f"<disp-formula>{formula}</disp-formula>"
        f"<p>Let <inline-formula>{formula}</inline-formula> be.</p>"
        f'<p>Note<xref ref-type="fn" rid="fn1">1</xref>.</p>'
        f'<fn-group><fn id="fn1"><p>See <inline-formula>{formula}</inline-formula>'
        " here.</p></fn></fn-group></sec>"
    )


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        pytest.param(_MML_FRAC_TEX, r"\frac{a}{b}", id="tex-annotation"),
        pytest.param(
            _MML.format(
                "<mml:semantics><mml:mrow>"
                + _MML_FRAC
                + '</mml:mrow><mml:annotation encoding="TeX">$a/b$</mml:annotation>'
                "</mml:semantics>"
            ),
            "a/b",
            id="tex-annotation-with-delimiters",
        ),
        pytest.param(
            f"<alternatives><graphic/>{_MML_FRAC_TEX}</alternatives>",
            r"\frac{a}{b}",
            id="tex-annotation-under-alternatives",
        ),
        pytest.param(
            f"<alternatives><tex-math>y^2</tex-math>{_MML_FRAC_TEX}</alternatives>",
            "y^2",
            id="tex-math-wins-over-mathml",
        ),
    ],
)
def test_jats_mathml_formula_with_tex_annotation_is_kept(formula, expected):
    doc = _convert_formula_shapes(formula)

    formulas = [t for t in doc.texts if t.label == DocItemLabel.FORMULA]
    assert [t.text for t in formulas] == [expected] * 3
    # the display formula is a block, the inline ones stay in their sentence
    assert formulas[0].parent.resolve(doc).label != GroupLabel.INLINE
    paragraph, footnote = _inline_group_items(doc)
    assert [_formatting_tuple(item) for item in paragraph] == [
        (DocItemLabel.TEXT, "Let ", None),
        (DocItemLabel.FORMULA, expected, None),
        (DocItemLabel.TEXT, " be.", None),
    ]
    assert [_formatting_tuple(item) for item in footnote] == [
        (DocItemLabel.FOOTNOTE, "See ", None),
        (DocItemLabel.FORMULA, expected, None),
        (DocItemLabel.FOOTNOTE, " here.", None),
    ]
    md = doc.export_to_markdown()
    assert f"Before.\n\n$${expected}$$\n\nLet ${expected}$ be." in md


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        pytest.param(_MML.format(_MML_SUM), "x+1", id="tokens"),
        pytest.param(
            # a/b as presentation markup has no TeX: its tokens are text, not math
            _MML.format(
                "<mml:semantics><mml:mrow>"
                + _MML_FRAC
                + '</mml:mrow><mml:annotation encoding="StarMath 5.0">a over b'
                "</mml:annotation></mml:semantics>"
            ),
            "ab",
            id="other-annotation-is-not-text",
        ),
        pytest.param(
            _MML.format(
                "<mml:mtext>if and only if</mml:mtext><mml:mi>x</mml:mi>"
                "<mml:mtext>holds</mml:mtext>"
            ),
            "if and only if x holds",
            id="mtext-keeps-its-spaces",
        ),
        pytest.param(
            # one digit per mtext, as some publishers write a number
            _MML.format(
                "<mml:mi>v</mml:mi><mml:mo>≥</mml:mo><mml:mtext>3</mml:mtext>"
                "<mml:mtext>0</mml:mtext><mml:mtext>mV</mml:mtext>"
            ),
            "v≥30 mV",
            id="mtext-without-letter-is-not-spaced",
        ),
        pytest.param(
            f"<alternatives><graphic/>{_MML.format(_MML_SUM)}</alternatives>",
            "x+1",
            id="tokens-under-alternatives",
        ),
    ],
)
def test_jats_mathml_formula_without_tex_reads_as_its_text(formula, expected):
    doc = _convert_formula_shapes(formula)

    assert [t for t in doc.texts if t.label == DocItemLabel.FORMULA] == []
    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    # the display formula is a text item in place, the inline ones are text runs
    assert texts == ["Before.", expected, f"Let {expected} be.", "Note1."]
    footnotes = [t.text for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert "".join(footnotes) == f"See {expected} here."


def test_jats_display_formula_given_as_mathml_keeps_its_place_in_a_paragraph():
    doc = convert_jats_body(
        f"<sec><title>T</title><p>See <disp-formula>{_MML_FRAC_TEX}</disp-formula>"
        " done.</p></sec>"
    )

    body = [
        (t.label, t.text)
        for t in doc.texts
        if t.label in (DocItemLabel.TEXT, DocItemLabel.FORMULA)
    ]
    assert body == [
        (DocItemLabel.TEXT, "See"),
        (DocItemLabel.FORMULA, r"\frac{a}{b}"),
        (DocItemLabel.TEXT, "done."),
    ]


@pytest.mark.parametrize(
    "wrap",
    [
        "<p>{}</p>",
        "<list><list-item><p>{}</p></list-item></list>",
        "<fn-group><fn><p>{}</p></fn></fn-group>",
    ],
)
def test_jats_bare_mathml_text_leaves_out_its_annotation(wrap):
    doc = convert_jats_body(
        "<sec><title>S</title>"
        + wrap.format(
            "A "
            + _MML.format(
                f"<mml:semantics><mml:mrow>{_MML_SUM}</mml:mrow>"
                '<mml:annotation encoding="application/x-tex">x+1</mml:annotation>'
                "</mml:semantics>"
            )
            + " b"
        )
        + "</sec>"
    )

    md = doc.export_to_markdown()
    assert "A x+1 b" in md
    assert "x+1x+1" not in md


_TEXTUAL_FORM_FORMULA = (
    "<inline-formula><alternatives>"
    + _MML.format("<mml:mi>β</mml:mi>")
    + "<textual-form><italic>β</italic></textual-form></alternatives>"
    "</inline-formula>"
)


@pytest.mark.parametrize(
    ("body", "line"),
    [
        pytest.param(
            "<sec><title>S</title><p>where {} is temp.</p></sec>",
            "where *β* is temp.",
            id="paragraph",
        ),
        pytest.param(
            "<list><list-item><p>where {} is temp.</p></list-item></list>",
            "- where *β* is temp.",
            id="list-item",
        ),
        pytest.param(
            "<p>a</p><fn-group><fn><p>where {} is temp.</p></fn></fn-group>",
            "- where *β* is temp.",
            id="footnote",
        ),
        pytest.param(
            '<fig id="f1"><label>Fig 1</label><caption><p>where {} is temp.</p>'
            "</caption></fig>",
            "Fig 1 where β is temp.",
            id="caption",
        ),
    ],
)
def test_jats_textual_form_is_read_instead_of_its_mathml(body, line):
    doc = convert_jats_body(body.format(_TEXTUAL_FORM_FORMULA))

    md = doc.export_to_markdown()
    assert line in md.splitlines()
    assert md.count("β") == 1


_MML_BETA = _MML.format("<mml:mi>β</mml:mi>")
_TEXTUAL_FORM = "<textual-form><italic>β</italic></textual-form>"


@pytest.mark.parametrize(
    ("body", "lines"),
    [
        pytest.param(
            "<sec><title>S</title><p>before</p><disp-formula><alternatives>"
            f"{_MML_BETA}{_TEXTUAL_FORM}</alternatives></disp-formula>"
            "<p>after</p></sec>",
            ["before", "β", "after"],
            id="alternatives",
        ),
        pytest.param(
            "<sec><title>S</title><p>before</p><disp-formula>"
            f"{_MML_BETA}{_TEXTUAL_FORM}</disp-formula><p>after</p></sec>",
            ["before", "β", "after"],
            id="in-wrapper",
        ),
        pytest.param(
            f"<sec><title>S</title><p>before</p><disp-formula>{_TEXTUAL_FORM}"
            "</disp-formula><p>after</p></sec>",
            ["before", "β", "after"],
            id="textual-form-only",
        ),
        pytest.param(
            "<sec><title>S</title><p>Minimize<disp-formula><alternatives>"
            f"{_MML_BETA}{_TEXTUAL_FORM}</alternatives></disp-formula>"
            "subject to y.</p></sec>",
            ["Minimize", "β", "subject to y."],
            id="in-paragraph",
        ),
        pytest.param(
            "<list><list-item><p>Do<disp-formula><alternatives>"
            f"{_MML_BETA}{_TEXTUAL_FORM}</alternatives></disp-formula>"
            "then.</p></list-item></list>",
            ["- Do", "β", "then."],
            id="in-list-item",
        ),
    ],
)
def test_jats_display_formula_keeps_its_textual_form(body, lines):
    doc = convert_jats_body(body)

    md_lines = [line for line in doc.export_to_markdown().splitlines() if line]
    positions = [md_lines.index(line) for line in lines]
    assert positions == sorted(positions)
    assert sum(line.count("β") for line in md_lines) == 1


def test_jats_display_formula_prefers_tex_math_over_textual_form():
    doc = convert_jats_body(
        "<sec><title>S</title><disp-formula><alternatives>"
        f"<tex-math>\\beta</tex-math>{_MML_BETA}{_TEXTUAL_FORM}"
        "</alternatives></disp-formula></sec>"
    )

    md = doc.export_to_markdown()
    assert "$$\\beta$$" in md
    assert "β" not in md


@pytest.mark.parametrize(
    ("float_xml", "text"),
    [
        pytest.param(
            '<fig id="f1"><label>Fig 1</label><caption><title>Cap</title>'
            "<p>Has <inline-formula><tex-math>x^2</tex-math></inline-formula>\n"
            '   here</p></caption><graphic xlink:href="a.png"/></fig>',
            "Fig 1 Cap Has $x^2$ here",
            id="figure",
        ),
        pytest.param(
            '<table-wrap id="t1"><label>Table 1</label><caption><p>Cap '
            "<inline-formula>"
            + _MML.format(
                "<mml:semantics><mml:mrow><mml:mi>y</mml:mi><mml:mo>=</mml:mo>"
                "<mml:mn>2</mml:mn></mml:mrow>"
                '<mml:annotation encoding="application/x-tex">y=2</mml:annotation>'
                "</mml:semantics>"
            )
            + "</inline-formula> z</p></caption>"
            "<table><tr><td>a</td></tr></table></table-wrap>",
            "Table 1 Cap $y=2$ z",
            id="table",
        ),
        pytest.param(
            '<table-wrap id="t1"><label>Table 2</label><caption><p>Img '
            "<inline-formula><tex-math>q_1</tex-math></inline-formula> cap</p>"
            '</caption><graphic xlink:href="t.png"/></table-wrap>',
            "Table 2 Img $q_1$ cap",
            id="table-as-image",
        ),
        pytest.param(
            '<table-wrap id="t1"><label>Table 3</label><caption><p>Cap '
            f"<inline-formula>{_MML.format('<mml:mi>y</mml:mi>')}</inline-formula>"
            " z</p></caption><table><tr><td>a</td></tr></table></table-wrap>",
            "Table 3 Cap y z",
            id="mathml-without-tex-is-text",
        ),
    ],
)
def test_jats_caption_keeps_its_inline_formula_as_tex(float_xml, text):
    doc = convert_jats_body(f"<sec><title>S</title>{float_xml}</sec>")

    floating = (doc.pictures or doc.tables)[0]
    [caption] = [ref.resolve(doc) for ref in floating.captions]
    # one plain caption item, so a chunker reads the caption once
    assert (caption.label, caption.text, caption.children) == (
        DocItemLabel.CAPTION,
        text,
        [],
    )
    assert floating.caption_text(doc) == text


def test_jats_caption_without_formula_stays_one_text_item():
    doc = convert_jats_body(
        '<sec><title>S</title><fig id="f1"><label>Fig 1</label><caption>'
        "<title>Cap</title><p>Plain <italic>it</italic> here</p></caption>"
        '<graphic xlink:href="a.png"/></fig></sec>'
    )

    [caption] = [ref.resolve(doc) for ref in doc.pictures[0].captions]
    assert caption.text == "Fig 1 Cap Plain it here"
    assert caption.children == []


def test_jats_boxed_text_title_is_a_heading():
    doc = convert_jats_body(
        "<sec><title>S</title><boxed-text><label>Box 1</label><caption>"
        "<title>Key points</title></caption><p>Boxed para</p></boxed-text>"
        "<p>After.</p></sec>"
    )

    items = [
        (item.label, item.text, level)
        for item, level in doc.iterate_items()
        if isinstance(item, TextItem)
    ]
    assert items[1:] == [
        (DocItemLabel.SECTION_HEADER, "S", 2),
        (DocItemLabel.SECTION_HEADER, "Box 1 Key points", 3),
        (DocItemLabel.TEXT, "Boxed para", 4),
        (DocItemLabel.TEXT, "After.", 3),
    ]
    assert doc.export_to_markdown().endswith(
        "## S\n\n### Box 1 Key points\n\nBoxed para\n\nAfter."
    )


@pytest.mark.parametrize(
    ("body", "texts"),
    [
        pytest.param(
            '<supplementary-material id="s1"><label>Supplementary file 1.</label>'
            "<caption><title>Primer list.</title><p>Detail.</p></caption>"
            '<media xlink:href="a.xlsx"/></supplementary-material>',
            ["Supplementary file 1. Primer list.", "Detail."],
            id="in-section",
        ),
        pytest.param(
            '<p>See <supplementary-material id="s2"><label>Data S1.</label>'
            "<caption><title>Raw data.</title></caption></supplementary-material>"
            " after.</p>",
            ["See", "Data S1. Raw data.", "after."],
            id="in-paragraph",
        ),
    ],
)
def test_jats_supplementary_material_keeps_its_label_and_title(body, texts):
    doc = convert_jats_body(f"<sec><title>S</title>{body}</sec>")

    assert [t.text for t in doc.texts if t.label == DocItemLabel.TEXT] == texts


def test_jats_boxed_text_with_a_section_nests_its_heading():
    doc = convert_jats_body(
        "<sec><title>S</title><boxed-text><caption><title>Box T</title></caption>"
        "<sec><title>Inner</title><p>x</p></sec></boxed-text>"
        "<sec><title>Sibling</title><p>y</p></sec></sec>"
    )

    headers = [
        (t.text, t.level) for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER
    ]
    assert headers == [("S", 1), ("Box T", 2), ("Inner", 3), ("Sibling", 2)]
    assert "## S\n\n### Box T\n\n#### Inner\n\nx\n\n### Sibling\n\ny" in (
        doc.export_to_markdown()
    )


def test_jats_content_after_a_definition_list_stays_in_place():
    doc = convert_jats_body(
        "<sec><title>S</title><def-list><def-item><term>T</term><def><p>D</p>"
        "</def></def-item></def-list><p>After.</p>"
        '<fig id="f1"><caption><p>Fig cap</p></caption>'
        '<graphic xlink:href="a.png"/></fig><p>Last.</p></sec>'
    )

    [heading] = [t for t in doc.texts if t.text == "S"]
    order = [
        getattr(child, "name", None) or getattr(child, "text", None) or child.label
        for child in (ref.resolve(doc) for ref in heading.children)
    ]
    assert order == ["description list", "After.", DocItemLabel.PICTURE, "Last."]


def test_jats_table_label_keeps_its_inline_markup():
    doc = convert_jats_body(
        '<sec><title>S</title><table-wrap id="t1"><label><bold>Table 9</bold>'
        "</label><caption><p>Cap</p></caption><table><tr><td>a</td></tr></table>"
        "</table-wrap></sec>"
    )

    assert doc.tables[0].caption_text(doc) == "Table 9 Cap"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "<p>Run it:</p><preformat>{code}</preformat><p>Done.</p>", id="in-section"
        ),
        pytest.param(
            "<p>Run it:<preformat>{code}</preformat>Done.</p>", id="in-paragraph"
        ),
        pytest.param(
            "<p>Run it:</p><code>{code}</code><p>Done.</p>", id="code-in-section"
        ),
    ],
)
def test_jats_preformat_is_code_block(body):
    # <preformat> and <code> hold verbatim text such as code or console output. A
    # block in a section was dropped, and one in a paragraph was merged into the
    # paragraph text without its line breaks.
    code = "for i in range(3):\n    print(i)"
    doc = convert_jats_body(f"<sec><title>T</title>{body.format(code=code)}</sec>")

    items = [
        (t.label, t.text)
        for t in doc.texts
        if t.label in (DocItemLabel.TEXT, DocItemLabel.CODE)
    ]
    assert items == [
        (DocItemLabel.TEXT, "Run it:"),
        (DocItemLabel.CODE, code),
        (DocItemLabel.TEXT, "Done."),
    ]


def test_jats_code_block_keeps_its_language():
    doc = convert_jats_body(
        '<sec><title>T</title><code language="python">print(1)</code></sec>'
    )

    [code] = [t for t in doc.texts if t.label == DocItemLabel.CODE]
    assert code.text == "print(1)"
    assert code.code_language == CodeLanguageLabel.PYTHON


def test_jats_table_header_rowspan_without_body_keeps_the_table():
    doc = convert_jats_body(
        "<sec><title>T</title><table-wrap><label>Table 1</label>"
        '<table><tr><th rowspan="2">h</th></tr></table>'
        "</table-wrap></sec>"
    )

    assert len(doc.tables) == 1
    table = doc.tables[0].data
    assert (table.num_rows, table.num_cols) == (0, 1)
    assert [cell.text for cell in table.table_cells] == ["h"]
    assert "Table 1" in doc.export_to_markdown()


def test_jats_table_oversized_spans_clamped_to_table_size():
    # Declared spans far beyond the table must not size the grid: the table
    # keeps the shape of its real cells, and the spans stop at its edges.
    doc = convert_jats_body(
        "<sec><title>T</title><table-wrap><table>"
        '<tr><td rowspan="100000000">A</td><td colspan="3000000">B</td></tr>'
        "<tr><td>C</td></tr>"
        "</table></table-wrap></sec>"
    )

    assert len(doc.tables) == 1
    data = doc.tables[0].data
    assert (data.num_rows, data.num_cols) == (2, 2)
    assert [(c.text, c.row_span, c.col_span) for c in data.table_cells] == [
        ("A", 2, 1),
        ("B", 1, 1),
        ("C", 1, 1),
    ]
    assert [[cell.text for cell in row] for row in data.grid] == [
        ["A", "B"],
        ["A", "C"],
    ]


def test_jats_footnotes_are_preserved():
    doc = convert_jats_body(
        """
        <sec>
          <title>Footnote Test</title>

          <fn-group>
                <fn id="fn1">
                    <label>1</label>
                    <p>First footnote</p>
                </fn>

                <fn id="fn2">
                    <label>2</label>
                    <p>Second footnote</p>
                </fn>
            </fn-group>
        </sec>
        """
    )

    md = doc.export_to_markdown()
    assert "First footnote" in md
    assert "Second footnote" in md


def test_jats_file_does_not_fetch_relative_figure_image_by_default(tmp_path: Path):
    _make_test_image(tmp_path / "figure.png")
    jats_path = tmp_path / "article.nxml"
    write_jats_body(jats_path, '<fig><graphic xlink:href="figure.png"/></fig>')

    doc = get_converter().convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None


def test_jats_file_embeds_relative_figure_image_when_enabled(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    _make_test_image(tmp_path / "images" / "figure.png")
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        "<fig><label>Figure 1</label><caption><p>A red rectangle.</p></caption>"
        '<graphic xlink:href="images/figure.png"/></fig>',
    )

    doc = local_fetch_converter.convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is not None
    assert pictures[0].captions[0].resolve(doc).text == "Figure 1 A red rectangle."
    image = pictures[0].get_image(doc)
    assert image is not None
    assert image.size == (7, 5)
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_jats_stream_does_not_resolve_relative_figure_image():
    doc = convert_jats_body(
        '<fig><graphic xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="figure.png"/></fig>',
        JatsBackendOptions(fetch_images=True, enable_local_fetch=True),
    )

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None


def test_jats_stream_resolves_figure_image_from_source_uri(tmp_path: Path):
    _make_test_image(tmp_path / "figure.png", color=(0, 255, 0))
    jats_path = tmp_path / "source.nxml"
    write_jats_body(jats_path, '<fig><graphic xlink:href="figure.png"/></fig>')
    stream = DocumentStream(name="article.nxml", stream=BytesIO(jats_path.read_bytes()))
    converter = get_converter(
        JatsBackendOptions(
            fetch_images=True,
            enable_local_fetch=True,
            source_uri=jats_path,
        )
    )

    doc = converter.convert(stream).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    image = pictures[0].get_image(doc)
    assert image is not None
    assert image.size == (7, 5)
    assert image.getpixel((0, 0)) == (0, 255, 0)


def test_jats_stream_does_not_fetch_relative_image_from_remote_source_uri():
    doc = convert_jats_body(
        '<fig><graphic xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xlink:href="figure.png"/></fig>',
        JatsBackendOptions(
            fetch_images=True,
            enable_remote_fetch=True,
            source_uri="https://example.com/article.nxml",
        ),
    )

    assert get_pictures(doc)[0].image is None


def test_jats_figure_image_requires_local_fetch_permission(tmp_path: Path):
    _make_test_image(tmp_path / "figure.png")
    jats_path = tmp_path / "article.nxml"
    write_jats_body(jats_path, '<fig><graphic xlink:href="figure.png"/></fig>')

    with pytest.warns(UserWarning, match="Fetching local resources"):
        doc = (
            get_converter(JatsBackendOptions(fetch_images=True, source_uri=jats_path))
            .convert(jats_path)
            .document
        )

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None


@pytest.mark.parametrize(
    "graphic",
    [
        pytest.param('<graphic xlink:href="images/figure"/>', id="direct"),
        pytest.param(
            "<alternatives>"
            '<graphic xlink:href="images/unsupported.svg"/>'
            '<graphic xlink:href="images/figure"/>'
            "</alternatives>",
            id="alternatives",
        ),
    ],
)
def test_jats_figure_image_resolves_extensionless_href(
    tmp_path: Path, graphic: str, local_fetch_converter: DocumentConverter
):
    _make_test_image(tmp_path / "images" / "figure.jpg", size=(9, 6), color=(0, 0, 255))
    jats_path = tmp_path / "article.nxml"
    write_jats_body(jats_path, f"<fig>{graphic}</fig>")

    doc = local_fetch_converter.convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    image = pictures[0].get_image(doc)
    assert image is not None
    assert image.size == (9, 6)


def test_jats_figure_image_falls_back_after_undecodable_alternative(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    broken_path = tmp_path / "images" / "broken.png"
    broken_path.parent.mkdir()
    broken_path.write_bytes(b"not an image")
    _make_test_image(tmp_path / "images" / "figure.png", size=(9, 6), color=(0, 0, 255))
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        "<fig><alternatives>"
        '<graphic xlink:href="images/broken.png"/>'
        '<graphic xlink:href="images/figure.png"/>'
        "</alternatives></fig>",
    )

    with pytest.warns(UserWarning, match="Could not process an image"):
        doc = local_fetch_converter.convert(jats_path).document

    image = get_pictures(doc)[0].get_image(doc)
    assert image is not None
    assert image.size == (9, 6)


def test_jats_figure_image_falls_back_after_absolute_alternative(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    absolute_path = _make_test_image(tmp_path / "absolute.png", color=(255, 0, 0))
    _make_test_image(
        tmp_path / "images" / "relative.png", size=(9, 6), color=(0, 0, 255)
    )
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        "<fig><alternatives>"
        f'<graphic xlink:href="{absolute_path.as_posix()}"/>'
        '<graphic xlink:href="images/relative.png"/>'
        "</alternatives></fig>",
    )

    with pytest.warns(UserWarning, match="Absolute paths are not allowed"):
        doc = local_fetch_converter.convert(jats_path).document

    image = get_pictures(doc)[0].get_image(doc)
    assert image is not None
    assert image.size == (9, 6)
    assert image.getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize(
    "graphic",
    [
        pytest.param("", id="no-graphic"),
        pytest.param("<graphic/>", id="missing-href"),
        pytest.param('<graphic xlink:href=" "/>', id="blank-href"),
        pytest.param(
            '<graphic xlink:href="https://example.com/figure.png"/>',
            id="remote-href",
        ),
        pytest.param('<graphic xlink:href="figure.svg"/>', id="unsupported-svg"),
    ],
)
def test_jats_file_skips_unavailable_figure_image(
    tmp_path: Path, graphic: str, local_fetch_converter: DocumentConverter
):
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        f"<fig>{graphic}</fig><p>Content after the unavailable figure.</p>",
    )

    doc = local_fetch_converter.convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None
    assert "Content after the unavailable figure." in doc.export_to_markdown()


def test_jats_file_warns_for_missing_figure_image(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    jats_path = tmp_path / "article.nxml"
    write_jats_body(jats_path, '<fig><graphic xlink:href="missing.png"/></fig>')

    with pytest.warns(UserWarning, match="no matching local file exists"):
        doc = local_fetch_converter.convert(jats_path).document

    assert get_pictures(doc)[0].image is None


def test_jats_figure_image_blocks_path_traversal(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    _make_test_image(tmp_path / "outside.png")
    article_dir = tmp_path / "article"
    article_dir.mkdir()
    _make_test_image(article_dir / "fallback.png", size=(9, 6), color=(0, 0, 255))
    jats_path = article_dir / "article.nxml"
    write_jats_body(
        jats_path,
        "<fig><alternatives>"
        '<graphic xlink:href="../outside.png"/>'
        '<graphic xlink:href="fallback.png"/>'
        "</alternatives></fig>"
        "<p>Content after the blocked figure.</p>",
    )

    with pytest.warns(UserWarning, match="Path traversal blocked"):
        doc = local_fetch_converter.convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None
    assert "Content after the blocked figure." in doc.export_to_markdown()


def test_jats_table_published_as_image_keeps_its_caption():
    """A <table-wrap> with a <graphic> instead of a <table> becomes a picture in place."""
    doc = convert_jats_body(
        "<sec><title>Methods</title>"
        "<p>The items are listed in Table 1.</p>"
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>Items of data extraction.</p></caption>"
        '<graphic xlink:href="table1.jpg"/></table-wrap>'
        "<p>Text after the table.</p></sec>"
    )

    assert doc.tables == []
    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].image is None
    caption = pictures[0].captions[0].resolve(doc)
    assert caption.text == "Table 1 Items of data extraction."
    markdown = doc.export_to_markdown()
    before = markdown.index("The items are listed in Table 1.")
    table_caption = markdown.index("Table 1 Items of data extraction.")
    picture = markdown.index("<!-- image -->")
    after = markdown.index("Text after the table.")
    assert before < table_caption < picture < after


def test_jats_table_published_as_alternative_images_is_one_picture():
    doc = convert_jats_body(
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>Two renditions.</p></caption>"
        '<alternatives><graphic xlink:href="table1.tif"/>'
        '<graphic xlink:href="table1.jpg"/></alternatives></table-wrap>'
    )

    assert doc.tables == []
    pictures = get_pictures(doc)
    assert len(pictures) == 1
    caption = pictures[0].captions[0].resolve(doc)
    assert caption.text == "Table 1 Two renditions."


def test_jats_table_with_cells_and_graphic_stays_a_table():
    """When a <table-wrap> has both cells and a <graphic>, only the table is added."""
    doc = convert_jats_body(
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>Cells and image.</p></caption>"
        '<graphic xlink:href="table1.jpg"/>'
        "<table><tr><td>A</td><td>B</td></tr></table></table-wrap>"
    )

    assert len(doc.tables) == 1
    assert get_pictures(doc) == []
    assert doc.tables[0].captions[0].resolve(doc).text == "Table 1 Cells and image."


def test_jats_table_published_as_image_without_caption_keeps_its_label():
    doc = convert_jats_body(
        '<table-wrap id="T2"><label>Table 2</label>'
        '<graphic xlink:href="table2.jpg"/></table-wrap>'
    )

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    assert pictures[0].captions[0].resolve(doc).text == "Table 2"


def test_jats_table_wrap_without_table_or_graphic_adds_nothing():
    doc = convert_jats_body(
        "<sec><title>S</title>"
        '<table-wrap id="T3"><label>Table 3</label>'
        "<caption><p>Only a caption.</p></caption></table-wrap></sec>"
    )

    assert doc.tables == []
    assert get_pictures(doc) == []


def test_jats_table_published_as_image_embeds_image_when_enabled(
    tmp_path: Path, local_fetch_converter: DocumentConverter
):
    _make_test_image(tmp_path / "table1.png", color=(0, 0, 255))
    jats_path = tmp_path / "article.nxml"
    write_jats_body(
        jats_path,
        '<table-wrap id="T1"><label>Table 1</label>'
        "<caption><p>A blue table.</p></caption>"
        '<graphic xlink:href="table1.png"/></table-wrap>',
    )

    doc = local_fetch_converter.convert(jats_path).document

    pictures = get_pictures(doc)
    assert len(pictures) == 1
    image = pictures[0].get_image(doc)
    assert image is not None
    assert image.getpixel((0, 0)) == (0, 0, 255)


def test_jats_element_citation_surname_only_does_not_crash():
    """Citations with only a surname are rendered with that surname."""
    doc = convert_jats_body(
        "<ref-list><ref><element-citation>"
        "<name><surname>Smith</surname></name>"
        "<article-title>Only a surname</article-title>"
        "<year>2020</year>"
        "</element-citation></ref></ref-list>"
    )
    markdown = doc.export_to_markdown()
    assert "Smith" in markdown
    assert "Only a surname" in markdown


def test_jats_element_citation_empty_name_parts_are_omitted():
    """Citations with empty surname and given-names elements omit that author."""
    doc = convert_jats_body(
        "<ref-list><ref><element-citation>"
        "<name><surname></surname><given-names></given-names></name>"
        "<article-title>Empty name parts</article-title>"
        "</element-citation></ref></ref-list>"
    )
    citations = [
        item
        for item, _level in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM
    ]
    assert len(citations) == 1
    assert citations[0].text == "Empty name parts. "


def test_jats_element_citation_given_names_only():
    """Citations with only given-names are rendered with that name."""
    doc = convert_jats_body(
        "<ref-list><ref><element-citation>"
        "<name><given-names>Ada</given-names></name>"
        "<article-title>Given names only</article-title>"
        "</element-citation></ref></ref-list>"
    )
    markdown = doc.export_to_markdown()
    assert "Ada" in markdown
    assert "Given names only" in markdown


def test_jats_element_citation_empty_year_does_not_crash():
    """Citations with an empty year element are rendered without a year."""
    doc = convert_jats_body(
        "<ref-list><ref><element-citation>"
        "<name><surname>Smith</surname><given-names>Jane</given-names></name>"
        "<article-title>Empty year</article-title>"
        "<year></year>"
        "</element-citation></ref></ref-list>"
    )
    markdown = doc.export_to_markdown()
    assert "Smith Jane" in markdown
    assert "Empty year" in markdown


def _citation_runs(doc: DoclingDocument) -> list[list[tuple[str, str | None]]]:
    """Return the (text, hyperlink) runs of each reference, in document order.

    A reference is a list item, either with its own text or with an inline
    group of runs as its only child.
    """
    citations: list[list[tuple[str, str | None]]] = []
    for item in _list_items(doc):
        runs = [item]
        if item.children:
            (group,) = [child.resolve(doc) for child in item.children]
            assert group.label == GroupLabel.INLINE
            runs = [child.resolve(doc) for child in group.children]
        citations.append(
            [
                (run.text, str(run.hyperlink) if run.hyperlink is not None else None)
                for run in runs
            ]
        )
    return citations


def test_jats_mixed_citation_keeps_external_link_target():
    doc = convert_jats_body(
        '<ref-list xmlns:xlink="http://www.w3.org/1999/xlink"><ref id="r1">'
        '<label>1</label><mixed-citation publication-type="other">'
        "BRAVO-CITATION. A work. 2020.\n"
        '<ext-link ext-link-type="uri" xlink:href="https://example.org/c2-citation">'
        "https://example.org/c2-citation</ext-link></mixed-citation></ref></ref-list>"
    )

    assert _citation_runs(doc) == [
        [
            ("BRAVO-CITATION. A work. 2020. ", None),
            ("https://example.org/c2-citation", "https://example.org/c2-citation"),
        ]
    ]
    assert (
        "- BRAVO-CITATION. A work. 2020. "
        "[https://example.org/c2-citation](https://example.org/c2-citation)"
    ) in doc.export_to_markdown()


def test_jats_mixed_citation_keeps_link_target_that_differs_from_its_text():
    doc = convert_jats_body(
        '<ref-list xmlns:xlink="http://www.w3.org/1999/xlink"><ref id="r1">'
        "<mixed-citation>Doe J. Survey data. "
        '<ext-link ext-link-type="uri" xlink:href="https://doi.org/10.5555/survey">'
        "doi:10.5555/survey</ext-link>. Accessed 2020.</mixed-citation></ref>"
        "</ref-list>"
    )

    assert _citation_runs(doc) == [
        [
            ("Doe J. Survey data. ", None),
            ("doi:10.5555/survey", "https://doi.org/10.5555/survey"),
            (". Accessed 2020.", None),
        ]
    ]


def test_jats_mixed_citation_with_one_run_stays_a_single_list_item():
    doc = convert_jats_body(
        '<ref-list xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<ref id="r1"><mixed-citation>Roe R. Plain work. 2019.</mixed-citation></ref>'
        '<ref id="r2"><mixed-citation><ext-link ext-link-type="uri" '
        'xlink:href="https://example.org/data">Example data set</ext-link>'
        "</mixed-citation></ref></ref-list>"
    )

    assert _inline_group_items(doc) == []
    assert _citation_runs(doc) == [
        [("Roe R. Plain work. 2019.", None)],
        [("Example data set", "https://example.org/data")],
    ]


def test_jats_mixed_citation_keeps_inline_styling():
    doc = convert_jats_body(
        '<ref-list><ref id="r1"><mixed-citation>Doe J. Growth of '
        "<italic>Escherichia coli</italic>. <source>Microbiology</source>. "
        "2020;<bold>12</bold>: 1-9.</mixed-citation></ref></ref-list>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (DocItemLabel.TEXT, "Doe J. Growth of ", None),
        (
            DocItemLabel.TEXT,
            "Escherichia coli",
            (False, True, False, False, Script.BASELINE),
        ),
        (DocItemLabel.TEXT, ". Microbiology. 2020;", None),
        (DocItemLabel.TEXT, "12", (True, False, False, False, Script.BASELINE)),
        (DocItemLabel.TEXT, ": 1-9.", None),
    ]


@pytest.mark.parametrize(
    ("contrib", "expected"),
    [
        (
            '<contrib contrib-type="author"><name><given-names>Jane</given-names><surname>Doe</surname></name></contrib>',
            "Jane Doe",
        ),
        (
            '<contrib contrib-type="author"><string-name>Jane Q. Doe</string-name></contrib>',
            "Jane Q. Doe",
        ),
        (
            '<contrib contrib-type="author"><name-alternatives><name><given-names>Jane</given-names><surname>Doe</surname></name><string-name>J. Doe</string-name></name-alternatives></contrib>',
            "Jane Doe",
        ),
        (
            '<contrib contrib-type="author"><collab-name>Example Working Group</collab-name></contrib>',
            "Example Working Group",
        ),
        (
            '<contrib contrib-type="author"><collab>Deprecated Working Group</collab></contrib>',
            "Deprecated Working Group",
        ),
        (
            '<contrib contrib-type="author"><collab-name-alternatives><collab-name>Primary Group</collab-name><collab>Legacy Group</collab></collab-name-alternatives></contrib>',
            "Primary Group",
        ),
        (
            '<contrib contrib-type="author"><collab-alternatives><collab>Alternative Group</collab></collab-alternatives></contrib>',
            "Alternative Group",
        ),
        ('<contrib contrib-type="author"><anonymous/></contrib>', "Anonymous"),
        (
            '<contrib contrib-type="author"><name><surname>Doe</surname></name></contrib>',
            "Doe",
        ),
        (
            '<contrib contrib-type="author"><name><given-names>Jane</given-names></name></contrib>',
            "Jane",
        ),
    ],
)
def test_jats_author_name_variants(contrib: str, expected: str):
    doc = convert_jats_contribs(contrib)

    assert expected in doc.export_to_markdown()


def test_jats_author_affiliations_still_map_from_xref():
    doc = convert_jats_contribs(
        """<contrib contrib-type="author"><name><given-names>Jane</given-names><surname>Doe</surname></name><xref ref-type="aff" rid="aff1">1</xref></contrib>""",
        """<aff id="aff1"><label>1</label><addr-line>Example University</addr-line></aff>""",
    )

    md = doc.export_to_markdown()
    assert "Jane Doe" in md
    assert "Example University" in md


def _furniture_statements(doc: DoclingDocument) -> list[list[tuple[str, str | None]]]:
    """Return the (text, hyperlink) runs of each furniture text, in document order.

    The runs of a text with styling or links share an inline group.
    """
    statements: dict[str, list[tuple[str, str | None]]] = {}
    for item in doc.texts:
        if item.content_layer != ContentLayer.FURNITURE:
            continue
        key = item.self_ref
        if item.parent is not None:
            parent = item.parent.resolve(doc)
            if parent.label == GroupLabel.INLINE:
                assert parent.content_layer == ContentLayer.FURNITURE
                key = parent.self_ref
        statements.setdefault(key, []).append(
            (item.text, str(item.hyperlink) if item.hyperlink is not None else None)
        )
    return list(statements.values())


def test_jats_license_and_funding_statements_are_furniture():
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Control</article-title></title-group>
      <permissions><license xlink:href="https://example.org/c3-licence"><license-p>CHARLIE-LICENCE terms of the
      <ext-link ext-link-type="uri" xlink:href="https://example.org/c3-licence">licence</ext-link>.</license-p></license></permissions>
      <abstract><p>CHARLIE-ABSTRACT paragraph.</p></abstract>
      <funding-group><funding-statement>CHARLIE-FUNDING statement.</funding-statement></funding-group>
""",
        "<sec><title>S</title><p>Body.</p></sec>",
    )

    assert _furniture_statements(doc) == [
        [
            ("CHARLIE-LICENCE terms of the ", None),
            ("licence", "https://example.org/c3-licence"),
            (".", None),
        ],
        [("CHARLIE-FUNDING statement.", None)],
    ]
    # The default export keeps to the body.
    body_markdown = doc.export_to_markdown()
    assert "CHARLIE-ABSTRACT paragraph." in body_markdown
    assert "CHARLIE-LICENCE" not in body_markdown
    assert "CHARLIE-FUNDING" not in body_markdown
    # On request, the statements follow the article content.
    markdown = doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    )
    assert (
        markdown.index("Body.")
        < markdown.index("CHARLIE-LICENCE")
        < markdown.index("CHARLIE-FUNDING")
    )


def test_jats_copyright_and_each_license_paragraph_are_kept_in_order():
    doc = convert_jats_article_meta(
        """
      <title-group><article-title>Permissions</article-title></title-group>
      <permissions>
        <copyright-statement>© The Authors 2026.</copyright-statement>
        <copyright-year>2026</copyright-year>
        <license license-type="cc-by" xlink:href="https://creativecommons.org/licenses/by/4.0/">
          <license-p>Distributed under the CC BY licence.</license-p>
          <license-p>Funded by SCOAP<sup>3</sup></license-p>
        </license>
      </permissions>
      <funding-group>
        <award-group><funding-source>Agency</funding-source></award-group>
        <funding-statement>First grant.</funding-statement>
        <funding-statement>Second grant.</funding-statement>
      </funding-group>
"""
    )

    assert _furniture_statements(doc) == [
        [("© The Authors 2026.", None)],
        [("Distributed under the CC BY licence.", None)],
        [("Funded by SCOAP", None), ("3", None)],
        [("First grant.", None)],
        [("Second grant.", None)],
    ]
    superscripts = [
        item.text
        for item in doc.texts
        if item.formatting is not None and item.formatting.script == Script.SUPER
    ]
    assert superscripts == ["3"]


@pytest.mark.parametrize("use_stream", [False, True])
def test_e2e_jats_conversions(use_stream: bool):
    jats_paths = get_jats_paths()
    converter = get_converter()

    for jats_path in jats_paths:
        gt_path = jats_path.parent.parent / "groundtruth" / jats_path.name
        if use_stream:
            buf = BytesIO(jats_path.open("rb").read())
            stream = DocumentStream(name=jats_path.name, stream=buf)
            conv_result: ConversionResult = converter.convert(stream)
        else:
            conv_result: ConversionResult = converter.convert(jats_path)
        doc: DoclingDocument = conv_result.document

        pred_md: str = doc.export_to_markdown(compact_tables=True)
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            "export to md"
        )

        pred_itxt: str = doc._export_to_indented_text(
            max_text_len=70, explicit_tables=False
        )
        # Core's hierarchy dump adds a separator space even for empty wrappers.
        # Remove only that separator in JATS fixtures; leave text runs unchanged.
        pred_itxt = re.sub(
            r"(?m)^( *item-\d+ at level \d+: footnote:) $", r"\1", pred_itxt
        )
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_jats_empty_article_title_does_not_crash():
    """An empty article-title element produces an empty document title without aborting conversion."""
    doc = convert_jats_article_meta(
        "<title-group><article-title></article-title></title-group>"
    )
    exported = doc.export_to_markdown()
    # Empty title is serialized as an H1 with no text.
    assert exported == "# "


def _caption_texts(doc: DoclingDocument) -> list[str]:
    return [t.text for t in doc.texts if t.label == DocItemLabel.CAPTION]


def test_jats_labelled_figure_without_a_caption_omits_the_word_none():
    """A <fig> may carry a <label> and no <caption>, which is legal and common."""
    doc = convert_jats_body(
        """<sec><p>Text.</p>
        <fig id="f1"><label>Figure 1.</label><graphic/></fig>
        </sec>"""
    )

    assert _caption_texts(doc) == ["Figure 1."]


def test_jats_table_with_an_empty_label_produces_no_caption():
    """An empty <label/> has .text of None in lxml."""
    doc = convert_jats_body(
        """<sec><p>Text.</p>
        <table-wrap id="t1"><label/>
          <table><tbody><tr><td>A</td><td>B</td></tr></tbody></table>
        </table-wrap>
        </sec>"""
    )

    assert _caption_texts(doc) == []


def test_jats_table_cell_keeps_scripts_and_their_whitespace():
    doc = convert_jats_body(
        """<table-wrap id="t1"><table><tbody>
        <tr><td> H<sub>2</sub>O in km<sup>2</sup> <!-- note --></td><td>plain</td></tr>
        </tbody></table></table-wrap>"""
    )

    rich, plain = doc.tables[0].data.table_cells
    assert not isinstance(plain, RichTableCell)
    assert isinstance(rich, RichTableCell)
    inline = rich.ref.resolve(doc).children[0].resolve(doc)
    runs = [child.resolve(doc) for child in inline.children]
    assert [run.text for run in runs] == ["H", "2", "O in km", "2"]
    assert [run.formatting.script if run.formatting else None for run in runs] == [
        None,
        Script.SUB,
        None,
        Script.SUPER,
    ]
    assert "H2O in km2" in doc.export_to_markdown()
    assert "H<sub>2</sub>O in km<sup>2</sup>" in doc.export_to_html()


def test_jats_table_cell_is_rich_only_when_more_than_plain_text():
    doc = convert_jats_body(
        """<table-wrap id="t1"><table><tbody><tr>
        <td>a <bold>b</bold></td>
        <td><ext-link xmlns:xlink="http://www.w3.org/1999/xlink"
            xlink:href="https://example.com/">site</ext-link></td>
        <td>a <inline-formula><alternatives><tex-math>$$x^2$$</tex-math>
            <mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML"><mml:mi>x</mml:mi></mml:math>
            </alternatives></inline-formula> b</td>
        <td>a <named-content>b</named-content></td>
        </tr></tbody></table></table-wrap>"""
    )

    bold, link, formula, plain = doc.tables[0].data.table_cells
    assert not isinstance(plain, RichTableCell)

    assert isinstance(bold, RichTableCell)
    inline = bold.ref.resolve(doc).children[0].resolve(doc)
    assert inline.children[1].resolve(doc).formatting.bold

    assert isinstance(link, RichTableCell)
    run = link.ref.resolve(doc).children[0].resolve(doc)
    assert (run.text, str(run.hyperlink)) == ("site", "https://example.com/")

    assert isinstance(formula, RichTableCell)
    inline = formula.ref.resolve(doc).children[0].resolve(doc)
    runs = [child.resolve(doc) for child in inline.children]
    assert [run.label for run in runs] == [
        DocItemLabel.TEXT,
        DocItemLabel.FORMULA,
        DocItemLabel.TEXT,
    ]
    assert runs[1].text == "x^2"


def test_jats_rich_table_cell_keeps_the_boundary_between_blocks():
    doc = convert_jats_body(
        """<table-wrap id="t1"><table><tbody>
        <tr><td><p>x<sup>1</sup></p><p>Next</p></td>
        <td><list><list-item><p>a<sup>2</sup></p></list-item>
        <list-item><p>b</p></list-item></list></td></tr>
        </tbody></table></table-wrap>"""
    )

    paragraphs, items = doc.tables[0].data.table_cells
    assert isinstance(paragraphs, RichTableCell)
    assert isinstance(items, RichTableCell)
    markdown = doc.export_to_markdown()
    assert "x1 Next" in markdown
    assert "a2 b" in markdown


def test_jats_table_cell_with_only_a_blank_script_stays_plain():
    doc = convert_jats_body(
        """<table-wrap id="t1"><table><tbody>
        <tr><td><sub> </sub></td><td> <sup> </sup>x</td></tr>
        </tbody></table></table-wrap>"""
    )

    blank, text = doc.tables[0].data.table_cells
    assert not isinstance(blank, RichTableCell)
    assert not isinstance(text, RichTableCell)
    assert doc.groups == []


def test_jats_nested_table_is_skipped_and_keeps_no_table_item():
    doc = convert_jats_body(
        """<sec><p>Text.</p><table-wrap id="t1"><table><tbody><tr><td>
        <table><tbody><tr><td>x<sub>1</sub></td></tr></tbody></table>
        </td></tr></tbody></table></table-wrap></sec>"""
    )

    assert doc.tables == []
