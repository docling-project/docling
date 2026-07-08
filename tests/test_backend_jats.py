import os
from io import BytesIO
from pathlib import Path

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, GroupLabel, TextItem
from docling_core.types.doc.document import Script

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter

from .test_data_gen_flag import GEN_TEST_DATA
from .verify_utils import verify_document, verify_export

GENERATE = GEN_TEST_DATA


def get_jats_paths():
    directory = Path(os.path.dirname(__file__) + "/data/jats/")
    xml_files = sorted(directory.rglob("*.nxml"))
    return xml_files


def get_converter():
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_JATS])
    return converter


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


def convert_jats_body(body: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <body>
    {body}
  </body>
</article>
"""
    stream = DocumentStream(
        name="body-test.nxml",
        stream=BytesIO(xml.encode()),
    )

    conv_result: ConversionResult = get_converter().convert(stream)
    return conv_result.document


def convert_jats_article_meta(article_meta: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <front>
    <article-meta>
      {article_meta}
    </article-meta>
  </front>
</article>
"""
    stream = DocumentStream(name="article-meta-test.nxml", stream=BytesIO(xml.encode()))
    conv_result: ConversionResult = get_converter().convert(stream)
    return conv_result.document


def convert_jats_body(body_content: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <front>
    <article-meta>
      <title-group>
        <article-title>Body Test</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    {body_content}
  </body>
</article>
"""
    stream = DocumentStream(name="body-test.nxml", stream=BytesIO(xml.encode()))
    conv_result: ConversionResult = get_converter().convert(stream)
    return conv_result.document


def convert_jats_contribs(contribs: str, affiliations: str = "") -> DoclingDocument:
    return convert_jats_article_meta(
        f"""
      <title-group><article-title>Author Variant Test</article-title></title-group>
      <contrib-group>{contribs}</contrib-group>
      {affiliations}
"""
    )


def test_jats_structured_abstract_sections_are_preserved():
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
    assert "## Abstract" in md
    assert "Background: Background text." in md
    assert "Methods: Methods text." in md


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


def _inline_group_items(doc: DoclingDocument) -> list[list]:
    """Return the resolved child items of every INLINE group, in document order."""
    return [
        [child.resolve(doc) for child in group.children]
        for group in doc.groups
        if group.label == GroupLabel.INLINE
    ]


@pytest.mark.parametrize(
    ("paragraph", "expected"),
    [
        pytest.param(
            "The mass energy relation <inline-formula><tex-math>$$E=mc^2$$</tex-math></inline-formula> is famous.",
            [
                (DocItemLabel.TEXT, "The mass energy relation", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, "is famous.", None),
            ],
            id="text-formula-text",
        ),
        pytest.param(
            "Given <inline-formula><tex-math>$$a^2$$</tex-math></inline-formula> and <inline-formula><tex-math>$$b^2$$</tex-math></inline-formula> we sum.",
            [
                (DocItemLabel.TEXT, "Given", None),
                (DocItemLabel.FORMULA, "a^2", None),
                (DocItemLabel.TEXT, "and", None),
                (DocItemLabel.FORMULA, "b^2", None),
                (DocItemLabel.TEXT, "we sum.", None),
            ],
            id="multiple-formulas",
        ),
        pytest.param(
            # loose text inside <inline-formula> around the <tex-math> is preserved
            # (adjacent unstyled runs are coalesced into one segment)
            "The relation <inline-formula>foo <tex-math>$$E=mc^2$$</tex-math> bar</inline-formula> holds.",
            [
                (DocItemLabel.TEXT, "The relation foo", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, "bar holds.", None),
            ],
            id="text-inside-inline-formula",
        ),
        pytest.param(
            # tex-math is not always wrapped in $$...$$ in real JATS files
            "The relation <inline-formula><tex-math>E=mc^2</tex-math></inline-formula> holds.",
            [
                (DocItemLabel.TEXT, "The relation", None),
                (DocItemLabel.FORMULA, "E=mc^2", None),
                (DocItemLabel.TEXT, "holds.", None),
            ],
            id="bare-tex-math",
        ),
        pytest.param(
            "We use <inline-formula><italic>x</italic> <tex-math>$$a^2$$</tex-math></inline-formula> here.",
            [
                (DocItemLabel.TEXT, "We use", None),
                (DocItemLabel.TEXT, "x", (False, True, False, False, Script.BASELINE)),
                (DocItemLabel.FORMULA, "a^2", None),
                (DocItemLabel.TEXT, "here.", None),
            ],
            id="italic-inside-formula",
        ),
        pytest.param(
            "Index <inline-formula>x<sub>i</sub> <tex-math>$$x_i$$</tex-math></inline-formula> shown.",
            [
                (DocItemLabel.TEXT, "Index x", None),
                (DocItemLabel.TEXT, "i", (False, False, False, False, Script.SUB)),
                (DocItemLabel.FORMULA, "x_i", None),
                (DocItemLabel.TEXT, "shown.", None),
            ],
            id="subscript-inside-formula",
        ),
        pytest.param(
            "Take <inline-formula><bold><italic>v</italic></bold> <tex-math>$$v$$</tex-math></inline-formula> next.",
            [
                (DocItemLabel.TEXT, "Take", None),
                (DocItemLabel.TEXT, "v", (True, True, False, False, Script.BASELINE)),
                (DocItemLabel.FORMULA, "v", None),
                (DocItemLabel.TEXT, "next.", None),
            ],
            id="nested-emphasis-inside-formula",
        ),
        pytest.param(
            # a tex-math nested inside an emphasis tag is still parsed as a
            # formula (not leaked as raw ``$$...$$`` text)
            "Val <inline-formula><italic><tex-math>$$x^2$$</tex-math></italic></inline-formula> shown.",
            [
                (DocItemLabel.TEXT, "Val", None),
                (DocItemLabel.FORMULA, "x^2", None),
                (DocItemLabel.TEXT, "shown.", None),
            ],
            id="tex-math-inside-emphasis",
        ),
        pytest.param(
            # emphasis outside the formula is now preserved (general text styling),
            # and adjacent unstyled runs are coalesced
            "Compare <italic>lhs</italic> <inline-formula><tex-math>$$a$$</tex-math> rhs</inline-formula> and <inline-formula><tex-math>$$b$$</tex-math></inline-formula> now.",
            [
                (DocItemLabel.TEXT, "Compare", None),
                (
                    DocItemLabel.TEXT,
                    "lhs",
                    (False, True, False, False, Script.BASELINE),
                ),
                (DocItemLabel.FORMULA, "a", None),
                (DocItemLabel.TEXT, "rhs and", None),
                (DocItemLabel.FORMULA, "b", None),
                (DocItemLabel.TEXT, "now.", None),
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


def test_jats_inline_formula_styled_content_markdown_rendering():
    doc = convert_jats_body(
        "<sec><title>T</title>"
        "<p>Use <inline-formula><italic>x</italic> "
        "<tex-math>$$a^2$$</tex-math></inline-formula> now.</p>"
        "</sec>"
    )

    md = doc.export_to_markdown()
    assert "*x*" in md
    assert "a^2" in md


def test_jats_paragraph_emphasis_is_preserved():
    doc = convert_jats_body(
        "<sec><title>T</title>"
        "<p>The species <italic>Homo sapiens</italic> is <bold>common</bold>.</p>"
        "</sec>"
    )

    groups = _inline_group_items(doc)
    assert len(groups) == 1
    assert [_formatting_tuple(item) for item in groups[0]] == [
        (DocItemLabel.TEXT, "The species", None),
        (
            DocItemLabel.TEXT,
            "Homo sapiens",
            (False, True, False, False, Script.BASELINE),
        ),
        (DocItemLabel.TEXT, "is", None),
        (DocItemLabel.TEXT, "common", (True, False, False, False, Script.BASELINE)),
        (DocItemLabel.TEXT, ".", None),
    ]


def test_jats_plain_paragraph_stays_a_single_text_item():
    doc = convert_jats_body(
        "<sec><title>T</title><p>Plain text with a <xref>1</xref> citation.</p></sec>"
    )

    # no emphasis (and coalesced xref) → a single TEXT item, no inline group
    assert _inline_group_items(doc) == []
    texts = [t.text for t in doc.texts if t.label == DocItemLabel.TEXT]
    assert texts == ["Plain text with a 1 citation."]


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


@pytest.mark.parametrize(
    ("contrib", "expected"),
    [
        (
            """<contrib contrib-type="author"><name><given-names>Jane</given-names><surname>Doe</surname></name></contrib>""",
            "Jane Doe",
        ),
        (
            """<contrib contrib-type="author"><string-name>Jane Q. Doe</string-name></contrib>""",
            "Jane Q. Doe",
        ),
        (
            """<contrib contrib-type="author"><name-alternatives><name><given-names>Jane</given-names><surname>Doe</surname></name><string-name>J. Doe</string-name></name-alternatives></contrib>""",
            "Jane Doe",
        ),
        (
            """<contrib contrib-type="author"><collab-name>Example Working Group</collab-name></contrib>""",
            "Example Working Group",
        ),
        (
            """<contrib contrib-type="author"><collab>Deprecated Working Group</collab></contrib>""",
            "Deprecated Working Group",
        ),
        (
            """<contrib contrib-type="author"><collab-name-alternatives><collab-name>Primary Group</collab-name><collab>Legacy Group</collab></collab-name-alternatives></contrib>""",
            "Primary Group",
        ),
        (
            """<contrib contrib-type="author"><collab-alternatives><collab>Alternative Group</collab></collab-alternatives></contrib>""",
            "Alternative Group",
        ),
        (
            """<contrib contrib-type="author"><anonymous/></contrib>""",
            "Anonymous",
        ),
        (
            """<contrib contrib-type="author"><name><surname>Doe</surname></name></contrib>""",
            "Doe",
        ),
        (
            """<contrib contrib-type="author"><name><given-names>Jane</given-names></name></contrib>""",
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


def test_e2e_jats_conversions(use_stream=False):
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
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            "export to indented-text"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), "export to json"


def test_e2e_jats_conversions_stream():
    test_e2e_jats_conversions(use_stream=True)


def test_e2e_jats_conversions_no_stream():
    test_e2e_jats_conversions(use_stream=False)
