# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the sub-articles of the JATS backend."""

from io import BytesIO

from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem

from docling.datamodel.base_models import DocumentStream

from .test_backend_jats import get_converter


def convert_jats_article(content: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Publishing DTD v1.3 20210610//EN" "JATS-journalpublishing1-3.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
  <front>
    <article-meta>
      <title-group><article-title>Main</article-title></title-group>
    </article-meta>
  </front>
  {content}
</article>
"""
    stream = DocumentStream(name="sub-article-test.nxml", stream=BytesIO(xml.encode()))
    return get_converter().convert(stream).document


def _text_items(doc: DoclingDocument) -> list[tuple[int, DocItemLabel, str]]:
    """Return the (level, label, text) of each text item, in document order."""
    return [
        (level, item.label, item.text)
        for item, level in doc.iterate_items()
        if isinstance(item, TextItem)
    ]


def test_jats_sub_article_content_is_preserved():
    doc = convert_jats_article(
        """
  <body><sec><title>Results</title><p>Main paragraph.</p></sec></body>
  <back><ack><p>Main acknowledgment.</p></ack></back>
  <sub-article article-type="decision-letter" id="sa1">
    <front-stub>
      <title-group><article-title>Decision letter</article-title></title-group>
    </front-stub>
    <body>
      <p>Reviewer paragraph.</p>
      <sec><title>Essential revisions</title><p>Revision paragraph.</p></sec>
    </body>
    <response response-type="reply" id="sa1-r1">
      <front-stub>
        <title-group><article-title>Author response</article-title></title-group>
      </front-stub>
      <body><p>Reply paragraph.</p></body>
    </response>
  </sub-article>
"""
    )

    assert _text_items(doc) == [
        (1, DocItemLabel.TITLE, "Main"),
        (2, DocItemLabel.SECTION_HEADER, "Results"),
        (3, DocItemLabel.TEXT, "Main paragraph."),
        (2, DocItemLabel.SECTION_HEADER, "Acknowledgments"),
        (3, DocItemLabel.TEXT, "Main acknowledgment."),
        (2, DocItemLabel.SECTION_HEADER, "Decision letter"),
        (3, DocItemLabel.TEXT, "Reviewer paragraph."),
        (3, DocItemLabel.SECTION_HEADER, "Essential revisions"),
        (4, DocItemLabel.TEXT, "Revision paragraph."),
        (3, DocItemLabel.SECTION_HEADER, "Author response"),
        (4, DocItemLabel.TEXT, "Reply paragraph."),
    ]


def test_jats_untitled_sub_article_content_is_preserved():
    doc = convert_jats_article(
        """
  <body><p>Main paragraph.</p></body>
  <sub-article article-type="author-comment" id="sa1">
    <front-stub><article-id pub-id-type="doi">10.0000/main.sa1</article-id></front-stub>
    <body><p>Comment paragraph.</p></body>
    <back><ack><p>Comment acknowledgment.</p></ack></back>
  </sub-article>
"""
    )

    assert _text_items(doc) == [
        (1, DocItemLabel.TITLE, "Main"),
        (2, DocItemLabel.TEXT, "Main paragraph."),
        (2, DocItemLabel.TEXT, "Comment paragraph."),
        (2, DocItemLabel.SECTION_HEADER, "Acknowledgments"),
        (3, DocItemLabel.TEXT, "Comment acknowledgment."),
    ]


def test_jats_sub_article_body_does_not_replace_missing_main_body():
    doc = convert_jats_article(
        """
  <sub-article article-type="decision-letter" id="sa1">
    <front-stub>
      <title-group><article-title>Decision letter</article-title></title-group>
    </front-stub>
    <body><p>Reviewer paragraph.</p></body>
  </sub-article>
"""
    )

    assert _text_items(doc) == [
        (1, DocItemLabel.TITLE, "Main"),
        (2, DocItemLabel.SECTION_HEADER, "Decision letter"),
        (3, DocItemLabel.TEXT, "Reviewer paragraph."),
    ]
