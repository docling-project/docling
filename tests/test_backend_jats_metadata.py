# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for the article metadata statements of the JATS backend."""

from io import BytesIO

from docling_core.types.doc import (
    ContentLayer,
    DoclingDocument,
    GroupLabel,
    Script,
)

from docling.datamodel.base_models import DocumentStream

from .test_backend_jats import get_converter


def convert_jats_front(article_meta: str, body: str = "") -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Publishing DTD v1.3 20210610//EN" "JATS-journalpublishing1-3.dtd">
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
  <front>
    <article-meta>{article_meta}</article-meta>
  </front>
  <body>{body}</body>
</article>
"""
    stream = DocumentStream(name="metadata-test.nxml", stream=BytesIO(xml.encode()))
    return get_converter().convert(stream).document


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
    doc = convert_jats_front(
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
            ("CHARLIE-LICENCE terms of the", None),
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
    doc = convert_jats_front(
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
