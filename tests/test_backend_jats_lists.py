# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Tests for where the JATS backend places lists and the content around them."""

from io import BytesIO

from docling_core.types.doc import DocItemLabel, DoclingDocument, ListItem

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter


def _convert_body(body: str) -> DoclingDocument:
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <body>
    {body}
  </body>
</article>
"""
    stream = DocumentStream(name="body-test.nxml", stream=BytesIO(xml.encode()))
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_JATS])
    return converter.convert(stream).document


def test_content_after_a_list_stays_out_of_the_list():
    doc = _convert_body(
        """
        <sec>
          <title>Steps</title>
          <list list-type="bullet">
            <list-item><p>first</p></list-item>
            <list-item><p>second</p></list-item>
          </list>
          <p>After the list.</p>
        </sec>
        """
    )

    heading = next(item for item in doc.texts if item.text == "Steps")
    after = next(item for item in doc.texts if item.text == "After the list.")
    assert after.label == DocItemLabel.TEXT
    assert after.parent == heading.get_ref()

    items = [item for item, _ in doc.iterate_items() if isinstance(item, ListItem)]
    list_group = items[0].parent.resolve(doc)
    assert [child.resolve(doc).text for child in list_group.children] == [
        "first",
        "second",
    ]
    assert "- second\n\nAfter the list." in doc.export_to_markdown()
