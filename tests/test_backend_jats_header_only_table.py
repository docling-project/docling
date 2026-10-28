# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter


def convert_jats_body(body: str):
    xml = f"""<!DOCTYPE article
PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article article-type="research-article">
  <body>
    {body}
  </body>
</article>
"""
    converter = DocumentConverter(allowed_formats=[InputFormat.XML_JATS])
    stream = DocumentStream(name="body-test.nxml", stream=BytesIO(xml.encode()))
    return converter.convert(stream).document


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
