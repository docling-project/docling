# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest
from docling_core.types.doc import DoclingDocument

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


def _convert(body: str) -> DoclingDocument:
    source = f"<html><body>{body}</body></html>".encode()
    stream = DocumentStream(name="page.html", stream=BytesIO(source))
    return DocumentConverter().convert(stream).document


@pytest.mark.parametrize(
    "heading",
    ["<h2></h2>", "<h2> &nbsp; </h2>", '<h2><a name="a"></a></h2>', "<h2><br></h2>"],
)
def test_heading_without_text_adds_no_item(heading: str):
    doc = _convert(f"<h1>Title</h1><h2>Setup</h2><p>one</p>{heading}<p>two</p>")

    assert doc.export_to_markdown() == "# Title\n\n## Setup\n\none\n\ntwo"
    # The next paragraph stays in the last section that has a title.
    assert doc.texts[-1].parent == doc.texts[1].get_ref()


def test_heading_without_text_keeps_its_image():
    image = "data:image/png;base64,iVBORw0KGgo="
    doc = _convert(f'<h1>Title</h1><h2><img src="{image}"></h2><p>after</p>')

    assert len(doc.pictures) == 1
    assert doc.export_to_markdown() == "# Title\n\n<!-- image -->\n\nafter"
