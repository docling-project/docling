# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument


def test_nodes_to_text_treats_missing_nodelist_as_empty():
    """LatexGroupNode.nodelist can be None; len(None) raised TypeError."""
    latex_content = b"\\documentclass{article}\\begin{document}Hello.\\end{document}"
    in_doc = InputDocument(
        path_or_stream=BytesIO(latex_content),
        format=InputFormat.LATEX,
        backend=LatexDocumentBackend,
        filename="test.tex",
    )
    backend = LatexDocumentBackend(in_doc=in_doc, path_or_stream=BytesIO(latex_content))
    assert backend._nodes_to_text(None) == ""
    assert backend._nodes_to_text([]) == ""
