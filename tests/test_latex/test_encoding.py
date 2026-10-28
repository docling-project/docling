# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO

import pytest

from docling.backend.latex.utils.encoding import decode_latex_content
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter


@pytest.mark.parametrize(
    ("encoding", "inputenc", "text"),
    [
        ("cp1252", "cp1252", "The \u201cquoted\u201d price is 5\u20ac \u2013 net."),
        ("latin-1", "latin1", "Résumé of the thesis."),
    ],
    ids=["cp1252", "latin-1"],
)
@pytest.mark.parametrize("from_path", [False, True], ids=["stream", "path"])
def test_latex_8bit_source_decoded(tmp_path, encoding, inputenc, text, from_path):
    """Windows-1252 punctuation is not decoded as Latin-1 control characters."""
    latex = (
        "\\documentclass{article}\n"
        f"\\usepackage[{inputenc}]{{inputenc}}\n"
        "\\begin{document}\n"
        f"{text}\n"
        "\\end{document}\n"
    ).encode(encoding)
    path = tmp_path / "main.tex"
    path.write_bytes(latex)
    source = (
        path if from_path else DocumentStream(name="main.tex", stream=BytesIO(latex))
    )

    converter = DocumentConverter(allowed_formats=[InputFormat.LATEX])
    md = converter.convert(source).document.export_to_markdown()

    assert text in md


def test_decode_latex_content_falls_back_to_latin1(tmp_path):
    """Bytes that Windows-1252 leaves undefined still decode as Latin-1."""
    raw = b"caf\xe9 \x81"
    path = tmp_path / "main.tex"
    path.write_bytes(raw)

    assert decode_latex_content(BytesIO(raw)) == "café \x81"
    assert decode_latex_content(path) == "café \x81"
