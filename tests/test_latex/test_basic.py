from pathlib import Path

import pytest

from docling.backend.latex_backend import LatexDocumentBackend
from docling.datamodel.backend_options import LatexBackendOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult, DoclingDocument

from ..test_data_gen_flag import GEN_TEST_DATA
from ..verify_utils import verify_document, verify_export
from ._utils import (
    LATEX_DATA_DIR,
    get_latex_converter,
    make_backend,
    make_backend_from_path,
)

GENERATE = GEN_TEST_DATA


def test_latex_basic_conversion():
    backend = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    Hello World.
    \\end{document}
    """
    )
    doc = backend.convert()

    headers = [t for t in doc.texts if t.label == "section_header"]
    paragraphs = [t for t in doc.texts if t.label != "section_header"]

    assert len(doc.texts) > 0
    assert len(headers) == 1
    assert headers[0].text == "Introduction"
    assert "Hello World" in paragraphs[0].text


def test_latex_preamble_filter():
    backend = make_backend(
        b"""
    \\documentclass{article}
    \\usepackage{test}
    \\title{Ignored Title}
    \\begin{document}
    Real Content
    \\end{document}
    """
    )
    doc = backend.convert()

    full_text = doc.export_to_markdown()
    assert "Real Content" in full_text
    assert "Ignored Title" in full_text
    assert "usepackage" not in full_text


def test_latex_is_valid():
    backend = make_backend(
        b"\\documentclass{article}\\begin{document}Content\\end{document}"
    )
    assert backend.is_valid() is True

    empty_backend = make_backend(b"   ", filename="empty.tex")
    assert empty_backend.is_valid() is False


def test_latex_supports_pagination():
    assert LatexDocumentBackend.supports_pagination() is False


def test_latex_supported_formats():
    formats = LatexDocumentBackend.supported_formats()
    assert InputFormat.LATEX in formats


def test_latex_file_path_loading(tmp_path):
    latex_file = tmp_path / "test.tex"
    latex_file.write_text(
        r"""
    \documentclass{article}
    \begin{document}
    File content here.
    \end{document}
    """
    )

    doc = make_backend_from_path(latex_file).convert()
    assert "File content here" in doc.export_to_markdown()


def test_latex_no_document_env():
    doc = make_backend(
        b"""
    \\section{Direct Section}
    Some direct content without document environment.
    """
    ).convert()

    md = doc.export_to_markdown()
    assert "Direct Section" in md or "direct content" in md


def test_latex_newline_macro():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    Line one\\\\
    Line two
    \\end{document}
    """
    ).convert()
    assert doc is not None


def test_latex_filecontents_ignored():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{filecontents}{sample.bib}
    @article{test, author={A}, title={B}}
    \\end{filecontents}
    \\begin{document}
    Actual content.
    \\end{document}
    """
    ).convert()

    md = doc.export_to_markdown()
    assert "Actual content" in md
    assert "@article" not in md


def test_latex_document_with_leading_comments():
    doc = make_backend(
        b"""% This is a leading comment
% Another comment line
\\documentclass{article}
\\begin{document}
\\section{Test Section}
This is test content.
\\end{document}
"""
    ).convert()

    assert len(doc.texts) > 0
    md = doc.export_to_markdown()
    assert "Test Section" in md
    assert "test content" in md


@pytest.fixture(scope="module")
def latex_paths() -> list[Path]:
    directory = LATEX_DATA_DIR
    if not directory.exists():
        return []

    paths = list(directory.glob("*.tex"))
    for subdir in directory.iterdir():
        if subdir.is_dir():
            if (subdir / "main.tex").exists():
                paths.append(subdir / "main.tex")
            elif (subdir / f"arxiv_{subdir.name}.tex").exists():
                paths.append(subdir / f"arxiv_{subdir.name}.tex")

    return sorted(paths)


def test_e2e_latex_conversions(latex_paths):
    if not latex_paths:
        pytest.skip("No LaTeX test files found")

    converter = get_latex_converter()
    for latex_path in latex_paths:
        if latex_path.parent.resolve() == LATEX_DATA_DIR.resolve():
            gt_name = latex_path.name
        else:
            gt_name = f"{latex_path.parent.name}_{latex_path.name}"

        gt_path = LATEX_DATA_DIR.parent / "groundtruth" / "docling_v2" / gt_name
        conv_result: ConversionResult = converter.convert(latex_path)
        doc: DoclingDocument = conv_result.document

        pred_md = doc.export_to_markdown()
        assert verify_export(pred_md, str(gt_path) + ".md", generate=GENERATE), (
            f"Markdown export mismatch for {latex_path}"
        )

        pred_itxt = doc._export_to_indented_text(max_text_len=70, explicit_tables=False)
        assert verify_export(pred_itxt, str(gt_path) + ".itxt", generate=GENERATE), (
            f"Indented text export mismatch for {latex_path}"
        )

        assert verify_document(doc, str(gt_path) + ".json", GENERATE), (
            f"Document JSON mismatch for {latex_path}"
        )


def test_latex_convert_error_fallback():
    latex_content = b"\\documentclass{article}\\begin{document}Hello\\end{document}"
    options = LatexBackendOptions(parse_timeout=0.05)
    backend = make_backend(latex_content, options=options)

    def _raise(doc):
        raise RuntimeError("Simulated parse failure")

    backend._do_parse_and_process = _raise  # type: ignore[method-assign]
    assert backend.convert() is not None


def test_latex_input_cycle_detection(tmp_path):
    file_a = tmp_path / "a.tex"
    file_b = tmp_path / "b.tex"
    file_a.write_text(
        "\\documentclass{article}\\begin{document}A content\\input{b}\\end{document}"
    )
    file_b.write_text("B content\\input{a}")

    doc = make_backend_from_path(file_a, filename="a.tex").convert()
    assert "A content" in doc.export_to_markdown()
