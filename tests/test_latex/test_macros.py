from docling_core.types.doc import DocItemLabel

from ._utils import make_backend


def test_latex_escaped_chars():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    value is 23\\% which is high.
    Costs \\$100.
    \\end{document}
    """
    ).convert()

    text_items = [
        t.text
        for t in doc.texts
        if t.label == DocItemLabel.TEXT or t.label == DocItemLabel.PARAGRAPH
    ]
    full_text = " ".join(text_items)
    assert "23%" in full_text or "23\\%" in full_text
    assert "which is high" in full_text
    assert "$100" in full_text or "\\$100" in full_text


def test_latex_unknown_macro_fallback():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    \\unknownmacro{Known Content}
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Known Content" in md


def test_latex_footnote():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    Main text\\footnote{This is a footnote}.
    \\end{document}
    """
    ).convert()

    footnotes = [t for t in doc.texts if t.label == DocItemLabel.FOOTNOTE]
    assert len(footnotes) >= 1
    assert "footnote" in footnotes[0].text


def test_latex_citet_macro():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    According to \\citet{author2020}, this is correct.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "[author2020]" in md
    assert "According to" in md


def test_latex_citations():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    As shown in \\cite{smith2020} and \\citep{jones2021}.
    Also see \\ref{fig:1} and \\eqref{eq:main}.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "[smith2020]" in md
    assert "[jones2021]" in md
    assert "[fig:1]" in md
    assert "[eq:main]" in md


def test_latex_title_macro():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\title{Document Title}
    \\maketitle
    Some content.
    \\end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.TITLE]) >= 1


def test_latex_label():
    backend = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\section{Introduction}
    \\label{sec:intro}
    Some content.
    \\end{document}
    """
    )
    backend.convert()
    assert "sec:intro" in backend.labels


def test_latex_text_formatting():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    This is \\textbf{bold} and \\textit{italic} and \\emph{emphasized}.
    Also \\texttt{monospace} and \\underline{underlined}.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "bold" in md
    assert "italic" in md
    assert "emphasized" in md


def test_latex_marginpar():
    assert (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    Main text with marginpar.
    \\end{document}
    """
        ).convert()
        is not None
    )


def test_latex_tilde_macro():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    Dr.~Smith arrived.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Smith" in md


def test_latex_citet_macro_2():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    \\citet{author2022} showed this.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "[author2022]" in md
    assert "showed this" in md


def test_latex_custom_macro_with_backslash():
    doc = make_backend(
        b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
\\end{document}
"""
    ).convert()
    md = doc.export_to_markdown()
    assert len(doc.texts) > 0
    assert "special term" in md and "42" in md


def extract_macro_name_old(raw_string):
    macro_name_raw = raw_string.strip("{} ")
    return macro_name_raw.lstrip("\\")


def extract_macro_name_new(raw_string):
    macro_name = raw_string.strip("{} \n\t\\")
    if macro_name.startswith("\\"):
        macro_name = macro_name[1:]
    return macro_name


def test_macro_extraction():
    test_cases = [
        (r"{\myterm}", "myterm"),
        (r"\myterm", "myterm"),
        (r"{ \myterm }", "myterm"),
        (r"{  \myvalue  }", "myvalue"),
        (r"{\important}", "important"),
        (r"{ \test }", "test"),
        (r"{\alpha}", "alpha"),
    ]
    assert all(
        extract_macro_name_new(input_str) == expected
        for input_str, expected in test_cases
    )


def test_edge_cases():
    edge_cases = [
        (r"{\cmd}", "cmd"),
        (r"{\\cmd}", "cmd"),
        (r"{ \cmd }", "cmd"),
        (r"{\   cmd   }", "cmd"),
        (r"{\my_macro}", "my_macro"),
        (r"{\MyMacro}", "MyMacro"),
        (r"{\MACRO}", "MACRO"),
    ]
    assert all(
        extract_macro_name_new(input_str) == expected
        for input_str, expected in edge_cases
    )


def test_debug_macro_extraction():
    backend = make_backend(
        b"""\\documentclass{article}
\\newcommand{\\myterm}{special term}
\\newcommand{\\myvalue}{42}
\\begin{document}
This is \\myterm and the value is \\myvalue.
\\end{document}
"""
    )
    md = backend.convert().export_to_markdown()
    assert "myterm" in backend._custom_macros
    assert backend._custom_macros["myterm"] == "special term"
    assert "special term" in md
    assert "42" in md


def test_latex_href_macro():
    md = (
        make_backend(
            rb"""
    \documentclass{article}
    \begin{document}
    Visit \href{https://example.com}{Example Site} for more.
    \end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Example Site" in md
    assert "https://example.com" in md


def test_latex_textcolor_macro():
    md = (
        make_backend(
            rb"""
    \documentclass{article}
    \begin{document}
    This is \textcolor{red}{important} text.
    Also \colorbox{yellow}{highlighted} here.
    \end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "important" in md
    assert "highlighted" in md
    assert "red" not in md
    assert "yellow" not in md


def test_latex_custom_macro_parameters():
    md = (
        make_backend(
            rb"""
    \documentclass{article}
    \newcommand{\highlight}[1]{\textcolor{white}{\textbf{#1}}}
    \newcommand{\metric}[2]{#1{\scriptsize$\_{#2}$}}
    \begin{document}
    \highlight{Result}
    \metric{Accuracy}{test}
    \end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Result" in md
    assert "Accuracy" in md
    assert "test" in md
    assert "#1" not in md
    assert "#2" not in md
    assert "\\textcolor" not in md
    assert "\\scriptsize" not in md


def test_latex_legacy_font_switches():
    md = (
        make_backend(
            rb"""
    \documentclass{article}
    \begin{document}
    {\bf bold text} and {\it italic text}.
    {\tt monospace} and {\large big} and {\tiny small}.
    Normal content here.
    \end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "bold text" in md
    assert "italic text" in md
    assert "Normal content here" in md


def test_latex_accent_macro():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    caf\'{e} and na\"{i}ve.
    \end{document}
    """
    ).convert()
    assert "caf" in doc.export_to_markdown()
    assert len(doc.texts) > 0


def test_latex_renewcommand():
    backend = make_backend(
        rb"""
    \documentclass{article}
    \newcommand{\foo}{original}
    \renewcommand{\foo}{replaced}
    \providecommand{\bar}{provided}
    \begin{document}
    Value is \foo{} and \bar{}.
    \end{document}
    """
    )
    backend.convert()
    assert "foo" in backend._custom_macros
    assert backend._custom_macros["foo"] == "replaced"
    assert "bar" in backend._custom_macros
    assert backend._custom_macros["bar"] == "provided"


def test_latex_author_date():
    md = (
        make_backend(
            b"""
    \\documentclass{article}
    \\begin{document}
    \\title{My Paper}
    \\author{Jane Doe}
    \\date{January 2025}
    Some content.
    \\end{document}
    """
        )
        .convert()
        .export_to_markdown()
    )
    assert "Jane Doe" in md
    assert "January 2025" in md


def test_vspace_argument_does_not_leak():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    Before text.
    \\vspace{-1mm}
    After text.
    \\end{document}
    """
    ).convert()

    all_text = " ".join(t.text for t in doc.texts)
    assert "-1mm" not in all_text, f"vspace argument leaked into text: {all_text!r}"
    assert "Before text" in all_text
    assert "After text" in all_text


def test_hspace_argument_does_not_leak():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    Left text.\\hspace{0.2cm}Right text.
    \\end{document}
    """
    ).convert()

    all_text = " ".join(t.text for t in doc.texts)
    assert "0.2cm" not in all_text, f"hspace argument leaked into text: {all_text!r}"
    assert "Left text" in all_text
    assert "Right text" in all_text
