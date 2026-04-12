from docling_core.types.doc import DocItemLabel

from ._utils import make_backend


def test_latex_math_parsing():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    Inline math: $E=mc^2$.
    Display math:
    $$
    x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}
    $$
    Aligned equations:
    \begin{align}
    a &= b + c \\
    d &= e + f
    \end{align}
    \end{document}
    """
    ).convert()

    formulas = [t for t in doc.texts if t.label == DocItemLabel.FORMULA]
    paragraphs = [
        t for t in doc.texts if t.label in [DocItemLabel.PARAGRAPH, DocItemLabel.TEXT]
    ]
    full_text = " ".join(p.text for p in paragraphs)
    md = doc.export_to_markdown()

    assert len(formulas) >= 2
    assert "$E=mc^2$" in full_text
    assert "$E=mc^2$" in md or r"\( E=mc^2 \)" in md
    assert r"\frac" in md
    assert r"\begin{align}" in md


def test_latex_various_math_environments():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    Equation starred:
    \begin{equation*}
    a = b
    \end{equation*}
    Gather:
    \begin{gather}
    x = y \\
    z = w
    \end{gather}
    Multline:
    \begin{multline}
    first \\
    second
    \end{multline}
    \end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.FORMULA]) >= 3


def test_latex_math_environment():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    Inline: \begin{math}a+b\end{math}.
    \end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.FORMULA]) >= 1


def test_latex_displaymath_brackets():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    Display: \[ x^2 + y^2 = z^2 \]
    \end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.FORMULA]) >= 1


def test_latex_subequations_environment():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{subequations}
    \begin{align}
    a &= b \\
    c &= d
    \end{align}
    \end{subequations}
    \end{document}
    """
    ).convert()
    assert len([t for t in doc.texts if t.label == DocItemLabel.FORMULA]) >= 1


def test_latex_split_cases_math():
    doc = make_backend(
        rb"""
    \documentclass{article}
    \begin{document}
    \begin{equation}
    \begin{cases}
    x & \text{if } x > 0 \\
    -x & \text{otherwise}
    \end{cases}
    \end{equation}
    \end{document}
    """
    ).convert()

    formulas = [t for t in doc.texts if t.label == DocItemLabel.FORMULA]
    formula_text = " ".join(f.text for f in formulas)
    assert len(formulas) >= 1
    assert "cases" in formula_text or "otherwise" in formula_text
