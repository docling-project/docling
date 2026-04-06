import tempfile
from pathlib import Path

from docling_core.types.doc import DocItemLabel
from PIL import Image as PILImage

from ._utils import make_backend, make_backend_from_path


def test_latex_caption():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{test.png}
    \\end{document}
    """
    ).convert()

    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(captions) >= 1
    assert "test.png" in captions[0].text


def test_latex_includegraphics():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tex_file = tmpdir_path / "test.tex"
        img_file = tmpdir_path / "test_image.png"

        test_img = PILImage.new("RGB", (100, 50), color="red")
        test_img.save(img_file, dpi=(96, 96))
        tex_file.write_bytes(
            b"""
        \\documentclass{article}
        \\begin{document}
        \\includegraphics{test_image.png}
        \\end{document}
        """
        )

        doc = make_backend_from_path(tex_file).convert()
        picture = doc.pictures[0]
        assert len(doc.pictures) >= 1
        assert picture.image is not None
        assert len(picture.captions) >= 1
        assert "test_image.png" in picture.captions[0].resolve(doc).text


def test_latex_includegraphics_missing_image():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\includegraphics{nonexistent_image.png}
    \\end{document}
    """
    ).convert()

    picture = doc.pictures[0]
    assert len(doc.pictures) >= 1
    assert picture.image is None
    assert len(picture.captions) >= 1
    assert "nonexistent_image.png" in picture.captions[0].resolve(doc).text


def test_latex_figure_environment():
    doc = make_backend(
        b"""
    \\documentclass{article}
    \\begin{document}
    \\begin{figure}
    \\includegraphics{test.png}
    \\caption{Test figure}
    \\end{figure}
    \\end{document}
    """
    ).convert()

    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(doc.pictures) >= 1
    assert len(captions) >= 1


def test_latex_figure_with_caption():
    doc = make_backend(
        b"""\\documentclass{article}
\\begin{document}
\\begin{figure}
\\includegraphics{test.png}
\\caption{This is a test figure caption}
\\label{fig:test}
\\end{figure}
\\end{document}
"""
    ).convert()

    figure_groups = [g for g in doc.groups if g.name == "figure"]
    captions = [t for t in doc.texts if t.label == DocItemLabel.CAPTION]
    assert len(figure_groups) >= 1
    assert len(doc.pictures) >= 1
    assert len(captions) >= 1
