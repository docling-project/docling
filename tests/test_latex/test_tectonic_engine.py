# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import subprocess
from pathlib import Path

import pytest

from docling.backend.latex.engines import tectonic
from docling.backend.latex.engines.tectonic import TectonicEngine


@pytest.fixture
def untrusted_engine(monkeypatch) -> TectonicEngine:
    """Engine with default options, using a stubbed system binary."""
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")
    return TectonicEngine(timeout=5.0)


def test_tectonic_engine_uses_system_binary(monkeypatch):
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")

    engine = TectonicEngine()

    assert engine.is_available() is True
    assert engine.binary_path == Path("/usr/bin/tectonic")


def test_tectonic_engine_logs_install_hint_when_missing(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with caplog.at_level(logging.WARNING):
        engine = TectonicEngine()

    assert engine.is_available() is False
    assert any(
        "Install Tectonic and make it available on PATH" in record.message
        for record in caplog.records
    )


def test_tectonic_render_times_out(monkeypatch):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 12.5
    engine.allow_shell_escape = True

    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == 12.5
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert engine.render(r"\begin{tikzpicture}\end{tikzpicture}") is None


def test_tectonic_sanitizes_assignment_only_pdftex_primitives():
    preamble = r"""
\usepackage{tikz}
\pdfcompresslevel=9
  \pdfminorversion = 7
\pdfobjcompresslevel=3 % keep compact
\ifdefined\pdfcompresslevel
  \typeout{pdftex-compatible}
\fi
"""

    sanitized = TectonicEngine._sanitize_preamble_for_tectonic(preamble)

    assert (
        "% docling: removed for Tectonic compatibility: \\pdfcompresslevel=9"
        in sanitized
    )
    assert (
        "% docling: removed for Tectonic compatibility: \\pdfminorversion = 7"
        in sanitized
    )
    assert (
        "% docling: removed for Tectonic compatibility: "
        "\\pdfobjcompresslevel=3 % keep compact" in sanitized
    )
    assert r"\ifdefined\pdfcompresslevel" in sanitized


def test_tectonic_render_uses_sanitized_preamble(monkeypatch):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = True

    captured_tex = {}

    def fake_run(cmd, **kwargs):
        tex_path = Path(cmd[-1])
        captured_tex["content"] = tex_path.read_text(encoding="utf-8")
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert (
        engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble="\\usepackage{tikz}\n\\pdfcompresslevel=9",
        )
        is None
    )
    assert (
        "% docling: removed for Tectonic compatibility: \\pdfcompresslevel=9"
        in captured_tex["content"]
    )


def test_tectonic_render_does_not_add_search_path(monkeypatch):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = True

    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert engine.render(r"\begin{tikzpicture}\end{tikzpicture}") is None
    assert not any(part.startswith("search-path=") for part in captured_cmd["cmd"])
    assert "-Z" in captured_cmd["cmd"]
    assert "shell-escape" in captured_cmd["cmd"]
    assert "--untrusted" not in captured_cmd["cmd"]


def test_tectonic_render_default_runs_untrusted_and_cache_only(
    monkeypatch, untrusted_engine
):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert untrusted_engine.render(r"\begin{tikzpicture}\end{tikzpicture}") is None
    cmd = captured["cmd"]
    assert "--untrusted" in cmd
    assert "--only-cached" in cmd
    assert "shell-escape" not in cmd
    assert Path(cmd[-1]).parent == Path(captured["kwargs"]["cwd"])


def test_tectonic_render_stages_explicit_local_dependencies(
    monkeypatch, untrusted_engine, tmp_path
):
    engine = untrusted_engine

    (tmp_path / "styles").mkdir()
    (tmp_path / "styles" / "tikz-macros.tex").write_text(
        "\\input{nested.tex}\n\\newcommand{\\foo}{bar}\n", encoding="utf-8"
    )
    (tmp_path / "nested.tex").write_text("\\newcommand{\\baz}{qux}\n", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "legend.png").write_bytes(b"png")

    captured = {}

    def fake_run(cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        captured["macro"] = (cwd / "styles" / "tikz-macros.tex").read_text(
            encoding="utf-8"
        )
        captured["nested_exists"] = (cwd / "nested.tex").exists()
        captured["asset_exists"] = (cwd / "assets" / "legend.png").exists()
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert (
        engine.render(
            r"\begin{tikzpicture}\includegraphics{assets/legend}\end{tikzpicture}",
            preamble="\\input{styles/tikz-macros}",
            source_root=tmp_path,
        )
        is None
    )
    assert "\\newcommand{\\foo}{bar}" in captured["macro"]
    assert captured["nested_exists"] is True
    assert captured["asset_exists"] is True


def test_tectonic_render_stages_dependencies_of_8bit_file(
    monkeypatch, untrusted_engine, tmp_path
):
    """A Latin-1 dependency is still scanned for the files it inputs."""
    engine = untrusted_engine

    (tmp_path / "macros.tex").write_bytes(
        "% Réglages\n\\input{nested}\n".encode("latin-1")
    )
    (tmp_path / "nested.tex").write_text("\\newcommand{\\baz}{qux}\n", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["nested_exists"] = (Path(kwargs["cwd"]) / "nested.tex").exists()
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert (
        engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble="\\input{macros}",
            source_root=tmp_path,
        )
        is None
    )
    assert captured["nested_exists"] is True


def test_tectonic_render_blocks_dependency_path_traversal(monkeypatch, tmp_path):
    # With shell escape the source pre-check is off, so staging alone must
    # refuse the parent-directory dependency.
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")
    engine = TectonicEngine(timeout=5.0, allow_shell_escape=True)

    outside_dir = tmp_path.parent
    outside_file = outside_dir / "secret.tex"
    outside_file.write_text("\\newcommand{\\secret}{1}\n", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        captured["staged_secret"] = (cwd / "secret.tex").exists()
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert (
        engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble="\\input{../secret}",
            source_root=tmp_path,
        )
        is None
    )
    assert captured["staged_secret"] is False


UNSAFE_TIKZ_SOURCES = [
    r"\input{/etc/passwd}",
    r"\input /etc/passwd ",
    r"\include{../outside}",
    r"\def\p{/etc/passwd}\input\p",
    r"\InputIfFileExists{~/notes.tex}{}{}",
    r"\includegraphics{/etc/image.png}",
    r"\includegraphics*[width=2cm]{figures/../../image.png}",
    r"\graphicspath{{C:/figures/}}",
    r"\openin1=notes.txt",
    r"\newwrite\f\immediate\openout\f=out.txt",
    r"\XeTeXpicfile image.png",
]


@pytest.mark.parametrize("source", UNSAFE_TIKZ_SOURCES)
@pytest.mark.parametrize("location", ["tikz", "preamble"])
def test_tectonic_render_skips_outside_file_references(
    monkeypatch, caplog, untrusted_engine, source, location
):
    calls = []
    monkeypatch.setattr(
        tectonic.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    tikz = rf"\begin{{tikzpicture}}{source}\end{{tikzpicture}}"
    preamble = "\\usepackage{tikz}"
    if location == "preamble":
        tikz = r"\begin{tikzpicture}\end{tikzpicture}"
        preamble = f"\\usepackage{{tikz}}\n{source}"

    with caplog.at_level(logging.WARNING):
        assert untrusted_engine.render(tikz, preamble=preamble) is None

    assert calls == []
    assert "Skipping TikZ rendering" in caplog.text


@pytest.mark.parametrize("staged_name", ["macros.tex", "macros.png"])
def test_tectonic_render_checks_every_staged_file(
    monkeypatch, untrusted_engine, tmp_path, staged_name
):
    # Staged files are checked whatever their extension, since \input can read
    # a file with any name.
    (tmp_path / staged_name).write_text("\\input{/etc/passwd}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        tectonic.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    assert (
        untrusted_engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble=f"\\input{{{staged_name}}}",
            source_root=tmp_path,
        )
        is None
    )
    assert calls == []


def test_tectonic_render_compiles_ordinary_tikz(monkeypatch, untrusted_engine):
    preamble = (
        "\\usepackage{amsmath,tikz,pgfplots}\n"
        "\\usetikzlibrary{arrows.meta,positioning}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\graphicspath{{figures/}{img/}}\n"
        "\\makeatletter\\newcommand{\\half}{0.5}\\makeatother"
    )
    tikz = (
        "\\begin{tikzpicture}\n"
        "\\draw[->, >=Stealth] (0,0) -- (1,1) node[above] {$a/b$};\n"
        "\\node {\\includegraphics[width=1cm]{figures/logo}};\n"
        "\\begin{axis}[xlabel={time / s}]\n"
        "\\addplot table {\nx y\n1 2\n3 4\n};\n"
        "\\addplot coordinates {(0,0) (1,\\half)};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}"
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    untrusted_engine.render(tikz, preamble=preamble)
    assert len(calls) == 1


def test_tectonic_shell_escape_optin_skips_source_check(monkeypatch):
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")
    engine = TectonicEngine(timeout=5.0, allow_shell_escape=True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    engine.render(r"\begin{tikzpicture}\input{/etc/hostname}\end{tikzpicture}")
    assert len(calls) == 1
    assert "shell-escape" in calls[0]
    assert "--untrusted" not in calls[0]
