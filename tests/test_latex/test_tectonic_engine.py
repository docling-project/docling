# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import logging
import subprocess
from pathlib import Path

import pytest

from docling.backend.latex.engines import tectonic
from docling.backend.latex.engines.tectonic import TectonicEngine


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


def test_tectonic_render_can_disable_shell_escape(monkeypatch):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = False

    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert engine.render(r"\begin{tikzpicture}\end{tikzpicture}") is None
    assert "shell-escape" not in captured_cmd["cmd"]


def test_tectonic_default_command_is_hardened(monkeypatch):
    # The default command has no shell escape and runs untrusted and cache-only.
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")

    engine = TectonicEngine()

    assert engine.allow_shell_escape is False

    cmd = engine._build_command(Path("/tmp/diagram.tex"))

    assert "--untrusted" in cmd
    assert "--only-cached" in cmd
    assert "-Z" not in cmd
    assert "shell-escape" not in cmd
    assert cmd.index("--untrusted") < cmd.index("/tmp/diagram.tex")


def test_tectonic_shell_escape_optin_command_is_trusted(monkeypatch):
    # With shell escape enabled, --untrusted is omitted because it disables it.
    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: "/usr/bin/tectonic")

    engine = TectonicEngine(allow_shell_escape=True)

    cmd = engine._build_command(Path("/tmp/diagram.tex"))

    assert "-Z" in cmd
    assert "shell-escape" in cmd
    assert "--untrusted" not in cmd


def test_tectonic_render_stages_explicit_local_dependencies(monkeypatch, tmp_path):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = False

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


def test_tectonic_render_blocks_dependency_path_traversal(monkeypatch, tmp_path):
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = False

    outside_dir = tmp_path.parent
    outside_file = outside_dir / "secret.tex"
    outside_file.write_text("\\newcommand{\\secret}{1}\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        tectonic.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    assert (
        engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble="\\input{../secret}",
            source_root=tmp_path,
        )
        is None
    )
    assert calls == []


def _untrusted_engine() -> TectonicEngine:
    engine = TectonicEngine.__new__(TectonicEngine)
    engine.binary_path = Path("/usr/bin/tectonic")
    engine._is_available = True
    engine.timeout = 5.0
    engine.allow_shell_escape = False
    return engine


UNSAFE_TIKZ_SOURCES = [
    r"\input{/etc/passwd}",
    r"\input /etc/passwd ",
    r"\include{../outside}",
    r"\def\p{/etc/passwd}\input\p",
    r"\expandafter\let\csname x\endcsname\relax",
    r"^^5cinput{/etc/passwd}",
    r"\begin{input}/etc/passwd \end{input}",
    r"\def\x{input}\begin\x /etc/passwd \end\x",
    r"\def\x{input}\begin{\x}/etc/passwd \end{\x}",
    r"\UseName{input}{/etc/passwd}",
    r"\csuse{input}",
    r"\catcode`\|=0 |input{/etc/passwd}",
    r"\makeatletter\@@input /etc/passwd",
    r"\openin1=/etc/passwd",
    r"\newwrite\f\immediate\openout\f=/tmp/out.txt",
    r"\includegraphics{/etc/image.png}",
    r"\includegraphics[width=2cm]{../image.png}",
    r"\graphicspath{{/etc/}}",
    r"\usepackage{../evil}",
    r"\lstinputlisting{notes.txt}",
    r"\pgfplotstableread{/etc/data.dat}\t",
    r"\begin{axis}\addplot table {/etc/data.dat};\end{axis}",
    r"\begin{axis}\addplot table[search path={/etc}] {data.dat};\end{axis}",
    r"\draw plot file {/etc/data.dat};",
    r"\begin{filecontents*}{/tmp/out.tex}x\end{filecontents*}",
    r"\special{pdf:image (/etc/image.png)}",
]


@pytest.mark.parametrize("source", UNSAFE_TIKZ_SOURCES)
@pytest.mark.parametrize("location", ["tikz", "preamble"])
def test_tectonic_render_skips_unsafe_file_access(
    monkeypatch, caplog, source, location
):
    engine = _untrusted_engine()
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
        assert engine.render(tikz, preamble=preamble) is None

    assert calls == []
    assert "Skipping TikZ rendering" in caplog.text


def test_tectonic_render_skips_unsafe_staged_dependency(monkeypatch, tmp_path):
    engine = _untrusted_engine()
    (tmp_path / "macros.tex").write_text("\\input{/etc/passwd}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        tectonic.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )

    assert (
        engine.render(
            r"\begin{tikzpicture}\end{tikzpicture}",
            preamble="\\input{macros}",
            source_root=tmp_path,
        )
        is None
    )
    assert calls == []


SAFE_TIKZ_SOURCES = [
    (
        "\\usepackage{amsmath,tikz,pgfplots}\n"
        "\\usetikzlibrary{arrows.meta,positioning}\n"
        "\\pgfplotsset{compat=1.18}\n"
        "\\graphicspath{{figures/}{img/}}\n"
        "\\tikzset{every node/.style={draw, rounded corners}}\n"
        "\\newcommand{\\half}{0.5}",
        "\\begin{tikzpicture}\n"
        "\\draw[->, >=Stealth] (0,0) -- (1,1) node[above] {$a/b$};\n"
        "\\foreach \\x in {1,...,5} \\node at (\\x,0) {\\x};\n"
        "\\begin{axis}[xlabel={time / s}]\n"
        "\\addplot table {\nx y\n1 2\n3 4\n};\n"
        "\\addplot table[row sep=\\\\] {x y\\\\ 1 2\\\\};\n"
        "\\addplot coordinates {(0,0) (1,\\half)};\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}",
    ),
]


@pytest.mark.parametrize(("preamble", "tikz"), SAFE_TIKZ_SOURCES)
def test_tectonic_render_compiles_ordinary_tikz(monkeypatch, preamble, tikz):
    engine = _untrusted_engine()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    assert TectonicEngine._find_unsafe_construct(preamble + "\n" + tikz) is None
    engine.render(tikz, preamble=preamble)
    assert len(calls) == 1


def test_tectonic_shell_escape_optin_skips_source_check(monkeypatch):
    engine = _untrusted_engine()
    engine.allow_shell_escape = True
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, output=b"", stderr=b"forced failure"
        )

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    engine.render(r"\begin{tikzpicture}\input{/etc/hostname}\end{tikzpicture}")
    assert len(calls) == 1
