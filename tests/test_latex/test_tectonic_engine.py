import subprocess
from pathlib import Path

from docling.backend.latex.engines import tectonic
from docling.backend.latex.engines.tectonic import TectonicEngine


def test_tectonic_engine_downloads_with_curl(monkeypatch, tmp_path):
    run_calls = []

    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def fake_run(*args, **kwargs):
        run_calls.append((args, kwargs))
        binary_path = tmp_path / ".cache" / "docling" / "tectonic" / "tectonic"
        binary_path.write_text("", encoding="utf-8")
        binary_path.chmod(0o755)
        return None

    monkeypatch.setattr(tectonic.subprocess, "run", fake_run)

    engine = TectonicEngine()

    assert engine.is_available() is True
    assert engine.binary_path.exists()
    assert len(run_calls) == 1
    assert run_calls[0][0][0].startswith("curl --proto '=https'")
    assert run_calls[0][1]["shell"] is True
    assert run_calls[0][1]["cwd"] == tmp_path / ".cache" / "docling" / "tectonic"


def test_tectonic_engine_can_disable_download(monkeypatch, tmp_path):
    run_calls = []

    monkeypatch.setattr(tectonic.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        tectonic.subprocess,
        "run",
        lambda *args, **kwargs: run_calls.append((args, kwargs)),
    )

    engine = TectonicEngine(allow_download=False)

    assert engine.is_available() is False
    assert run_calls == []


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
