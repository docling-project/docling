# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import subprocess
from io import BytesIO
from pathlib import Path

import pytest

from docling.backend.docx.drawingml import utils as drawingml_utils


def _track_mkdtemp(monkeypatch) -> list[Path]:
    created_dirs: list[Path] = []
    real_mkdtemp = drawingml_utils.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(path))
        return path

    monkeypatch.setattr(drawingml_utils, "mkdtemp", tracking_mkdtemp)
    return created_dirs


class _FakePopen:
    """Minimal stand-in for ``subprocess.Popen`` used by ``_run_hardened_soffice``.

    ``side_effect`` mirrors LibreOffice's real behavior of writing the output
    file into ``--outdir`` while the "conversion" runs.
    """

    def __init__(self, args, side_effect=None, returncode=0, wait_exc=None, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.pid = 4242
        self._returncode = returncode
        self._wait_exc = wait_exc
        self.wait_timeout = None
        self.wait_calls = 0
        if side_effect is not None:
            side_effect(self.args)

    def wait(self, timeout=None):
        self.wait_calls += 1
        # The first wait is the timeout-bounded one; a killpg cleanup wait
        # (timeout=5) may follow.
        if self.wait_calls == 1:
            self.wait_timeout = timeout
            if self._wait_exc is not None:
                raise self._wait_exc
        return self._returncode

    def kill(self):  # pragma: no cover - only on the no-killpg fallback path
        pass


def _install_fake_popen(monkeypatch, *, side_effect=None, returncode=0, wait_exc=None):
    instances: list[_FakePopen] = []

    def factory(args, **kwargs):
        proc = _FakePopen(
            args,
            side_effect=side_effect,
            returncode=returncode,
            wait_exc=wait_exc,
            **kwargs,
        )
        instances.append(proc)
        return proc

    monkeypatch.setattr(drawingml_utils.subprocess, "Popen", factory)
    return instances


# ---------------------------------------------------------------------------
# Pure-helper unit tests (no LibreOffice binary required)
# ---------------------------------------------------------------------------


def test_build_soffice_command_contains_hardening_flags():
    args = drawingml_utils._build_soffice_command(
        "/usr/bin/soffice",
        "-env:UserInstallation=file:///tmp/p",
        target_format="pdf",
        outdir="/tmp/out",
        input_path="/tmp/in.docx",
    )

    # Every hardening flag is present.
    for flag in (
        "--headless",
        "--norestore",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
    ):
        assert flag in args, f"missing hardening flag {flag}"

    # The isolated-profile env argument and the conversion request survive.
    assert "-env:UserInstallation=file:///tmp/p" in args
    assert args[args.index("--convert-to") + 1] == "pdf"
    assert args[args.index("--outdir") + 1] == "/tmp/out"
    assert args[0] == "/usr/bin/soffice"
    assert args[-1] == "/tmp/in.docx"


def test_registrymodifications_xcu_locks_down_security():
    xcu = drawingml_utils._registrymodifications_xcu()

    # Well-formed XML.
    import xml.etree.ElementTree as ET

    ET.fromstring(xcu)

    # Macros: maximum security level and execution disabled.
    assert 'oor:name="MacroSecurityLevel"' in xcu
    assert "<value>3</value>" in xcu
    assert 'oor:name="DisableMacrosExecution"' in xcu
    assert "<value>true</value>" in xcu

    # External link updates disabled for Writer and Calc.
    assert "/org.openoffice.Office.Writer/Content/Update" in xcu
    assert "/org.openoffice.Office.Calc/Content/Update" in xcu
    assert 'oor:name="Link"' in xcu
    assert "<value>0</value>" in xcu


def test_isolated_profile_seeds_registrymodifications(monkeypatch):
    created = _track_mkdtemp(monkeypatch)

    with drawingml_utils._isolated_libreoffice_profile() as profile_arg:
        profile_dir = created[-1]
        xcu_path = profile_dir / "user" / "registrymodifications.xcu"
        assert xcu_path.exists()
        assert profile_dir.as_uri() in profile_arg
        assert xcu_path.read_text(encoding="utf-8") == (
            drawingml_utils._registrymodifications_xcu()
        )

    # Cleaned up afterwards.
    assert not profile_dir.exists()


def test_get_libreoffice_cmd_honors_env_var(monkeypatch):
    monkeypatch.setenv("DOCLING_LIBREOFFICE_CMD", "/opt/custom/soffice")
    assert drawingml_utils.get_libreoffice_cmd() == "/opt/custom/soffice"


# ---------------------------------------------------------------------------
# Integration of helpers via the public conversion entry points
# ---------------------------------------------------------------------------


def test_convert_with_libreoffice_uses_timeout_and_isolated_profile(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        drawingml_utils, "get_libreoffice_cmd", lambda: "/usr/bin/soffice"
    )
    created_profile_dirs = _track_mkdtemp(monkeypatch)

    def side_effect(args):
        # The isolated profile dir must exist while the "conversion" runs.
        assert created_profile_dirs[-1].exists()
        (tmp_path / "drawing_only.pdf").write_bytes(b"%PDF-1.4")

    instances = _install_fake_popen(monkeypatch, side_effect=side_effect)

    converter = drawingml_utils.get_docx_to_pdf_converter()
    assert converter is not None

    input_path = tmp_path / "drawing_only.docx"
    output_path = tmp_path / "drawing_only.pdf"
    input_path.write_bytes(b"")

    converter(input_path, output_path)

    # Launched with a bounded wait, its own session, and the hardening flags.
    assert instances[-1].wait_timeout == drawingml_utils.LIBREOFFICE_TIMEOUT_S
    assert instances[-1].kwargs.get("start_new_session") is True
    for flag in drawingml_utils.LIBREOFFICE_HARDENING_FLAGS:
        assert flag in instances[-1].args

    profile_flag = next(
        a for a in instances[-1].args if str(a).startswith("-env:UserInstallation=")
    )
    assert created_profile_dirs[-1].as_uri() in profile_flag

    # The isolated profile directory is cleaned up after the call.
    assert not created_profile_dirs[-1].exists()


def test_convert_with_libreoffice_kills_process_group_on_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        drawingml_utils, "get_libreoffice_cmd", lambda: "/usr/bin/soffice"
    )
    created_profile_dirs = _track_mkdtemp(monkeypatch)

    timeout_exc = subprocess.TimeoutExpired(cmd="soffice", timeout=5)
    _install_fake_popen(monkeypatch, wait_exc=timeout_exc)

    killed: dict = {}

    monkeypatch.setattr(drawingml_utils.os, "getpgid", lambda pid: pid)

    def fake_killpg(pgid, sig):
        killed["pgid"] = pgid
        killed["sig"] = sig

    monkeypatch.setattr(drawingml_utils.os, "killpg", fake_killpg)

    converter = drawingml_utils.get_docx_to_pdf_converter()
    assert converter is not None

    input_path = tmp_path / "drawing_only.docx"
    output_path = tmp_path / "drawing_only.pdf"
    input_path.write_bytes(b"")

    with pytest.raises(subprocess.TimeoutExpired):
        converter(input_path, output_path)

    # The whole process group was SIGKILLed, not just the wrapper.
    assert killed["pgid"] == 4242
    assert killed["sig"] == drawingml_utils.signal.SIGKILL

    # A hung/killed conversion must not leak its profile directory.
    assert not created_profile_dirs[-1].exists()


def test_convert_to_modern_format_uses_isolated_profile(monkeypatch):
    monkeypatch.setattr(
        drawingml_utils, "get_libreoffice_cmd", lambda: "/usr/bin/soffice"
    )
    created_profile_dirs = _track_mkdtemp(monkeypatch)

    def side_effect(args):
        # The isolated profile dir (most recently created) must exist while
        # the "conversion" runs, separate from the outer working tmp_dir.
        assert created_profile_dirs[-1].exists()
        outdir = Path(args[args.index("--outdir") + 1])
        (outdir / "input.docx").write_bytes(b"PK\x03\x04")

    instances = _install_fake_popen(monkeypatch, side_effect=side_effect)

    result = drawingml_utils.convert_to_modern_format(
        BytesIO(b"legacy doc bytes"), "doc", "docx", timeout_s=5
    )

    assert isinstance(result, BytesIO)
    assert instances[-1].wait_timeout == 5
    for flag in drawingml_utils.LIBREOFFICE_HARDENING_FLAGS:
        assert flag in instances[-1].args

    # Both the outer working directory and the isolated profile directory
    # are cleaned up once the conversion completes.
    assert not created_profile_dirs[0].exists()
    assert not created_profile_dirs[-1].exists()


def test_run_hardened_soffice_raises_on_nonzero_exit(monkeypatch):
    _install_fake_popen(monkeypatch, returncode=1)

    with pytest.raises(subprocess.CalledProcessError):
        drawingml_utils._run_hardened_soffice(["/usr/bin/soffice"], timeout_s=5)
