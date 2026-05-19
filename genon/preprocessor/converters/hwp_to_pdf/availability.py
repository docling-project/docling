from __future__ import annotations

import os
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def pdf_sdk_home() -> Path:
    env = os.environ.get("PDF_SDK_HOME")
    if env:
        return Path(env)
    return _repo_root() / "pdf_sdk"


def pdf_sdk_binary() -> Path:
    return pdf_sdk_home() / "pdfConverter"


def rhwp_binary() -> Path:
    return Path(os.environ.get("RHWP_BIN", "/usr/local/bin/rhwp"))


def pdf_sdk_available() -> bool:
    p = pdf_sdk_binary()
    return p.exists() and os.access(p, os.X_OK)


def rhwp_available() -> bool:
    p = rhwp_binary()
    return p.exists() and os.access(p, os.X_OK)


def libreoffice_available() -> bool:
    return shutil.which("soffice") is not None
