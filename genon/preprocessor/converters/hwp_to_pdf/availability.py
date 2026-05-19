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


def rhwp_pdf_api_url() -> str | None:
    """genos-rhwp 서버의 base URL (예: http://rhwp-pdf-api:7878).

    None / 빈 문자열이면 backend 미사용으로 간주.
    """
    raw = os.environ.get("RHWP_PDF_API_URL", "").strip()
    return raw or None


def pdf_sdk_available() -> bool:
    p = pdf_sdk_binary()
    return p.exists() and os.access(p, os.X_OK)


def rhwp_available() -> bool:
    """rhwp 가용성 = HTTP endpoint URL 설정 여부.

    실제 health check 는 별도 단계에서 수행 (이슈 #199 — genos 가 운영하는
    rhwp-pdf-api Deployment/Service 를 client 로 호출하는 패턴).
    """
    return rhwp_pdf_api_url() is not None


def libreoffice_available() -> bool:
    return shutil.which("soffice") is not None
