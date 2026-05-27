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
    """rhwp 바이너리 경로. 이미지 빌드 시 multi-stage 로 /usr/local/bin/rhwp 에 설치.

    RHWP_BIN 환경변수로 override 가능 (로컬 dev 시 다른 위치 빌드).
    빈/공백 값은 unset 으로 취급해 기본 경로를 쓴다.
    """
    env = os.environ.get("RHWP_BIN")
    if env and env.strip():
        return Path(env.strip())
    return Path("/usr/local/bin/rhwp")


def pdf_sdk_available() -> bool:
    p = pdf_sdk_binary()
    return p.exists() and os.access(p, os.X_OK)


def rhwp_available() -> bool:
    """rhwp 가용성 = 바이너리 존재 + 실행 권한.

    이미지 빌드 시 Dockerfile 의 rhwp_builder stage 에서 cargo build 한 결과를
    runtime stage 로 COPY. 미설치(standard/pro 가 아닌 환경 등)거나
    실행권한 없으면 chain 에서 자동 제외.
    """
    p = rhwp_binary()
    return p.is_file() and os.access(p, os.X_OK)


def libreoffice_available() -> bool:
    return shutil.which("soffice") is not None
