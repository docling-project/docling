from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

import requests

from .availability import rhwp_available, rhwp_pdf_api_url
from .base import BackendName

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 600
CONVERT_PATH = "/api/convert/hwp-to-pdf"
PDF_MAGIC = b"%PDF-"


class RhwpConverter:
    """genos-rhwp 의 serve-pdf HTTP API 를 호출하는 client.

    OCR/LLM 등 다른 외부 서비스와 동일한 패턴 — endpoint URL 만 옵션으로 받아
    HWP 바이트를 POST 하고 PDF 바이트를 받아 파일로 저장한다.

    매니페스트: genos-rhwp `k8s/rhwp-pdf-api.yaml` (ClusterIP Service `rhwp-pdf-api:7878`).
    프로토콜: POST {base_url}/api/convert/hwp-to-pdf
        Request  Content-Type: application/octet-stream, body=HWP bytes
        Response Content-Type: application/pdf, body=PDF bytes
    """

    name: ClassVar[BackendName] = "rhwp"

    def is_available(self) -> bool:
        return rhwp_available()

    def convert(self, file_path: str) -> str | None:
        base_url = rhwp_pdf_api_url()
        if not base_url:
            _log.warning("[hwp_to_pdf:rhwp] RHWP_PDF_API_URL not set; skipping")
            return None

        endpoint = base_url.rstrip("/") + CONVERT_PATH
        timeout = float(os.environ.get("HWP_TO_PDF_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC))

        try:
            in_path = Path(file_path).resolve()
            if not in_path.exists():
                _log.warning("[hwp_to_pdf:rhwp] input not found: %s", in_path)
                return None
            hwp_bytes = in_path.read_bytes()
        except OSError as e:
            _log.error("[hwp_to_pdf:rhwp] cannot read input %s: %s", file_path, e)
            return None

        _log.info(
            "[hwp_to_pdf:rhwp] POST %s (%d bytes, timeout=%ss)",
            endpoint, len(hwp_bytes), timeout,
        )
        try:
            resp = requests.post(
                endpoint,
                data=hwp_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=timeout,
            )
        except requests.Timeout:
            _log.error("[hwp_to_pdf:rhwp] timeout after %ss for %s", timeout, file_path)
            return None
        except requests.RequestException as e:
            _log.error("[hwp_to_pdf:rhwp] request error: %s", e)
            return None

        if resp.status_code != 200:
            body_preview = resp.text[:500] if resp.text else ""
            _log.warning(
                "[hwp_to_pdf:rhwp] FAILED status=%s body=%r", resp.status_code, body_preview,
            )
            return None

        pdf_bytes = resp.content
        if not pdf_bytes or pdf_bytes[:5] != PDF_MAGIC:
            _log.warning(
                "[hwp_to_pdf:rhwp] response is not a PDF (len=%d, head=%r)",
                len(pdf_bytes), pdf_bytes[:8],
            )
            return None

        out_path = in_path.with_suffix(".pdf")
        try:
            out_path.write_bytes(pdf_bytes)
        except OSError as e:
            _log.error("[hwp_to_pdf:rhwp] failed to write output %s: %s", out_path, e)
            return None

        _log.info("[hwp_to_pdf:rhwp] success -> %s (%d bytes)", out_path, len(pdf_bytes))
        return str(out_path)
