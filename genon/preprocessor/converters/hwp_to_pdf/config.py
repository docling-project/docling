from __future__ import annotations

import logging
import os
from typing import Iterable

from .availability import libreoffice_available, pdf_sdk_available, rhwp_available
from .base import BACKEND_NAMES, BackendName, HwpToPdfConverter
from .chain import ConverterChain
from .libreoffice import LibreOfficeConverter
from .pdf_sdk import PdfSdkConverter
from .rhwp import RhwpConverter

_log = logging.getLogger(__name__)

_BACKEND_FACTORIES: dict[BackendName, type] = {
    "pdf_sdk": PdfSdkConverter,
    "rhwp": RhwpConverter,
    "libreoffice": LibreOfficeConverter,
}

_AVAILABILITY: dict[BackendName, "callable[[], bool]"] = {
    "pdf_sdk": pdf_sdk_available,
    "rhwp": rhwp_available,
    "libreoffice": libreoffice_available,
}


def _coerce_name(value: str | None) -> BackendName | None:
    if not value:
        return None
    v = value.strip().lower()
    if v in BACKEND_NAMES:
        return v  # type: ignore[return-value]
    _log.warning("[hwp_to_pdf] unknown backend name ignored: %r", value)
    return None


def _parse_env_order(raw: str | None) -> list[BackendName]:
    if not raw:
        return []
    out: list[BackendName] = []
    for tok in raw.split(","):
        n = _coerce_name(tok)
        if n and n not in out:
            out.append(n)
    return out


def _auto_default_order() -> list[BackendName]:
    if pdf_sdk_available():
        return ["pdf_sdk", "rhwp", "libreoffice"]
    if rhwp_available():
        return ["rhwp", "libreoffice"]
    return ["libreoffice"]


def _resolve_order(
    primary: BackendName | None,
    order: Iterable[BackendName] | None,
) -> list[BackendName]:
    env_order = _parse_env_order(os.environ.get("HWP_TO_PDF_ORDER"))
    env_primary = _coerce_name(os.environ.get("HWP_TO_PDF_PRIMARY"))

    if order is not None:
        chosen = list(order)
    elif env_order:
        chosen = env_order
    elif primary is not None:
        chosen = [primary] + [b for b in _auto_default_order() if b != primary]
    elif env_primary is not None:
        chosen = [env_primary] + [b for b in _auto_default_order() if b != env_primary]
    else:
        chosen = _auto_default_order()

    seen: set[BackendName] = set()
    deduped: list[BackendName] = []
    for b in chosen:
        if b in BACKEND_NAMES and b not in seen:
            seen.add(b)
            deduped.append(b)
    return deduped


def _env_disable_fallback() -> bool:
    raw = os.environ.get("HWP_TO_PDF_DISABLE_FALLBACK", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def build_chain(
    primary: BackendName | None = None,
    order: Iterable[BackendName] | None = None,
    disable_fallback: bool = False,
) -> ConverterChain:
    resolved = _resolve_order(primary, order)
    if disable_fallback or _env_disable_fallback():
        resolved = resolved[:1]

    converters: list[HwpToPdfConverter] = []
    skipped: list[BackendName] = []
    for name in resolved:
        avail = _AVAILABILITY[name]()
        if not avail:
            skipped.append(name)
            continue
        converters.append(_BACKEND_FACTORIES[name]())

    if skipped:
        _log.warning(
            "[hwp_to_pdf] skipping unavailable backends: %s (resolved order=%s)",
            skipped, resolved,
        )

    return ConverterChain(converters)
