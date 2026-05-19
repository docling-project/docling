from __future__ import annotations

from typing import Iterable

from .base import BACKEND_NAMES, BackendName, HwpToPdfConverter
from .chain import ConverterChain
from .config import build_chain

__all__ = [
    "BACKEND_NAMES",
    "BackendName",
    "ConverterChain",
    "HwpToPdfConverter",
    "build_chain",
    "convert_hwp_to_pdf",
]


def convert_hwp_to_pdf(
    file_path: str,
    *,
    primary: BackendName | None = None,
    order: Iterable[BackendName] | None = None,
    disable_fallback: bool = False,
) -> str | None:
    chain = build_chain(primary=primary, order=order, disable_fallback=disable_fallback)
    return chain.try_each(file_path)
