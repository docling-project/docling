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
    "convert_hwp_to_pdf_from_options",
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


def convert_hwp_to_pdf_from_options(file_path: str, options) -> str | None:
    """`docling.datamodel.pipeline_options.PipelineOptions` 인스턴스로부터
    `hwp_to_pdf_*` 필드를 읽어 chain 을 구성하고 변환을 수행한다.

    PipelineOptions 와 본 모듈 사이의 결합을 호출지점에서 명시적으로 끊기 위한
    얇은 helper. opts 가 None 이면 auto-default chain 으로 동작.
    """
    if options is None:
        return convert_hwp_to_pdf(file_path)
    return convert_hwp_to_pdf(
        file_path,
        primary=getattr(options, "hwp_to_pdf_primary", None),
        order=getattr(options, "hwp_to_pdf_order", None),
        disable_fallback=bool(getattr(options, "hwp_to_pdf_disable_fallback", False)),
    )
