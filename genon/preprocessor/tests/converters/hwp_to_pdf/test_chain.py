"""hwp_to_pdf.chain 단위 테스트."""
from __future__ import annotations

from typing import ClassVar

import pytest

from genon.preprocessor.converters.hwp_to_pdf.base import BackendName
from genon.preprocessor.converters.hwp_to_pdf.chain import ConverterChain


class _StubConverter:
    def __init__(self, name: BackendName, result: str | None = None, raises: Exception | None = None):
        self.name: BackendName = name  # type: ignore[assignment]
        self._result = result
        self._raises = raises
        self.calls: int = 0

    def is_available(self) -> bool:
        return True

    def convert(self, file_path: str) -> str | None:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.mark.unit
def test_try_each_returns_first_success():
    a = _StubConverter("pdf_sdk", result="/tmp/out.pdf")
    b = _StubConverter("rhwp", result="/tmp/other.pdf")
    chain = ConverterChain([a, b])
    assert chain.try_each("/tmp/in.hwp") == "/tmp/out.pdf"
    assert a.calls == 1
    assert b.calls == 0


@pytest.mark.unit
def test_try_each_falls_back_when_first_returns_none():
    a = _StubConverter("pdf_sdk", result=None)
    b = _StubConverter("rhwp", result="/tmp/out.pdf")
    chain = ConverterChain([a, b])
    assert chain.try_each("/tmp/in.hwp") == "/tmp/out.pdf"
    assert a.calls == 1
    assert b.calls == 1


@pytest.mark.unit
def test_try_each_swallows_exception_and_continues():
    a = _StubConverter("pdf_sdk", raises=RuntimeError("boom"))
    b = _StubConverter("rhwp", result="/tmp/out.pdf")
    chain = ConverterChain([a, b])
    assert chain.try_each("/tmp/in.hwp") == "/tmp/out.pdf"
    assert a.calls == 1
    assert b.calls == 1


@pytest.mark.unit
def test_try_each_returns_none_when_all_fail():
    a = _StubConverter("pdf_sdk", result=None)
    b = _StubConverter("rhwp", result=None)
    c = _StubConverter("libreoffice", result=None)
    chain = ConverterChain([a, b, c])
    assert chain.try_each("/tmp/in.hwp") is None
    assert a.calls == b.calls == c.calls == 1


@pytest.mark.unit
def test_empty_chain_returns_none_without_error():
    chain = ConverterChain([])
    assert chain.try_each("/tmp/in.hwp") is None
    assert chain.backends == []


@pytest.mark.unit
def test_backends_property_reflects_order():
    a = _StubConverter("rhwp")
    b = _StubConverter("libreoffice")
    chain = ConverterChain([a, b])
    assert chain.backends == ["rhwp", "libreoffice"]
