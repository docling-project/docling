"""PipelineOptions 기반 chain 구성 단위 테스트 (이슈 #199)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from genon.preprocessor.converters.hwp_to_pdf import (
    config as cfg,
    convert_hwp_to_pdf_from_options,
)


@pytest.fixture
def all_available(monkeypatch):
    cfg._AVAILABILITY["pdf_sdk"] = lambda: True
    cfg._AVAILABILITY["rhwp"] = lambda: True
    cfg._AVAILABILITY["libreoffice"] = lambda: True


@pytest.fixture
def clear_env(monkeypatch):
    for k in ("HWP_TO_PDF_PRIMARY", "HWP_TO_PDF_ORDER", "HWP_TO_PDF_DISABLE_FALLBACK"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.unit
def test_enum_value_coerced(all_available, clear_env):
    """Enum 인스턴스를 primary 로 전달해도 string 처럼 처리되어야 함."""
    enum_like = SimpleNamespace(value="rhwp")
    chain = cfg.build_chain(primary=enum_like)
    assert chain.backends[0] == "rhwp"


@pytest.mark.unit
def test_enum_list_for_order(all_available, clear_env):
    order = [SimpleNamespace(value="libreoffice"), SimpleNamespace(value="rhwp")]
    chain = cfg.build_chain(order=order)
    assert chain.backends == ["libreoffice", "rhwp"]


@pytest.mark.unit
def test_helper_with_none_options(all_available, clear_env, tmp_path, monkeypatch):
    """options=None 이면 명시값 없이 auto-default chain 사용."""
    captured = {}

    def fake_build_chain(*, primary=None, order=None, disable_fallback=False):
        captured["primary"] = primary
        captured["order"] = order
        captured["disable_fallback"] = disable_fallback
        return cfg.ConverterChain([])

    monkeypatch.setattr(
        "genon.preprocessor.converters.hwp_to_pdf.build_chain", fake_build_chain
    )

    convert_hwp_to_pdf_from_options(str(tmp_path / "x.hwp"), None)
    assert captured == {"primary": None, "order": None, "disable_fallback": False}


@pytest.mark.unit
def test_helper_forwards_pipeline_options_fields(all_available, clear_env, tmp_path, monkeypatch):
    captured = {}

    def fake_build_chain(*, primary=None, order=None, disable_fallback=False):
        captured["primary"] = primary
        captured["order"] = order
        captured["disable_fallback"] = disable_fallback
        return cfg.ConverterChain([])

    monkeypatch.setattr(
        "genon.preprocessor.converters.hwp_to_pdf.build_chain", fake_build_chain
    )

    opts = SimpleNamespace(
        hwp_to_pdf_primary=SimpleNamespace(value="rhwp"),
        hwp_to_pdf_order=[SimpleNamespace(value="rhwp"), SimpleNamespace(value="libreoffice")],
        hwp_to_pdf_disable_fallback=True,
    )

    convert_hwp_to_pdf_from_options(str(tmp_path / "x.hwp"), opts)

    assert captured["primary"].value == "rhwp"
    assert [b.value for b in captured["order"]] == ["rhwp", "libreoffice"]
    assert captured["disable_fallback"] is True


@pytest.mark.unit
def test_helper_handles_missing_attributes(all_available, clear_env, tmp_path):
    """opts 객체에 hwp_to_pdf_* 속성이 없어도 안전하게 처리."""
    opts = SimpleNamespace(save_images=True)  # 무관 속성만 있음
    # 예외 없이 None 반환 (chain 이 빈 환경)
    cfg._AVAILABILITY["pdf_sdk"] = lambda: False
    cfg._AVAILABILITY["rhwp"] = lambda: False
    cfg._AVAILABILITY["libreoffice"] = lambda: False
    assert convert_hwp_to_pdf_from_options(str(tmp_path / "x.hwp"), opts) is None
