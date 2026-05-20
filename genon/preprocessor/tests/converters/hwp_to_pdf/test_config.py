"""hwp_to_pdf.config.build_chain 단위 테스트."""
from __future__ import annotations

import pytest

from genon.preprocessor.converters.hwp_to_pdf import config as cfg


@pytest.fixture
def all_available(monkeypatch):
    """세 backend 모두 가용한 환경으로 모킹."""
    monkeypatch.setattr(cfg, "pdf_sdk_available", lambda: True)
    monkeypatch.setattr(cfg, "rhwp_available", lambda: True)
    monkeypatch.setattr(cfg, "libreoffice_available", lambda: True)
    cfg._AVAILABILITY["pdf_sdk"] = lambda: True
    cfg._AVAILABILITY["rhwp"] = lambda: True
    cfg._AVAILABILITY["libreoffice"] = lambda: True


@pytest.fixture
def clear_env(monkeypatch):
    for k in ("HWP_TO_PDF_PRIMARY", "HWP_TO_PDF_ORDER", "HWP_TO_PDF_DISABLE_FALLBACK"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.unit
def test_auto_default_with_pdf_sdk_available(all_available, clear_env):
    chain = cfg.build_chain()
    assert chain.backends == ["pdf_sdk", "rhwp", "libreoffice"]


@pytest.mark.unit
def test_auto_default_without_pdf_sdk(monkeypatch, clear_env):
    cfg._AVAILABILITY["pdf_sdk"] = lambda: False
    cfg._AVAILABILITY["rhwp"] = lambda: True
    cfg._AVAILABILITY["libreoffice"] = lambda: True
    monkeypatch.setattr(cfg, "pdf_sdk_available", lambda: False)
    monkeypatch.setattr(cfg, "rhwp_available", lambda: True)
    monkeypatch.setattr(cfg, "libreoffice_available", lambda: True)
    chain = cfg.build_chain()
    assert chain.backends == ["rhwp", "libreoffice"]


@pytest.mark.unit
def test_auto_default_libreoffice_only(monkeypatch, clear_env):
    cfg._AVAILABILITY["pdf_sdk"] = lambda: False
    cfg._AVAILABILITY["rhwp"] = lambda: False
    cfg._AVAILABILITY["libreoffice"] = lambda: True
    monkeypatch.setattr(cfg, "pdf_sdk_available", lambda: False)
    monkeypatch.setattr(cfg, "rhwp_available", lambda: False)
    monkeypatch.setattr(cfg, "libreoffice_available", lambda: True)
    chain = cfg.build_chain()
    assert chain.backends == ["libreoffice"]


@pytest.mark.unit
def test_primary_arg_promotes_backend_to_front(all_available, clear_env):
    chain = cfg.build_chain(primary="libreoffice")
    assert chain.backends[0] == "libreoffice"
    assert set(chain.backends) == {"pdf_sdk", "rhwp", "libreoffice"}


@pytest.mark.unit
def test_disable_fallback_keeps_only_primary(all_available, clear_env):
    chain = cfg.build_chain(primary="pdf_sdk", disable_fallback=True)
    assert chain.backends == ["pdf_sdk"]


@pytest.mark.unit
def test_env_primary_used_when_arg_missing(all_available, clear_env, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_PRIMARY", "rhwp")
    chain = cfg.build_chain()
    assert chain.backends[0] == "rhwp"


@pytest.mark.unit
def test_env_order_overrides_default(all_available, clear_env, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_ORDER", "libreoffice,rhwp")
    chain = cfg.build_chain()
    assert chain.backends == ["libreoffice", "rhwp"]


@pytest.mark.unit
def test_explicit_order_arg_overrides_env(all_available, clear_env, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_ORDER", "pdf_sdk")
    chain = cfg.build_chain(order=["rhwp", "libreoffice"])
    assert chain.backends == ["rhwp", "libreoffice"]


@pytest.mark.unit
def test_env_disable_fallback_truncates_chain(all_available, clear_env, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_DISABLE_FALLBACK", "1")
    chain = cfg.build_chain(primary="rhwp")
    assert chain.backends == ["rhwp"]


@pytest.mark.unit
def test_unavailable_backend_silently_dropped(monkeypatch, clear_env):
    cfg._AVAILABILITY["pdf_sdk"] = lambda: False
    cfg._AVAILABILITY["rhwp"] = lambda: True
    cfg._AVAILABILITY["libreoffice"] = lambda: True
    monkeypatch.setattr(cfg, "pdf_sdk_available", lambda: False)
    monkeypatch.setattr(cfg, "rhwp_available", lambda: True)
    monkeypatch.setattr(cfg, "libreoffice_available", lambda: True)
    chain = cfg.build_chain(primary="pdf_sdk")
    assert "pdf_sdk" not in chain.backends
    assert chain.backends[0] == "rhwp"


@pytest.mark.unit
def test_unknown_backend_in_env_ignored(all_available, clear_env, monkeypatch):
    monkeypatch.setenv("HWP_TO_PDF_ORDER", "garbage,rhwp")
    chain = cfg.build_chain()
    assert chain.backends == ["rhwp"]
