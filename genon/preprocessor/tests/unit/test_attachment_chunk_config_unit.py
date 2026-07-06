"""
attachment_processor 의 청킹/포맷 설정 → _default_kwargs 매핑 단위 테스트.

이번 리팩터로 바뀐 동작을 검증한다:
  - chunk_size 공통화(chunking.chunk_size → recursive/hybrid 동시 반영, per-block override)
  - chunker_type 위치 이동(chunking.chunker_type, 없으면 defaults.chunker_type 폴백)
  - HWP 옵션 이동(formats.hwp.*, 없으면 defaults.* 폴백)
  - token_chunk_size_cap 완전 제거

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""

from pathlib import Path

import pytest
import yaml

_CONFIG_NAME = "attachment_processor_config.yaml"


def _load_processor():
    mod = pytest.importorskip("facade.attachment_processor")
    return mod.DocumentProcessor


def _write_config(tmp_path: Path, mutate) -> str:
    """출고 attachment config 를 복사하고 mutate(cfg) 로 수정한 임시 config 경로 반환."""
    repo_resource = Path(__file__).resolve().parents[2] / "resource" / _CONFIG_NAME
    cfg = yaml.safe_load(repo_resource.read_text(encoding="utf-8"))
    cfg.setdefault("chunking", {})
    cfg.setdefault("defaults", {})
    cfg.setdefault("formats", {})
    mutate(cfg)
    out = tmp_path / _CONFIG_NAME
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def _init(tmp_path: Path, mutate):
    DocumentProcessor = _load_processor()
    try:
        return DocumentProcessor(config_path=_write_config(tmp_path, mutate))
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 등 환경 의존
        pytest.skip(f"DocumentProcessor init unavailable: {e}")


# ---------------------------------------------------------------------------
# 공통 chunk_size
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_common_chunk_size_applies_to_recursive_and_hybrid(tmp_path):
    """chunking.chunk_size(공통)가 recursive/hybrid 양쪽 기본값으로 반영된다."""
    def m(cfg):
        cfg["chunking"]["chunk_size"] = 2000
        cfg["chunking"].get("recursive", {}).pop("chunk_size", None)
        cfg["chunking"].get("hybrid", {}).pop("chunk_size", None)
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["recursive_chunk_size"] == 2000
    assert proc._default_kwargs["hybrid_chunk_size"] == 2000


@pytest.mark.unit
def test_recursive_chunk_size_overrides_common(tmp_path):
    """per-block recursive.chunk_size 가 공통 chunk_size 를 덮어쓴다(hybrid 는 공통 유지)."""
    def m(cfg):
        cfg["chunking"]["chunk_size"] = 2000
        cfg["chunking"].setdefault("recursive", {})["chunk_size"] = 500
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["recursive_chunk_size"] == 500
    assert proc._default_kwargs["hybrid_chunk_size"] == 2000


@pytest.mark.unit
def test_chunk_size_zero_kept(tmp_path):
    """chunk_size=0(전체 1청크) 이 그대로 로드된다(강제 기본값으로 치환되지 않음)."""
    def m(cfg):
        cfg["chunking"]["chunk_size"] = 0
        cfg["chunking"].get("recursive", {}).pop("chunk_size", None)
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["recursive_chunk_size"] == 0


# ---------------------------------------------------------------------------
# chunker_type 위치 이동 + 폴백
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunker_type_from_chunking(tmp_path):
    def m(cfg):
        cfg["chunking"]["chunker_type"] = "hybrid"
        cfg["defaults"].pop("chunker_type", None)
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["chunker_type"] == "hybrid"


@pytest.mark.unit
def test_chunker_type_fallback_to_defaults(tmp_path):
    """chunking.chunker_type 부재 시 구 위치 defaults.chunker_type 로 폴백."""
    def m(cfg):
        cfg["chunking"].pop("chunker_type", None)
        cfg["defaults"]["chunker_type"] = "hybrid"
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["chunker_type"] == "hybrid"


# ---------------------------------------------------------------------------
# HWP 옵션 formats.hwp 이동 + 폴백
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_hwp_options_from_formats(tmp_path):
    def m(cfg):
        cfg["formats"]["hwp"] = {
            "use_hwp_sdk": False, "dump_sdk_output": True, "save_images": False,
        }
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["use_hwp_sdk"] is False
    assert proc._default_kwargs["save_images"] is False
    # dump_sdk_output 은 use_hwp_sdk=False 여도 _default_kwargs 자체엔 설정값이 반영됨
    assert proc._default_kwargs["dump_sdk_output"] is True


@pytest.mark.unit
def test_hwp_options_fallback_to_defaults(tmp_path):
    """formats.hwp 부재 시 구 위치 defaults.* 로 폴백."""
    def m(cfg):
        cfg["formats"].pop("hwp", None)
        cfg["defaults"]["use_hwp_sdk"] = False
        cfg["defaults"]["save_images"] = False
    proc = _init(tmp_path, m)
    assert proc._default_kwargs["use_hwp_sdk"] is False
    assert proc._default_kwargs["save_images"] is False


# ---------------------------------------------------------------------------
# token_chunk_size_cap 완전 제거
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_token_chunk_size_cap_removed(tmp_path):
    """token cap 관련 키가 _default_kwargs 에 더 이상 존재하지 않는다."""
    proc = _init(tmp_path, lambda cfg: None)
    assert "recursive_token_chunk_size_cap" not in proc._default_kwargs
    assert "recursive_tokenizer_id" not in proc._default_kwargs
    assert "hybrid_tokenizer_id" not in proc._default_kwargs
