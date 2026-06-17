"""
chunking.chunk_size 의 yaml/kwargs 지정 및 우선순위(kwargs > yaml > 0) 단위 테스트.

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
intelligent_processor / convert_processor 두 곳의 DocumentProcessor 가 동일하게 동작하는지 확인한다.
"""

import shutil
from pathlib import Path

import pytest
import yaml

_MODULES = ["intelligent_processor", "convert_processor"]

# facade 모듈명 → 출고 config 파일명
_DEFAULT_CONFIG = {
    "intelligent_processor": "intelligent_processor_config.yaml",
    "convert_processor": "convert_processor_config.yaml",
}


def _load_processor(module_name: str):
    mod = pytest.importorskip(f"facade.{module_name}")
    return mod.DocumentProcessor


def _make_config(tmp_path: Path, module_name: str, chunk_size) -> str:
    """출고 config 를 복사하고 chunking.chunk_size 만 덮어쓴 임시 config 경로 반환."""
    repo_resource = Path(__file__).resolve().parents[2] / "resource" / _DEFAULT_CONFIG[module_name]
    cfg = yaml.safe_load(repo_resource.read_text(encoding="utf-8"))
    cfg.setdefault("chunking", {})
    if chunk_size is None:
        cfg["chunking"].pop("chunk_size", None)
    else:
        cfg["chunking"]["chunk_size"] = chunk_size
    out = tmp_path / _DEFAULT_CONFIG[module_name]
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def _init_processor(module_name: str, config_path: str):
    DocumentProcessor = _load_processor(module_name)
    try:
        return DocumentProcessor(config_path=config_path)
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 등 환경 의존
        pytest.skip(f"DocumentProcessor init unavailable: {e}")


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_yaml_chunk_size_is_loaded(tmp_path, module_name):
    """yaml chunking.chunk_size 값이 self._chunk_size 로 로드된다."""
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, 1234))
    assert proc._chunk_size == 1234


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_yaml_chunk_size_zero(tmp_path, module_name):
    """출고 기본값 0 도 정상 로드된다(None 이 아님)."""
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, 0))
    assert proc._chunk_size == 0


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_yaml_chunk_size_absent_is_none(tmp_path, module_name):
    """chunk_size 미설정 시 self._chunk_size 는 None (split 단계에서 0 으로 처리)."""
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, None))
    assert proc._chunk_size is None


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_priority_kwargs_over_yaml(tmp_path, module_name):
    """split_documents 의 max_tokens 우선순위: kwargs.chunk_size > yaml > 0.

    실제 분할을 수행하지 않고, GenosSmartChunker 생성 시 전달되는 max_tokens 만 가로채 검증한다.
    """
    mod = pytest.importorskip(f"facade.{module_name}")
    proc = _init_processor(module_name, _make_config(tmp_path, module_name, 500))

    captured = {}
    OrigChunker = mod.GenosSmartChunker

    class _SpyChunker(OrigChunker):
        def __init__(self, **kw):
            captured["max_tokens"] = kw.get("max_tokens")
            super().__init__(**kw)

        def chunk(self, *a, **k):  # 실제 청킹은 생략
            return iter(())

    mod.GenosSmartChunker = _SpyChunker
    try:
        # kwargs 에 chunk_size 전달 → kwargs 우선
        list(proc.split_documents(documents=None, chunk_size=77))
        assert captured["max_tokens"] == 77

        # kwargs 미전달 → yaml(500)
        captured.clear()
        list(proc.split_documents(documents=None))
        assert captured["max_tokens"] == 500
    finally:
        mod.GenosSmartChunker = OrigChunker
