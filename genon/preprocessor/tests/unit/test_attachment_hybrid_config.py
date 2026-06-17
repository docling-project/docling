"""
attachment_processor 의 chunking 섹션 config 로딩 단위 테스트.

chunking.hybrid.chunk_size / chunking.hybrid.tokenizer_type 가 _default_kwargs 로,
chunking.tokenizer_path/tokenizer_id 가 self._tokenizer 로 흐르는지 확인한다.
의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""

from pathlib import Path

import pytest
import yaml

_CONFIG_NAME = "attachment_processor_config.yaml"


def _load_processor():
    mod = pytest.importorskip("facade.attachment_processor")
    return mod.DocumentProcessor


def _make_config(tmp_path: Path, *, hybrid_chunk_size=None, hybrid_tokenizer_type=None,
                 tokenizer_path=None) -> str:
    """출고 attachment config 를 복사하고 chunking 하위 키만 덮어쓴 임시 config 경로 반환."""
    repo_resource = Path(__file__).resolve().parents[2] / "resource" / _CONFIG_NAME
    cfg = yaml.safe_load(repo_resource.read_text(encoding="utf-8"))
    chunking = cfg.setdefault("chunking", {})
    hybrid = chunking.setdefault("hybrid", {})
    if hybrid_chunk_size is not None:
        hybrid["chunk_size"] = hybrid_chunk_size
    if hybrid_tokenizer_type is not None:
        hybrid["tokenizer_type"] = hybrid_tokenizer_type
    if tokenizer_path is not None:
        chunking["tokenizer_path"] = tokenizer_path
    out = tmp_path / _CONFIG_NAME
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(out)


def _init_processor(config_path: str):
    DocumentProcessor = _load_processor()
    try:
        return DocumentProcessor(config_path=config_path)
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 등 환경 의존
        pytest.skip(f"DocumentProcessor init unavailable: {e}")


@pytest.mark.unit
def test_hybrid_chunk_size_loaded(tmp_path):
    """chunking.hybrid.chunk_size 가 _default_kwargs['hybrid_chunk_size'] 로 로드된다."""
    proc = _init_processor(_make_config(tmp_path, hybrid_chunk_size=4096))
    assert proc._default_kwargs["hybrid_chunk_size"] == 4096


@pytest.mark.unit
def test_hybrid_tokenizer_type_default_char(tmp_path):
    """tokenizer_type 미지정 시 기본 char 로 로드된다."""
    proc = _init_processor(_make_config(tmp_path))
    assert proc._default_kwargs["hybrid_tokenizer_type"] == "char"


@pytest.mark.unit
def test_hybrid_tokenizer_type_explicit(tmp_path):
    """chunking.hybrid.tokenizer_type 값이 그대로 로드된다."""
    proc = _init_processor(_make_config(tmp_path, hybrid_tokenizer_type="huggingface"))
    assert proc._default_kwargs["hybrid_tokenizer_type"] == "huggingface"


@pytest.mark.unit
def test_hybrid_tokenizer_type_invalid_falls_back(tmp_path):
    """알 수 없는 tokenizer_type 은 char 로 폴백한다."""
    proc = _init_processor(_make_config(tmp_path, hybrid_tokenizer_type="bogus"))
    assert proc._default_kwargs["hybrid_tokenizer_type"] == "char"


@pytest.mark.unit
def test_tokenizer_resolved_from_chunking_section(tmp_path):
    """chunking.tokenizer_path 미존재 시 tokenizer_id(HF) 로 폴백하여 self._tokenizer 결정."""
    proc = _init_processor(_make_config(tmp_path, tokenizer_path="/nonexistent/path/xyz"))
    # 경로가 없으므로 HF id(str) 로 폴백
    assert proc._tokenizer == "sentence-transformers/all-MiniLM-L6-v2"
