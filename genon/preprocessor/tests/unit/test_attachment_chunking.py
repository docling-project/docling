"""
attachment_processor 청킹/설정 변경에 대한 단위 테스트 (로컬 실행 가능 부분).

로컬 환경은 vendored docling 충돌로 `facade.attachment_processor` import 가 불가하므로,
아래 두 검증은 **모듈 import 없이** 수행한다:
  1) 통합 문자수 헬퍼 `_char_split_text` — 소스에서 함수만 AST 추출해 stub 의존성으로 실행.
  2) shipped config(yaml) 구조 — chunk_size/chunker_type 공통화, generic·token cap 제거,
     formats.hwp 이동 등 리팩터 결과를 yaml 로드만으로 검증.

(DocumentProcessor.__init__ 파싱 → _default_kwargs 매핑 검증은 docling 가용 CI 에서 실행되는
 test_attachment_chunk_config_unit.py 가 담당한다.)
"""

import ast
from pathlib import Path

import pytest
import yaml

_FACADE_DIR = Path(__file__).resolve().parents[2] / "facade"
_ATTACH_SRC = _FACADE_DIR / "attachment_processor.py"
_RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resource"
_RESOURCE_DEV_DIR = Path(__file__).resolve().parents[2] / "resource_dev"
_CONFIG_NAME = "attachment_processor_config.yaml"


# ---------------------------------------------------------------------------
# _char_split_text: 실제 소스에서 함수만 추출 + stub 로 실행 (import 불필요)
# ---------------------------------------------------------------------------

class _FakeSplitter:
    """RecursiveCharacterTextSplitter 대체: 문자수 기준 고정 크기 분할."""

    def __init__(self, chunk_size, chunk_overlap):
        self.cs = int(chunk_size)
        self.co = int(chunk_overlap)

    def split_text(self, text):
        step = max(self.cs - self.co, 1)
        return [text[i:i + self.cs] for i in range(0, len(text), step)]


def _load_char_split_text():
    src = _ATTACH_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_char_split_text"
    )
    ns = {"RecursiveCharacterTextSplitter": _FakeSplitter}
    exec(ast.get_source_segment(src, fn), ns)  # noqa: S102 - 테스트 내 신뢰된 소스
    return ns["_char_split_text"]


_TXT = "abcdefghij" * 5  # 50자


@pytest.mark.unit
class TestCharSplitText:
    def setup_method(self):
        self.f = _load_char_split_text()

    def test_empty_returns_empty(self):
        assert self.f("", chunk_size=0, chunk_overlap=0) == []

    def test_chunk_size_zero_single_chunk(self):
        """chunk_size=0 → 문서 전체가 1청크(분할 안 함)."""
        assert self.f(_TXT, chunk_size=0, chunk_overlap=100) == [_TXT]

    def test_chunk_size_none_single_chunk(self):
        """chunk_size 미지정(None) → 0 과 동일하게 1청크."""
        assert self.f(_TXT, chunk_size=None, chunk_overlap=None) == [_TXT]

    def test_positive_chunk_size_splits_by_chars(self):
        """chunk_size>0 → 문자수 단위 분할, 각 청크는 chunk_size 이하, 원문 보존."""
        out = self.f(_TXT, chunk_size=20, chunk_overlap=0)
        assert len(out) > 1
        assert all(len(c) <= 20 for c in out)
        assert "".join(out) == _TXT

    def test_signature_has_no_token_cap(self):
        """token cap 제거 확인: 파라미터가 (text, chunk_size, chunk_overlap) 뿐."""
        import inspect
        params = list(inspect.signature(self.f).parameters)
        assert params == ["text", "chunk_size", "chunk_overlap"]


# ---------------------------------------------------------------------------
# shipped config 구조 검증 (yaml 로드만; import 불필요)
# ---------------------------------------------------------------------------

def _config_paths():
    paths = []
    for d in (_RESOURCE_DIR, _RESOURCE_DEV_DIR):
        p = d / _CONFIG_NAME
        if p.exists():
            paths.append(p)
    return paths


@pytest.mark.unit
@pytest.mark.parametrize("cfg_path", _config_paths(), ids=lambda p: p.parent.name)
class TestConfigStructure:
    def _load(self, cfg_path):
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def test_chunking_common_keys(self, cfg_path):
        """chunker_type·chunk_size 는 chunking 공통 레벨에 있다."""
        ch = self._load(cfg_path)["chunking"]
        assert ch.get("chunker_type") in ("recursive", "hybrid")
        assert "chunk_size" in ch  # 공통 chunk_size

    def test_generic_block_removed(self, cfg_path):
        assert "generic" not in self._load(cfg_path)["chunking"]

    def test_recursive_block_minimal(self, cfg_path):
        """recursive 블록엔 chunk_overlap 만(=chunk_size·tokenizer_id·token cap 제거)."""
        rec = self._load(cfg_path)["chunking"]["recursive"]
        assert "chunk_size" not in rec
        assert "tokenizer_id" not in rec
        assert "token_chunk_size_cap" not in rec

    def test_hybrid_block_no_moved_keys(self, cfg_path):
        """hybrid 블록엔 chunk_size·tokenizer_id 없음(공통/최상위로 이동)."""
        hyb = self._load(cfg_path)["chunking"]["hybrid"]
        assert "chunk_size" not in hyb
        assert "tokenizer_id" not in hyb
        assert hyb.get("tokenizer_type") in ("char", "huggingface")

    def test_tokenizer_single_source(self, cfg_path):
        ch = self._load(cfg_path)["chunking"]
        assert "tokenizer_path" in ch and "tokenizer_id" in ch

    def test_defaults_minimal(self, cfg_path):
        """defaults 엔 전역 옵션만(log_level, use_pdf_sdk). hwp 옵션은 formats 로 이동."""
        df = self._load(cfg_path)["defaults"]
        assert set(df.keys()) <= {"log_level", "use_pdf_sdk"}
        for moved in ("use_hwp_sdk", "dump_sdk_output", "save_images"):
            assert moved not in df

    def test_formats_hwp_present(self, cfg_path):
        """HWP 전용 옵션은 formats.hwp 로 이동."""
        hwp = self._load(cfg_path)["formats"]["hwp"]
        assert set(hwp.keys()) == {"use_hwp_sdk", "dump_sdk_output", "save_images"}
