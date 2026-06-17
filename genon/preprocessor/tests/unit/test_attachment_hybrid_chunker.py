"""
attachment_processor 의 HybridChunker tokenizer_type(문자 수 기반 vs HF 토크나이저) 선택 기능 테스트.

의존성(docling 등) 미가용 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
"""

import pytest


def _load_hybrid_chunker():
    mod = pytest.importorskip("facade.attachment_processor")
    return mod.HybridChunker


@pytest.mark.unit
class TestHybridChunkerTokenizerType:
    def test_default_is_char(self):
        """tokenizer_type 미지정 시 기본값은 char 이며 HF 토크나이저를 로드하지 않는다."""
        HybridChunker = _load_hybrid_chunker()
        chunker = HybridChunker(max_tokens=100)
        assert chunker.tokenizer_type == "char"
        assert chunker._tokenizer is None

    def test_count_text_tokens_is_char_length(self):
        """char 모드에서 _count_text_tokens 는 문자 수(len)를 반환한다(str/list 모두)."""
        HybridChunker = _load_hybrid_chunker()
        chunker = HybridChunker(max_tokens=100)
        assert chunker._count_text_tokens(None) == 0
        assert chunker._count_text_tokens("가나다라") == 4
        assert chunker._count_text_tokens(["ab", "cde"]) == 5  # 2 + 3

    def test_invalid_value_falls_back_to_char(self):
        HybridChunker = _load_hybrid_chunker()
        chunker = HybridChunker(max_tokens=100, tokenizer_type="bogus")
        assert chunker.tokenizer_type == "char"
        assert chunker._tokenizer is None

    def test_value_is_normalized(self):
        HybridChunker = _load_hybrid_chunker()
        chunker = HybridChunker(max_tokens=100, tokenizer_type="  CHAR ")
        assert chunker.tokenizer_type == "char"

    def test_huggingface_mode_loads_tokenizer(self):
        """huggingface 모드에서는 HF 토크나이저가 로드된다(미가용 시 skip)."""
        HybridChunker = _load_hybrid_chunker()
        try:
            chunker = HybridChunker(max_tokens=100, tokenizer_type="huggingface")
        except Exception as e:  # noqa: BLE001 - 모델/네트워크 미가용
            pytest.skip(f"HF tokenizer unavailable: {e}")
        assert chunker.tokenizer_type == "huggingface"
        assert chunker._tokenizer is not None
        assert chunker._count_text_tokens("hello world example text") > 0
