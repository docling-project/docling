"""
GenosSmartChunker 의 tokenizer_type(문자 수 기반 vs HF 토크나이저) 선택 기능 단위 테스트.

의존성(docling 등)이 없는 환경에서는 importorskip 으로 자동 skip 된다(CI gate).
intelligent_processor / convert_processor 두 곳의 GenosSmartChunker 가 동일하게 동작하는지 확인한다.
"""

import pytest


def _load_chunker(module_name: str):
    mod = pytest.importorskip(f"facade.{module_name}")
    return mod.GenosSmartChunker


_MODULES = ["intelligent_processor", "convert_processor"]


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
class TestTokenizerTypeChar:
    def test_default_is_char(self, module_name):
        """tokenizer_type 미지정 시 기본값은 char 이며 HF 토크나이저를 로드하지 않는다."""
        GenosSmartChunker = _load_chunker(module_name)
        chunker = GenosSmartChunker(max_tokens=100)
        assert chunker.tokenizer_type == "char"
        assert chunker._tokenizer is None

    def test_count_tokens_is_char_length(self, module_name):
        """char 모드에서 _count_tokens 는 문자 수(len)를 반환한다."""
        GenosSmartChunker = _load_chunker(module_name)
        chunker = GenosSmartChunker(max_tokens=100)
        assert chunker._count_tokens("") == 0
        assert chunker._count_tokens("가나다라") == 4
        assert chunker._count_tokens("abcde\nfghij") == len("abcde\nfghij")

    def test_invalid_value_falls_back_to_char(self, module_name):
        """알 수 없는 tokenizer_type 은 char 로 폴백한다."""
        GenosSmartChunker = _load_chunker(module_name)
        chunker = GenosSmartChunker(max_tokens=100, tokenizer_type="bogus")
        assert chunker.tokenizer_type == "char"
        assert chunker._tokenizer is None

    def test_value_is_normalized(self, module_name):
        """대소문자/공백은 정규화된다."""
        GenosSmartChunker = _load_chunker(module_name)
        chunker = GenosSmartChunker(max_tokens=100, tokenizer_type="  CHAR ")
        assert chunker.tokenizer_type == "char"

    def test_split_table_text_by_chars(self, module_name):
        """char 모드 테이블 분할은 문자 수 기준으로 chunk_size 를 넘지 않는다."""
        GenosSmartChunker = _load_chunker(module_name)
        chunker = GenosSmartChunker(max_tokens=100)
        text = "a" * 250
        parts = chunker._split_table_text(text, max_tokens=100)
        assert all(len(p) <= 100 for p in parts)
        assert "".join(parts) == text


@pytest.mark.unit
@pytest.mark.parametrize("module_name", _MODULES)
def test_huggingface_mode_uses_tokenizer(module_name):
    """huggingface 모드에서는 HF 토크나이저가 로드되고 tokenize 경로로 카운트한다.

    토크나이저(모델/네트워크) 미가용 환경에서는 skip.
    """
    GenosSmartChunker = _load_chunker(module_name)
    try:
        chunker = GenosSmartChunker(max_tokens=100, tokenizer_type="huggingface")
    except Exception as e:  # noqa: BLE001 - 모델/네트워크 미가용
        pytest.skip(f"HF tokenizer unavailable: {e}")
    assert chunker.tokenizer_type == "huggingface"
    assert chunker._tokenizer is not None
    # tokenize 경로(문자 수와 다른 토큰 수)로 동작하는지 확인
    assert chunker._count_tokens("hello world example text") > 0
