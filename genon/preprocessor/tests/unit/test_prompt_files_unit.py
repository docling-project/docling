"""prompt 파일 분리 + default system prompt 단위 테스트.

- prompt_files.resolve_prompt_path / read_prompt_file (경로 해석/읽기)
- EnrichmentConfig 의 system_prompt_file / user_prompt_file 파싱 + 우선순위
- built-in default system prompt fallback
- has_custom_metadata 게이트 플래그

순수 로직(facade.enrichment.*)만 import 하므로 docling/httpx 없이 실행된다.
"""
from pathlib import Path

import pytest

from facade.enrichment.prompt_files import resolve_prompt_path, read_prompt_file
from facade.enrichment.enrichment_config import (
    EnrichmentConfig,
    _resolve_prompt,
    _DEFAULT_METADATA_SYSTEM_PROMPT,
)


# ── prompt_files 헬퍼 ───────────────────────────────────────────────────────────

@pytest.mark.unit
class TestPromptFiles:
    def test_relative_resolves_under_base(self, tmp_path):
        (tmp_path / "prompts").mkdir()
        f = tmp_path / "prompts" / "p.md"
        f.write_text("hello", encoding="utf-8")
        resolved = resolve_prompt_path("prompts/p.md", tmp_path)
        assert resolved == f.resolve()

    def test_absolute_allowed(self, tmp_path):
        f = tmp_path / "abs.md"
        f.write_text("x", encoding="utf-8")
        resolved = resolve_prompt_path(str(f), Path("/some/other/base"))
        assert resolved == f.resolve()

    def test_traversal_outside_base_rejected(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(ValueError):
            resolve_prompt_path("../secret.md", base)

    def test_read_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_prompt_file("nope.md", tmp_path)

    def test_read_strips_content(self, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("\n  내용  \n\n", encoding="utf-8")
        assert read_prompt_file("p.md", tmp_path) == "내용"


# ── _resolve_prompt 우선순위 ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestResolvePrompt:
    def test_file_beats_inline_and_default(self, tmp_path):
        f = tmp_path / "p.md"
        f.write_text("FROM_FILE", encoding="utf-8")
        out = _resolve_prompt("INLINE", "p.md", "DEFAULT", tmp_path)
        assert out == "FROM_FILE"

    def test_inline_beats_default(self, tmp_path):
        out = _resolve_prompt("INLINE", None, "DEFAULT", tmp_path)
        assert out == "INLINE"

    def test_default_when_no_inline_no_file(self, tmp_path):
        out = _resolve_prompt(None, None, "DEFAULT", tmp_path)
        assert out == "DEFAULT"

    def test_user_prompt_default_none(self, tmp_path):
        # user prompt 는 built-in default 가 None
        out = _resolve_prompt("  ", "", None, tmp_path)
        assert out is None


# ── EnrichmentConfig 통합: file 파싱 + default system + has_custom_metadata ──────

@pytest.mark.unit
class TestEnrichmentConfigPromptFiles:
    def _write(self, d: Path, name: str, text: str) -> None:
        (d / name).write_text(text, encoding="utf-8")

    def test_metadata_user_file_only_uses_default_system(self, tmp_path):
        self._write(tmp_path, "mu.md", "USER {{raw_text}}")
        ec = EnrichmentConfig.from_raw(
            [{"metadata": {
                "enable": True, "url": "http://m", "model": "m",
                "user_prompt_file": "mu.md", "output_fields": ["created_date"],
            }}],
            tmp_path,
        )
        assert ec.metadata.user_prompt == "USER {{raw_text}}"
        # system_prompt_file/inline 둘 다 없으므로 built-in default 사용
        assert ec.metadata.system_prompt == _DEFAULT_METADATA_SYSTEM_PROMPT
        assert ec.metadata.has_custom_metadata is True

    def test_metadata_system_file_beats_inline(self, tmp_path):
        self._write(tmp_path, "ms.md", "SYS_FROM_FILE")
        ec = EnrichmentConfig.from_raw(
            [{"metadata": {
                "enable": True, "url": "http://m", "model": "m",
                "system_prompt": "SYS_INLINE", "system_prompt_file": "ms.md",
                "output_fields": ["x"],
            }}],
            tmp_path,
        )
        assert ec.metadata.system_prompt == "SYS_FROM_FILE"

    def test_missing_file_raises_at_parse(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EnrichmentConfig.from_raw(
                [{"metadata": {"enable": True, "user_prompt_file": "ghost.md",
                               "output_fields": ["x"]}}],
                tmp_path,
            )

    def test_no_custom_signal_means_builtin_metadata(self, tmp_path):
        # url/model 만 있고 prompt/output_fields/parser 미지정 → has_custom_metadata False
        ec = EnrichmentConfig.from_raw(
            [{"metadata": {"enable": True, "url": "http://m", "model": "m"}}],
            tmp_path,
        )
        assert ec.metadata.do_metadata is True
        assert ec.metadata.has_custom_metadata is False
        # default system prompt 는 채워지지만 게이트는 흔들리지 않음
        assert ec.metadata.system_prompt == _DEFAULT_METADATA_SYSTEM_PROMPT

    def test_output_fields_alone_opts_in(self, tmp_path):
        ec = EnrichmentConfig.from_raw(
            [{"metadata": {"enable": True, "url": "http://m", "model": "m",
                           "output_fields": ["created_date"]}}],
            tmp_path,
        )
        assert ec.metadata.has_custom_metadata is True

    def test_toc_prompt_files(self, tmp_path):
        self._write(tmp_path, "ts.md", "TOC_SYS")
        self._write(tmp_path, "tu.md", "TOC_USER {{raw_text}}")
        ec = EnrichmentConfig.from_raw(
            [{"toc": {"enable": True, "url": "http://t", "model": "m",
                      "system_prompt_file": "ts.md", "user_prompt_file": "tu.md"}}],
            tmp_path,
        )
        assert ec.toc.system_prompt == "TOC_SYS"
        assert ec.toc.user_prompt == "TOC_USER {{raw_text}}"
        # toc 는 built-in default 가 없음(미지정 시 None)
        ec2 = EnrichmentConfig.from_raw(
            [{"toc": {"enable": True, "url": "http://t", "model": "m"}}], tmp_path,
        )
        assert ec2.toc.system_prompt is None

    def test_dict_format_file_parsing(self, tmp_path):
        self._write(tmp_path, "ms.md", "DICT_SYS")
        ec = EnrichmentConfig.from_raw(
            {"metadata": {"system_prompt_file": "ms.md", "output_fields": ["x"]}},
            tmp_path,
            parent_cfg={},
        )
        assert ec.metadata.system_prompt == "DICT_SYS"
        assert ec.metadata.has_custom_metadata is True
