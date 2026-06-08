"""MetadataEnricher / CustomFieldsEnricher 내부 로직 단위 테스트.

브랜치 task/109-enrichment-enrichment-prompt 의 핵심인 enrichment 프롬프트/파싱 로직을
검증한다. 두 enricher 는 stdlib + httpx 만 의존하며, 여기서는 httpx.AsyncClient 를 mock 하여
**실제 내부 LLM 서버를 호출하지 않는다**. 따라서 로컬/GitHub CI(내부망 미접근) 어디서든
pass(또는 의존성 미설치 시 skip)된다.

검증 대상:
- _default_parse: 3단계 JSON fallback / extract_pattern / 비-dict·비-str → {}
- _normalize_message_content: str / list[dict] / list[str] / 기타
- _preprocess_text: <!-- image --> 제거 (MetadataEnricher)
- _call_llm: {{raw_text}}/{raw_text} 치환, system+user 메시지, payload/Authorization 헤더 (요청 캡처)
- enrich(): url/model 미설정 시 no-op (네트워크 호출 없음)
- _load_external_parser: parser 경로 traversal 가드 (ValueError)
- _extract_raw_text: 기본 첫 4페이지 vs 명시 pages (MetadataEnricher)
- CustomFieldsEnricher._load_config: 경로 해석/누락 처리
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("httpx")
_me = pytest.importorskip("facade.enrichment.metadata_enricher")
_cf = pytest.importorskip("facade.enrichment.custom_fields_enricher")

MetadataEnricher = _me.MetadataEnricher
CustomFieldsEnricher = _cf.CustomFieldsEnricher


# ── httpx.AsyncClient mock 헬퍼 ────────────────────────────────────────────────

def _patch_async_client(module_path: str, captured: dict, content="{}"):
    """`<module>.httpx.AsyncClient` 를 async context manager mock 으로 패치한다.

    실제 전송 대신 post() 호출 인자를 captured 에 기록하고, LLM 응답 스키마를 흉내낸
    가짜 response 를 돌려준다.
    """
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={"choices": [{"message": {"content": content}}]}
    )

    async def _post(url, json=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return resp

    client = MagicMock()
    client.post = _post

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock(return_value=ctx)
    return patch(f"{module_path}.httpx.AsyncClient", factory)


def _make_metadata_enricher(**overrides):
    kwargs = dict(
        url="http://llm.internal/v1/chat/completions",
        api_key="secret",
        model="gpt-x",
        system_prompt="SYS",
        user_prompt="문서 내용: {{raw_text}}",
        output_fields=["created_date"],
        parser={"type": "json"},
    )
    kwargs.update(overrides)
    return MetadataEnricher(**kwargs)


def _make_custom_fields_enricher(**overrides):
    kwargs = dict(
        url="http://llm.internal/v1/chat/completions",
        api_key="secret",
        model="gpt-x",
        system_prompt="SYS",
        user_prompt="문서 내용: {{raw_text}}",
        output_fields=["authors"],
        parser={"type": "json"},
    )
    kwargs.update(overrides)
    return CustomFieldsEnricher(**kwargs)


# 두 enricher 모두에서 동일하게 검증할 (생성자, 모듈경로) 쌍
_ENRICHERS = [
    pytest.param(_make_metadata_enricher, "facade.enrichment.metadata_enricher", id="metadata"),
    pytest.param(_make_custom_fields_enricher, "facade.enrichment.custom_fields_enricher", id="custom_fields"),
]


# ── _default_parse ─────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("factory,_modpath", _ENRICHERS)
class TestDefaultParse:
    def test_direct_json_object(self, factory, _modpath):
        enr = factory()
        assert enr._default_parse('{"a": 1}') == {"a": 1}

    def test_json_array_is_not_dict_returns_empty(self, factory, _modpath):
        enr = factory()
        assert enr._default_parse('[1, 2, 3]') == {}

    def test_markdown_code_block(self, factory, _modpath):
        enr = factory()
        out = "설명입니다.\n```json\n{\"created_date\": \"2025-01-01\"}\n```\n끝."
        assert enr._default_parse(out) == {"created_date": "2025-01-01"}

    def test_raw_decode_scan_amid_prose(self, factory, _modpath):
        enr = factory()
        out = "결과는 다음과 같습니다 {\"k\": \"v\"} 이상입니다."
        assert enr._default_parse(out) == {"k": "v"}

    def test_non_string_non_dict_returns_empty(self, factory, _modpath):
        enr = factory()
        assert enr._default_parse(12345) == {}

    def test_dict_input_passthrough(self, factory, _modpath):
        enr = factory()
        assert enr._default_parse({"x": 1}) == {"x": 1}

    def test_garbage_returns_empty(self, factory, _modpath):
        enr = factory()
        assert enr._default_parse("그냥 평범한 텍스트, JSON 없음") == {}

    def test_extract_pattern_applied(self, factory, _modpath):
        enr = factory(parser={"type": "json", "extract_pattern": r"<json>([\s\S]*?)</json>"})
        out = "blah <json>{\"a\": 2}</json> blah"
        assert enr._default_parse(out) == {"a": 2}

    def test_extract_pattern_no_match_returns_empty(self, factory, _modpath):
        enr = factory(parser={"type": "json", "extract_pattern": r"<json>([\s\S]*?)</json>"})
        assert enr._default_parse("no markers here {\"a\": 2}") == {}


# ── _normalize_message_content ───────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("factory,_modpath", _ENRICHERS)
class TestNormalizeMessageContent:
    def test_plain_string(self, factory, _modpath):
        enr = factory()
        assert enr._normalize_message_content("hello") == "hello"

    def test_list_of_text_dicts(self, factory, _modpath):
        enr = factory()
        content = [{"text": "a"}, {"text": "b"}]
        assert enr._normalize_message_content(content) == "a\nb"

    def test_list_of_strings(self, factory, _modpath):
        enr = factory()
        assert enr._normalize_message_content(["x", "y"]) == "x\ny"

    def test_other_type_str_cast(self, factory, _modpath):
        enr = factory()
        assert enr._normalize_message_content(123) == "123"


# ── _preprocess_text (MetadataEnricher 전용) ──────────────────────────────────

@pytest.mark.unit
def test_metadata_preprocess_removes_image_tags():
    assert MetadataEnricher._preprocess_text("a <!-- image --> b <!--image--> c") == "a  b  c"


# ── _call_llm 프롬프트 빌드 (httpx mock, 실제 전송 없음) ───────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("factory,modpath", _ENRICHERS)
class TestCallLlmPromptBuild:
    def test_double_brace_placeholder_substituted(self, factory, modpath):
        enr = factory(user_prompt="문서 내용: {{raw_text}}")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("RAWDOC"))
        messages = captured["json"]["messages"]
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "문서 내용: RAWDOC"

    def test_single_brace_placeholder_substituted(self, factory, modpath):
        enr = factory(user_prompt="내용={raw_text}")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("RAWDOC"))
        assert captured["json"]["messages"][-1]["content"] == "내용=RAWDOC"

    def test_system_prompt_included_first(self, factory, modpath):
        enr = factory(system_prompt="SYSTEM")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("X"))
        messages = captured["json"]["messages"]
        assert messages[0] == {"role": "system", "content": "SYSTEM"}

    def test_empty_user_prompt_falls_back_to_raw_text(self, factory, modpath):
        enr = factory(user_prompt="")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("RAWONLY"))
        assert captured["json"]["messages"][-1]["content"] == "RAWONLY"

    def test_payload_carries_model_and_sampling(self, factory, modpath):
        enr = factory(model="my-model")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("X"))
        payload = captured["json"]
        assert payload["model"] == "my-model"
        assert "max_tokens" in payload
        assert "temperature" in payload

    def test_authorization_bearer_header(self, factory, modpath):
        enr = factory(api_key="topsecret")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("X"))
        assert captured["headers"]["Authorization"] == "Bearer topsecret"

    def test_response_content_returned_normalized(self, factory, modpath):
        enr = factory()
        captured = {}
        with _patch_async_client(modpath, captured, content="MODEL_OUTPUT"):
            result = asyncio.run(enr._call_llm("X"))
        assert result == "MODEL_OUTPUT"


# ── enrich() 비활성 no-op (url/model 미설정 → 네트워크 호출 없음) ──────────────

@pytest.mark.unit
@pytest.mark.parametrize("factory,modpath", _ENRICHERS)
class TestEnrichDisabledNoop:
    def test_empty_url_is_noop(self, factory, modpath):
        enr = factory(url="")
        doc = MagicMock()
        with _patch_async_client(modpath, {}) as _:
            # 패치되어 있어도 비활성 경로라면 호출되지 않아야 한다.
            result = asyncio.run(enr.enrich(doc))
        assert result is doc

    def test_empty_model_is_noop(self, factory, modpath):
        enr = factory(model="")
        doc = MagicMock()
        result = asyncio.run(enr.enrich(doc))
        assert result is doc


# ── parser 경로 traversal 가드 ─────────────────────────────────────────────────

@pytest.mark.unit
class TestParserPathTraversalGuard:
    def test_metadata_python_parser_outside_base_raises(self, tmp_path):
        with pytest.raises(ValueError):
            MetadataEnricher(
                url="http://x", api_key="", model="m",
                system_prompt="", user_prompt="",
                output_fields=[], parser={"type": "python", "file": "../evil.py"},
                config_dir=tmp_path,
            )

    def test_metadata_python_parser_missing_file_raises(self, tmp_path):
        # base 안의 경로지만 파일이 없으면 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            MetadataEnricher(
                url="http://x", api_key="", model="m",
                system_prompt="", user_prompt="",
                output_fields=[], parser={"type": "python", "file": "nope.py"},
                config_dir=tmp_path,
            )

    def test_metadata_python_parser_requires_file(self, tmp_path):
        with pytest.raises(ValueError):
            MetadataEnricher(
                url="http://x", api_key="", model="m",
                system_prompt="", user_prompt="",
                output_fields=[], parser={"type": "python"},
                config_dir=tmp_path,
            )

    def test_custom_fields_python_parser_outside_base_raises(self, tmp_path):
        with pytest.raises(ValueError):
            CustomFieldsEnricher(
                url="http://x", model="m", resource_path=str(tmp_path),
                parser={"type": "python", "file": "../evil.py"},
            )


# ── MetadataEnricher._extract_raw_text 페이지 선택 ─────────────────────────────

def _make_paged_doc(num_pages: int):
    doc = MagicMock()
    doc.pages = {i: object() for i in range(1, num_pages + 1)}
    doc.export_to_markdown = MagicMock(side_effect=lambda page_no: f"[{page_no}]")
    return doc


@pytest.mark.unit
class TestMetadataExtractRawText:
    def test_default_reads_first_four_pages(self):
        enr = _make_metadata_enricher(pages=None)
        doc = _make_paged_doc(6)
        text = enr._extract_raw_text(doc)
        assert text == "[1][2][3][4]"

    def test_fewer_than_four_pages(self):
        enr = _make_metadata_enricher(pages=None)
        doc = _make_paged_doc(2)
        assert enr._extract_raw_text(doc) == "[1][2]"

    def test_explicit_pages_respected(self):
        enr = _make_metadata_enricher(pages=[2, 3])
        doc = _make_paged_doc(6)
        assert enr._extract_raw_text(doc) == "[2][3]"


# ── CustomFieldsEnricher._load_config 경로 해석 ────────────────────────────────

@pytest.mark.unit
class TestCustomFieldsLoadConfig:
    def test_no_config_file_returns_empty(self):
        enr = object.__new__(CustomFieldsEnricher)
        assert enr._load_config("", None) == {}

    def test_yaml_with_resource_path(self, tmp_path):
        cfg = tmp_path / "fields.yaml"
        cfg.write_text("url: http://x\nmodel: m\n", encoding="utf-8")
        enr = object.__new__(CustomFieldsEnricher)
        loaded = enr._load_config("fields.yaml", str(tmp_path))
        assert loaded == {"url": "http://x", "model": "m"}

    def test_bare_name_with_resource_path(self, tmp_path):
        cfg = tmp_path / "authors.yaml"
        cfg.write_text("output_fields: [authors]\n", encoding="utf-8")
        enr = object.__new__(CustomFieldsEnricher)
        loaded = enr._load_config("authors", str(tmp_path))
        assert loaded == {"output_fields": ["authors"]}

    def test_missing_config_raises(self, tmp_path):
        enr = object.__new__(CustomFieldsEnricher)
        with pytest.raises(FileNotFoundError):
            enr._load_config("missing.yaml", str(tmp_path))


@pytest.mark.unit
class TestCustomFieldsPromptFiles:
    """custom_fields 의 system_prompt_file / user_prompt_file 지원 + default."""

    def test_prompt_files_loaded(self, tmp_path):
        (tmp_path / "cfs.md").write_text("CF_SYS", encoding="utf-8")
        (tmp_path / "cfu.md").write_text("CF_USER {{raw_text}}", encoding="utf-8")
        enr = _make_custom_fields_enricher(
            system_prompt="", user_prompt="",
            system_prompt_file="cfs.md", user_prompt_file="cfu.md",
            resource_path=str(tmp_path),
        )
        assert enr._system_prompt == "CF_SYS"
        assert enr._user_prompt == "CF_USER {{raw_text}}"

    def test_file_beats_inline_kwarg(self, tmp_path):
        (tmp_path / "cfs.md").write_text("FROM_FILE", encoding="utf-8")
        enr = _make_custom_fields_enricher(
            system_prompt="INLINE", system_prompt_file="cfs.md",
            resource_path=str(tmp_path),
        )
        assert enr._system_prompt == "FROM_FILE"

    def test_default_system_prompt_when_unset(self, tmp_path):
        from facade.enrichment.custom_fields_enricher import _DEFAULT_CUSTOM_FIELDS_SYSTEM_PROMPT
        enr = _make_custom_fields_enricher(
            system_prompt="", user_prompt="USER", resource_path=str(tmp_path),
        )
        assert enr._system_prompt == _DEFAULT_CUSTOM_FIELDS_SYSTEM_PROMPT

    def test_missing_prompt_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _make_custom_fields_enricher(
                user_prompt_file="ghost.md", resource_path=str(tmp_path),
            )


@pytest.mark.unit
@pytest.mark.parametrize("factory,modpath", _ENRICHERS)
class TestTemplateVariablesAndMode:
    """PromptTemplate 통합: user-defined 변수 / strict·lenient."""

    def test_strict_unknown_variable_raises_at_init(self, factory, modpath):
        with pytest.raises(ValueError):
            factory(user_prompt="안녕 {{unknown_var}}")

    def test_user_defined_variable_substituted(self, factory, modpath):
        enr = factory(
            user_prompt="회사={{company}} 문서={{raw_text}}",
            variables={"company": "GenON"},
        )
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("DOC"))
        assert captured["json"]["messages"][-1]["content"] == "회사=GenON 문서=DOC"

    def test_lenient_unknown_variable_renders_empty(self, factory, modpath):
        enr = factory(user_prompt="[{{unknown_var}}] {{raw_text}}", template_mode="lenient")
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("DOC"))
        assert captured["json"]["messages"][-1]["content"] == "[] DOC"

    def test_json_braces_in_prompt_preserved(self, factory, modpath):
        enr = factory(user_prompt='추출: {{raw_text}} 형식: {"date": "Y"}')
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("DOC"))
        assert captured["json"]["messages"][-1]["content"] == '추출: DOC 형식: {"date": "Y"}'

    def test_doc_context_fills_reserved(self, factory, modpath):
        """document 이 주어지면 filename/page_count 등 reserved 가 채워진다."""
        enr = factory(user_prompt="{{filename}} p{{page_count}} {{raw_text}}")
        doc = MagicMock()
        doc.origin.filename = "a.pdf"
        doc.num_pages.return_value = 3
        captured = {}
        with _patch_async_client(modpath, captured):
            asyncio.run(enr._call_llm("DOC", doc))
        assert captured["json"]["messages"][-1]["content"] == "a.pdf p3 DOC"
