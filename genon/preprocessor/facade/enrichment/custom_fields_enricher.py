import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml
from docling_core.types import DoclingDocument

from .base_enricher import BaseEnricher
from .prompt_files import read_prompt_file
from .prompt_template import PromptTemplate

_log = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "enrich" / "custom_fields"

# user 가 user_prompt 만 지정한 경우 사용할 built-in default system prompt.
_DEFAULT_CUSTOM_FIELDS_SYSTEM_PROMPT = (
    "너는 문서 정보추출 전문가다. 주어진 문서에서 요청한 필드를 정확하게 추출하라."
)


class CustomFieldsEnricher(BaseEnricher):
    """문서 단위 커스텀 메타데이터 추출 enricher.

    - LLM을 1회 호출해 다중 필드를 추출한다.
    - 파싱 로직은 parser 설정에 따라 외부 파이썬 파일/함수로 위임 가능하다.
    - parser 파일 경로는 config 파일과 동일한 위치(resource_path) 기준으로 해석한다.
    """

    def __init__(
        self,
        api_key: str = "",
        config_file: str = "",
        resource_path: str | None = None,
        url: str = "",
        model: str = "",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        timeout: int = 60,
        system_prompt: str = "",
        user_prompt: str = "",
        system_prompt_file: str = "",
        user_prompt_file: str = "",
        output_fields: list[str] | None = None,
        parser: dict | None = None,
        pages: list[int] | None = None,
        variables: dict | None = None,
        template: dict | None = None,
        template_mode: str = "strict",
    ):
        cfg = self._load_config(config_file, resource_path)
        prompt_cfg = cfg.get("prompt", {}) if isinstance(cfg.get("prompt"), dict) else {}

        # prompt 파일/parser 파일 경로 해석 기준 디렉토리.
        self._parser_base_dir = self._resolve_parser_base_dir(config_file, resource_path)

        self._url = url or cfg.get("url", "")
        self._model = model or cfg.get("model", "")
        self._max_tokens = max_tokens if max_tokens != 1000 else cfg.get("max_tokens", max_tokens)
        self._temperature = temperature if temperature != 0.0 else cfg.get("temperature", temperature)
        self._timeout = timeout if timeout != 60 else cfg.get("timeout", timeout)
        # 우선순위: file > 생성자 kwarg > cfg["*_prompt"] > cfg["prompt"][*] > built-in default
        self._system_prompt = (
            self._maybe_read_prompt(system_prompt_file or cfg.get("system_prompt_file"))
            or system_prompt
            or cfg.get("system_prompt", "").strip()
            or str(prompt_cfg.get("system", "")).strip()
            or _DEFAULT_CUSTOM_FIELDS_SYSTEM_PROMPT
        )
        self._user_prompt = (
            self._maybe_read_prompt(user_prompt_file or cfg.get("user_prompt_file"))
            or user_prompt
            or cfg.get("user_prompt", "").strip()
            or str(prompt_cfg.get("user", "")).strip()
        )
        self._output_fields = list(output_fields or cfg.get("output_fields", []))
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        resolved_key = api_key or cfg.get("api_key", "")
        if resolved_key:
            self._headers["Authorization"] = f"Bearer {resolved_key}"

        self._parser_cfg = parser or cfg.get("parser", {}) or {}
        self._parser_callable = self._build_parser_callable()
        self._extract_pattern: str = self._parser_cfg.get("extract_pattern", "")

        cfg_pages = cfg.get("pages")
        self._pages: list[int] | None = pages or (cfg_pages if isinstance(cfg_pages, list) and cfg_pages else None)

        # 변수 치환 템플릿. user-defined 변수는 reserved 와 함께 strict 검증에 허용된다.
        self._variables = dict(variables or cfg.get("variables") or {})
        _tmpl_cfg = template if isinstance(template, dict) else cfg.get("template")
        _mode = (_tmpl_cfg.get("mode") if isinstance(_tmpl_cfg, dict) else None) or template_mode or "strict"
        _allowed = set(self._variables.keys())
        self._system_tpl = PromptTemplate(self._system_prompt, mode=_mode, allowed_names=_allowed)
        self._user_tpl = PromptTemplate(self._user_prompt, mode=_mode, allowed_names=_allowed)

    def _maybe_read_prompt(self, file_ref: Any) -> str:
        """prompt 파일 경로가 지정된 경우 읽어서 반환, 없으면 빈 문자열."""
        if isinstance(file_ref, str) and file_ref.strip():
            return read_prompt_file(file_ref.strip(), self._parser_base_dir)
        return ""

    def _load_config(self, config_file: str, resource_path: str | None = None) -> dict:
        if not config_file:
            return {}
        cfg_path = Path(config_file)
        if cfg_path.is_absolute():
            path = cfg_path
        elif cfg_path.suffix in {".yaml", ".yml"} or len(cfg_path.parts) > 1:
            if resource_path:
                path = (Path(resource_path) / cfg_path).resolve()
            else:
                path = cfg_path
        else:
            if resource_path:
                candidate = (Path(resource_path) / f"{config_file}.yaml").resolve()
                if candidate.exists():
                    path = candidate
                else:
                    path = _CONFIG_DIR / f"{config_file}.yaml"
            else:
                path = _CONFIG_DIR / f"{config_file}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"custom_fields config 없음: {path}")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _resolve_parser_base_dir(config_file: str, resource_path: str | None) -> Path:
        if config_file:
            cfg_path = Path(config_file)
            if cfg_path.is_absolute():
                return cfg_path.resolve().parent
            if cfg_path.suffix in {".yaml", ".yml"} or len(cfg_path.parts) > 1:
                if resource_path:
                    return (Path(resource_path) / cfg_path).resolve().parent
                return cfg_path.resolve().parent
        if resource_path:
            return Path(resource_path).resolve()
        return Path.cwd().resolve()

    def _build_parser_callable(self) -> Callable[..., dict]:
        parser_type = str(self._parser_cfg.get("type", "json")).strip().lower()
        if parser_type == "python":
            return self._load_external_parser(self._parser_cfg)
        return self._default_parse

    def _load_external_parser(self, parser_cfg: dict) -> Callable[..., dict]:
        parser_file = parser_cfg.get("file")
        parser_callable = parser_cfg.get("callable", "parse")
        if not parser_file:
            raise ValueError("custom_fields parser.type=python 인 경우 parser.file 값이 필요합니다.")

        parser_path = (self._parser_base_dir / parser_file).resolve()
        try:
            parser_path.relative_to(self._parser_base_dir)
        except ValueError as exc:
            raise ValueError(
                f"parser 경로가 허용 범위를 벗어났습니다: {parser_path} (base: {self._parser_base_dir})"
            ) from exc

        if not parser_path.exists():
            raise FileNotFoundError(f"parser 파일이 없습니다: {parser_path}")

        module_name = f"custom_fields_parser_{abs(hash(str(parser_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, parser_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"parser 모듈 로딩 실패: {parser_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, parser_callable, None)
        if not callable(fn):
            raise TypeError(f"parser callable을 찾을 수 없거나 호출 불가: {parser_callable}")
        return fn

    def _extract_text_for_json(self, text: str) -> str:
        if not self._extract_pattern:
            return text
        m = re.search(self._extract_pattern, text, re.DOTALL)
        if not m:
            return text
        return m.group(1) if m.lastindex else m.group(0)

    def _default_parse(self, llm_output: str, **kwargs) -> dict:
        if isinstance(llm_output, dict):
            return llm_output
        if not isinstance(llm_output, str):
            return {}

        # extract_pattern 지정 시 먼저 적용
        if self._extract_pattern:
            candidate = self._extract_text_for_json(llm_output)
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
            return {}

        # 자동 fallback — 3단계
        # 1단계: 직접 파싱
        try:
            parsed = json.loads(llm_output)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # 2단계: 마크다운 코드블록
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", llm_output):
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # 3단계: raw_decode 스캔 — 설명문구 앞뒤에 JSON이 섞인 경우
        decoder = json.JSONDecoder()
        for i, ch in enumerate(llm_output):
            if ch in "{[":
                try:
                    parsed, _ = decoder.raw_decode(llm_output, i)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

        return {}

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item and isinstance(item["text"], str):
                        chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks).strip()
        return str(content)

    def _render_prompts(self, raw_text: str, document: DoclingDocument | None) -> "tuple[str, str]":
        needed = self._user_tpl.referenced | self._system_tpl.referenced
        if document is not None:
            ctx = PromptTemplate.doc_context(
                document, needed=needed, raw_text=raw_text, **self._variables
            )
        else:
            ctx = {"raw_text": raw_text, **self._variables}
        user = raw_text if self._user_tpl.is_empty else self._user_tpl.render(**ctx)
        system = "" if self._system_tpl.is_empty else self._system_tpl.render(**ctx)
        return system, user

    async def _call_llm(self, raw_text: str, document: DoclingDocument | None = None) -> str:
        system, prompt = self._render_prompts(raw_text, document)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._url, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()
            return self._normalize_message_content(data["choices"][0]["message"]["content"])

    def _parse_with_custom_parser(self, llm_output: str, document: DoclingDocument, **kwargs) -> dict:
        try:
            parsed = self._parser_callable(
                llm_output,
                output_fields=self._output_fields,
                parser_config=self._parser_cfg,
                document=document,
                **kwargs,
            )
        except TypeError:
            parsed = self._parser_callable(llm_output)
        if not isinstance(parsed, dict):
            raise TypeError("custom_fields parser 결과는 dict 이어야 합니다.")
        return parsed

    def _normalize_output_fields(self, parsed: dict) -> dict:
        if not self._output_fields:
            return parsed
        return {key: parsed.get(key) for key in self._output_fields}

    def _extract_raw_text(self, document: DoclingDocument) -> str:
        if not self._pages:
            return document.export_to_text()
        from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
        serializer = MarkdownDocSerializer(
            doc=document,
            params=MarkdownParams(pages=set(self._pages)),
        )
        return serializer.serialize().text

    async def enrich(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
        if not self._url or not self._model:
            _log.warning("custom_fields enricher 비활성: url/model 설정이 비어있습니다.")
            return document

        raw_text = self._extract_raw_text(document)
        try:
            llm_output = await self._call_llm(raw_text, document)
            parsed = self._parse_with_custom_parser(llm_output, document, **kwargs)
            normalized = self._normalize_output_fields(parsed)
        except Exception as e:
            _log.warning(f"custom_fields 추출 실패: {e}")
            normalized = {key: None for key in self._output_fields}

        context = kwargs.get("_enrichment_context")
        if isinstance(context, dict):
            context.setdefault("metadata", {}).update(normalized)

        return document
