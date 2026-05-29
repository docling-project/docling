import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from docling_core.types import DoclingDocument

from .base_enricher import BaseEnricher

_log = logging.getLogger(__name__)


class MetadataEnricher(BaseEnricher):
    """YAML 설정 기반 커스텀 메타데이터 enricher.

    docling 내장 metadata enricher를 대체한다.
    - 프롬프트를 yaml에서 지정할 수 있다.
    - 파싱 로직을 yaml parser 설정으로 교체할 수 있다 (json/python).
    - output_fields로 추출 필드를 지정할 수 있다.
    - 기존 동작(첫 4페이지, 파일명 주입, 이미지태그 제거, add_key_values 저장)을 보존한다.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        output_fields: list[str],
        parser: dict,
        pages: Optional[list[int]] = None,
        max_tokens: int = 10000,
        temperature: float = 0.0,
        timeout: int = 3600,
        config_dir: Optional[Path] = None,
    ):
        self._url = url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._system_prompt = (system_prompt or "").strip()
        self._user_prompt = (user_prompt or "").strip()
        self._output_fields = list(output_fields or [])
        self._pages = pages  # None → 첫 4페이지 (docling 기존 동작)

        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._parser_base_dir = Path(config_dir).resolve() if config_dir else Path.cwd().resolve()
        self._parser_cfg = dict(parser or {})
        self._extract_pattern: str = self._parser_cfg.get("extract_pattern", "")
        parser_type = str(self._parser_cfg.get("type", "json")).strip().lower()
        if parser_type == "python":
            self._parser_callable = self._load_external_parser(self._parser_cfg)
        else:
            self._parser_callable = self._default_parse

    # ── 파서 ──────────────────────────────────────────────────────────────────

    def _load_external_parser(self, parser_cfg: dict) -> Callable[..., dict]:
        parser_file = parser_cfg.get("file")
        parser_callable = parser_cfg.get("callable", "parse")
        if not parser_file:
            raise ValueError("metadata parser.type=python 인 경우 parser.file 값이 필요합니다.")

        parser_path = (self._parser_base_dir / parser_file).resolve()
        try:
            parser_path.relative_to(self._parser_base_dir)
        except ValueError as exc:
            raise ValueError(
                f"parser 경로가 허용 범위를 벗어났습니다: {parser_path}"
            ) from exc

        if not parser_path.exists():
            raise FileNotFoundError(f"parser 파일이 없습니다: {parser_path}")

        module_name = f"metadata_parser_{abs(hash(str(parser_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, parser_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"parser 모듈 로딩 실패: {parser_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, parser_callable, None)
        if not callable(fn):
            raise TypeError(f"parser callable을 찾을 수 없거나 호출 불가: {parser_callable}")
        return fn

    def _default_parse(self, llm_output: str, **kwargs) -> dict:
        if isinstance(llm_output, dict):
            return llm_output
        if not isinstance(llm_output, str):
            return {}

        if self._extract_pattern:
            m = re.search(self._extract_pattern, llm_output, re.DOTALL)
            candidate = (m.group(1) if m and m.lastindex else m.group(0)) if m else llm_output
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
            return {}

        # 3단계 자동 fallback
        try:
            parsed = json.loads(llm_output)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", llm_output):
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

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

    def _parse(self, llm_output: str, document: DoclingDocument, **kwargs) -> dict:
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
            raise TypeError("metadata parser 결과는 dict 이어야 합니다.")
        return parsed

    # ── 텍스트 추출 ───────────────────────────────────────────────────────────

    def _extract_raw_text(self, document: DoclingDocument) -> str:
        total_pages = len(document.pages)
        if self._pages:
            pages_to_read = self._pages
        else:
            # docling 기존 동작: 첫 4페이지
            pages_to_read = list(range(1, min(5, total_pages + 1)))

        text = ""
        for page in pages_to_read:
            text += document.export_to_markdown(page_no=page)
        return text

    @staticmethod
    def _preprocess_text(text: str) -> str:
        return re.sub(r"<!--\s*image\s*-->", "", text)

    # ── LLM 호출 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                    chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks).strip()
        return str(content)

    async def _call_llm(self, raw_text: str) -> str:
        prompt = (
            self._user_prompt
            .replace("{{raw_text}}", raw_text)
            .replace("{raw_text}", raw_text)
        ) if self._user_prompt else raw_text

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
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

    # ── 결과 저장 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _store_in_document(document: DoclingDocument, metadata: dict) -> None:
        """docling 내장과 동일한 방식으로 KeyValueItem을 문서에 삽입한다."""
        try:
            from docling_core.types.doc.document import GraphData, GraphCell, GraphCellLabel
        except ImportError:
            _log.warning("GraphData/GraphCell import 실패 — 문서 저장 생략")
            return

        graph_cells = []
        cell_id = 0
        for key, value in metadata.items():
            graph_cells.append(GraphCell(
                label=GraphCellLabel.KEY,
                cell_id=cell_id,
                text=key,
                orig=key,
            ))
            cell_id += 1
            value_str = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            graph_cells.append(GraphCell(
                label=GraphCellLabel.VALUE,
                cell_id=cell_id,
                text=value_str,
                orig=value_str,
            ))
            cell_id += 1

        graph_data = GraphData(cells=graph_cells, links=[])
        document.add_key_values(graph=graph_data, prov=None, parent=None)

    # ── enrich ────────────────────────────────────────────────────────────────

    async def enrich(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
        if not self._url or not self._model:
            _log.warning("metadata enricher 비활성: url/model 설정이 비어있습니다.")
            return document

        empty_result = {field: None for field in self._output_fields}

        # 텍스트 추출 및 전처리 (docling 내장과 동일)
        raw_text = self._extract_raw_text(document)
        raw_text = self._preprocess_text(raw_text)

        # 파일명 주입 (docling 내장과 동일)
        org_filename = kwargs.get("org_filename")
        if org_filename:
            raw_text = f"filename: {org_filename}\n\n{raw_text}"

        if not raw_text.strip():
            self._store_in_document(document, empty_result)
            return document

        try:
            llm_output = await self._call_llm(raw_text)
            parsed = self._parse(llm_output, document, **kwargs)
            normalized = (
                {field: parsed.get(field) for field in self._output_fields}
                if self._output_fields
                else parsed
            )
        except Exception as e:
            _log.warning(f"metadata 추출 실패: {e}")
            normalized = empty_result

        _log.info(f"추출된 메타데이터: {json.dumps(normalized, ensure_ascii=False)}")

        # docling 내장과 동일한 저장 방식
        self._store_in_document(document, normalized)

        # 파이프라인 내 접근용 (custom_fields와 동일한 경로)
        context = kwargs.get("_enrichment_context")
        if isinstance(context, dict):
            context.setdefault("metadata", {}).update(normalized)

        return document
