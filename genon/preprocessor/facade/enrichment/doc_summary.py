"""doc_summary.py — 문서 본문요약을 요청당 1회만 계산·공유하는 전용 enrichment 단계.

기존에는 image/table description enricher 가 각각 `summarize_body()` 를 호출해 동일 문서에
LLM 요약을 중복 수행했다. 이 모듈은 doc_summary 를 단일 설정 블록(enrichment.doc_summary)으로
정의하고, 파이프라인 시작에 **한 번** 계산해 요청당 공유 dict(`_enrichment_context`)에 저장한다.

- image/table 은 `_enrichment_context["doc_summary"]` 를 읽어 `{{doc_summary}}` 로 소비만 한다.
- 계산된 요약은 출력 metadata 에도 노출된다(`_enrichment_context["metadata"]["doc_summary"]`).

실제 요약 계산은 `body_summary.summarize_body()` 에 위임한다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from docling_core.types import DoclingDocument

from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.body_summary import (
    DEFAULT_BODY_SUMMARY_PROMPT,
    summarize_body,
)

_log = logging.getLogger(__name__)

# 요청당 공유 dict(`_enrichment_context`) 안에서 요약 텍스트를 담는 키.
DOC_SUMMARY_CONTEXT_KEY = "doc_summary"


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _parse_optional_bool(value: Any, key: str = "") -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    if key:
        _log.warning(f"[DocSummaryOptions] Invalid bool value for '{key}': {value!r}. Fallback to default.")
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[DocSummaryOptions] Invalid int value for '{key}': {value!r}. Fallback to default.")
        return None


def _parse_optional_float(value: Any, key: str = "") -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[DocSummaryOptions] Invalid float value for '{key}': {value!r}. Fallback to default.")
        return None


@dataclass(frozen=True)
class DocSummaryOptions:
    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = "model"
    prompt_template: str = ""
    max_chars: int = 6000
    timeout: float = 360.0
    headers: dict[str, str] = field(default_factory=dict)
    provenance: str = "facade_doc_summary"

    @classmethod
    def from_config(
        cls,
        *,
        doc_summary_cfg: dict,
        fallback_api_url: str,
        fallback_api_key: str,
        fallback_model: str,
        config_dir: "Optional[Path]" = None,
    ) -> "DocSummaryOptions":
        doc_summary_cfg = _as_dict(doc_summary_cfg)
        base_dir = config_dir if config_dir is not None else Path.cwd()

        # prompt 우선순위: prompt_file > inline prompt_template > built-in default.
        # prompt_file 이 지정됐지만 읽기에 실패하면(파일 없음 등) 기본 프롬프트로 폴백(비차단).
        prompt_file = doc_summary_cfg.get("prompt_file")
        prompt_template = None
        if isinstance(prompt_file, str) and prompt_file.strip():
            try:
                prompt_template = read_prompt_file(prompt_file.strip(), base_dir)
            except Exception as exc:
                _log.warning("[DocSummaryOptions] prompt read failed: file=%s error=%s", prompt_file, exc)
                prompt_template = None
        if not isinstance(prompt_template, str) or not prompt_template.strip():
            inline = doc_summary_cfg.get("prompt_template")
            prompt_template = inline if isinstance(inline, str) and inline.strip() else DEFAULT_BODY_SUMMARY_PROMPT

        enabled = _parse_optional_bool(doc_summary_cfg.get("enable"), "enable")
        if enabled is None:
            enabled = _parse_optional_bool(doc_summary_cfg.get("enabled"), "enabled")
        max_chars = _parse_optional_int(doc_summary_cfg.get("max_chars"), "max_chars")
        if max_chars is None or max_chars <= 0:
            max_chars = 6000
        timeout = _parse_optional_float(doc_summary_cfg.get("timeout"), "timeout")
        timeout = 360.0 if timeout is None or timeout <= 0 else timeout

        return cls(
            enabled=False if enabled is None else enabled,
            api_url=str(doc_summary_cfg.get("api_url") or doc_summary_cfg.get("url") or fallback_api_url or "").strip(),
            api_key=str(doc_summary_cfg.get("api_key") or fallback_api_key or "").strip(),
            model=str(doc_summary_cfg.get("model") or fallback_model or "model").strip(),
            prompt_template=prompt_template,
            max_chars=max_chars,
            timeout=timeout,
            headers=_as_dict(doc_summary_cfg.get("headers")),
            provenance=str(
                doc_summary_cfg.get("provenance", "facade_doc_summary")
            ).strip()
            or "facade_doc_summary",
        )


def resolve_runtime_doc_summary_options(
    base: DocSummaryOptions,
    *,
    doc_summary: int,
) -> DocSummaryOptions:
    """런타임 kwargs(0/1)로 base 옵션의 enabled 를 override 한다."""
    return replace(base, enabled=(doc_summary == 1))


class DocSummaryEnricher:
    """문서 본문요약을 1회 계산해 `_enrichment_context` 에 공유·노출하는 enricher.

    다른 enricher(image/table)보다 먼저 실행되어야 하며, 결과를
    `context["doc_summary"]`(소비용) 및 `context["metadata"]["doc_summary"]`(출력용)에 저장한다.
    """

    def __init__(self, options: DocSummaryOptions):
        self.options = options

    def enrich(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        if not self.options.enabled:
            return document

        context = kwargs.get("_enrichment_context")
        if not isinstance(context, dict):
            # 공유 context 가 없으면 저장할 곳이 없어 계산 자체를 생략(비차단).
            _log.debug("[DocSummaryEnricher] no _enrichment_context; skip")
            return document

        # 이미 계산된 요약이 있으면 재계산하지 않는다(멱등).
        if DOC_SUMMARY_CONTEXT_KEY in context:
            return document

        if not self.options.api_url:
            _log.warning("[DocSummaryEnricher] enabled=true but api_url is empty; skip")
            context[DOC_SUMMARY_CONTEXT_KEY] = ""
            return document

        stage_started_at = time.perf_counter()
        summary = summarize_body(
            document,
            api_url=self.options.api_url,
            api_key=self.options.api_key,
            model=self.options.model,
            prompt_template=self.options.prompt_template,
            max_chars=self.options.max_chars,
            timeout=self.options.timeout,
            headers=self.options.headers,
        )
        context[DOC_SUMMARY_CONTEXT_KEY] = summary
        # 계산된 요약을 출력 metadata 에도 노출(metadata enricher 의 setdefault().update() 와 병합됨).
        if summary:
            context.setdefault("metadata", {})["doc_summary"] = summary

        elapsed = time.perf_counter() - stage_started_at
        _log.info(
            "[DocSummaryEnricher] doc summary done: has_summary=%s elapsed=%.3fs",
            bool(summary),
            elapsed,
        )
        _log.debug("[DocSummaryEnricher] doc summary: %s", summary)
        return document
