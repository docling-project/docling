"""body_summary.py — 문서 본문 요약 공용 모듈.

이미지·차트 description 의 공통 컨텍스트({{doc_summary}} 변수)로 쓰기 위해, 문서 BODY
텍스트를 LLM 으로 1회 요약한다. page_description.py 와 동일하게 "옵션 없이 명시 인자를
받는 sync 함수" 스타일.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from docling.utils.llm_cache import cached_call, remaining_timeout

_log = logging.getLogger(__name__)

# {{full_text}} 를 본문으로 치환하는 기본 요약 프롬프트.
DEFAULT_BODY_SUMMARY_PROMPT = (
    "다음 문서 본문을 3~5문장으로 핵심만 한국어로 요약해줘.\n\n{{full_text}}"
)


def _to_single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collect_body_text(document: Any, max_chars: int) -> str:
    """DoclingDocument 의 BODY 텍스트를 결합해 max_chars 로 자른다. 실패/빈값이면 ""."""
    from docling_core.types.doc.document import ContentLayer

    texts: list[str] = []
    try:
        for item, _ in document.iterate_items(included_content_layers={ContentLayer.BODY}):
            text = _to_single_line(str(getattr(item, "text", "") or ""))
            if text:
                texts.append(text)
    except Exception:
        return ""
    full_text = "\n".join(texts).strip()
    if not full_text:
        return ""
    return full_text[:max_chars]


def summarize_body(
    document: Any,
    *,
    api_url: str,
    api_key: str = "",
    model: str = "model",
    prompt_template: str = "",
    max_chars: int = 6000,
    timeout: float = 360.0,
    headers: Optional[dict] = None,
) -> str:
    """문서 본문을 1회 LLM(text-only chat) 호출로 요약. 실패 시 "" (파이프라인 비차단)."""
    full_text = collect_body_text(document, max_chars)
    if not full_text:
        return ""

    prompt_tmpl = prompt_template or DEFAULT_BODY_SUMMARY_PROMPT
    prompt = prompt_tmpl.replace("{{full_text}}", full_text)

    req_headers = dict(headers or {})
    req_headers.setdefault("Content-Type", "application/json")
    if api_key and "Authorization" not in req_headers:
        req_headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model or "model",
        "messages": [{"role": "user", "content": prompt}],
    }
    def _produce() -> str:
        # #329: llm_cache opt-in 시 캐시 경유. 빈 결과("")는 cached_call 이 저장하지 않는다.
        with httpx.Client(timeout=httpx.Timeout(remaining_timeout(timeout))) as client:
            response = client.post(api_url, headers=req_headers, json=body)
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"] or "").strip()

    # 실패 시 ""(파이프라인 비차단)는 캐시 밖에 유지한다.
    try:
        doc_summary = cached_call(api_url, body, _produce)
        _log.debug("[body_summary] body summary success: %s", doc_summary)
        return doc_summary
    except Exception as exc:
        _log.warning("[body_summary] body summary failed: %s", exc)
        return ""
