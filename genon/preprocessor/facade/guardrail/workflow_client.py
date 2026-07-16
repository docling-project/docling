"""가드레일 분류 워크플로우 호출 (#315).

문서 전체 텍스트를 분류 워크플로우(run/v2)에 1회 보내 sensitive_infos[] 를 받는다.
실패·미설정 시 빈 리스트(fail-open) — 전처리기가 절대 안 터지게.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .config import GuardrailConfig

_log = logging.getLogger(__name__)


def classify_document(
    text: str,
    url: str,
    workflow_id,
    api_key: str,
    timeout: int = 60,
) -> list:
    """분류 워크플로우(run/v2) 문서당 1회 호출 → sensitive_infos[] 반환.

    입력: {"question": <문서 전체>} / 응답: data.sensitive_infos
    (data.text 에 JSON 문자열로 실려오면 파싱 fallback). 실패/미설정 시 [](fail-open).
    """
    import requests

    if not text:
        return []
    if not url or workflow_id is None or not api_key:
        _log.warning("[guardrail] url/workflow_id/api_key 미설정 — 분류 skip(fail-open)")
        return []
    try:
        endpoint = f"{url.rstrip('/')}/workflow/{workflow_id}/run/v2"
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"question": text},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"workflow code={body.get('code')} {body.get('errMsg')}")
        data = body.get("data") or {}
        infos = data.get("sensitive_infos")
        if infos is None:  # text 필드에 JSON 문자열로 실려온 경우 파싱 시도
            try:
                infos = (json.loads(data.get("text") or "{}")).get("sensitive_infos")
            except Exception:
                infos = None
        return infos if isinstance(infos, list) else []
    except Exception as exc:
        _log.warning(f"[guardrail] 분류 워크플로우 호출 실패 — skip(fail-open): {exc}")
        return []


def classify_with_config(text: str, cfg: GuardrailConfig) -> list:
    """GuardrailConfig 로 classify_document 를 호출하는 편의 래퍼."""
    return classify_document(text, cfg.url, cfg.workflow_id, cfg.api_key, cfg.timeout)
