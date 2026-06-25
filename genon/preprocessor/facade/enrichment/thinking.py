"""thinking.py — LLM thinking(추론) 모드 토글 공유 헬퍼.

모델별로 thinking 토글 키 이름이 다르다. 키 이름은 vLLM 옵션이 아니라 모델의
chat template 이 정의하므로, dialect 로 흡수한다.

dialect:
  - "standard" (기본): chat_template_kwargs={"enable_thinking": bool}
    (Qwen3 / GLM / DeepSeek 등 비-hcx 진영 공통 관례)
  - "hcx": chat_template_kwargs={"force_reasoning": true} / {"skip_reasoning": true}
    (HyperCLOVAX-SEED-Think 전용)

mode 미설정(None/auto)이면 아무것도 전송하지 않아 기존 동작을 보존한다.

이 모듈은 모던 enricher, 레거시 docling 레이어(prompt_manager), 레거시 BOK facade가
모두 단일 소스로 import 해 재사용한다.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_ON_VALUES = {"on", "true", "1", "yes", "force", "enable", "enabled"}
_NEUTRAL_VALUES = {"", "auto", "hybrid", "default", "none"}


def resolve_thinking_kwargs(
    mode: Optional[str], dialect: str = "standard"
) -> Optional[dict]:
    """thinking 설정 → payload['chat_template_kwargs'] dict, 또는 None(미전송).

    Args:
        mode: "on" | "off" | "auto" | None
        dialect: "standard"(기본) | "hcx"

    Returns:
        chat_template_kwargs 로 넣을 dict. mode 가 None/auto/빈값이면 None
        (= chat_template_kwargs 미전송, 기존 동작 보존).
    """
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m in _NEUTRAL_VALUES:
        return None
    on = m in _ON_VALUES
    if str(dialect).strip().lower() == "hcx":
        return {"force_reasoning": True} if on else {"skip_reasoning": True}
    return {"enable_thinking": on}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(message: Any) -> str:
    """응답 message(dict) 또는 content(str)에서 본문만 반환.

    - reasoning parser 가 켜져 분리된 경우: message["content"] 에 본문만 있음.
    - 분리 안 된 경우: content 안에 <think>...</think> 가 섞일 수 있어 제거한다.
    - content 가 multimodal list 인 경우 text 파트만 이어붙인다.
    """
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = message or ""
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        content = "\n".join(chunks)
    elif not isinstance(content, str):
        content = str(content)
    return _THINK_RE.sub("", content).strip()
