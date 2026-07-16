"""가드레일(민감정보 분류/마스킹, #315) 설정 + 요청 플래그 파싱.

facade 의 각 processor 가 공유하는 guardrail 모듈. yaml 의 `guardrail:` 섹션을 읽어
접속 정보를 담고, 요청 kwargs 의 `guardrail_call`(0/1) 로 호출 여부를 판단한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    """0/1(int) · true/false(bool) · "0"/"1"/"true" 등 문자열을 bool 로 해석."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y", "t")
    return False


def call_enabled(kwargs: dict) -> bool:
    """요청 kwargs 의 `guardrail_call` 로 가드레일 호출 여부를 결정.
    GenOS 규약상 boolean 대신 정수 0/1 로 전달되므로 int/bool/문자열 모두 허용."""
    return _to_bool((kwargs or {}).get("guardrail_call"))


@dataclass
class GuardrailConfig:
    """전처리기 config yaml 의 `guardrail:` 섹션 (환경 종속 접속 정보)."""
    url: str = ""
    workflow_id: Optional[int] = None
    api_key: str = ""
    timeout: int = 60
    masking_enabled: bool = False

    @classmethod
    def from_cfg(cls, cfg: dict) -> "GuardrailConfig":
        gm = _as_dict((cfg or {}).get("guardrail"))
        timeout = _to_int(gm.get("timeout"))
        return cls(
            url=str(gm.get("url") or "").strip(),
            workflow_id=_to_int(gm.get("workflow_id")),
            api_key=str(gm.get("api_key") or "").strip(),
            timeout=timeout if timeout and timeout > 0 else 60,
            masking_enabled=_to_bool(gm.get("masking_enabled")),
        )

    @property
    def configured(self) -> bool:
        """url·workflow_id·api_key 가 모두 채워졌는지(하나라도 비면 호출 skip = fail-open)."""
        return bool(self.url and self.workflow_id is not None and self.api_key)
