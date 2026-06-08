"""prompt_template.py — enrichment prompt 변수 치환 단일 유틸.

각 enricher 가 제각각 `.replace()` 하던 placeholder 치환을 한 곳으로 모은다.

설계 요점
---------
- 구문 통일: `{{variable}}` (mustache). prompt 본문의 JSON 예시(`{"key": "value"}`)와
  충돌하지 않는다 — single-brace `{...}` 는 토큰으로 보지 않으므로 그대로 통과한다.
- escape: `{{{{literal}}}}` → 리터럴 `{{literal}}` (4중 중괄호).
- 단일 패스(single `re.sub`): escape 와 token 을 좌→우 1회 처리하므로, 치환된 값 안에
  `{{...}}` 가 들어있어도 재확장되지 않는다.
- 하위호환 shim: 과거에 지원하던 단일 중괄호 `{raw_text}` 같은 표기도 (값이 주입되는
  변수에 한해) 계속 치환하되 deprecation 경고를 남긴다.
- strict / lenient mode:
  - strict(default): template 이 (reserved ∪ user-defined) 에 없는 `{{foo}}` 를 참조하면
    생성 시점(config load)에 ValueError. 변수 "이름"은 정적이라 문서 없이 검증 가능.
  - lenient: 미정의 변수는 렌더 시 빈 문자열로 치환하고 warning.

reserved 변수의 "값"은 DoclingDocument 가 있어야 만들 수 있으므로 `doc_context()` 가
런타임에 추출한다(value-based). 이름 검증은 load-time, 값 채움은 render-time 이다.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Optional

_log = logging.getLogger(__name__)


# ── reserved 변수 카탈로그 ──────────────────────────────────────────────────────

# DoclingDocument 에서 직접 도출 가능한 변수
DOC_RESERVED = frozenset({
    "raw_text", "full_text", "filename", "mimetype",
    "page_count", "table_count", "picture_count", "section_headers",
})
# 이미지 description item 단위(개별 PictureItem 컨텍스트)에서만 의미있는 변수
ITEM_RESERVED = frozenset({
    "before_context", "after_context", "caption", "section_header",
})
RESERVED_VAR_NAMES = DOC_RESERVED | ITEM_RESERVED

_NAME = r"[A-Za-z_][A-Za-z0-9_.]*"
# 단일 패스 정규식: escape 우선, 그다음 double-brace token.
_RENDER_RE = re.compile(
    r"(?P<esc_open>\{\{\{\{)"
    r"|(?P<esc_close>\}\}\}\})"
    r"|\{\{\s*(?P<tok>" + _NAME + r")\s*\}\}"
)


class PromptTemplate:
    """`{{var}}` 치환 템플릿.

    Args:
        template: 원본 prompt 문자열.
        mode: "strict" | "lenient".
        allowed_names: reserved 외에 추가로 허용할 user-defined 변수 이름.
    """

    def __init__(
        self,
        template: Optional[str],
        *,
        mode: str = "strict",
        allowed_names: Optional[Iterable[str]] = None,
    ):
        self._template = template or ""
        self._mode = (mode or "strict").strip().lower()
        if self._mode not in {"strict", "lenient"}:
            self._mode = "strict"
        self.allowed_names = set(RESERVED_VAR_NAMES)
        if allowed_names:
            self.allowed_names |= {str(n) for n in allowed_names}

        # template 이 실제 참조하는 token 이름(escape 안의 이름은 제외).
        self.referenced = frozenset(
            m.group("tok") for m in _RENDER_RE.finditer(self._template) if m.group("tok")
        )
        if self._mode == "strict":
            unknown = self.referenced - self.allowed_names
            if unknown:
                raise ValueError(
                    f"prompt template 에 정의되지 않은 변수 {sorted(unknown)} 가 있습니다. "
                    f"(reserved + user-defined 허용 변수: {sorted(self.allowed_names)})"
                )

    @property
    def is_empty(self) -> bool:
        return not self._template.strip()

    def render(self, **values: Any) -> str:
        """변수를 치환한 최종 prompt 문자열을 반환한다."""
        if not self._template:
            return ""

        # 하위호환: 값이 주입되는 변수에 한해 단일 중괄호 `{name}` 표기도 치환한다.
        shim_keys = [re.escape(k) for k in values.keys()]
        if shim_keys:
            pattern = re.compile(
                _RENDER_RE.pattern + r"|\{(?P<single>" + "|".join(shim_keys) + r")\}"
            )
        else:
            pattern = _RENDER_RE

        used_single = False

        def _repl(m: "re.Match") -> str:
            nonlocal used_single
            gd = m.groupdict()
            if gd.get("esc_open"):
                return "{{"
            if gd.get("esc_close"):
                return "}}"
            tok = gd.get("tok")
            if tok is not None:
                if tok not in values:
                    if self._mode != "strict" and tok not in self.allowed_names:
                        _log.warning(
                            f"[PromptTemplate] 미정의 변수 '{{{{{tok}}}}}' → 빈 문자열로 치환(lenient)."
                        )
                    return ""
                val = values.get(tok)
                return "" if val is None else str(val)
            single = gd.get("single")
            if single is not None:
                used_single = True
                val = values.get(single)
                return "" if val is None else str(val)
            return m.group(0)

        out = pattern.sub(_repl, self._template)
        if used_single:
            _log.warning(
                "[PromptTemplate] 단일 중괄호 placeholder(예: '{raw_text}')는 deprecated 입니다. "
                "이중 중괄호 '{{raw_text}}' 사용을 권장합니다."
            )
        return out

    # ── DoclingDocument 기반 reserved 변수 추출 ────────────────────────────────

    @classmethod
    def doc_context(
        cls,
        document: Any,
        *,
        needed: Optional[Iterable[str]] = None,
        **overrides: Any,
    ) -> dict:
        """DoclingDocument 에서 reserved 변수 값을 추출한다.

        `raw_text` 는 enricher 마다 추출 방식(첫 N페이지/지정 페이지/파일명 주입 등)이
        다르므로 여기서 재계산하지 않는다 — 호출자가 override 로 전달한다.

        Args:
            document: DoclingDocument.
            needed: 계산할 변수 이름 집합. None 이면 전부. (full_text 처럼 비싼 추출을
                불필요하게 수행하지 않도록 template 이 참조하는 변수만 넘기면 된다.)
            **overrides: reserved 추출값을 덮어쓸 값(raw_text 등) + user-defined 변수.
        """
        need = (lambda k: True) if needed is None else (lambda k: k in set(needed))
        ctx: dict[str, Any] = {}
        origin = getattr(document, "origin", None)

        if need("filename"):
            ctx["filename"] = getattr(origin, "filename", None) or ""
        if need("mimetype"):
            ctx["mimetype"] = getattr(origin, "mimetype", None) or ""
        if need("page_count"):
            ctx["page_count"] = cls._num_pages(document)
        if need("table_count"):
            ctx["table_count"] = len(getattr(document, "tables", []) or [])
        if need("picture_count"):
            ctx["picture_count"] = len(getattr(document, "pictures", []) or [])
        if need("section_headers"):
            ctx["section_headers"] = cls._section_headers(document)
        if need("full_text"):
            try:
                ctx["full_text"] = document.export_to_text()
            except Exception:  # noqa: BLE001
                ctx["full_text"] = ""

        ctx.update(overrides)
        return ctx

    @staticmethod
    def _num_pages(document: Any) -> int:
        try:
            return document.num_pages()
        except Exception:  # noqa: BLE001
            return len(getattr(document, "pages", {}) or {})

    @staticmethod
    def _section_headers(document: Any) -> str:
        headers: list[str] = []
        for t in getattr(document, "texts", []) or []:
            label = getattr(t, "label", None)
            label_value = label.value if hasattr(label, "value") else str(label or "")
            if label_value in {"section_header", "title"}:
                text = str(getattr(t, "text", "") or "").strip()
                if text:
                    headers.append(text)
        return "\n".join(headers)
