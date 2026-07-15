"""table_description.py — TableItem 단위 표 description 공용 모듈.

각 TableItem 을 앞뒤 문맥·캡션·섹션헤더(+선택적 본문요약)와 함께 VLM 에 보내 한국어
요약을 생성하고 DescriptionAnnotation 으로 붙인다. refine 옵션이 켜지면 같은 VLM 호출에서
표 구조를 충실한 HTML 로 재구성해 MiscAnnotation(content["refined_html"]) 으로 함께 붙인다.

- `TableDescriptionOptions`      : config(enrichment.table_description) → 옵션 dataclass
- `TableDescriptionEnricher`     : 문서 순회 + 표별 VLM 호출(ThreadPoolExecutor)
- `resolve_runtime_table_options`: 런타임 kwargs(0/1) → base 옵션 override
- `TableDescriptionExtractor`    : parse-format 출력용 요약/재구성 HTML 추출

문서 본문요약({{doc_summary}})은 별도 `doc_summary` 단계가 `_enrichment_context` 에 넣어둔 값을
공유해서 사용한다. image_description.py 를 1:1 미러링한다.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    DescriptionAnnotation,
    DocItem,
    PictureItem,
    TableItem,
)
from docling_core.types.doc.document import ContentLayer, MiscAnnotation
from docling.utils.api_image_request import api_image_request

from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.prompt_template import PromptTemplate

_log = logging.getLogger(__name__)


# 표 description enricher 가 부착하는 annotation 의 기본 provenance(무관 annotation 배제용).
TABLE_DESCRIPTION_PROVENANCE = "facade_table_description"

# refine 통합 응답 마커 규약: 프롬프트가 아래 두 마커를 정확히 출력하도록 강제한다.
TABLE_HTML_MARKER = "[[[TABLE_HTML]]]"
TABLE_SUMMARY_MARKER = "[[[TABLE_SUMMARY]]]"
_REFINE_SPLIT_RE = re.compile(
    r"\[\[\[TABLE_HTML\]\]\](?P<html>.*?)\[\[\[TABLE_SUMMARY\]\]\](?P<summary>.*)",
    re.DOTALL,
)


# ── 소형 파싱 헬퍼(image_description.py 와 동일 컨벤션) ────────────────────────
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
        _log.warning(f"[TableDescriptionOptions] Invalid bool value for '{key}': {value!r}. Fallback to default.")
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[TableDescriptionOptions] Invalid int value for '{key}': {value!r}. Fallback to default.")
        return None


def _as_int_flag(value: Any, default: int = 0) -> int:
    """Normalize runtime feature flags to 0 or 1."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return 1
        if normalized in {"0", "false", "no", "n", "off"}:
            return 0
    return default


def _read_optional_prompt(config_dir: Path, prompt_file: str | None, default: str = "") -> str:
    if not prompt_file:
        return default
    try:
        return read_prompt_file(str(prompt_file), config_dir)
    except Exception as exc:
        _log.warning("[table_description] prompt read failed: file=%s error=%s", prompt_file, exc)
        return default


_DEFAULT_TABLE_DESCRIPTION_PROMPT_TEMPLATE = (
    "문서의 표 이미지를 설명해줘. "
    "아래 문맥을 참고해서 표가 담고 있는 핵심 정보를 2~4문장으로 간결하게 작성해줘.\n\n"
    "[문서 요약]\n{{doc_summary}}\n\n"
    "[섹션 헤더]\n{{section_header}}\n\n"
    "[캡션]\n{{caption}}\n\n"
    "[앞 문맥]\n{{before_context}}\n\n"
    "[뒤 문맥]\n{{after_context}}\n\n"
    "요구사항:\n"
    "1) 추측은 최소화하고 표에서 확인 가능한 사실(항목/수치/단위) 중심으로 작성\n"
    "2) 판독 불가한 값은 '판독 불가'로 표기\n"
    "3) 한국어로 작성"
)


@dataclass(frozen=True)
class TableDescriptionOptions:
    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = "model"
    timeout: float = 360.0
    concurrency: int = 8
    before_items: int = 3
    after_items: int = 2
    max_context_chars: int = 1500
    include_caption: bool = True
    include_section_header: bool = True
    same_page_first: bool = True
    provenance: str = TABLE_DESCRIPTION_PROVENANCE
    prompt_template: str = _DEFAULT_TABLE_DESCRIPTION_PROMPT_TEMPLATE
    template_mode: str = "strict"
    variables: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # 구조 재구성(refine): enabled 이면 통합 프롬프트로 재구성 HTML + 요약을 함께 생성
    refine_enabled: bool = False
    refine_prompt_template: str = ""

    @classmethod
    def from_config(
        cls,
        *,
        table_desc_cfg: dict,
        fallback_api_url: str,
        fallback_api_key: str,
        fallback_model: str,
        config_dir: "Optional[Path]" = None,
    ) -> "TableDescriptionOptions":
        table_desc_cfg = _as_dict(table_desc_cfg)
        base_dir = config_dir if config_dir is not None else Path.cwd()

        # prompt_template 우선순위: prompt_template_file > inline prompt_template > built-in default
        prompt_template_file = table_desc_cfg.get("prompt_template_file")
        if isinstance(prompt_template_file, str) and prompt_template_file.strip():
            prompt_template = read_prompt_file(prompt_template_file.strip(), base_dir)
        else:
            prompt_template = table_desc_cfg.get("prompt_template")
            if not isinstance(prompt_template, str):
                prompt_template = _DEFAULT_TABLE_DESCRIPTION_PROMPT_TEMPLATE

        tbl_variables = table_desc_cfg.get("variables")
        tbl_variables = dict(tbl_variables) if isinstance(tbl_variables, dict) else {}
        _tmpl_cfg = table_desc_cfg.get("template")
        tbl_mode = (_tmpl_cfg.get("mode") if isinstance(_tmpl_cfg, dict) else None) \
            or table_desc_cfg.get("template_mode") or "strict"

        def _parse_optional_float(value: Any, key: str) -> Optional[float]:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                _log.warning(
                    f"[TableDescriptionOptions] Invalid float value for '{key}': {value!r}. Fallback to default."
                )
                return None

        enabled = _parse_optional_bool(table_desc_cfg.get("enabled"), "enabled")
        timeout = _parse_optional_float(table_desc_cfg.get("timeout"), "timeout")
        concurrency = _parse_optional_int(table_desc_cfg.get("concurrency"), "concurrency")
        before_items = _parse_optional_int(table_desc_cfg.get("before_items"), "before_items")
        after_items = _parse_optional_int(table_desc_cfg.get("after_items"), "after_items")
        max_context_chars = _parse_optional_int(
            table_desc_cfg.get("max_context_chars"), "max_context_chars"
        )
        include_caption = _parse_optional_bool(table_desc_cfg.get("include_caption"), "include_caption")
        include_section_header = _parse_optional_bool(
            table_desc_cfg.get("include_section_header"), "include_section_header"
        )
        same_page_first = _parse_optional_bool(table_desc_cfg.get("same_page_first"), "same_page_first")

        timeout = 360.0 if timeout is None or timeout <= 0 else timeout
        if concurrency is None or concurrency <= 0:
            concurrency = 4
        if before_items is None or before_items < 0:
            before_items = 3
        if after_items is None or after_items < 0:
            after_items = 2
        if max_context_chars is None or max_context_chars <= 0:
            max_context_chars = 1500

        # ── 구조 재구성(refine) 하위 블록 ──
        refine_cfg = _as_dict(table_desc_cfg.get("refine"))
        refine_enabled = _parse_optional_bool(refine_cfg.get("enable"), "refine.enable")
        refine_prompt_template = _read_optional_prompt(
            base_dir, refine_cfg.get("prompt_file"), default=""
        )

        return cls(
            enabled=False if enabled is None else enabled,
            api_url=str(table_desc_cfg.get("api_url") or table_desc_cfg.get("url") or fallback_api_url or "").strip(),
            api_key=str(table_desc_cfg.get("api_key") or fallback_api_key or "").strip(),
            model=str(table_desc_cfg.get("model") or fallback_model or "model").strip(),
            timeout=timeout,
            concurrency=concurrency,
            before_items=before_items,
            after_items=after_items,
            max_context_chars=max_context_chars,
            include_caption=True if include_caption is None else include_caption,
            include_section_header=True if include_section_header is None else include_section_header,
            same_page_first=True if same_page_first is None else same_page_first,
            provenance=str(
                table_desc_cfg.get("provenance", "facade_table_description")
            ).strip()
            or "facade_table_description",
            prompt_template=prompt_template,
            template_mode=str(tbl_mode).strip().lower(),
            variables=tbl_variables,
            headers=_as_dict(table_desc_cfg.get("headers")),
            params=_as_dict(table_desc_cfg.get("params")),
            refine_enabled=False if refine_enabled is None else refine_enabled,
            refine_prompt_template=refine_prompt_template,
        )


def resolve_runtime_table_options(
    base: TableDescriptionOptions,
    *,
    table_desc: int,
    table_refine: int,
) -> TableDescriptionOptions:
    """런타임 kwargs(0/1)로 base 옵션을 override 한 새 옵션을 반환한다.

    - enabled        = table_desc==1 or table_refine==1  (refine 단독 지정도 표 enrichment 활성)
    - refine_enabled = table_refine==1

    doc_summary 는 별도 doc_summary 단계가 _enrichment_context 로 공유하므로 여기서 다루지 않는다.
    """
    return replace(
        base,
        enabled=(table_desc == 1 or table_refine == 1),
        refine_enabled=(table_refine == 1),
    )


def refined_html_to_format(refined_html: str, table_format: str, compact_tables: bool = True) -> str:
    """refine 재구성 HTML 표를 output table_format 에 맞춰 반환한다.

    - table_format == "markdown": HTML 을 docling TableData(grid) 로 재파싱해 markdown 표로 변환.
      compact_tables=True 면 컬럼 정렬 패딩을 제거한다(대형 표 축소).
    - 그 외(html 등): 원본 HTML 그대로.
    - 변환 실패/표 미검출/예외 시 원본 HTML 로 폴백(내용 손실·파이프라인 차단 방지).
    """
    if not refined_html:
        return refined_html
    if str(table_format).strip().lower() != "markdown":
        return refined_html
    try:
        md = _refined_html_to_markdown(refined_html, compact_tables)
        return md or refined_html
    except Exception as exc:
        _log.warning("[table_description] refined html→markdown 변환 실패, HTML 유지: %s", exc)
        return refined_html


def _parse_refined_table_data(html: str):
    """재구성 <table> HTML 을 docling TableData(grid) 로 복원한다. 구조적으로 유효하지 않으면 None.

    유효 = <table> 존재 + grid 비어있지 않음 + 2행 이상(헤더+데이터) + 첫 행 non-empty.
    백엔드 파서는 중첩표를 거부하므로 내부 표는 평문으로 평탄화한다.
    """
    if not html or not html.strip():
        return None
    # <table> 여닫음 쌍 검사: bs4 는 누락된 </table> 를 자동 보정하므로, 잘린(truncated) HTML 을
    # 걸러내려면 원문에서 여는/닫는 태그 수가 일치(≥1)하는지 먼저 확인한다.
    open_cnt = len(re.findall(r"<table\b", html, re.IGNORECASE))
    close_cnt = len(re.findall(r"</table\s*>", html, re.IGNORECASE))
    if open_cnt == 0 or open_cnt != close_cnt:
        return None
    # 지연 import (모듈 로드 시 하드 의존 회피)
    from bs4 import BeautifulSoup, Tag
    from docling.backend.genos_vlm_html_backend import GenosVlmHTMLDocumentBackend

    soup = BeautifulSoup(html, "html.parser")
    table_tag = soup.find("table")
    if not isinstance(table_tag, Tag):
        return None
    nested = table_tag.find("table")
    while isinstance(nested, Tag):
        txt = nested.get_text(" ", strip=True)
        if txt:
            nested.replace_with(txt)
        else:
            nested.decompose()
        nested = table_tag.find("table")

    table_data = GenosVlmHTMLDocumentBackend.parse_table_data(table_tag)
    grid = getattr(table_data, "grid", None)
    # grid 없음 / 데이터 행 없음(헤더만) 은 유효한 표가 아님.
    if table_data is None or not grid or len(grid) < 2:
        return None
    # 첫 행(헤더)에 내용 있는 셀이 하나도 없으면(전부 빈 셀) 유효한 표가 아님.
    if not any(str(getattr(cell, "text", "") or "").strip() for cell in grid[0]):
        return None
    return table_data


def is_valid_refined_html(refined_html: str) -> bool:
    """재구성 HTML 이 구조적으로 유효한 표인지 검사. False 면 원본 표로 폴백해야 한다."""
    try:
        return _parse_refined_table_data(refined_html) is not None
    except Exception as exc:
        _log.warning("[table_description] refined html 검증 실패, 원본 사용: %s", exc)
        return False


def is_valid_table_summary(summary: str) -> bool:
    """표 요약이 정상 산문인지 검사. 마커나 원문 <table> 이 섞이면(refine 파싱 잔재) False → 요약 폐기."""
    if not summary or not summary.strip():
        return False
    if TABLE_HTML_MARKER in summary or TABLE_SUMMARY_MARKER in summary:
        return False
    if re.search(r"</?table\b", summary, re.IGNORECASE):
        return False
    return True


def _refined_html_to_markdown(html: str, compact_tables: bool = True) -> str:
    """<table> HTML 을 docling 파서로 grid 복원 후 markdown 표로 렌더.

    복원한 TableData 를 임시 DoclingDocument 에 넣고 docling `MarkdownDocSerializer`
    (compact_tables 반영)로 직렬화한다. native docling 표의 markdown 경로와 동일 serializer 라
    포맷이 완전히 일치하며, compact_tables=True 면 컬럼 정렬 패딩이 제거된다. 병합셀은 grid 에 반영.
    """
    # 지연 import (모듈 로드 시 하드 의존 회피)
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.transforms.serializer.markdown import (
        MarkdownDocSerializer,
        MarkdownParams,
    )

    table_data = _parse_refined_table_data(html)
    if table_data is None:
        return ""

    doc = DoclingDocument(name="refined_table")
    tbl = doc.add_table(data=table_data)
    return MarkdownDocSerializer(
        doc=doc,
        params=MarkdownParams(compact_tables=compact_tables),
    ).serialize(item=tbl).text


class TableDescriptionExtractor:
    """TableItem 에 부착된 annotation 에서 요약/재구성 HTML 을 추출한다.

    parse-format 출력(파서)에서 표 설명 텍스트·재구성 HTML 을 뽑을 때 사용.
    """

    @staticmethod
    def extract_summary(item: TableItem) -> str:
        # 표 description enricher 가 부착한 것(provenance 일치)만 대상 — docling/타 enricher annotation 배제.
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, DescriptionAnnotation):
                continue
            if getattr(annotation, "provenance", "") != TABLE_DESCRIPTION_PROVENANCE:
                continue
            text = str(getattr(annotation, "text", "") or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def extract_refined_html(item: TableItem) -> str:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, MiscAnnotation):
                continue
            content = getattr(annotation, "content", None) or {}
            html = str(content.get("refined_html", "") or "").strip()
            if html:
                return html
        return ""


class TableDescriptionEnricher:
    def __init__(self, options: TableDescriptionOptions):
        self.options = options
        # {{doc_summary}} 는 표 프롬프트 공통 변수 → strict 모드 로드 허용
        allowed_names = set(getattr(options, "variables", {}) or {})
        allowed_names.add("doc_summary")
        mode = getattr(options, "template_mode", "strict")
        self._summary_prompt_tpl = PromptTemplate(
            options.prompt_template,
            mode=mode,
            allowed_names=allowed_names,
        )
        # refine 전용 통합 프롬프트(비어 있으면 요약 프롬프트로 폴백)
        refine_prompt = getattr(options, "refine_prompt_template", "") or options.prompt_template
        self._refine_prompt_tpl = PromptTemplate(
            refine_prompt,
            mode=mode,
            allowed_names=allowed_names,
        )
        # 같은 페이지에 표가 여러 개면 ThreadPoolExecutor 가 공유 page.image 를 동시에 crop/디코드하며
        # PIL lazy-load 가 깨진다. crop+디코드만 직렬화한다(느린 VLM 호출은 lock 밖에서 병렬 유지).
        self._page_image_lock = threading.Lock()

    @staticmethod
    def _get_item_page_no(item: DocItem, default_page_no: int = 1) -> int:
        prov_list = getattr(item, "prov", None) or []
        if not prov_list:
            return default_page_no
        page_no = getattr(prov_list[0], "page_no", None)
        if isinstance(page_no, int) and page_no > 0:
            return page_no
        return default_page_no

    @staticmethod
    def _to_single_line(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _is_context_candidate(self, item: DocItem) -> bool:
        # 표/그림 자체는 문맥 텍스트 후보에서 제외
        if isinstance(item, (TableItem, PictureItem)):
            return False

        text = self._to_single_line(str(getattr(item, "text", "") or ""))
        if not text:
            return False

        label = getattr(item, "label", None)
        label_value = label.value if hasattr(label, "value") else str(label or "")
        if label_value in {"page_header", "page_footer"}:
            return False
        return True

    def _collect_neighbor_context(
        self,
        items: list[DocItem],
        table_index: int,
        table_page_no: int,
        max_items: int,
        direction: str,
    ) -> list[str]:
        if max_items <= 0:
            return []

        if direction == "before":
            scan_range = range(table_index - 1, -1, -1)
        else:
            scan_range = range(table_index + 1, len(items))

        sequential: list[str] = []
        same_page: list[str] = []
        cross_page: list[str] = []

        for idx in scan_range:
            candidate = items[idx]
            if not self._is_context_candidate(candidate):
                continue
            text = self._to_single_line(str(getattr(candidate, "text", "") or ""))
            if not text:
                continue

            if not self.options.same_page_first:
                sequential.append(text)
                if len(sequential) >= max_items:
                    break
                continue

            candidate_page_no = self._get_item_page_no(
                candidate, default_page_no=table_page_no
            )
            if candidate_page_no == table_page_no:
                same_page.append(text)
            else:
                cross_page.append(text)

            if len(same_page) + len(cross_page) >= max_items:
                break

        if self.options.same_page_first:
            if direction == "before":
                same_page = list(reversed(same_page))
                cross_page = list(reversed(cross_page))
            selected = (same_page + cross_page)[:max_items]
        else:
            selected = sequential[:max_items]
            if direction == "before":
                selected.reverse()
        return selected

    def _collect_section_header_context(
        self,
        items: list[DocItem],
        table_index: int,
    ) -> str:
        for idx in range(table_index - 1, -1, -1):
            candidate = items[idx]
            label = getattr(candidate, "label", None)
            label_value = label.value if hasattr(label, "value") else str(label or "")
            if label_value in {"section_header", "title"}:
                text = self._to_single_line(str(getattr(candidate, "text", "") or ""))
                if text:
                    return text
        return ""

    def _truncate_context(self, text: str) -> str:
        if len(text) <= self.options.max_context_chars:
            return text
        return text[: self.options.max_context_chars].rstrip() + " ..."

    def _build_prompt(
        self,
        before_context: str,
        after_context: str,
        caption: str,
        section_header: str = "",
        *,
        use_refine: bool = False,
        doc_summary: str = "",
    ) -> str:
        # 컨텍스트(앞/뒤 문맥)만 개별 절단한다. 완성 프롬프트 전체를 자르면 refine 템플릿 말미의
        # 출력 마커/재구성 규칙이 잘려 응답 파싱이 실패하므로 프롬프트는 그대로 반환한다.
        safe_before = self._truncate_context(before_context) if before_context else "-"
        safe_after = self._truncate_context(after_context) if after_context else "-"
        safe_caption = caption or "-"
        safe_header = section_header or "-"
        safe_summary = doc_summary or "-"
        tpl = self._refine_prompt_tpl if use_refine else self._summary_prompt_tpl
        try:
            prompt = tpl.render(
                before_context=safe_before,
                after_context=safe_after,
                caption=safe_caption,
                section_header=safe_header,
                doc_summary=safe_summary,
                **(self.options.variables or {}),
            )
        except Exception as exc:
            _log.warning(
                f"[TableDescriptionEnricher] Invalid prompt_template, fallback to default: {exc}"
            )
            prompt = (
                "문맥을 참고해서 표를 설명해줘.\n\n"
                f"[앞 문맥]\n{safe_before}\n\n"
                f"[캡션]\n{safe_caption}\n\n"
                f"[뒤 문맥]\n{safe_after}"
            )
        return prompt

    @staticmethod
    def _parse_refine_output(output_text: str) -> "tuple[str, str]":
        """refine 통합 응답을 (summary, refined_html) 로 파싱.

        마커가 하나라도 있는데 정상 HTML…SUMMARY 구조로 매칭되지 않으면(응답이 잘리거나
        degeneration 으로 깨진 경우) refine 실패로 간주해 ("", "") 를 반환한다(→ 원본 표 폴백).
        마커가 전혀 없으면 일반 요약으로 간주(refined_html="") — 파이프라인 비차단.
        """
        match = _REFINE_SPLIT_RE.search(output_text)
        if not match:
            if TABLE_HTML_MARKER in output_text or TABLE_SUMMARY_MARKER in output_text:
                _log.warning("[TableDescriptionEnricher] refine 응답 마커 불완전 → 폐기(원본 표 사용)")
                return "", ""
            return output_text.strip(), ""
        refined_html = (match.group("html") or "").strip()
        # 코드펜스가 섞여 오면 제거
        refined_html = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", refined_html).strip()
        summary = (match.group("summary") or "").strip()
        return summary, refined_html

    def _annotate_single_table(
        self,
        document: DoclingDocument,
        table_item: TableItem,
        prompt: str,
        use_refine: bool,
    ) -> "Optional[tuple[str, str]]":
        """표 1개를 VLM 으로 처리해 (summary, refined_html) 반환. 실패/빈값이면 None."""
        # 공유 page.image 의 동시 crop/디코드(PIL lazy-load race) 방지: 취득+강제 로드만 직렬화.
        with self._page_image_lock:
            image = table_item.get_image(document, prov_index=0)
            if image is not None:
                image = image.copy()  # lock 안에서 강제 decode/load → 공유 버퍼와 분리된 독립 이미지
        if image is None:
            return None

        headers = dict(self.options.headers)
        if self.options.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.options.api_key}"

        params = dict(self.options.params)
        if self.options.model and "model" not in params:
            params["model"] = self.options.model

        output = api_image_request(
            image=image,
            prompt=prompt,
            url=self.options.api_url,
            timeout=self.options.timeout,
            headers=headers,
            **params,
        )
        output_text = str(output or "").strip()
        if not output_text:
            return None

        if use_refine:
            summary, refined_html = self._parse_refine_output(output_text)
        else:
            summary, refined_html = output_text, ""

        _log.debug(
            "[TableDescriptionEnricher] table description result: "
            f"seq={getattr(table_item, 'seq', '?')}, "
            f"summary={summary}, refined_html={refined_html}"
        )

        if not summary and not refined_html:
            return None
        return summary, refined_html

    def enrich(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        if not self.options.enabled:
            return document

        stage_started_at = time.perf_counter()

        if not self.options.api_url:
            _log.warning(
                "[TableDescriptionEnricher] enabled=true but api_url is empty; skip"
            )
            return document

        items: list[DocItem] = [
            item
            for item, _ in document.iterate_items(
                included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
            )
        ]
        if not items:
            return document

        # 문서 본문요약은 doc_summary 단계가 _enrichment_context 에 넣어둔 값을 공유한다.
        doc_summary = str((kwargs.get("_enrichment_context") or {}).get("doc_summary", "") or "")

        use_refine = self.options.refine_enabled

        targets: list[tuple[int, TableItem, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, TableItem):
                continue

            page_no = self._get_item_page_no(item, default_page_no=1)
            before_context_items = self._collect_neighbor_context(
                items=items,
                table_index=idx,
                table_page_no=page_no,
                max_items=self.options.before_items,
                direction="before",
            )
            after_context_items = self._collect_neighbor_context(
                items=items,
                table_index=idx,
                table_page_no=page_no,
                max_items=self.options.after_items,
                direction="after",
            )

            section_header = ""
            if self.options.include_section_header:
                section_header = self._collect_section_header_context(
                    items=items, table_index=idx
                )
                if section_header:
                    before_context_items = [section_header] + before_context_items

            caption = ""
            if self.options.include_caption:
                try:
                    caption = self._to_single_line(item.caption_text(document))
                except Exception:
                    caption = ""

            before_context = "\n".join(before_context_items)
            after_context = "\n".join(after_context_items)

            _log.debug(
                f"[TableDescriptionEnricher] table target: seq={len(targets)+1}, page={page_no}, "
                f"caption={'(empty)' if not caption else caption[:30]}, "
                f"section_header={'(empty)' if not section_header else section_header[:30]}, "
                f"use_refine={use_refine}, "
                f"before_context_items={len(before_context_items)}, after_context_items={len(after_context_items)}"
            )

            prompt = self._build_prompt(
                before_context=before_context,
                after_context=after_context,
                caption=caption,
                section_header=section_header,
                use_refine=use_refine,
                doc_summary=doc_summary,
            )
            table_seq = len(targets) + 1
            targets.append((table_seq, item, prompt))

        if not targets:
            elapsed = time.perf_counter() - stage_started_at
            _log.info(
                f"[TableDescriptionEnricher] no table target for table description; "
                f"elapsed={elapsed:.3f}s"
            )
            return document

        total_targets = len(targets)
        _log.info(
            f"[TableDescriptionEnricher] table description start: "
            f"targets={total_targets}, concurrency={self.options.concurrency}, refine={use_refine}"
        )

        stats_lock = threading.Lock()
        success_count = 0
        failed_count = 0
        skipped_count = 0

        def _annotate_target(target: tuple[int, TableItem, str]) -> None:
            nonlocal success_count, failed_count, skipped_count
            seq, tbl, prompt = target
            table_started_at = time.perf_counter()
            page_no = self._get_item_page_no(tbl, default_page_no=1)
            try:
                result = self._annotate_single_table(document, tbl, prompt, use_refine)
            except Exception as exc:
                elapsed = time.perf_counter() - table_started_at
                with stats_lock:
                    failed_count += 1
                _log.warning(
                    f"[TableDescriptionEnricher] table description failed: "
                    f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s, error={exc}"
                )
                return
            if result is None:
                elapsed = time.perf_counter() - table_started_at
                with stats_lock:
                    skipped_count += 1
                _log.debug(
                    f"[TableDescriptionEnricher] table description empty: "
                    f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s"
                )
                return

            summary, refined_html = result
            # 동일 provenance 로 앞서 붙인 annotation 을 제거(재실행 안전)
            tbl.annotations = [
                ann
                for ann in tbl.annotations
                if getattr(ann, "provenance", "") != self.options.provenance
                and not (
                    isinstance(ann, MiscAnnotation)
                    and (getattr(ann, "content", None) or {}).get("provenance") == self.options.provenance
                )
            ]
            if summary and is_valid_table_summary(summary):
                tbl.annotations.append(
                    DescriptionAnnotation(text=summary, provenance=self.options.provenance)
                )
            elif summary:
                # 마커/원문 <table> 이 섞인 비정상 요약: 부착하지 않음(원본 표/설명 없음으로 폴백).
                _log.warning(
                    "[TableDescriptionEnricher] invalid table summary (markup/marker), drop: "
                    f"seq={seq}, page={page_no}"
                )
            if refined_html and is_valid_refined_html(refined_html):
                tbl.annotations.append(
                    MiscAnnotation(content={"refined_html": refined_html, "provenance": self.options.provenance})
                )
            elif refined_html:
                # 재구성 표가 구조적으로 깨진 경우: 부착하지 않음 → 다운스트림이 원본 표로 폴백.
                _log.warning(
                    "[TableDescriptionEnricher] refined table invalid, fallback to original: "
                    f"seq={seq}, page={page_no}"
                )
            elapsed = time.perf_counter() - table_started_at
            with stats_lock:
                success_count += 1
            _log.debug(
                f"[TableDescriptionEnricher] table description done: "
                f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s"
            )

        with ThreadPoolExecutor(max_workers=self.options.concurrency) as executor:
            list(executor.map(_annotate_target, targets))

        total_elapsed = time.perf_counter() - stage_started_at
        _log.info(
            f"[TableDescriptionEnricher] table description done: "
            f"targets={total_targets}, success={success_count}, skipped={skipped_count}, "
            f"failed={failed_count}, elapsed={total_elapsed:.3f}s"
        )

        return document
