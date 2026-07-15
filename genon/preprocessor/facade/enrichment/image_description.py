"""image_description.py — PictureItem 단위 이미지/차트 description 공용 모듈.

각 PictureItem 을 앞뒤 문맥·캡션·섹션헤더(+선택적 본문요약)와 함께 VLM 에 보내 설명을
생성하고 DescriptionAnnotation 으로 붙인다. 차트로 분류된(또는 전체) 이미지에는 차트 전용
프롬프트를 적용할 수 있다.

- `ImageDescriptionOptions`        : config(enrichment.image_description) → 옵션 dataclass
- `ImageDescriptionEnricher` : 문서 순회 + 이미지별 VLM 호출(ThreadPoolExecutor)
- `resolve_runtime_image_options`  : 런타임 kwargs(0/1) → base 옵션 override

차트 판별은 `chart_detection` 모듈에 위임한다. 문서 본문요약({{doc_summary}})은 별도
`doc_summary` 단계가 `_enrichment_context` 에 넣어둔 값을 공유해서 사용한다.
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
)
from docling_core.types.doc.document import ContentLayer
from docling.utils.api_image_request import api_image_request

from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.prompt_template import PromptTemplate
from genon.preprocessor.facade.enrichment.chart_detection import is_chart

_log = logging.getLogger(__name__)


# ── 소형 파싱 헬퍼(패키지 자기완결; page_description 컨벤션과 동일) ─────────────
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
        _log.warning(f"[ImageDescriptionOptions] Invalid bool value for '{key}': {value!r}. Fallback to default.")
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[ImageDescriptionOptions] Invalid int value for '{key}': {value!r}. Fallback to default.")
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
        _log.warning("[image_description] prompt read failed: file=%s error=%s", prompt_file, exc)
        return default


_DEFAULT_IMAGE_DESCRIPTION_PROMPT_TEMPLATE = (
    "문서의 일부 이미지를 설명해줘. "
    "아래 문맥을 참고해서 핵심 정보를 2~4문장으로 간결하게 작성해줘.\n\n"
    "[앞 문맥]\n{{before_context}}\n\n"
    "[캡션]\n{{caption}}\n\n"
    "[뒤 문맥]\n{{after_context}}\n\n"
    "요구사항:\n"
    "1) 추측은 최소화하고 이미지에서 확인 가능한 사실 중심으로 작성\n"
    "2) 문서 문맥과의 연결점을 포함\n"
    "3) 한국어로 작성"
)


@dataclass(frozen=True)
class ImageDescriptionOptions:
    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = "model"
    timeout: float = 360.0
    concurrency: int = 16
    before_items: int = 3
    after_items: int = 2
    max_context_chars: int = 1500
    include_caption: bool = True
    include_section_header: bool = True
    same_page_first: bool = True
    provenance: str = "facade_image_description"
    prompt_template: str = _DEFAULT_IMAGE_DESCRIPTION_PROMPT_TEMPLATE
    template_mode: str = "strict"
    variables: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    # 차트 처리: enabled 이면 detection 방식에 따라 chart 프롬프트로 전환
    chart_enabled: bool = False
    chart_detection: str = "auto"  # "auto"=docling 자동판별 | "all"=모든 이미지를 차트로 처리
    chart_prompt_template: str = ""

    @classmethod
    def from_config(
        cls,
        *,
        image_desc_cfg: dict,
        fallback_api_url: str,
        fallback_api_key: str,
        fallback_model: str,
        config_dir: "Optional[Path]" = None,
    ) -> "ImageDescriptionOptions":
        image_desc_cfg = _as_dict(image_desc_cfg)
        base_dir = config_dir if config_dir is not None else Path.cwd()

        # prompt_template 우선순위: prompt_template_file > inline prompt_template > built-in default
        prompt_template_file = image_desc_cfg.get("prompt_template_file")
        if isinstance(prompt_template_file, str) and prompt_template_file.strip():
            prompt_template = read_prompt_file(prompt_template_file.strip(), base_dir)
        else:
            prompt_template = image_desc_cfg.get("prompt_template")
            if not isinstance(prompt_template, str):
                prompt_template = _DEFAULT_IMAGE_DESCRIPTION_PROMPT_TEMPLATE

        img_variables = image_desc_cfg.get("variables")
        img_variables = dict(img_variables) if isinstance(img_variables, dict) else {}
        _tmpl_cfg = image_desc_cfg.get("template")
        img_mode = (_tmpl_cfg.get("mode") if isinstance(_tmpl_cfg, dict) else None) \
            or image_desc_cfg.get("template_mode") or "strict"

        def _parse_optional_float(value: Any, key: str) -> Optional[float]:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                _log.warning(
                    f"[ImageDescriptionOptions] Invalid float value for '{key}': {value!r}. Fallback to default."
                )
                return None

        enabled = _parse_optional_bool(image_desc_cfg.get("enabled"), "enabled")
        timeout = _parse_optional_float(image_desc_cfg.get("timeout"), "timeout")
        concurrency = _parse_optional_int(image_desc_cfg.get("concurrency"), "concurrency")
        before_items = _parse_optional_int(image_desc_cfg.get("before_items"), "before_items")
        after_items = _parse_optional_int(image_desc_cfg.get("after_items"), "after_items")
        max_context_chars = _parse_optional_int(
            image_desc_cfg.get("max_context_chars"), "max_context_chars"
        )
        include_caption = _parse_optional_bool(image_desc_cfg.get("include_caption"), "include_caption")
        include_section_header = _parse_optional_bool(
            image_desc_cfg.get("include_section_header"), "include_section_header"
        )
        same_page_first = _parse_optional_bool(image_desc_cfg.get("same_page_first"), "same_page_first")

        timeout = 360.0 if timeout is None or timeout <= 0 else timeout
        if concurrency is None or concurrency <= 0:
            concurrency = 4
        if before_items is None or before_items < 0:
            before_items = 3
        if after_items is None or after_items < 0:
            after_items = 2
        if max_context_chars is None or max_context_chars <= 0:
            max_context_chars = 1500

        # ── 차트(chart) 하위 블록 ──
        chart_cfg = _as_dict(image_desc_cfg.get("chart"))
        chart_enabled = _parse_optional_bool(chart_cfg.get("enable"), "chart.enable")
        chart_detection = str(chart_cfg.get("detection") or "auto").strip().lower()
        if chart_detection not in {"auto", "all"}:
            _log.warning(
                f"[ImageDescriptionOptions] Unknown chart.detection '{chart_detection}', fallback to 'auto'."
            )
            chart_detection = "auto"
        chart_prompt_template = _read_optional_prompt(
            base_dir, chart_cfg.get("chart_prompt_file"), default=""
        )

        return cls(
            enabled=False if enabled is None else enabled,
            api_url=str(image_desc_cfg.get("api_url") or image_desc_cfg.get("url") or fallback_api_url or "").strip(),
            api_key=str(image_desc_cfg.get("api_key") or fallback_api_key or "").strip(),
            model=str(image_desc_cfg.get("model") or fallback_model or "model").strip(),
            timeout=timeout,
            concurrency=concurrency,
            before_items=before_items,
            after_items=after_items,
            max_context_chars=max_context_chars,
            include_caption=True if include_caption is None else include_caption,
            include_section_header=True if include_section_header is None else include_section_header,
            same_page_first=True if same_page_first is None else same_page_first,
            provenance=str(
                image_desc_cfg.get("provenance", "facade_image_description")
            ).strip()
            or "facade_image_description",
            prompt_template=prompt_template,
            template_mode=str(img_mode).strip().lower(),
            variables=img_variables,
            headers=_as_dict(image_desc_cfg.get("headers")),
            params=_as_dict(image_desc_cfg.get("params")),
            chart_enabled=False if chart_enabled is None else chart_enabled,
            chart_detection=chart_detection,
            chart_prompt_template=chart_prompt_template,
        )

    @classmethod
    def from_legacy_processor(cls, processor: Any) -> "ImageDescriptionOptions":
        def _safe_int(value: Any, default: int, min_value: int) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return default
            return parsed if parsed >= min_value else default

        def _safe_float(value: Any, default: float, min_value: float) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return default
            return parsed if parsed >= min_value else default

        return cls(
            enabled=bool(getattr(processor, "image_description_enabled", False)),
            api_url=str(getattr(processor, "image_description_api_url", "") or "").strip(),
            api_key=str(getattr(processor, "image_description_api_key", "") or "").strip(),
            model=str(getattr(processor, "image_description_model", "model") or "model").strip(),
            timeout=_safe_float(getattr(processor, "image_description_timeout", 20.0), 20.0, 0.00001),
            concurrency=_safe_int(getattr(processor, "image_description_concurrency", 4), 4, 1),
            before_items=_safe_int(getattr(processor, "image_description_before_items", 3), 3, 0),
            after_items=_safe_int(getattr(processor, "image_description_after_items", 2), 2, 0),
            max_context_chars=_safe_int(
                getattr(processor, "image_description_max_context_chars", 1500), 1500, 1
            ),
            include_caption=bool(getattr(processor, "image_description_include_caption", True)),
            include_section_header=bool(
                getattr(processor, "image_description_include_section_header", True)
            ),
            same_page_first=bool(getattr(processor, "image_description_same_page_first", True)),
            provenance=str(
                getattr(processor, "image_description_provenance", "facade_image_description")
                or "facade_image_description"
            ).strip(),
            prompt_template=str(
                getattr(
                    processor,
                    "image_description_prompt_template",
                    _DEFAULT_IMAGE_DESCRIPTION_PROMPT_TEMPLATE,
                )
            ),
            headers=dict(getattr(processor, "image_description_headers", {}) or {}),
            params=dict(getattr(processor, "image_description_params", {}) or {}),
        )


def resolve_runtime_image_options(
    base: ImageDescriptionOptions,
    *,
    img_desc: int,
    chart_desc: int,
    chart_detection: int,
    classification_available: bool,
) -> ImageDescriptionOptions:
    """런타임 kwargs(0/1)로 base 옵션을 override 한 새 옵션을 반환한다.

    - enabled            = img_desc==1 or chart_desc==1
    - chart_enabled      = chart_desc==1
    - chart_detection    = "auto" if chart_detection==1 else "all"
      (auto 지만 변환 단계 그림 분류가 꺼져 있으면 annotation 이 없으므로 all 로 강등)

    doc_summary 는 별도 doc_summary 단계가 _enrichment_context 로 공유하므로 여기서 다루지 않는다.
    """
    detection = "auto" if chart_detection == 1 else "all"
    if chart_desc == 1 and detection == "auto" and not classification_available:
        _log.warning(
            "[runtime_feature] chart_detection=auto 요청이나 그림 분류 미활성(config chart.enable=false) "
            "→ detection=all 로 강등"
        )
        detection = "all"
    return replace(
        base,
        enabled=(img_desc == 1 or chart_desc == 1),
        chart_enabled=(chart_desc == 1),
        chart_detection=detection,
    )


class PictureDescriptionExtractor:
    """PictureItem 에 부착된 DescriptionAnnotation 의 텍스트를 추출한다.

    parse-format 출력(파서)에서 그림 설명 텍스트를 뽑을 때 사용. 첫 번째 유효한
    DescriptionAnnotation.text 를 반환하고, 없으면 빈 문자열.
    """

    @staticmethod
    def extract(item: PictureItem) -> str:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, DescriptionAnnotation):
                continue
            text = str(getattr(annotation, "text", "") or "").strip()
            if text:
                return text
        return ""


class ImageDescriptionEnricher:
    def __init__(self, options: ImageDescriptionOptions):
        self.options = options
        # {{doc_summary}} 는 이미지·차트 프롬프트 공통 변수 → strict 모드 로드 허용
        allowed_names = set(getattr(options, "variables", {}) or {})
        allowed_names.add("doc_summary")
        mode = getattr(options, "template_mode", "strict")
        self._prompt_tpl = PromptTemplate(
            options.prompt_template,
            mode=mode,
            allowed_names=allowed_names,
        )
        # 차트 전용 프롬프트(비어 있으면 base 프롬프트로 폴백)
        chart_prompt = getattr(options, "chart_prompt_template", "") or options.prompt_template
        self._chart_prompt_tpl = PromptTemplate(
            chart_prompt,
            mode=mode,
            allowed_names=allowed_names,
        )

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
        if isinstance(item, PictureItem):
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
        picture_index: int,
        picture_page_no: int,
        max_items: int,
        direction: str,
    ) -> list[str]:
        if max_items <= 0:
            return []

        if direction == "before":
            scan_range = range(picture_index - 1, -1, -1)
        else:
            scan_range = range(picture_index + 1, len(items))

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
                candidate, default_page_no=picture_page_no
            )
            if candidate_page_no == picture_page_no:
                same_page.append(text)
            else:
                cross_page.append(text)

            if len(same_page) + len(cross_page) >= max_items:
                break

        if self.options.same_page_first:
            if direction == "before":
                # 그룹 우선순위(same page -> cross page)는 유지하면서 문서 순서로 정렬
                same_page = list(reversed(same_page))
                cross_page = list(reversed(cross_page))
            selected = (same_page + cross_page)[:max_items]
        else:
            selected = sequential[:max_items]
            if direction == "before":
                # 앞 문맥은 문서 순서(먼저 나온 텍스트 → 최근 텍스트)로 정렬한다.
                selected.reverse()
        return selected

    def _collect_section_header_context(
        self,
        items: list[DocItem],
        picture_index: int,
    ) -> str:
        for idx in range(picture_index - 1, -1, -1):
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
        use_chart: bool = False,
        doc_summary: str = "",
    ) -> str:
        safe_before = before_context or "-"
        safe_after = after_context or "-"
        safe_caption = caption or "-"
        safe_header = section_header or "-"
        safe_summary = doc_summary or "-"
        tpl = self._chart_prompt_tpl if use_chart else self._prompt_tpl
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
                f"[ImageDescriptionEnricher] Invalid prompt_template, fallback to default: {exc}"
            )
            prompt = (
                "문맥을 참고해서 이미지를 설명해줘.\n\n"
                f"[앞 문맥]\n{safe_before}\n\n"
                f"[캡션]\n{safe_caption}\n\n"
                f"[뒤 문맥]\n{safe_after}"
            )
        return self._truncate_context(prompt)

    def _annotate_single_picture(
        self,
        document: DoclingDocument,
        picture_item: PictureItem,
        prompt: str,
    ) -> Optional[DescriptionAnnotation]:
        image = picture_item.get_image(document, prov_index=0)
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
        return DescriptionAnnotation(
            text=output_text,
            provenance=self.options.provenance,
        )

    def enrich(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        if not self.options.enabled:
            return document

        stage_started_at = time.perf_counter()

        if not self.options.api_url:
            _log.warning(
                "[ImageDescriptionEnricher] enabled=true but api_url is empty; skip"
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

        chart_enabled = self.options.chart_enabled
        chart_all = chart_enabled and self.options.chart_detection == "all"
        chart_auto = chart_enabled and self.options.chart_detection == "auto"

        targets: list[tuple[int, PictureItem, str]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, PictureItem):
                continue

            page_no = self._get_item_page_no(item, default_page_no=1)
            before_context_items = self._collect_neighbor_context(
                items=items,
                picture_index=idx,
                picture_page_no=page_no,
                max_items=self.options.before_items,
                direction="before",
            )
            after_context_items = self._collect_neighbor_context(
                items=items,
                picture_index=idx,
                picture_page_no=page_no,
                max_items=self.options.after_items,
                direction="after",
            )

            section_header = ""
            if self.options.include_section_header:
                section_header = self._collect_section_header_context(
                    items=items, picture_index=idx
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
            use_chart = chart_all or (chart_auto and is_chart(item))

            _log.debug(
                f"[ImageDescriptionEnricher] picture target: seq={len(targets)+1}, page={page_no}, "
                f"caption={'(empty)' if not caption else caption[:30]}, "
                f"section_header={'(empty)' if not section_header else section_header[:30]}, "
                f"use_chart={use_chart}, "
                f"before_context_items={len(before_context_items)}, after_context_items={len(after_context_items)}"
            )

            prompt = self._build_prompt(
                before_context=before_context,
                after_context=after_context,
                caption=caption,
                section_header=section_header,
                use_chart=use_chart,
                doc_summary=doc_summary,
            )
            picture_seq = len(targets) + 1
            targets.append((picture_seq, item, prompt))

        if not targets:
            elapsed = time.perf_counter() - stage_started_at
            _log.info(
                f"[ImageDescriptionEnricher] no picture target for image description; "
                f"elapsed={elapsed:.3f}s"
            )
            return document

        total_targets = len(targets)
        _log.info(
            f"[ImageDescriptionEnricher] image description start: "
            f"targets={total_targets}, concurrency={self.options.concurrency}"
        )

        stats_lock = threading.Lock()
        success_count = 0
        failed_count = 0
        skipped_count = 0

        def _annotate_target(target: tuple[int, PictureItem, str]) -> None:
            nonlocal success_count, failed_count, skipped_count
            seq, pic, prompt = target
            picture_started_at = time.perf_counter()
            page_no = self._get_item_page_no(pic, default_page_no=1)
            try:
                annotation = self._annotate_single_picture(document, pic, prompt)
            except Exception as exc:
                elapsed = time.perf_counter() - picture_started_at
                with stats_lock:
                    failed_count += 1
                _log.warning(
                    f"[ImageDescriptionEnricher] image description failed: "
                    f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s, error={exc}"
                )
                return
            if annotation is None:
                elapsed = time.perf_counter() - picture_started_at
                with stats_lock:
                    skipped_count += 1
                _log.debug(
                    f"[ImageDescriptionEnricher] image description empty: "
                    f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s"
                )
                return
            pic.annotations = [
                ann
                for ann in pic.annotations
                if not (
                    isinstance(ann, DescriptionAnnotation)
                    and getattr(ann, "provenance", "") == self.options.provenance
                )
            ]
            pic.annotations.append(annotation)
            elapsed = time.perf_counter() - picture_started_at
            with stats_lock:
                success_count += 1
            _log.debug(
                f"[ImageDescriptionEnricher] image description done: "
                f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s"
            )

        with ThreadPoolExecutor(max_workers=self.options.concurrency) as executor:
            list(executor.map(_annotate_target, targets))

        total_elapsed = time.perf_counter() - stage_started_at
        _log.info(
            f"[ImageDescriptionEnricher] image description done: "
            f"targets={total_targets}, success={success_count}, skipped={skipped_count}, "
            f"failed={failed_count}, elapsed={total_elapsed:.3f}s"
        )

        return document
