# 변환용 전처리기 v.2.2.1.3 (2026-06-26 Release)
from __future__ import annotations

import json
import os
import logging
import math, bisect
import yaml
from pathlib import Path

from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple

from fastapi import Request

_log = logging.getLogger(__name__)

# from utils import assert_cancelled
import fitz
import math, bisect
import uuid
import shutil
import subprocess
import tempfile
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    # TextLoader,                       # TXT
    PyMuPDFLoader,  # PDF
    DataFrameLoader,  # DataFrame
    UnstructuredWordDocumentLoader,  # DOC and DOCX
    UnstructuredPowerPointLoader,  # PPT and PPTX
    UnstructuredImageLoader,  # JPG, PNG
    UnstructuredMarkdownLoader,  # Markdown
    UnstructuredFileLoader,  # Generic fallback
)
from langchain_core.documents import Document
# docling imports

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.genos_msword_backend import GenosMsWordDocumentBackend
# HWP/HWPX 레거시 백엔드 (GenosHwp SDK 실패 시 폴백용; olefile/xml 순수 파이썬, SDK 미사용)
from docling.backend.hwp_backend import HwpDocumentBackend
from docling.backend.xml.hwpx_backend import HwpxDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
# from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    # OcrEngine,
    # PdfBackend,
    LayoutModelType,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureModelType,
    PipelineOptions,
    PaddleOcrOptions,
    UpstageOcrOptions,
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    FormatOption,
    WordFormatOption,
    HwpxFormatOption
)
from docling.datamodel.pipeline_options import DataEnrichmentOptions
from docling.prompts.prompt_manager import LLMApiError
from docling.utils.document_enrichment import enrich_document, check_document
from docling.datamodel.document import ConversionResult
from docling.exceptions import HwpConversionError
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)

from docling_core.types import DoclingDocument

from pandas import DataFrame
import asyncio
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocumentOrigin,
    LevelNumber,
    ListItem,
    CodeItem,
    ContentLayer,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import (
    BoundingBox,
    DescriptionAnnotation,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    DocItem,
    ImageRef,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem,
    ProvenanceItem
)
from docling_core.types.doc.utils import relative_path
from docling.datamodel.settings import settings
from docling.utils.api_image_request import api_image_request

from collections import Counter
import re
import json
import time
import threading
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    import semchunk
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:
    raise RuntimeError(
        "Module requires 'chunking' extra; to install, run: "
        "`pip install 'docling-core[chunking]'`"
    )

try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None

from genon.preprocessor.facade.enrichment.enrichment_config import EnrichmentConfig
from genon.preprocessor.facade.enrichment.field_transforms import (
    DEFAULT_METADATA_FIELD_TRANSFORMS,
    apply_field_transforms,
    extract_metadata_from_document,
    serialize_metadata_value_for_output,
)
try:
    from genon.preprocessor.facade.enrichment.custom_fields_enricher import CustomFieldsEnricher
except ImportError:
    CustomFieldsEnricher = None  # type: ignore[assignment,misc]
try:
    from genon.preprocessor.facade.enrichment.metadata_enricher import MetadataEnricher
except ImportError:
    MetadataEnricher = None  # type: ignore[assignment,misc]

from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.prompt_template import PromptTemplate


# ============================================================
# 설정 로딩 헬퍼 (from parser_processor.py)
# ============================================================

def _warn_unresolved_placeholders(cfg: dict, config_path: str) -> None:
    """config 에 남아있는 미치환 플레이스홀더(<UPPER_SNAKE>)를 탐지해 경고한다.

    Site 배포 시 OCR/Layout/Enrichment endpoint·serving ID 등의 치환 누락을 조기에
    드러내기 위함. fail-fast 하지 않고(기동 보존) WARNING 로그만 남긴다.
    """
    pattern = re.compile(r"<[A-Z0-9_]+>")
    found = []

    def _scan(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _scan(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _scan(v, f"{path}[{i}]")
        elif isinstance(node, str):
            for ph in pattern.findall(node):
                found.append((path, ph))

    _scan(cfg, "")
    if found:
        lines = "\n".join(f"  - {path}: {ph}" for path, ph in found)
        _log.warning(
            "[DocumentProcessor] 미치환 설정 플레이스홀더가 발견되었습니다 "
            f"(config='{config_path}'). Site 배포 시 실제 값으로 변경하세요:\n{lines}"
        )


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: expected mapping, got {type(cfg).__name__}")
    _warn_unresolved_placeholders(cfg, config_path)
    return cfg


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
        _log.warning(f"[DocumentProcessor] Invalid bool value for '{key}': {value!r}. Fallback to default.")
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[DocumentProcessor] Invalid int value for '{key}': {value!r}. Fallback to default.")
        return None


def _parse_optional_float(value: Any, key: str = "") -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[DocumentProcessor] Invalid float value for '{key}': {value!r}. Fallback to default.")
        return None


# ============================================================
# Facade Image Description
# ============================================================

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


class FacadeImageDescriptionEnricher:
    def __init__(self, options: ImageDescriptionOptions):
        self.options = options
        self._prompt_tpl = PromptTemplate(
            options.prompt_template,
            mode=getattr(options, "template_mode", "strict"),
            allowed_names=set(getattr(options, "variables", {}) or {}),
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
    ) -> str:
        safe_before = before_context or "-"
        safe_after = after_context or "-"
        safe_caption = caption or "-"
        safe_header = section_header or "-"
        try:
            prompt = self._prompt_tpl.render(
                before_context=safe_before,
                after_context=safe_after,
                caption=safe_caption,
                section_header=safe_header,
                **(self.options.variables or {}),
            )
        except Exception as exc:
            _log.warning(
                f"[FacadeImageDescriptionEnricher] Invalid prompt_template, fallback to default: {exc}"
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
                "[FacadeImageDescriptionEnricher] enabled=true but api_url is empty; skip"
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
            prompt = self._build_prompt(
                before_context=before_context,
                after_context=after_context,
                caption=caption,
                section_header=section_header,
            )
            picture_seq = len(targets) + 1
            targets.append((picture_seq, item, prompt))

        if not targets:
            elapsed = time.perf_counter() - stage_started_at
            _log.info(
                f"[FacadeImageDescriptionEnricher] no picture target for image description; "
                f"elapsed={elapsed:.3f}s"
            )
            return document

        total_targets = len(targets)
        _log.info(
            f"[FacadeImageDescriptionEnricher] image description start: "
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
                    f"[FacadeImageDescriptionEnricher] image description failed: "
                    f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s, error={exc}"
                )
                return
            if annotation is None:
                elapsed = time.perf_counter() - picture_started_at
                with stats_lock:
                    skipped_count += 1
                _log.debug(
                    f"[FacadeImageDescriptionEnricher] image description empty: "
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
                f"[FacadeImageDescriptionEnricher] image description done: "
                f"seq={seq}, page={page_no}, elapsed={elapsed:.3f}s"
            )

        with ThreadPoolExecutor(max_workers=self.options.concurrency) as executor:
            list(executor.map(_annotate_target, targets))

        total_elapsed = time.perf_counter() - stage_started_at
        _log.info(
            f"[FacadeImageDescriptionEnricher] image description done: "
            f"targets={total_targets}, success={success_count}, skipped={skipped_count}, "
            f"failed={failed_count}, elapsed={total_elapsed:.3f}s"
        )

        return document


# pdf_pipeline.device / pdf_pipeline.table_structure_mode 의 yaml 문자열 → docling enum 매핑.
# 키가 없거나 알 수 없는 값이면 호출부에서 경고 + 기본값으로 폴백한다 (startup 견고성).
_ACCELERATOR_DEVICE_MAP = {
    "auto": AcceleratorDevice.AUTO,
    "cpu": AcceleratorDevice.CPU,
    "cuda": AcceleratorDevice.CUDA,
    "mps": AcceleratorDevice.MPS,
}

_TABLE_FORMER_MODE_MAP = {
    "accurate": TableFormerMode.ACCURATE,
    "fast": TableFormerMode.FAST,
}


def _resolve_default_convert_config_path() -> str:
    base_dir = Path(__file__).resolve().parent
    local_config = (base_dir / "../resource_dev/convert_processor_config.yaml").resolve()
    default_config = (base_dir / "../resource/convert_processor_config.yaml").resolve()

    if local_config.exists():
        return str(local_config)
    return str(default_config)


# 청킹용 토크나이저 기본 경로 (config 미지정 시 현행 동작 유지)
_DEFAULT_TOKENIZER_LOCAL_PATH = "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
_DEFAULT_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"

# tabular 모드로 직접 처리(행=벡터)할 엑셀 계열 포맷(이슈 #288).
# docling 모드(기본)에서는 xlsx 가 self.converter 의 docling 기본 백엔드(MsExcel)로 처리되므로
# 별도 인터셉트 없이 기존 경로를 그대로 탄다.
_XLSX_DIRECT_EXTS = {".xlsx", ".xlsm"}


def _resolve_tokenizer(chunking_cfg: dict):
    """chunking config 로부터 토크나이저를 결정한다.

    tokenizer_path 가 실제 존재하면 그 로컬 경로를, 없으면 tokenizer_id(HF) 로 폴백한다
    (외부 네트워크 차단 환경 대비). config 미지정 시 기본값은 현행 하드코딩 값과 동일.
    """
    local = chunking_cfg.get("tokenizer_path") or _DEFAULT_TOKENIZER_LOCAL_PATH
    hf_id = chunking_cfg.get("tokenizer_id") or _DEFAULT_TOKENIZER_ID
    return Path(local) if Path(local).exists() else hf_id


# ============================================
#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""
CONVERTIBLE_EXTENSIONS = ['.txt', '.json', '.md', '.docx', '.ppt', '.pptx']


def convert_to_pdf(file_path: str, use_pdf_sdk: bool = True) -> str | None:
    """
    PDF 변환을 시도한다. 실패해도 예외를 던지지 않고 None을 반환한다.

    chain (HWP/HWPX 입력):
      use_pdf_sdk=True  → pdf_sdk → rhwp → libreoffice
      use_pdf_sdk=False → rhwp → libreoffice
    chain (그 외 입력, 예: docx/pptx):
      use_pdf_sdk=True  → pdf_sdk → libreoffice
      use_pdf_sdk=False → libreoffice

    rhwp 는 HWP/HWPX 전용이라 비-HWP 입력에는 chain 에 들어가지 않는다. HWP/HWPX
    변환은 rhwp 를 libreoffice 보다 우선한다 (pdf_sdk 가 있으면 그 다음 순위).
    내부 구현은 `genon.preprocessor.converters.hwp_to_pdf` 모듈에 통합되어 있다.
    """
    from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf
    # 이슈 #286 — 변환 backend(pdf_sdk/rhwp/libreoffice)가 전무하면(빌드 시 OFF) 변환 시도가
    # 무의미하므로, PDF 직접 입력을 안내하는 warning 한 번만 남기고 None 을 반환한다.
    if not _has_any_pdf_converter():
        _log.warning(
            "[convert_to_pdf] PDF 변환기(rhwp/LibreOffice/PDF SDK)가 설치되어 있지 않습니다 "
            f"(이슈 #286). '{os.path.basename(file_path)}' 변환을 건너뜁니다. PDF 로 변환된 "
            "파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 빌드하세요 (genon/README.md 참고)."
        )
        return None
    ext = os.path.splitext(file_path)[1].lower()
    is_hwp = ext in (".hwp", ".hwpx")
    if use_pdf_sdk:
        order = ["pdf_sdk", "rhwp", "libreoffice"] if is_hwp else ["pdf_sdk", "libreoffice"]
    else:
        order = ["rhwp", "libreoffice"] if is_hwp else ["libreoffice"]
    return convert_hwp_to_pdf(file_path, order=order)


def _has_any_pdf_converter() -> bool:
    """PDF 변환 backend(pdf_sdk / rhwp / libreoffice) 가 하나라도 가용한지 확인 (이슈 #286).

    빌드 시 INSTALL_LIBREOFFICE / INSTALL_RHWP 를 끄거나 PDF SDK 미포함(standard)이면
    변환 backend 가 0개가 될 수 있다. 가용성 판단 자체가 불가하면(import 실패 등) True 를
    반환해 기존 동작을 유지한다.
    """
    try:
        from genon.preprocessor.converters.hwp_to_pdf.availability import (
            libreoffice_available,
            pdf_sdk_available,
            rhwp_available,
        )
        return bool(pdf_sdk_available() or rhwp_available() or libreoffice_available())
    except ImportError:
        # facade 단일 파일 실행 등으로 모듈 import 가 안 되는 경우 → 기존 동작 유지(가용 가정)
        return True
    except Exception as exc:
        # 가용성 probe 자체가 예기치 못하게 실패하면 로그만 남기고 파이프라인은 막지 않는다
        _log.warning(f"[_has_any_pdf_converter] PDF 변환기 가용성 확인 실패: {exc}")
        return True


def _get_pdf_path(file_path: str) -> str:
    """
    다양한 파일 확장자를 PDF 확장자로 변경하는 공통 함수
    Args:
        file_path (str): 원본 파일 경로
    Returns:
        str: PDF 확장자로 변경된 파일 경로
    """
    pdf_path = file_path
    for ext in CONVERTIBLE_EXTENSIONS:
        pdf_path = pdf_path.replace(ext, '.pdf')
    return pdf_path


class GenosSmartChunker(BaseChunker):
    """토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 청커 (v2)"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Union[PreTrainedTokenizerBase, str, Path] = (
            Path(_DEFAULT_TOKENIZER_LOCAL_PATH)
            if Path(_DEFAULT_TOKENIZER_LOCAL_PATH).exists()
            else _DEFAULT_TOKENIZER_ID
        )
    max_tokens: int = 1024
    merge_peers: bool = True
    # 토큰 수 계산 방식. "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
    tokenizer_type: str = "char"

    # _inner_chunker: BaseChunker = None
    _tokenizer: PreTrainedTokenizerBase = None
    merge_list_items: bool = True

    @model_validator(mode="after")
    def _initialize_components(self) -> Self:
        # 토크나이저 초기화
        mode = (self.tokenizer_type or "char").strip().lower()
        if mode not in {"char", "huggingface"}:
            _log.warning(f"[GenosSmartChunker] Unknown tokenizer_type '{mode}', fallback to 'char'.")
            mode = "char"
        self.tokenizer_type = mode
        if mode == "char":
            # 문자 수 기반: HF 토크나이저 로드 불필요 (외부 모델 의존 제거)
            self._tokenizer = None
        else:
            self._tokenizer = (
                self.tokenizer
                if isinstance(self.tokenizer, PreTrainedTokenizerBase)
                else AutoTokenizer.from_pretrained(self.tokenizer)
            )
        return self

    def preprocess(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """문서의 모든 아이템을 헤더 정보와 함께 청크로 생성

        Args:
            dl_doc: 청킹할 문서

        Yields:
            문서의 모든 아이템을 포함하는 하나의 청크
        """
        # 모든 아이템과 헤더 정보 수집
        all_items = []
        all_header_info = []  # 각 아이템의 헤더 정보
        current_heading_by_level: dict[LevelNumber, str] = {}
        all_header_short_info = []  # 각 아이템의 짧은 헤더 정보
        current_heading_short_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []

        # iterate_items()로 수집된 아이템들의 self_ref 추적
        processed_refs = set()

        # 모든 아이템 순회
        for item, level in dl_doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}):
            if hasattr(item, 'self_ref'):
                processed_refs.add(item.self_ref)

            if not isinstance(item, DocItem):
                continue

            # 리스트 아이템 병합 처리
            if self.merge_list_items:
                if isinstance(item, ListItem) or (
                    isinstance(item, TextItem) and item.label == DocItemLabel.LIST_ITEM
                ):
                    list_items.append(item)
                    continue
                elif list_items:
                    # 누적된 리스트 아이템들을 추가
                    for list_item in list_items:
                        all_items.append(list_item)
                        # 리스트 아이템의 헤더 정보 저장
                        all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                        all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
                    list_items = []

            # 섹션 헤더 처리
            if isinstance(item, SectionHeaderItem) or (
                isinstance(item, TextItem) and
                item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
            ):
                # 새로운 헤더 레벨 설정
                header_level = (
                    item.level if isinstance(item, SectionHeaderItem)
                    else (0 if item.label == DocItemLabel.TITLE else 1)
                )
                current_heading_by_level[header_level] = item.text
                current_heading_short_by_level[header_level] = item.orig  # 첫 단어로 짧은 헤더 정보 설정

                # 더 깊은 레벨의 헤더들 제거
                keys_to_del = [k for k in current_heading_by_level if k > header_level]
                for k in keys_to_del:
                    current_heading_by_level.pop(k, None)
                keys_to_del_short = [k for k in current_heading_short_by_level if k > header_level]
                for k in keys_to_del_short:
                    current_heading_short_by_level.pop(k, None)

                # 헤더 아이템도 추가 (헤더 자체도 아이템임)
                all_items.append(item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})
                continue

            if (isinstance(item, TextItem) or
                isinstance(item, ListItem) or
                isinstance(item, CodeItem) or
                isinstance(item, TableItem) or
                isinstance(item, PictureItem)):
                # if item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                #     item.text = ""
                all_items.append(item)
                # 현재 아이템의 헤더 정보 저장
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})

        # 마지막 리스트 아이템들 처리
        if list_items:
            for list_item in list_items:
                all_items.append(list_item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                all_header_short_info.append({k: v for k, v in current_heading_short_by_level.items()})

        # iterate_items()에서 누락된 테이블들을 별도로 추가
        missing_tables = []
        for table in dl_doc.tables:
            table_ref = getattr(table, 'self_ref', None)
            if table_ref not in processed_refs:
                missing_tables.append(table)

        # 누락된 테이블들을 문서 앞부분에 추가 (페이지 1의 테이블들일 가능성이 높음)
        if missing_tables:
            for missing_table in missing_tables:
                # 첫 번째 위치에 삽입 (헤더 테이블일 가능성이 높음)
                all_items.insert(0, missing_table)
                all_header_info.insert(0, {})  # 빈 헤더 정보
                all_header_short_info.insert(0, {})  # 빈 짧은 헤더 정보

        # 아이템이 없으면 빈 문서
        if not all_items:
            return

        # 모든 아이템을 하나의 청크로 반환 (HybridChunker에서 분할)
        # headings는 None으로 설정하고, 헤더 정보는 별도로 관리
        chunk = DocChunk(
            text="",  # 텍스트는 HybridChunker에서 생성
            meta=DocMeta(
                doc_items=all_items,
                headings=None,  # DocMeta의 원래 형식 유지
                captions=None,
                origin=dl_doc.origin,
            ),
        )
        # 헤더 정보를 별도 속성으로 저장
        chunk._header_info_list = all_header_info
        chunk._header_short_info_list = all_header_short_info  # 짧은 헤더 정보도 저장
        yield chunk

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산 (안전한 분할 처리)"""
        if not text:
            return 0

        if self._tokenizer is None:   # 문자 수 기반
            return len(text)

        # 텍스트를 더 작은 단위로 분할하여 계산
        max_chunk_length = 300  # 더 안전한 길이로 설정
        total_tokens = 0

        # 텍스트를 줄 단위로 먼저 분할
        lines = text.split('\n')
        current_chunk = ""

        for line in lines:
            # 현재 청크에 줄을 추가했을 때 길이 확인
            temp_chunk = current_chunk + '\n' + line if current_chunk else line

            if len(temp_chunk) <= max_chunk_length:
                current_chunk = temp_chunk
            else:
                # 현재 청크가 있으면 토큰 계산
                if current_chunk:
                    try:
                        total_tokens += len(self._tokenizer.tokenize(current_chunk))
                    except Exception:
                        total_tokens += int(len(current_chunk.split()) * 1.3)  # 대략적인 계산

                # 새로운 청크 시작
                current_chunk = line

        # 마지막 청크 처리
        if current_chunk:
            try:
                total_tokens += len(self._tokenizer.tokenize(current_chunk))
            except Exception:
                total_tokens += int(len(current_chunk.split()) * 1.3)  # 대략적인 계산

        return total_tokens

    def _generate_text_from_items_with_headers(self, items: list[DocItem],
                                              header_info_list: list[dict],
                                              dl_doc: DoclingDocument,
                                              **kwargs) -> str:
        """DocItem 리스트로부터 헤더 정보를 포함한 텍스트 생성"""
        text_parts = []
        current_section_headers = {}  # 현재 섹션의 헤더 정보

        for i, item in enumerate(items):
            item_headers = header_info_list[i] if i < len(header_info_list) else {}

            # 헤더 정보가 변경된 경우 (새로운 섹션 시작)
            if item_headers != current_section_headers:
                # 변경된 헤더 레벨들만 추가
                headers_to_add = []
                for level in sorted(item_headers.keys()):
                    # 이전 섹션과 다른 헤더만 추가
                    if (level not in current_section_headers or
                        current_section_headers[level] != item_headers[level]):
                        # 해당 레벨까지의 모든 상위 헤더 포함
                        for l in sorted(item_headers.keys()):
                            if l < level:
                                headers_to_add.append(item_headers[l])
                            elif l == level:
                                headers_to_add.append('')

                        break

                # 헤더가 있으면 추가
                if headers_to_add:
                    header_text = ", ".join(headers_to_add)
                    if header_text not in text_parts:
                        text_parts.append(header_text)

                current_section_headers = item_headers.copy()

            # 아이템 텍스트 추가
            if isinstance(item, TableItem):
                table_text = self._extract_table_text(item, dl_doc, **kwargs)
                if table_text:
                    text_parts.append(table_text)
            elif hasattr(item, 'text') and item.text:
                # 타이틀과 섹션 헤더 처리 개선
                # is_section_header = (
                #     isinstance(item, SectionHeaderItem) or
                #     (isinstance(item, TextItem) and
                #      item.label in [DocItemLabel.SECTION_HEADER])  # TITLE은 제외
                # )

                # 타이틀은 항상 포함, 섹션 헤더는 중복 방지를 위해 스킵
                # if not is_section_header:
                # 20250909, shkim, text_parts에 없는 경우만 추가. 섹션헤더가 반복해서 추가되는 것 방지
                if item.text not in text_parts:
                    text_parts.append(item.text)
            elif isinstance(item, PictureItem):
                text_parts.append("")  # 이미지는 빈 텍스트

        result_text = self.delim.join(text_parts)
        return result_text

    def _extract_table_text(self, table_item: TableItem, dl_doc: DoclingDocument, **kwargs) -> str:
        """테이블에서 텍스트를 추출하는 일반화된 메서드"""
        try:
            # 먼저 export_to_markdown 시도
            export_to_html = kwargs.get('export_to_html', 1)
            if export_to_html == 1:
                table_text = table_item.export_to_html(dl_doc)
            else:
                table_text = table_item.export_to_markdown(dl_doc)
            if table_text and table_text.strip():
                return table_text
        except Exception:
            pass

        # export_to_markdown 실패 시 테이블 셀 데이터에서 직접 텍스트 추출
        try:
            if hasattr(table_item, 'data') and table_item.data:
                cell_texts = []

                # table_cells에서 텍스트 추출
                if hasattr(table_item.data, 'table_cells'):
                    for cell in table_item.data.table_cells:
                        if hasattr(cell, 'text') and cell.text and cell.text.strip():
                            cell_texts.append(cell.text.strip())

                # grid에서 텍스트 추출 (table_cells가 없는 경우)
                elif hasattr(table_item.data, 'grid') and table_item.data.grid:
                    for row in table_item.data.grid:
                        if isinstance(row, list):
                            for cell in row:
                                if hasattr(cell, 'text') and cell.text and cell.text.strip():
                                    cell_texts.append(cell.text.strip())

                # 추출된 셀 텍스트들을 결합
                if cell_texts:
                    return ' '.join(cell_texts)
        except Exception:
            pass

        # 모든 방법 실패 시 item.text 사용 (있는 경우)
        if hasattr(table_item, 'text') and table_item.text:
            return table_item.text

        return ""

    @staticmethod
    def _render_table_row_html(row: list, num_cols: int) -> str:
        """grid 한 행을 <tr>..</tr> HTML 로 렌더(docling HTMLTableSerializer 형식 모방).
        colspan 중복 셀은 제거하고 헤더 계열 셀은 <th>, 그 외는 <td> 로 낸다.
        (row_span==1 전제 — 호출부에서 세로 병합 표는 분할하지 않음)
        """
        import html as _html
        cells = []
        for j in range(num_cols):
            cell = row[j]
            if cell.start_col_offset_idx != j:  # colspan 으로 이미 렌더된 셀 스킵
                continue
            is_header = bool(
                getattr(cell, "column_header", False)
                or getattr(cell, "row_header", False)
                or getattr(cell, "row_section", False)
            )
            tag = "th" if is_header else "td"
            attrs = f' colspan="{cell.col_span}"' if cell.col_span > 1 else ""
            cells.append(f"<{tag}{attrs}>{_html.escape((cell.text or '').strip())}</{tag}>")
        return "<tr>" + "".join(cells) + "</tr>"

    @staticmethod
    def _sheet_prefix(table_item: TableItem, dl_doc: DoclingDocument) -> str:
        """xlsx docling 표의 부모 그룹(name='sheet: X')에서 시트명을 뽑아 '시트명: X\\n' 접두 생성.
        시트 그룹이 없으면 '' 반환(PDF 등 비-xlsx 문서엔 실질 미적용)."""
        try:
            parent = table_item.parent.resolve(dl_doc) if getattr(table_item, "parent", None) else None
            name = getattr(parent, "name", None)
        except Exception:
            name = None
        if not name:
            return ""
        if name.startswith("sheet: "):
            name = name[len("sheet: "):]
        name = name.strip()
        return f"시트명: {name}\n" if name else ""

    def _table_item_to_texts(self, table_item: TableItem, dl_doc: DoclingDocument,
                             h_short: dict, **kwargs) -> list[str]:
        """표를 청크 텍스트 목록으로 변환. chunk_size(max_tokens) 초과 시 row 단위로 분할하고
        각 분할 청크에 헤더 행(선두 column_header 행 + 다음 컬럼명 행)을 반복 포함한다.

        미초과(또는 max_tokens<=0)면 현행과 동일하게 단일 청크(docling export_to_html) 1개를 반환.
        모든 청크(단일/분할)에 시트명 접두(`시트명: X\\n`)를 붙인다.
        """
        sheet_prefix = self._sheet_prefix(table_item, dl_doc)
        single = sheet_prefix + self._generate_section_text_with_heading([table_item], [h_short], dl_doc, **kwargs)

        if self.max_tokens is None or self.max_tokens <= 0:
            return [single]
        if self._count_tokens(single) <= self.max_tokens:
            return [single]

        try:
            grid = table_item.data.grid
            num_cols = table_item.data.num_cols
        except Exception:
            return [single]
        if not grid or not num_cols:
            return [single]

        # 세로 병합(row_span>1)이 있으면 row 분할이 구조를 깨뜨리므로 분할하지 않는다.
        if any(getattr(c, "row_span", 1) > 1 for r in grid for c in r):
            return [single]

        # 헤더 행 수: 선두의 연속된 헤더 플래그 행 + 바로 다음 행(컬럼명 추정)
        flag_n = 0
        for row in grid:
            if any(getattr(c, "column_header", False) or getattr(c, "row_header", False)
                   or getattr(c, "row_section", False) for c in row):
                flag_n += 1
            else:
                break
        header_n = flag_n + 1
        if header_n >= len(grid):  # 데이터 행이 없음 → 분할 불가
            return [single]

        header_rows = grid[:header_n]
        data_rows = grid[header_n:]

        # heading 접두(_generate_section_text_with_heading 과 동일 규칙). xlsx 는 보통 공백.
        merged = {lvl: t for lvl, t in (h_short or {}).items() if t}
        heading = ", ".join(merged[l] for l in sorted(merged)) if merged else ""
        prefix = (heading + ", ") if heading else ""

        header_inner = "".join(self._render_table_row_html(r, num_cols) for r in header_rows)

        def wrap(inner: str) -> str:
            return sheet_prefix + prefix + "<table><tbody>" + header_inner + inner + "</tbody></table>"

        texts: list[str] = []
        cur = ""
        for r in data_rows:
            tr = self._render_table_row_html(r, num_cols)
            if cur and self._count_tokens(wrap(cur + tr)) > self.max_tokens:
                texts.append(wrap(cur))
                cur = tr
            else:
                cur += tr
        if cur:
            texts.append(wrap(cur))
        return texts or [single]

    def _extract_used_headers(self, header_info_list: list[dict]) -> Optional[list[str]]:
        """헤더 정보 리스트에서 실제 사용되는 모든 헤더들을 level 순서대로 추출하고 ', '로 연결"""
        if not header_info_list:
            return None

        all_headers = [] # header 순서대로 추가
        seen_headers = set()  # 중복 방지용

        for header_info in header_info_list:
            if header_info:
                for level in sorted(header_info.keys()):
                    header_text = header_info[level]
                    if header_text and header_text not in seen_headers:
                        all_headers.append(header_text)
                        seen_headers.add(header_text)

        return all_headers if all_headers else None

    def _split_table_text(self, table_text: str, max_tokens: int) -> list[str]:
        """테이블 텍스트를 토큰 제한에 맞게 분할 (단순 토큰 수 기준)"""
        if not table_text:
            return [table_text]

        # 전체 테이블이 토큰 제한 내인지 확인
        if self._count_tokens(table_text) <= max_tokens:
            return [table_text]

        # 단순히 토큰 수 기준으로 텍스트 분할
        # semchunk 사용하여 토큰 제한에 맞게 분할 (char 모드는 문자 수 카운터 len 사용)
        counter = len if self._tokenizer is None else self._tokenizer
        chunker = semchunk.chunkerify(counter, chunk_size=max_tokens)
        chunks = chunker(table_text)
        return chunks if chunks else [table_text]

    def _is_section_header(self, item: DocItem) -> bool:
        """아이템이 section header인지 확인"""
        return (isinstance(item, SectionHeaderItem) or
                (isinstance(item, TextItem) and
                 item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]))

    def _get_section_header_level(self, item: DocItem) -> Optional[int]:
        """Section header의 level을 반환"""
        if isinstance(item, SectionHeaderItem):
            return item.level
        elif isinstance(item, TextItem):
            if item.label == DocItemLabel.TITLE:
                return 0
            elif item.label == DocItemLabel.SECTION_HEADER:
                return 1
        return None

    def _generate_section_text_with_heading(self, section_items: list[DocItem],
                                            section_header_infos: list[dict],
                                            dl_doc: DoclingDocument,
                                            **kwargs) -> str:
        """섹션의 텍스트를 생성하되, 앞에 heading을 붙임"""
        # 첫 번째 item의 header_info에서 heading 추출
        if section_header_infos and section_header_infos[0]:
            merged_headers = {}
            for level, header_text in section_header_infos[0].items():
                if header_text:
                    merged_headers[level] = header_text

            # level 순서대로 정렬해서 ', '로 연결
            if merged_headers:
                sorted_levels = sorted(merged_headers.keys())
                headers = [merged_headers[level] for level in sorted_levels]
                heading_text = ', '.join(headers)
            else:
                heading_text = ""
        else:
            heading_text = ""

        # 섹션의 일반 텍스트 생성
        section_text = self._generate_text_from_items_with_headers(
            section_items, section_header_infos, dl_doc, **kwargs
        )

        # heading이 있으면 앞에 붙이기
        if heading_text:
            return heading_text + ", " + section_text
        else:
            return section_text

    def _split_document_by_tokens(self, doc_chunk: DocChunk, dl_doc: DoclingDocument, **kwargs) -> list[DocChunk]:
        """문서를 토큰 제한에 맞게 분할 (v2: 섹션 헤더 기준으로 분할 후 max_tokens로 병합)"""
        items = doc_chunk.meta.doc_items
        header_info_list = getattr(doc_chunk, '_header_info_list', [])
        header_short_info_list = getattr(doc_chunk, '_header_short_info_list', [])

        if not items:
            return []

        # ================================================================
        # 헬퍼 함수들
        # ================================================================

        def get_header_level(header_infos, *, first=False, default=-1):
            """header_infos에서 최종 레벨 계산"""
            if not header_infos:
                return default
            info = header_infos[0] if first else header_infos[-1]
            return max(info.keys(), default=default)

        def get_current_chunk(doc_chunk: DocChunk, merged_texts: list[str], merged_header_short_infos: list[dict], merged_items: list[DocItem]):
            """현재까지 병합된 내용으로 DocChunk 생성"""
            if not merged_texts:
                return None
            chunk_text = "\n".join(merged_texts)
            used_headers = self._extract_used_headers(merged_header_short_infos)

            return DocChunk(
                    text=chunk_text,
                    meta=DocMeta(
                        doc_items=merged_items,
                        headings=used_headers,
                        captions=None,
                        origin=doc_chunk.meta.origin,
                    )
                )

        def get_text_from_item(item: DocItem) -> str:
            """DocItem에서 텍스트 추출"""
            if isinstance(item, TableItem):
                return self._extract_table_text(item, dl_doc, **kwargs)
            elif hasattr(item, 'text') and item.text:
                return item.text
            elif isinstance(item, PictureItem):
                text = ""
                for annotation in item.annotations:
                    if hasattr(annotation, 'text'):
                        text += annotation.text
                return text
            return ""

        def split_items_evenly_by_tokens(item_token_counts, max_tokens):
            n = len(item_token_counts)
            total = sum(item_token_counts)
            if n == 0:
                return []
            if total <= max_tokens:
                return [(0, n)]   # ✅ 항상 (a,b)

            k = math.ceil(total / max_tokens)
            target = total / k

            P = [0]
            for c in item_token_counts:
                P.append(P[-1] + c)

            cuts = [0]
            used = {0}
            for t in range(1, k):
                goal = t * target
                j = bisect.bisect_left(P, goal)

                cand = []
                if 0 < j < len(P): cand.append(j)
                if 0 <= j-1 < len(P): cand.append(j-1)

                best = None
                best_dist = float("inf")
                for x in cand:
                    if x in used:
                        continue
                    if x <= cuts[-1]:
                        continue
                    if x >= len(P)-1:  # n
                        continue
                    dist = abs(P[x] - goal)
                    if dist < best_dist:
                        best_dist = dist
                        best = x

                if best is None:
                    best = min(max(cuts[-1] + 1, 1), len(P)-2)

                cuts.append(best)
                used.add(best)

            cuts.append(n)

            return [(a, b) for a, b in zip(cuts[:-1], cuts[1:])]

        def adjust_captions(items_group):

            b_modified = False
            for idx, group in enumerate(items_group):
                if group is None:
                    continue
                item = group[0][0]
                ref_idx_list = []
                if hasattr(item, 'captions') and item.captions:
                    for cap in item.captions:
                        cap_ref = cap.cref
                        cap_idx = -1
                        for j, it in enumerate(items_group):
                            if it is None:
                                continue
                            if getattr(it[0][0], 'self_ref', None) == cap_ref:
                                cap_idx = j
                                break
                        if cap_idx != -1:
                            ref_idx_list.append(cap_idx)
                if ref_idx_list:
                    ref_idx_list = sorted(ref_idx_list)

                if not ref_idx_list:
                    continue

                # caption 아이템들을 부모 아이템 바로 뒤로 이동
                for cap_idx in ref_idx_list:
                    for g in items_group[cap_idx]:
                        items_group[idx].append(g)
                    items_group[cap_idx] = None  # 나중에 None 제거
                    b_modified = True

            if b_modified:
                items_group = [it for it in items_group if it is not None]

            return items_group

        def adjust_pictures_in_tables(items_group):
            # picture in table 처리

            b_modified = False
            for idx, group in enumerate(items_group):
                if group is None:
                    continue
                item = group[0][0]
                pic_idx_list = []
                if isinstance(item, TableItem):
                    table_bbox = item.prov[0].bbox
                    table_page_no = item.prov[0].page_no

                    for j in range(len(items_group)):
                        if items_group[j] is None:
                            continue
                        pic_item = items_group[j][0][0]
                        if isinstance(pic_item, PictureItem):
                            # table 안의 picture인지 확인. iou 사용
                            pic_bbox = pic_item.prov[0].bbox
                            pic_page_no = pic_item.prov[0].page_no
                            if pic_page_no != table_page_no:
                                continue
                            ios = pic_bbox.intersection_over_self(table_bbox)
                            if ios > 0.5:  # picture가 50% 이상 table 안에 포함되면 table 안의 picture로 간주
                                pic_idx_list.append(j)
                    if pic_idx_list:
                        pic_idx_list = sorted(pic_idx_list)

                if not pic_idx_list:
                    continue

                for pic_idx in pic_idx_list:
                    for g in items_group[pic_idx]:
                        items_group[idx].append(g)
                    items_group[pic_idx] = None  # 나중에 None 제거
                    b_modified = True

            if b_modified:
                items_group = [it for it in items_group if it is not None]

            return items_group

        # ================================================================
        # 표 단위 청크 분리 (xlsx docling 전용, kwargs: table_as_chunk)
        #   각 TableItem 을 독립 청크로, 사이의 연속 비표 아이템은 별도 청크로 묶는다.
        #   chunk_size(max_tokens) 와 무관하게 표가 병합되지 않도록 토큰 단계 이전에 확정 반환한다.
        # ================================================================
        if kwargs.get("table_as_chunk"):
            table_chunks: list[DocChunk] = []
            buf_items: list[DocItem] = []
            buf_short: list[dict] = []

            def _flush_buf():
                if buf_items:
                    text = self._generate_section_text_with_heading(buf_items, buf_short, dl_doc, **kwargs)
                    # 빈 문서 방어용 "." placeholder 등 무의미한 텍스트 run 은 청크로 만들지 않는다.
                    if text and text.strip() and text.strip() != ".":
                        ch = get_current_chunk(doc_chunk, [text], list(buf_short), list(buf_items))
                        if ch:
                            table_chunks.append(ch)
                    buf_items.clear()
                    buf_short.clear()

            for i, item in enumerate(items):
                h_short = header_short_info_list[i] if i < len(header_short_info_list) else {}
                if isinstance(item, TableItem):
                    _flush_buf()
                    # 행이 많아 chunk_size 를 초과하는 표는 row 단위로 분할(각 청크에 헤더 반복 포함).
                    for text in self._table_item_to_texts(item, dl_doc, h_short, **kwargs):
                        ch = get_current_chunk(doc_chunk, [text], [h_short], [item])
                        if ch:
                            table_chunks.append(ch)
                else:
                    buf_items.append(item)
                    buf_short.append(h_short)
            _flush_buf()

            if table_chunks:
                return table_chunks

        # ================================================================
        # 1단계: 섹션 헤더 기준으로 분할
        # ================================================================

        sections = []  # [(items, header_infos, header_short_infos), ...]
        cur_items, cur_h_infos, cur_h_short = [], [], []

        for i, item in enumerate(items):
            h_info = header_info_list[i] if i < len(header_info_list) else {}
            h_short = header_short_info_list[i] if i < len(header_short_info_list) else {}

            # 섹션 헤더를 만나면
            if self._is_section_header(item):
                # 이전 섹션이 있으면 저장
                if cur_items:
                    sections.append((cur_items, cur_h_infos, cur_h_short))

                # 새로운 섹션 시작
                cur_items = [item]
                cur_h_infos = [h_info]
                cur_h_short = [h_short]
            else:
                # 섹션 헤더가 아니면 현재 섹션에 추가
                cur_items.append(item)
                cur_h_infos.append(h_info)
                cur_h_short.append(h_short)

        # 마지막 섹션 저장
        if cur_items:
            sections.append((cur_items, cur_h_infos, cur_h_short))

        # ================================================================
        # 2단계: 각 섹션의 텍스트에 heading 붙이기
        # ================================================================

        sections_with_text = []
        for items, header_infos, header_short_infos in sections:
            text = self._generate_section_text_with_heading(
                items, header_short_infos, dl_doc, **kwargs
            )
            sections_with_text.append((
                text,
                items,
                header_infos,
                header_short_infos
            ))

        # ================================================================
        # 2.5단계: 너무 긴 청크는 분할
        # ================================================================
        if self.max_tokens > 0:
            for i in range(len(sections_with_text)):
                text, items, h_infos, h_short = sections_with_text[i]
                token_count = self._count_tokens(text)
                if token_count < self.max_tokens:
                    continue

                # caption 및 table 내 그림은 같은 섹션에 있도록 조정
                items_group=[[(item, info, short)] for item, info, short in zip(items, h_infos, h_short)]
                items_group = adjust_captions(items_group)
                items_group = adjust_pictures_in_tables(items_group)

                # 너무 긴 섹션은 분할
                # 각 아이템 별 token 수 계산
                item_token_counts = []
                for group in items_group:
                    cur_count = 0
                    for g in group:
                        cur_count += self._count_tokens(get_text_from_item(g[0]))
                    item_token_counts.append(cur_count)

                # 아이템 그룹들을 토큰 기준으로 균등 분할
                split_info = split_items_evenly_by_tokens(item_token_counts, self.max_tokens)

                # item_groups를 섹션으로 다시 구성
                new_sections = []
                for (a, b) in split_info:

                    # 각 그룹에서 items, h_infos, h_short로 분리
                    group_items = []
                    group_h_infos = []
                    group_h_short = []
                    for idx in range(a, b):
                        for g in items_group[idx]:
                            group_items.append(g[0])
                            group_h_infos.append(g[1])
                            group_h_short.append(g[2])

                    new_text = self._generate_section_text_with_heading(
                        group_items, group_h_short, dl_doc, **kwargs
                    )
                    new_sections.append((new_text, group_items, group_h_infos, group_h_short))

                # 원래 섹션을 새로 분할된 섹션들로 교체
                sections_with_text.pop(i)
                for new_section in reversed(new_sections):
                    sections_with_text.insert(i, new_section)

        # ================================================================
        # 3단계: 단독 타이틀(1줄만) → 다음 섹션으로 병합
        # ================================================================

        for i in range(len(sections_with_text) - 2, -1, -1):
            text, items, h_infos, h_short = sections_with_text[i]

            # 아이템이 하나인 섹션 헤더만 검사
            if len(items) != 1 or not self._is_section_header(items[0]):
                continue

            # 문단이 이미 구성된 것은 제외 (문자 수가 30자 이상이면 문단을 구성했다고 간주)
            item_text = "".join(getattr(it, "text", "") for it in items)
            if len(item_text) > 30:
                continue

            # 현재 섹션헤더 레벨이 다음 섹션헤더 레벨보다 더 높은 경우에만 병합 (높은 레벨이 더 작은 숫자)
            n_text, n_items, n_h_infos, n_h_short = sections_with_text[i + 1]
            current_level = get_header_level(h_infos, first=False)
            next_level = get_header_level(n_h_infos, first=True)
            if 0 <= next_level < current_level:
                continue

            # 다음 섹션과 병합
            sections_with_text[i] = (text + '\n' + n_text, items + n_items, h_infos + n_h_infos, h_short + n_h_short)
            sections_with_text.pop(i + 1)

        # ================================================================
        # 4단계: 토큰 기준 병합 (1차 — 섹션 구조 경계 기준 그룹 생성)
        # ================================================================

        groups: list[dict] = []
        merged_texts, merged_items = [], []
        merged_header_infos, merged_header_short_infos = [], []

        def flush_group():
            if merged_texts:
                groups.append({
                    "texts": list(merged_texts),
                    "items": list(merged_items),
                    "h_infos": list(merged_header_infos),
                    "h_short": list(merged_header_short_infos),
                })

        for text, items, header_infos, header_short_infos in sections_with_text:

            b_new_chunk = False

            #----------------------------------
            # 병합 가능 여부 판단

            # 병합 가능 토큰 수 계산
            test_tokens = self._count_tokens("\n".join(merged_texts + [text]))

            # 현재 섹션헤더 레벨과 병합된 섹션헤더 레벨
            section_level = get_header_level(header_infos, first=True)
            merged_level = get_header_level(merged_header_infos, first=False)

            # 토큰 수 초과 시 새로운 청크 생성
            if test_tokens > self.max_tokens and len(merged_texts) > 0:
                b_new_chunk = True
            # 현재 섹션헤더 레벨이 더 높으면 새로운 청크 생성
            elif 0 <= section_level < merged_level:
                b_new_chunk = True
            #----------------------------------

            # 새로운 청크 생성
            if b_new_chunk:
                flush_group()

                # 새로운 병합 시작
                merged_texts = [text]
                merged_items = list(items)
                merged_header_infos = list(header_infos)
                merged_header_short_infos = list(header_short_infos)
            else:
                # 현재 섹션 병합
                merged_texts.append(text)
                merged_items.extend(items)
                merged_header_infos.extend(header_infos)
                merged_header_short_infos.extend(header_short_infos)

        # 마지막 병합된 items 처리
        flush_group()

        # ================================================================
        # 5단계: chunk_size 한도 내 인접 그룹 greedy 병합
        #   1차 결과(구조 경계 기준 그룹)를 순서대로, 합산 크기가 chunk_size 이하인 동안
        #   인접 그룹끼리 결합한다. (크기는 HEADER 라인 포함 최종 텍스트 기준)
        # ================================================================
        if self.max_tokens > 0 and groups:
            def _size(g):
                text = "\n".join(g["texts"])
                headings = self._extract_used_headers(g["h_short"]) or []
                header_line = ("HEADER: " + ", ".join(headings) + "\n") if headings else ""
                # char 모드면 문자 수, huggingface 모드면 토큰 수로 산정 (max_tokens 단위와 일치)
                return self._count_tokens(header_line + text)

            def _merge(a, b):
                return {
                    "texts": a["texts"] + b["texts"],
                    "items": a["items"] + b["items"],
                    "h_infos": a["h_infos"] + b["h_infos"],
                    "h_short": a["h_short"] + b["h_short"],
                }

            merged_groups = [groups[0]]
            for g in groups[1:]:
                cand = _merge(merged_groups[-1], g)
                if _size(cand) <= self.max_tokens:
                    merged_groups[-1] = cand
                else:
                    merged_groups.append(g)
            groups = merged_groups

        # ================================================================
        # 6단계: 최종 DocChunk 생성
        # ================================================================
        result_chunks = []
        for g in groups:
            cur_chunk = get_current_chunk(doc_chunk, g["texts"], g["h_short"], g["items"])
            if cur_chunk:
                result_chunks.append(cur_chunk)

        return result_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """문서를 청킹하여 반환

        Args:
            dl_doc: 청킹할 문서

        Yields:
            토큰 제한에 맞게 분할된 청크들
        """
        doc_chunks = list(self.preprocess(dl_doc=dl_doc, **kwargs))

        if not doc_chunks:
            return iter([])

        doc_chunk = doc_chunks[0]  # preprocess는 하나의 청크만 반환

        final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc, **kwargs)

        return iter(final_chunks)

class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    e_page: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    created_date: Optional[int] = None  # YYYYMMDD 형식의 정수
    authors: Optional[str] = None      # 팀 리스트
    title: Optional[str] = None         # 문서 제목

class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        self.n_word: Optional[int] = None
        self.n_line: Optional[int] = None
        self.i_page: Optional[int] = None
        self.e_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_page: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_page: Optional[int] = None
        self.reg_date: Optional[str] = None
        self.chunk_bboxes: Optional[str] = None
        self.media_files: Optional[str] = None
        self.created_date: Optional[int] = None
        self.authors: Optional[str] = None      # 팀 리스트
        self.title: Optional[str] = None
        self.extra_metadata: dict[str, Any] = {}

    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련된 데이터를 설정"""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self

    def set_page_info(
            self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int
    ) -> "GenOSVectorMetaBuilder":
        """페이지 정보 설정"""
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_page = n_chunk_of_page
        return self

    def set_chunk_index(self, i_chunk_on_doc: int) -> "GenOSVectorMetaBuilder":
        """문서 전체의 청크 인덱스 설정"""
        self.i_chunk_on_doc = i_chunk_on_doc
        return self

    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        """글로벌 메타데이터 병합"""
        for key, value in global_metadata.items():
            if key in self.__dict__ and key != "extra_metadata":
                setattr(self, key, value)
            else:
                self.extra_metadata[key] = value
        return self

    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                label = item.self_ref
                type_ = item.label
                size = document.pages.get(prov.page_no).size
                page_no = prov.page_no
                bbox = prov.bbox
                bbox_data = {'l': bbox.l / size.width,
                             't': bbox.t / size.height,
                             'r': bbox.r / size.width,
                             'b': bbox.b / size.height,
                             'coord_origin': bbox.coord_origin.value}
                chunk_bboxes.append({'page': page_no, 'bbox': bbox_data, 'type': type_, 'ref': label})
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else 0
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self

    def set_media_files(self, doc_items: list, include_tables: bool = False) -> "GenOSVectorMetaBuilder":
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
            elif include_tables and isinstance(item, TableItem) and item.image:
                # 표 이미지는 picture 와 구분되도록 type='table_image' 로 기록한다.
                # ref(self_ref)는 chunk_bboxes 의 table 엔트리 ref 와 동일 → 조인 가능.
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'table_image', 'ref': item.self_ref})
        self.media_files = json.dumps(temp_list)
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        payload = {
            "text": self.text,
            "n_char": self.n_char,
            "n_word": self.n_word,
            "n_line": self.n_line,
            "i_page": self.i_page,
            "e_page": self.e_page,
            "i_chunk_on_page": self.i_chunk_on_page,
            "n_chunk_of_page": self.n_chunk_of_page,
            "i_chunk_on_doc": self.i_chunk_on_doc,
            "n_chunk_of_doc": self.n_chunk_of_doc,
            "n_page": self.n_page,
            "reg_date": self.reg_date,
            "chunk_bboxes": self.chunk_bboxes,
            "media_files": self.media_files,
            "created_date": self.created_date,
            "authors": self.authors,      # 팀 리스트
            "title": self.title,
            **self.extra_metadata,
        }
        return GenOSVectorMeta.model_validate(payload)


class DocumentProcessor:

    def __init__(self, config_path: str | None = None):
        '''
        initialize Document Converter (config 기반)

        config_path 가 None 이면 resource_dev/convert_processor_config.yaml
        (없으면 resource/convert_processor_config.yaml) 을 사용한다.
        GenOS 는 DocumentProcessor() 무인자로 호출하므로 기본 경로 resolve 필수.
        '''
        if config_path is None:
            config_path = _resolve_default_convert_config_path()

        cfg = _load_config(config_path)
        self._config_dir = Path(config_path).resolve().parent

        defaults_cfg = _as_dict(cfg.get("defaults"))
        log_level = _parse_optional_int(defaults_cfg.get("log_level"), "defaults.log_level")
        if log_level is None:
            log_level = 4
        self._log_level = log_level

        ocr_cfg = _as_dict(cfg.get("ocr"))
        layout_cfg = _as_dict(cfg.get("layout"))
        pdf_cfg = _as_dict(cfg.get("pdf_pipeline"))
        models_cfg = _as_dict(cfg.get("models"))
        chunking_cfg = _as_dict(cfg.get("chunking"))
        ec = EnrichmentConfig.from_raw(cfg.get("enrichment"), self._config_dir, parent_cfg=cfg)

        # 청킹용 토크나이저 (chunking config 기반; 미지정 시 현행 기본값)
        self._tokenizer = _resolve_tokenizer(chunking_cfg)

        # 토큰 수 계산 방식 (chunking 섹션). "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
        self._tokenizer_type = str(chunking_cfg.get("tokenizer_type", "char")).strip().lower()
        if self._tokenizer_type not in {"char", "huggingface"}:
            _log.warning(
                f"[DocumentProcessor] Unknown chunking.tokenizer_type '{self._tokenizer_type}', fallback to 'char'."
            )
            self._tokenizer_type = "char"

        # 청크 최대 크기(GenosSmartChunker.max_tokens) 기본값. kwargs 의 chunk_size 가 우선.
        self._chunk_size = _parse_optional_int(chunking_cfg.get("chunk_size"), "chunking.chunk_size")

        # xlsx(엑셀) 처리 설정(이슈 #288).
        #   docling(기본): xlsx 를 docling MsExcel 백엔드로 처리(현행) → 기존 청킹/벡터 파이프라인.
        #   tabular: 데이터 행마다 1청크(벡터) + 컬럼 헤더→메타(병합셀 unmerge+forward-fill).
        xlsx_cfg = _as_dict(cfg.get("xlsx"))
        xlsx_mode = str(xlsx_cfg.get("processing_mode", "docling")).strip().lower()
        if xlsx_mode not in {"docling", "tabular"}:
            _log.warning(
                f"[DocumentProcessor] Unknown xlsx.processing_mode '{xlsx_mode}', fallback to 'docling'."
            )
            xlsx_mode = "docling"
        self._xlsx_cfg = {
            "processing_mode": xlsx_mode,
            "header_row": _parse_optional_int(xlsx_cfg.get("header_row"), "xlsx.header_row") or 0,
            "encoding": (str(xlsx_cfg.get("encoding")).strip() or None),
            "intercept_csv": bool(_parse_optional_bool(xlsx_cfg.get("intercept_csv"), "xlsx.intercept_csv")),
        }

        # OCR 엔드포인트는 ocr.paddle.ocr_endpoint 가 정식 위치.
        # 구버전 호환: ocr.ocr_endpoint(상위) / 최상위 ocr_endpoint 도 폴백으로 인식.
        paddle_cfg = _as_dict(ocr_cfg.get("paddle"))
        ocr_ep = (
            paddle_cfg.get("ocr_endpoint")
            or ocr_cfg.get("ocr_endpoint")
            or cfg.get("ocr_endpoint", "http://192.168.73.172:48080/ocr")
        )

        # OCR 수행 모드. "auto"(default)=휴리스틱 기반 재OCR / "force"=무조건 전체 OCR / "disable"=OCR 안 함
        # (PDF 입력에만 적용. DOCX/기타 포맷은 ocr_mode 무관)
        raw_ocr_mode = str(ocr_cfg.get("ocr_mode", cfg.get("ocr_mode", "auto"))).lower().strip()
        if raw_ocr_mode not in {"auto", "force", "disable"}:
            _log.warning(f"[DocumentProcessor] Unknown ocr_mode '{raw_ocr_mode}', fallback to 'auto'")
            raw_ocr_mode = "auto"
        self.ocr_mode = raw_ocr_mode

        # 테이블 셀 재OCR HTTP timeout (ocr_all_table_cells). 잘못된 값은 60 으로 폴백.
        table_cell_ocr_timeout = _parse_optional_int(
            ocr_cfg.get("table_cell_ocr_timeout"), "ocr.table_cell_ocr_timeout"
        )
        self._table_cell_ocr_timeout = (
            table_cell_ocr_timeout if table_cell_ocr_timeout and table_cell_ocr_timeout > 0 else 60
        )

        # 글리프 기반 auto-OCR 재트리거 임계값.
        glyph_cfg = _as_dict(ocr_cfg.get("glyph_detection"))
        glyph_cell_th = _parse_optional_int(
            glyph_cfg.get("table_cell_threshold"), "ocr.glyph_detection.table_cell_threshold"
        )
        self._glyph_table_cell_threshold = glyph_cell_th if glyph_cell_th and glyph_cell_th > 0 else 1
        glyph_doc_th = _parse_optional_int(
            glyph_cfg.get("document_threshold"), "ocr.glyph_detection.document_threshold"
        )
        self._glyph_document_threshold = glyph_doc_th if glyph_doc_th and glyph_doc_th > 0 else 10

        ocr_options = self._build_ocr_options(ocr_cfg, paddle_endpoint=ocr_ep)
        if isinstance(ocr_options, UpstageOcrOptions):
            self.ocr_endpoint = ocr_options.api_endpoint
        else:
            self.ocr_endpoint = ocr_ep

        self.page_chunk_counts = defaultdict(int)

        device_str = str(pdf_cfg.get("device", "auto")).lower().strip()
        device = _ACCELERATOR_DEVICE_MAP.get(device_str)
        if device is None:
            _log.warning(f"[DocumentProcessor] Unknown pdf_pipeline.device '{device_str}', fallback to 'auto'")
            device = AcceleratorDevice.AUTO

        num_threads = _parse_optional_int(pdf_cfg.get("num_threads"), "pdf_pipeline.num_threads")
        if num_threads is None or num_threads <= 0:
            num_threads = 8
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)

        images_scale = _parse_optional_int(pdf_cfg.get("images_scale"), "pdf_pipeline.images_scale")
        if images_scale is None or images_scale <= 0:
            images_scale = 2

        generate_page_images = _parse_optional_bool(
            pdf_cfg.get("generate_page_images"), "pdf_pipeline.generate_page_images"
        )
        generate_picture_images = _parse_optional_bool(
            pdf_cfg.get("generate_picture_images"), "pdf_pipeline.generate_picture_images"
        )

        # 표 이미지(table_image) 옵션: 표를 picture 와 동일하게 이미지로 잘라 저장하고,
        # media_files 에 type='table_image' 로 기록한다(검색=청크 텍스트 / 답변=표 이미지).
        # 기본 False 라 미설정 시 기존 동작과 동일(하위 호환).
        table_image_cfg = _as_dict(cfg.get("table_image"))
        self.table_image_enabled = bool(
            _parse_optional_bool(table_image_cfg.get("enable"), "table_image.enable")
        )

        table_mode_str = str(pdf_cfg.get("table_structure_mode", "accurate")).lower().strip()
        table_structure_mode = _TABLE_FORMER_MODE_MAP.get(table_mode_str)
        if table_structure_mode is None:
            _log.warning(
                f"[DocumentProcessor] Unknown pdf_pipeline.table_structure_mode '{table_mode_str}', fallback to 'accurate'"
            )
            table_structure_mode = TableFormerMode.ACCURATE

        # PDF 파이프라인 옵션 설정
        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = (
            True if generate_page_images is None else generate_page_images
        )
        self.pipe_line_options.generate_picture_images = (
            True if generate_picture_images is None else generate_picture_images
        )
        # 표 이미지 크롭(TableItem.get_image)은 페이지 이미지를 소스로 하므로,
        # table_image 가 켜지면 generate_page_images 를 True 로 강제 보장한다.
        if self.table_image_enabled:
            self.pipe_line_options.generate_page_images = True
        self.pipe_line_options.do_ocr = False
        self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.images_scale = images_scale

        # layout 모델 선택. "genos_layout"(default) / "docling_layout". 잘못된 값은 경고 후 폴백.
        layout_model_type_str = str(
            layout_cfg.get("layout_model_type", cfg.get("layout_model_type", "genos_layout"))
        ).lower().strip()
        if layout_model_type_str == LayoutModelType.DOCLING_LAYOUT.value:
            layout_model_type = LayoutModelType.DOCLING_LAYOUT
        else:
            if layout_model_type_str != LayoutModelType.GENOS_LAYOUT.value:
                _log.warning(
                    f"[DocumentProcessor] Unknown layout_model_type '{layout_model_type_str}', "
                    f"fallback to '{LayoutModelType.GENOS_LAYOUT.value}'"
                )
            layout_model_type = LayoutModelType.GENOS_LAYOUT
        self.pipe_line_options.layout_options.layout_model_type = layout_model_type
        self.pipe_line_options.layout_options.genos_layout_options.endpoint = _as_dict(
            layout_cfg.get("genos_layout")
        ).get("endpoint", "http://192.168.75.174:26001/v1/chat/completions")
        self.pipe_line_options.layout_options.genos_layout_options.api_key = _as_dict(
            layout_cfg.get("genos_layout")
        ).get("api_key", "")

        # genos layout 모델은 batch size를 32로 설정
        page_batch_size = _parse_optional_int(
            _as_dict(layout_cfg.get("genos_layout")).get("page_batch_size"), "layout.genos_layout.page_batch_size"
        )
        if page_batch_size is None or page_batch_size <= 0:
            page_batch_size = 128
        settings.perf.page_batch_size = page_batch_size

        max_completion_tokens = _parse_optional_int(
            _as_dict(layout_cfg.get("genos_layout")).get("max_completion_tokens"),
            "layout.genos_layout.max_completion_tokens",
        )
        if max_completion_tokens is None or max_completion_tokens <= 0:
            max_completion_tokens = 16384
        self.pipe_line_options.layout_options.genos_layout_options.max_completion_tokens = max_completion_tokens

        # DotsOCR VLM 호출/생성 파라미터 (yaml 누락·무효 시 기본값 폴백)
        genos_layout_cfg = _as_dict(layout_cfg.get("genos_layout"))
        layout_model = genos_layout_cfg.get("model") or "dots-mocr"
        layout_timeout = _parse_optional_int(
            genos_layout_cfg.get("timeout"), "layout.genos_layout.timeout"
        )
        if layout_timeout is None or layout_timeout <= 0:
            layout_timeout = 3600
        layout_retry_count = _parse_optional_int(
            genos_layout_cfg.get("retry_count"), "layout.genos_layout.retry_count"
        )
        if layout_retry_count is None or layout_retry_count < 0:
            layout_retry_count = 2
        layout_temperature = _parse_optional_float(
            genos_layout_cfg.get("temperature"), "layout.genos_layout.temperature"
        )
        if layout_temperature is None or layout_temperature < 0:
            layout_temperature = 0.1
        layout_top_p = _parse_optional_float(
            genos_layout_cfg.get("top_p"), "layout.genos_layout.top_p"
        )
        if layout_top_p is None or not (0 < layout_top_p <= 1):
            layout_top_p = 0.9
        layout_repetition_penalty = _parse_optional_float(
            genos_layout_cfg.get("repetition_penalty"),
            "layout.genos_layout.repetition_penalty",
        )
        if layout_repetition_penalty is None or layout_repetition_penalty <= 0:
            layout_repetition_penalty = 1.15
        self.pipe_line_options.layout_options.genos_layout_options.model = layout_model
        self.pipe_line_options.layout_options.genos_layout_options.timeout = layout_timeout
        self.pipe_line_options.layout_options.genos_layout_options.retry_count = layout_retry_count
        self.pipe_line_options.layout_options.genos_layout_options.temperature = layout_temperature
        self.pipe_line_options.layout_options.genos_layout_options.top_p = layout_top_p
        self.pipe_line_options.layout_options.genos_layout_options.repetition_penalty = layout_repetition_penalty

        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = table_structure_mode
        self.pipe_line_options.accelerator_options = accelerator_options

        # docling 모델(TableFormer 등) 로컬 경로. config 에 값이 있을 때만 설정하고,
        # 비어있으면 설정하지 않아 docling 기본 캐시 동작을 그대로 유지(backward compat).
        # (아래 ocr_pipe_line_options 는 pipe_line_options 의 deep copy 라 자동 전파됨)
        artifacts_path = models_cfg.get("artifacts_path")
        if artifacts_path:
            self.pipe_line_options.artifacts_path = Path(artifacts_path)

        # Simple 파이프라인 옵션을 인스턴스 변수로 저장
        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False

        # ocr 파이프라인 옵션
        self.ocr_pipe_line_options = PdfPipelineOptions()
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True

        # 기본 컨버터들 생성
        self._create_converters()

        self.image_description_options = ImageDescriptionOptions.from_config(
            image_desc_cfg=ec.image_description_cfg,
            fallback_api_url=ec.api_url,
            fallback_api_key=ec.api_key,
            fallback_model=ec.model,
            config_dir=self._config_dir,
        )
        self.image_description_enricher = FacadeImageDescriptionEnricher(
            self.image_description_options
        )
        self.custom_fields_enrichers: list = (
            [CustomFieldsEnricher(**c) for c in ec.custom_fields_cfgs]
            if CustomFieldsEnricher is not None
            else []
        )
        self.metadata_enricher = (
            MetadataEnricher(
                url=ec.metadata.url,
                api_key=ec.metadata.api_key,
                model=ec.metadata.model,
                system_prompt=ec.metadata.system_prompt,
                user_prompt=ec.metadata.user_prompt,
                output_fields=ec.metadata.output_fields,
                parser=ec.metadata.parser,
                pages=ec.metadata.pages,
                max_tokens=ec.metadata.max_tokens,
                temperature=ec.metadata.temperature,
                timeout=ec.metadata.timeout,
                config_dir=self._config_dir,
                variables=ec.metadata.variables,
                template_mode=ec.metadata.template_mode,
                thinking=ec.metadata.thinking,
                thinking_dialect=ec.metadata.thinking_dialect,
            )
            if MetadataEnricher is not None and ec.metadata.do_metadata and ec.metadata.has_custom_metadata
            else None
        )
        # 추출 메타데이터 → typed 벡터 필드 매핑(설정 기반). 설정이 비어있으면
        # 기존 created_date 동작을 그대로 재현한다(하위 호환).
        self._metadata_field_transforms = (
            ec.metadata.field_transforms or DEFAULT_METADATA_FIELD_TRANSFORMS
        )

        # enrichment 옵션 설정 (yaml 의 enrichment 섹션을 EnrichmentConfig 로 파싱)
        self.enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=ec.toc.do_toc,
            toc_doc_type=ec.toc.doc_type,
            extract_metadata=ec.metadata.do_metadata and self.metadata_enricher is None,
            toc_api_provider="custom",
            metadata_api_provider="custom",
            toc_api_base_url=ec.toc.url,
            metadata_api_base_url=ec.metadata.url,
            toc_api_key=ec.toc.api_key,
            metadata_api_key=ec.metadata.api_key,
            toc_model=ec.toc.model,
            metadata_model=ec.metadata.model,
            toc_temperature=ec.toc.temperature,
            toc_top_p=ec.toc.top_p,
            toc_seed=ec.toc.seed,
            toc_max_tokens=ec.toc.max_tokens,
            toc_repetition_penalty=ec.toc.repetition_penalty,
            toc_precheck_enabled=ec.toc.precheck_enabled,
            toc_max_context_tokens=ec.toc.precheck_max_context_tokens,
            toc_completion_reserved_tokens=ec.toc.precheck_completion_reserved_tokens,
            toc_split_enabled=ec.toc.split_enabled,
            toc_pages_per_chunk=ec.toc.split_pages_per_chunk,
            toc_page_overlap=ec.toc.split_page_overlap,
            toc_carryover_max_tokens=ec.toc.split_carryover_max_tokens,
            metadata_precheck_enabled=ec.metadata.precheck_enabled,
            metadata_max_context_tokens=ec.metadata.precheck_max_context_tokens,
            metadata_completion_reserved_tokens=ec.metadata.precheck_completion_reserved_tokens,
            toc_system_prompt=ec.toc.system_prompt,
            toc_user_prompt=ec.toc.user_prompt,
            toc_thinking=ec.toc.thinking,
            toc_thinking_dialect=ec.toc.thinking_dialect,
            metadata_thinking=ec.metadata.thinking,
            metadata_thinking_dialect=ec.metadata.thinking_dialect,
        )

    @staticmethod
    def _build_ocr_options(ocr_cfg: dict, paddle_endpoint: str):
        """Build OcrOptions based on ocr.engine key in yaml.

        Returns PaddleOcrOptions or UpstageOcrOptions. Default engine is "paddle".
        For "upstage", api_key falls back to UPSTAGE_API_KEY env var when empty.
        Unknown engine values fall back to "paddle" with a warning.
        """
        ocr_cfg = ocr_cfg if isinstance(ocr_cfg, dict) else {}
        ocr_engine = str(ocr_cfg.get("engine", "paddle")).lower().strip()
        if ocr_engine not in {"paddle", "upstage"}:
            _log.warning(f"[DocumentProcessor] Unknown ocr.engine '{ocr_engine}', fallback to 'paddle'")
            ocr_engine = "paddle"

        if ocr_engine == "upstage":
            upstage_cfg = _as_dict(ocr_cfg.get("upstage"))
            upstage_api_key = upstage_cfg.get("api_key", "") or os.getenv("UPSTAGE_API_KEY", "")

            raw_timeout = upstage_cfg.get("timeout", 60)
            try:
                upstage_timeout = int(raw_timeout)
                if upstage_timeout <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                _log.warning(f"[DocumentProcessor] Invalid ocr.upstage.timeout '{raw_timeout}', fallback to 60")
                upstage_timeout = 60

            raw_text_score = upstage_cfg.get("text_score", 0.5)
            try:
                upstage_text_score = float(raw_text_score)
            except (TypeError, ValueError):
                _log.warning(f"[DocumentProcessor] Invalid ocr.upstage.text_score '{raw_text_score}', fallback to 0.5")
                upstage_text_score = 0.5

            return UpstageOcrOptions(
                force_full_page_ocr=False,
                lang=upstage_cfg.get("lang", ["ko", "en"]),
                api_endpoint=upstage_cfg.get(
                    "api_endpoint",
                    "https://api.upstage.ai/v1/document-digitization",
                ),
                api_key=upstage_api_key,
                model=upstage_cfg.get("model", "ocr"),
                timeout=upstage_timeout,
                text_score=upstage_text_score,
            )

        paddle_cfg = _as_dict(ocr_cfg.get("paddle"))

        raw_lang = paddle_cfg.get("lang", ["korean"])
        if isinstance(raw_lang, list) and raw_lang:
            paddle_lang = raw_lang
        else:
            if raw_lang not in (None, [], ["korean"]):
                _log.warning(f"[DocumentProcessor] Invalid ocr.paddle.lang '{raw_lang}', fallback to ['korean']")
            paddle_lang = ["korean"]

        raw_text_score = paddle_cfg.get("text_score", 0.3)
        try:
            paddle_text_score = float(raw_text_score)
        except (TypeError, ValueError):
            _log.warning(f"[DocumentProcessor] Invalid ocr.paddle.text_score '{raw_text_score}', fallback to 0.3")
            paddle_text_score = 0.3

        return PaddleOcrOptions(
            force_full_page_ocr=False,
            lang=paddle_lang,
            ocr_endpoint=paddle_endpoint,
            text_score=paddle_text_score,
        )

    def _create_converters(self):
        """컨버터들을 생성하는 헬퍼 메서드"""
        self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipe_line_options,
                        backend=PyPdfiumDocumentBackend
                    ),
                }
            )
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )
        self.ocr_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.ocr_pipe_line_options,
                        backend=DoclingParseV4DocumentBackend
                    ),
                }
            )
        self.ocr_second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            },
        )

    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        try:
            conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def load_documents_with_docling_ocr(self, file_path: str, **kwargs: dict) -> DoclingDocument:

        try:
            conv_result: ConversionResult = self.ocr_converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.ocr_second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def _load_hwp_with_legacy_backend(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        """HWP/HWPX 레거시 백엔드(SDK 미사용) 전용 변환 — GenosHwp SDK 폴백용.

        HWP/HWPX 는 docling 기본 백엔드(GenosHwpDocumentBackend=convtext SDK)로 처리되는데,
        SDK 가 UNSUPPORTED_TYPE(exit code 3) 등으로 실패할 때 이 메서드로 폴백한다.
        .hwp → HwpDocumentBackend, .hwpx → HwpxDocumentBackend (둘 다 olefile/xml 기반
        순수 파이썬이라 SDK 의존이 없음). attachment_processor.HwpProcessor 의 폴백과 동일 구성.
        """
        pipeline_options = PipelineOptions()
        pipeline_options.save_images = kwargs.get('save_images', True)
        converter = DocumentConverter(
            format_options={
                InputFormat.HWP: HwpxFormatOption(
                    pipeline_options=pipeline_options,
                    backend=HwpDocumentBackend,
                ),
                InputFormat.XML_HWPX: HwpxFormatOption(
                    pipeline_options=pipeline_options,
                    backend=HwpxDocumentBackend,
                ),
            }
        )
        conv_result: ConversionResult = converter.convert(
            Path(file_path).resolve(), raises_on_error=True
        )
        return conv_result.document

    @staticmethod
    def _hwp_sdk_text_is_empty(document: DoclingDocument) -> bool:
        """GenosHwp SDK 결과 문서에 본문 텍스트가 전혀 없는지 판단(레거시 폴백 트리거용).

        SDK 가 exit 0 으로 "성공"해도 본문을 한 글자도 못 뽑는 경우가 있다(일부 .hwp/.hwpx).
        이때 doc_items 가 비어 다운스트림 GenosSmartChunker 의 DocMeta(min_length=1) 검증이
        깨진다(too_short). 텍스트 run 이 하나도 없으면 True.
        """
        texts = getattr(document, "texts", None) or []
        return not any((getattr(t, "text", "") or "").strip() for t in texts)

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        ext = os.path.splitext(file_path)[-1].lower()
        is_hwp = ext in ('.hwp', '.hwpx')
        backend_name = "HwpDocumentBackend" if ext == '.hwp' else "HwpxDocumentBackend"
        try:
            document = self.load_documents_with_docling(file_path, **kwargs)
        except Exception as sdk_err:
            # (1) GenosHwp SDK 가 예외로 실패(예: exit code 3) → HWP/HWPX 만 레거시 폴백.
            if not is_hwp:
                raise
            _log.warning(f"[DocumentProcessor] GenosHwp SDK 변환 실패: {sdk_err}")
            try:
                _log.info(f"[DocumentProcessor] {backend_name}로 폴백 시도: {file_path}")
                document = self._load_hwp_with_legacy_backend(file_path, **kwargs)
                _log.info(f"[DocumentProcessor] {backend_name} 폴백 성공")
                return document
            except Exception as fallback_err:
                _log.warning(f"[DocumentProcessor] {backend_name} 폴백도 실패: {fallback_err}")
                raise sdk_err

        # (2) SDK 가 예외 없이(exit 0) 끝났지만 본문 텍스트가 비어 있으면(빈 doc_items 로
        #     다운스트림 DocMeta 검증이 깨지는 케이스) 레거시 백엔드로 폴백 시도한다.
        #     폴백 결과도 비었거나 폴백이 실패하면 원 SDK 결과를 그대로 유지(무회귀).
        if is_hwp and self._hwp_sdk_text_is_empty(document):
            _log.warning(
                f"[DocumentProcessor] GenosHwp SDK 결과에 본문 텍스트가 없어 {backend_name} 폴백 시도: {file_path}"
            )
            try:
                fallback_doc = self._load_hwp_with_legacy_backend(file_path, **kwargs)
                if not self._hwp_sdk_text_is_empty(fallback_doc):
                    _log.info(f"[DocumentProcessor] {backend_name} 폴백 성공(본문 텍스트 확보)")
                    return fallback_doc
                _log.info(f"[DocumentProcessor] {backend_name} 폴백 결과도 비어 상위 PDF 폴백으로 위임")
            except Exception as fallback_err:
                _log.warning(f"[DocumentProcessor] {backend_name} 폴백 실패, 상위 PDF 폴백으로 위임: {fallback_err}")
            # SDK·레거시 모두 본문을 못 얻음 → 예외로 올려 __call__ 의 PDF 변환 폴백에 위임한다.
            raise HwpConversionError(
                f"HWP/HWPX SDK 결과가 비어 있고 레거시 백엔드로도 본문 복구 실패: {file_path}"
            )
        return document

    def get_loader_langchain(self, file_path: str, use_pdf_sdk: bool = True):
        """PPT 파일용 langchain 로더"""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.ppt':
            convert_to_pdf(file_path, use_pdf_sdk=use_pdf_sdk)
            return UnstructuredPowerPointLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

    def load_documents_langchain(self, file_path: str, **kwargs: dict):
        """langchain으로 문서 로드"""
        loader = self.get_loader_langchain(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
        documents = loader.load()
        return documents

    def split_documents_langchain(self, documents, **kwargs: dict):
        """langchain 문서를 청킹"""

        splitter_params = {}
        chunk_size = kwargs.get('chunk_size')
        chunk_overlap = kwargs.get('chunk_overlap')

        if chunk_size is not None:
            splitter_params['chunk_size'] = chunk_size

        if chunk_overlap is not None:
            splitter_params['chunk_overlap'] = chunk_overlap

        text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            page = chunk.metadata.get('page', 1)

            source = chunk.metadata.get('source', '')
            file_ext = os.path.splitext(source)[-1].lower() if source else ''

            if file_ext in ['.jpg', '.jpeg', '.png']:
                # 이미지 파일: 이미 1-based이므로 그대로 사용
                if isinstance(page, int) and page <= 0:
                    page = 1  # 0이거나 음수인 경우에만 1로 설정
            else:
                # 다른 파일들: 0-based를 1-based로 변환
                if isinstance(page, int) and page >= 0:
                    page += 1

            chunk.metadata['page'] = page
            self.page_chunk_counts[page] += 1
        return chunks

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        # chunk_size 우선순위: kwargs > yaml(chunking.chunk_size) > 0
        chunk_size = _parse_optional_int(kwargs.get('chunk_size'), 'chunk_size')
        if chunk_size is None:
            chunk_size = self._chunk_size
        chunker: GenosSmartChunker = GenosSmartChunker(
            max_tokens = chunk_size if chunk_size is not None else 0,
            merge_peers = True,
            tokenizer = self._tokenizer,
            tokenizer_type = self._tokenizer_type,
        )

        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        try:
            # 새로운 enriched result 받기
            document = enrich_document(document, self.enrichment_options, **kwargs)
            return document
        except LLMApiError as e:
            # Preserve provider error payload as-is for load status error message.
            raise GenosServiceException("1", e.raw_error_message) from e

    def _get_or_create_image_description_enricher(self):
        enricher = getattr(self, "image_description_enricher", None)
        if enricher is None:
            # 테스트 등에서 __init__ 우회 시 legacy attribute 기반으로 재구성
            legacy_options = ImageDescriptionOptions.from_legacy_processor(self)
            enricher = FacadeImageDescriptionEnricher(legacy_options)
            self.image_description_enricher = enricher
        return enricher

    def enrich_image_descriptions(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = self._get_or_create_image_description_enricher()
        if enricher is None:
            return document
        return enricher.enrich(document, **kwargs)

    async def enrich_metadata(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = getattr(self, "metadata_enricher", None)
        if enricher is not None:
            document = await enricher.enrich(document, **kwargs)
        return document

    async def enrich_custom_fields(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        for enricher in self.custom_fields_enrichers:
            document = await enricher.enrich(document, **kwargs)
        return document

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request, **kwargs: dict) -> \
            list[dict]:
        title = ""
        enrichment_context = kwargs.get("_enrichment_context")
        context_metadata = (
            dict(enrichment_context.get("metadata", {}))
            if isinstance(enrichment_context, dict) and isinstance(enrichment_context.get("metadata"), dict)
            else {}
        )
        document_metadata = extract_metadata_from_document(document)
        merged_metadata = dict(document_metadata)
        merged_metadata.update(context_metadata)
        # 설정 기반 typed 필드 변환 (created_date 등). source/target 키는 passthrough 에서 제외.
        typed_values, consumed_keys = apply_field_transforms(
            self._metadata_field_transforms, merged_metadata, document)

        for item, _ in document.iterate_items():
            if hasattr(item, 'label'):
                if item.label == DocItemLabel.TITLE:
                    title = item.text.strip() if item.text else ""
                    break

        passthrough_metadata = dict(merged_metadata)
        # GenOSVectorMeta 스키마 예약 필드 + transform 이 소비한 source/target 키는 passthrough 제외.
        reserved_keys = {
            "text", "n_char", "n_word", "n_line", "e_page", "i_page",
            "i_chunk_on_page", "n_chunk_of_page", "i_chunk_on_doc", "n_chunk_of_doc",
            "n_page", "reg_date", "chunk_bboxes", "media_files", "title",
            "created_date",
        } | consumed_keys
        for reserved_key in reserved_keys:
            passthrough_metadata.pop(reserved_key, None)
        passthrough_metadata = {
            key: serialize_metadata_value_for_output(value)
            for key, value in passthrough_metadata.items()
        }

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
            title=title,
        )
        global_metadata.update(typed_values)  # 설정 기반 typed 필드 (created_date 등)
        global_metadata.update(passthrough_metadata)

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
            # header 앞에 헤더 마커 추가 (HEADER: )
            headers_text = "HEADER: " + ", ".join(chunk.meta.headings) + '\n' if chunk.meta.headings else ''
            content = headers_text + chunk.text

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items, include_tables=self.table_image_enabled)
                      ).build()
            vectors.append(vector)

            chunk_index_on_page += 1
            if upload_files:
                file_list = self.get_media_files(chunk.meta.doc_items, include_tables=self.table_image_enabled)
                upload_tasks.append(asyncio.create_task(
                    upload_files(file_list, request=request)
                ))

        if upload_tasks:
            await asyncio.gather(*upload_tasks)

        return vectors

    def compose_vectors_langchain(self, chunks, file_path: str, **kwargs: dict) -> list[dict]:
        """langchain 청크를 벡터로 변환 (PPT용)"""
        pdf_path = _get_pdf_path(file_path)
        doc = None
        total_pages = 0

        try:
            if os.path.exists(pdf_path):
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
        except Exception as e:
            print(f"Failed to open PDF {pdf_path}: {e}")

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=max([chunk.metadata['page'] for chunk in chunks]),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []

        chunk_bboxes_data = []
        i_page_value = None
        e_page_value = None

        for chunk_idx, chunk in enumerate(chunks):
            page = chunk.metadata['page']
            text = chunk.page_content

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            i_page_value = page  # 디폴트값
            e_page_value = page  # 디폴트값

            if doc and total_pages > 0:
                page_index = page - 1
                if 0 <= page_index < total_pages:
                    fitz_page = doc.load_page(page_index)
                    try:
                        from genos_utils import merge_overlapping_bboxes
                        merged_bboxes = merge_overlapping_bboxes([
                            {
                                'page': page,
                                'type': 'text',
                                'bbox': {
                                    'l': rect[0] / fitz_page.rect.width,
                                    't': rect[1] / fitz_page.rect.height,
                                    'r': rect[2] / fitz_page.rect.width,
                                    'b': rect[3] / fitz_page.rect.height,
                                }
                            } for rect in fitz_page.search_for(text)
                        ], x_tolerance=1 / fitz_page.rect.width,
                            y_tolerance=1 / fitz_page.rect.height)
                    except ImportError:
                        merged_bboxes = []
                        for rect in fitz_page.search_for(text):
                            bbox_data = {
                                'page': page,
                                'type': 'text',
                                'bbox': {
                                    'l': rect[0] / fitz_page.rect.width,
                                    't': rect[1] / fitz_page.rect.height,
                                    'r': rect[2] / fitz_page.rect.width,
                                    'b': rect[3] / fitz_page.rect.height,
                                }
                            }
                            merged_bboxes.append(bbox_data)

                    chunk_bboxes_data = merged_bboxes
                    global_metadata['chunk_bboxes'] = json.dumps(merged_bboxes)

                    if merged_bboxes:
                        bbox_pages = [bbox.get('page') for bbox in merged_bboxes if bbox.get('page') is not None]
                        if bbox_pages:
                            i_page_value = min(bbox_pages)  # 최소값
                            e_page_value = max(bbox_pages)  # 최대값

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_chars': len(text),
                'n_words': len(text.split()),
                'n_lines': len(text.splitlines()),
                'i_page': i_page_value,
                'e_page': e_page_value,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                'chunk_bboxes': json.dumps(chunk_bboxes_data),
                'media_files': json.dumps([]),
                **global_metadata
            }))
            chunk_index_on_page += 1

        if doc:
            doc.close()

        return vectors

    async def _extract_page_images(self, pdf_path: str, request: Request) -> dict[int, list[dict]]:
        if not os.path.exists(pdf_path):
            return {}

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Failed to open PDF {pdf_path}: {e}")
            return {}
        file_list: list[dict] = []
        page_meta: dict[int, list[dict]] = defaultdict(list)

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Convert to RGB if needed
                    if pix.n >= 5:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.n == 4:  # RGBA
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.n < 3:  # Grayscale
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_name = f"{uuid.uuid4()}.png"
                    img_path = os.path.join("/tmp", img_name)

                    pix.save(img_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    continue
                finally:
                    pix = None  # Free memory

        if file_list and upload_files:
            await upload_files(file_list, request=request)

        doc.close()
        return page_meta

    def _save_table_images(
        self,
        document: DoclingDocument,
        image_dir: Path,
        reference_path: Optional[Path] = None,
    ) -> None:
        """표 영역을 PNG 로 저장하고 TableItem.image.uri 를 설정한다(in-place).

        docling 의 DoclingDocument._with_pictures_refs 가 PictureItem 만 디스크에
        저장하므로, 동일 로직을 TableItem 에 대해 미러링한다. TableItem.get_image 는
        item.image 가 없으면 페이지 이미지에서 prov bbox 로 잘라 반환한다
        (generate_page_images 가 True 여야 함 — __init__ 에서 보장).
        """
        image_dir.mkdir(parents=True, exist_ok=True)
        if not image_dir.is_dir():
            return

        img_count = 0
        for item, _ in document.iterate_items(with_groups=False):
            if not isinstance(item, TableItem):
                continue
            img = item.get_image(doc=document)
            if img is None:
                continue
            hexhash = PictureItem._image_to_hexhash(img)
            if hexhash is None:
                continue
            loc_path = image_dir / f"table_{img_count:06}_{hexhash}.png"
            img.save(loc_path)
            if reference_path is not None:
                obj_path = relative_path(reference_path.resolve(), loc_path.resolve())
            else:
                obj_path = loc_path
            # 파이프라인이 표 이미지를 미리 크롭하지 않으므로(generate_table_images 미사용)
            # item.image 는 보통 None 이다. ImageRef 를 생성하되 uri 는 반드시 저장한
            # PNG 파일 경로로 설정한다(from_pil 의 base64 data URI 가 남지 않도록).
            if item.image is None:
                scale = img.size[0] / item.prov[0].bbox.width
                item.image = ImageRef.from_pil(image=img, dpi=round(72 * scale))
            item.image.uri = Path(obj_path)
            img_count += 1

    def get_media_files(self, doc_items: list, include_tables: bool = False):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
            elif include_tables and isinstance(item, TableItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
        """텍스트에 GLYPH 항목이 있는지 확인하는 메서드"""
        if not text:
            return False

        # GLYPH 항목이 있는지 정규식으로 확인
        matches = re.findall(r'GLYPH\w*', text)
        if len(matches) >= threshold:
            # print(f"Text has glyphs. len(matches): {len(matches)}. ")
            return True

        return False

    def check_glyphs(self, document: DoclingDocument) -> bool:
        """문서에 글리프가 있는지 확인하는 메서드"""
        for item, level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no
                # page_texts += item.text

                # GLYPH 항목이 있는지 확인. 정규식사용
                matches = re.findall(r'GLYPH\w*', item.text)
                if len(matches) > self._glyph_document_threshold:
                    # print(f"Document has glyphs on page {page_no}. len(matches): {len(matches)}. ")
                    return True

        return False

    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> List[Dict[str, Any]]:
        """
        글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR을 수행합니다.
        Args:
            document: DoclingDocument 객체
            pdf_path: PDF 파일 경로
        Returns:
            OCR이 완료된 문서의 DoclingDocument 객체
        """
        import fitz
        import base64
        import requests

        def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
            HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"file": base64.b64encode(img_bytes).decode("ascii"), "fileType": 1, "visualize": False}
            r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
            if not r.ok:
                # 진단에 도움되도록 본문 일부 출력
                raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
            return r.json()

        def extract_ocr_fields(resp: dict):
            """
            resp: 위와 같은 OCR 응답 JSON(dict)
            return: (rec_texts, rec_scores, rec_boxes) — 모두 list
            """
            if resp is None:
                return [], [], []

            # 최상위 상태 체크
            if resp.get("errorCode") not in (0, None):
                return [], [], []

            ocr_results = (
                resp.get("result", {})
                    .get("ocrResults", [])
            )
            if not ocr_results:
                return [], [], []

            pruned = (
                ocr_results[0]
                .get("prunedResult", {})
            )
            if not pruned:
                return [], [], []

            rec_texts  = pruned.get("rec_texts", [])   # list[str]
            rec_scores = pruned.get("rec_scores", [])  # list[float]
            rec_boxes  = pruned.get("rec_boxes", [])   # list[[x1,y1,x2,y2]]

            # 길이 불일치 방어: 최소 길이에 맞춰 자르기
            n = min(len(rec_texts), len(rec_scores), len(rec_boxes))
            return rec_texts[:n], rec_scores[:n], rec_boxes[:n]

        try:
            doc = fitz.open(pdf_path)

            for table_idx, table_item in enumerate(document.tables):
                if not table_item.data or not table_item.data.table_cells:
                    continue

                b_ocr = False
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if self.check_glyph_text(cell.text, threshold=self._glyph_table_cell_threshold):
                        b_ocr = True
                        break

                if b_ocr is False:
                    # 글리프 깨진 텍스트가 없는 경우, OCR을 수행하지 않음
                    continue

                for cell_idx, cell in enumerate(table_item.data.table_cells):

                    # Provenance 정보에서 위치 정보 추출
                    if not table_item.prov:
                        continue

                    page_no = table_item.prov[0].page_no - 1
                    bbox = cell.bbox

                    page = doc.load_page(page_no)

                    # 셀의 바운딩 박스를 사용하여 이미지에서 해당 영역을 잘라냄
                    cell_bbox = fitz.Rect(
                        bbox.l, min(bbox.t, bbox.b),
                        bbox.r, max(bbox.t, bbox.b)
                    )

                    # bbox 높이 계산 (PDF 좌표계 단위)
                    bbox_height = cell_bbox.height

                    # 목표 픽셀 높이
                    target_height = 20

                    # zoom factor 계산
                    # (너무 작은 bbox일 경우 0으로 나누는 걸 방지)
                    zoom_factor = target_height / bbox_height if bbox_height > 0 else 1.0
                    zoom_factor = min(zoom_factor, 4.0)  # 최대 확대 비율 제한
                    zoom_factor = max(zoom_factor, 1)  # 최소 확대 비율 제한

                    # 페이지를 이미지로 렌더링
                    mat = fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat, clip=cell_bbox)
                    img_data = pix.tobytes("png")

                    result = post_ocr_bytes(img_data, timeout=self._table_cell_ocr_timeout)
                    rec_texts, rec_scores, rec_boxes = extract_ocr_fields(result)

                    cell.text = ""
                    for t in rec_texts:
                        if len(cell.text) > 0:
                            cell.text += " "
                        cell.text += t if t else ""
        except Exception as e:
            print(f"OCR processing failed: {e}")
            pass

        return document

    def setup_logging(self, level_num: int):
        """
            5"DEBUG", 4"INFO", 3"WARNING", 2"ERROR", 1"CRITICAL", 0"NOLOG" 중 하나를 받아서 로깅 레벨을 설정하는 메서드
        """
        def get_level_name(level_num: int) -> str:
            level_map = {
                5: "DEBUG",
                4: "INFO",
                3: "WARNING",
                2: "ERROR",
                1: "CRITICAL",
                0: "NOLOG"
            }
            return level_map.get(level_num, "INFO")
        level_name = get_level_name(level_num)
        print(f"Setting log level to: {level_name}")

        if level_name == "NOLOG" or not hasattr(logging, level_name):
            logging.disable(logging.CRITICAL)  # 모든 로그 비활성화
            return

        level = getattr(logging, level_name.upper())

        # root logger 설정 (핸들러는 main에서만 설정)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]   # 콘솔 출력
        )

        # root logger level 적용
        logging.getLogger().setLevel(level)

    async def _process_request(self, request: Request, file_path: str, **kwargs: dict):
        runtime_level = kwargs.get('log_level')
        self.setup_logging(runtime_level if runtime_level is not None else self._log_level)

        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")

        ext = Path(file_path).suffix.lower()

        # 직접 처리 가능한 엑셀 계열 포맷(이슈 #288). csv 는 intercept_csv 옵션 시 포함.
        xlsx_direct_exts = set(_XLSX_DIRECT_EXTS)
        if self._xlsx_cfg["intercept_csv"]:
            xlsx_direct_exts.add(".csv")

        # 포맷별 처리: 엑셀 계열은 직접 처리, ppt 는 langchain, 그 외는 docling.
        if ext in xlsx_direct_exts:
            return await self._process_xlsx(request, file_path, **kwargs)
        if ext == ".ppt":
            return await self._process_ppt(request, file_path, **kwargs)
        return await self._process_docling(request, file_path, **kwargs)

    async def _process_xlsx(self, request: Request, file_path: str, **kwargs: dict):
        """xlsx/csv 직접 처리(이슈 #288): PDF 변환 없이 처리해 행 분할 버그 방지.
          - tabular: 데이터 행마다 1청크(벡터)로 만들어 즉시 반환
          - docling(기본): MsExcel 백엔드로 DoclingDocument 생성 후 공유 파이프라인으로 합류
        """
        from genon.preprocessor.converters.xlsx_processor import (
            build_docling_document,
            build_tabular_vectors,
        )
        if self._xlsx_cfg["processing_mode"] == "tabular":
            _log.info(f"[DocumentProcessor] xlsx tabular 직접 처리: {file_path}")
            vectors = build_tabular_vectors(
                file_path,
                header_row=self._xlsx_cfg["header_row"],
                encoding=self._xlsx_cfg["encoding"],
            )
            if not vectors:
                raise GenosServiceException("1", f"chunk length is 0")
            return vectors

        _log.info(f"[DocumentProcessor] xlsx docling 직접 처리: {file_path}")
        try:
            document = build_docling_document(
                file_path, save_images=kwargs.get('save_images', False)
            )
        except Exception as e:
            raise GenosServiceException(
                "1", f"xlsx 처리 실패: {os.path.basename(file_path)} ({e})"
            )
        # openpyxl 텍스트라 글리프 깨짐이 없고 렌더 PDF 도 없으므로 테이블셀 재OCR 은 생략.
        # table_as_chunk=True: 시트/표마다 별도 청크로 분리(엑셀은 표 단위가 논리 단위).
        return await self._document_to_vectors(
            document, file_path, request, ocr_table_cells=False, table_as_chunk=True, **kwargs
        )

    async def _process_ppt(self, request: Request, file_path: str, **kwargs: dict):
        """PPT(.ppt)는 langchain 로더로 처리하고 페이지 이미지 메타를 부착한다."""
        documents = self.load_documents_langchain(file_path, **kwargs)
        chunks = self.split_documents_langchain(documents, **kwargs)

        pdf_path = _get_pdf_path(file_path)
        page_image_meta = {}
        try:
            page_image_meta = await self._extract_page_images(pdf_path, request)
        except:
            pass

        vectors = self.compose_vectors_langchain(chunks, file_path, **kwargs)
        for v in vectors:
            if v.i_page in page_image_meta:
                v.media_files = json.dumps(page_image_meta[v.i_page], ensure_ascii=False)
            else:
                v.media_files = json.dumps([])
        return vectors

    async def _process_docling(self, request: Request, file_path: str, **kwargs: dict):
        """PDF/DOCX/PPTX/HWP/기타를 docling 으로 로딩 후 공유 파이프라인으로 처리."""
        ext = Path(file_path).suffix.lower()
        document = self._load_document(file_path, **kwargs)

        # DOCX/PPTX 는 미리보기용 PDF 아티팩트를 생성한다(부수효과; 처리 결과엔 미사용).
        if ext in ['.docx', '.pptx']:
            convert_to_pdf(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))

        return await self._document_to_vectors(
            document, file_path, request, ocr_table_cells=(ext == '.pdf'), **kwargs
        )

    def _load_document(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        """docling 문서 로딩. pdf 는 ocr_mode 분기, 그 외 포맷은 기본 백엔드로 로딩한다.
        ocr_mode: "force"=무조건 전체 OCR / "auto"=휴리스틱 기반 재OCR / "disable"=OCR 안 함
        """
        ext = Path(file_path).suffix.lower()
        if ext != '.pdf':
            return self.load_documents(file_path, **kwargs)

        if self.ocr_mode == "force":
            return self.load_documents_with_docling_ocr(file_path, **kwargs)
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        if self.ocr_mode == "auto":
            if not check_document(document, self.enrichment_options) or self.check_glyphs(document):
                # OCR이 필요하다고 판단되면 OCR 수행
                document = self.load_documents_with_docling_ocr(file_path, **kwargs)
        return document

    async def _document_to_vectors(self, document: DoclingDocument, file_path: str,
                                   request: Request, *, ocr_table_cells: bool, **kwargs: dict) -> list:
        """DoclingDocument → enrichment → 청킹 → 벡터 생성(공유 파이프라인).

        ocr_table_cells: 글리프 깨진 테이블 셀 재OCR 수행 여부(pdf 만 True).
        """
        # 글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR 수행 (청크토큰 8k이상 발생 방지)
        if ocr_table_cells and self.ocr_mode != "disable" and self.ocr_endpoint:
            document = self.ocr_all_table_cells(document, file_path)

        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent

        document = document._with_pictures_refs(image_dir=artifacts_dir, page_no=None, reference_path=reference_path)

        # 표 이미지 저장 옵션이 켜진 경우, picture 와 동일하게 표 영역을 PNG 로 저장하고
        # TableItem.image.uri 를 설정한다(_with_pictures_refs 미러).
        if self.table_image_enabled:
            self._save_table_images(document, image_dir=artifacts_dir, reference_path=reference_path)

        document = self.enrichment(document, **kwargs)
        enrichment_kwargs = dict(kwargs)
        enrichment_kwargs["_enrichment_context"] = {}
        try:
            document = self.enrich_image_descriptions(document, **enrichment_kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] facade image enrichment skipped: {exc}")
        try:
            document = await self.enrich_metadata(document, **enrichment_kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] metadata enrichment skipped: {exc}")
        try:
            document = await self.enrich_custom_fields(document, **enrichment_kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] custom_fields enrichment skipped: {exc}")

        # Extract Chunk from DoclingDocument
        chunks: List[DocChunk] = self.split_documents(document, **kwargs)

        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(
                document,
                chunks,
                file_path,
                request,
                **enrichment_kwargs,
            )
        else:
            raise GenosServiceException("1", f"chunk length is 0")

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        # HWP/HWPX: docling(SDK + 레거시 백엔드) 처리가 전체 실패하면 PDF 변환으로 최종 폴백한다.
        # attachment_processor.DocumentProcessor.__call__ 의 PDF 폴백과 동일 취지 —
        # convert_to_pdf 는 rhwp ↔ LibreOffice HWP→PDF 체인이며, 변환된 PDF 를 PDF 경로로 재처리한다.
        # (헌법.hwp 처럼 SDK 가 exit 3 으로 거부하거나 02.hwp 처럼 빈 결과를 내는 경우를 살린다.)
        ext = Path(file_path).suffix.lower()
        if ext in ('.hwp', '.hwpx'):
            try:
                return await self._process_request(request, file_path, **kwargs)
            except Exception as hwp_err:
                _log.warning(f"[DocumentProcessor] HWP/HWPX 처리 실패, PDF 변환 폴백 시도: {hwp_err}")
                converted = convert_to_pdf(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
                if converted:
                    _log.info(f"[DocumentProcessor] PDF 변환 성공, PDF 경로로 재처리: {converted}")
                    return await self._process_request(request, converted, **kwargs)
                raise hwp_err
        return await self._process_request(request, file_path, **kwargs)


class GenosServiceException(Exception):
    # GenOS 와의 의존성 부분 제거를 위해 추가
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"


# GenOS 와의 의존성 제거를 위해 추가
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")
