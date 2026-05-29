"""field_transforms.py — 추출 메타데이터를 typed 벡터 필드로 변환하는 재사용 로직.

intelligent_processor 의 created_date 전용 코드를 일반화한 모듈이다.
설정(field_transforms)에 따라 "어떤 출력 키 → 어떤 벡터 필드 → 어떤 변환" 을 매핑한다.

순수 함수 모음이라 docling/fastapi 등 무거운 의존성이 없다(enrichment_config.py 와 동일 성격).
docling 타입은 타입 힌트 용도로만 참조하므로 TYPE_CHECKING 으로 import 한다.

신규 변환기/보조추출은 함수 작성 후 VALUE_TRANSFORMS / FALLBACK_STRATEGIES 에 등록만 하면 된다.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from docling_core.types import DoclingDocument

_log = logging.getLogger(__name__)


# 추출 메타데이터 → typed 벡터 필드 변환의 기본값.
# yaml metadata.field_transforms 가 비어있을 때 적용되어 기존 created_date 동작을 보존한다.
# 각 항목: source(후보 키, str 또는 list) → target(벡터 필드) → type(값 변환기) / fallback(보조 추출).
DEFAULT_METADATA_FIELD_TRANSFORMS = [
    {
        "source": ["created_date", "작성일"],
        "target": "created_date",
        "type": "date_int",
        "fallback": "doc_text_scan",
    },
]


# ── 값 변환기 ───────────────────────────────────────────────────────────────

def parse_created_date(date_text: str) -> int:
    """작성일 텍스트를 파싱하여 YYYYMMDD 형식의 정수로 변환.

    Args:
        date_text: 작성일 텍스트 (YYYY-MM 또는 YYYY-MM-DD 형식)

    Returns:
        YYYYMMDD 형식의 정수, 파싱 실패시 0
    """
    if not date_text or not isinstance(date_text, str) or date_text == "None":
        return 0

    # 공백 제거 및 정리
    date_text = date_text.strip()

    # 1) YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD / YYYY년 MM월 DD일 (문장 내부 포함) 우선
    for pattern in (
        r'(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})',
        r'(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?',
    ):
        match_full = re.search(pattern, date_text)
        if match_full:
            year, month, day = match_full.groups()
            try:
                datetime(int(year), int(month), int(day))
                return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
            except ValueError:
                continue

    # 2) YYYY-MM / YYYY.MM / YYYY/MM / YYYY년 MM월 (문장 내부 포함) → 일자는 01
    for pattern in (
        r'(\d{4})\s*[-./]\s*(\d{1,2})',
        r'(\d{4})\s*년\s*(\d{1,2})\s*월',
    ):
        match_month = re.search(pattern, date_text)
        if match_month:
            year, month = match_month.groups()
            try:
                datetime(int(year), int(month), 1)
                return int(f"{year}{month.zfill(2)}01")
            except ValueError:
                continue

    # 3) YYYY 형식 → 월일은 0101
    match_year = re.search(r'(\d{4})', date_text)
    if match_year:
        year = match_year.group(1)
        try:
            datetime(int(year), 1, 1)
            return int(f"{year}0101")
        except ValueError:
            pass

    return 0


def transform_date_int(value: Any) -> int:
    """date_int 변환기: 날짜 텍스트/정수를 YYYYMMDD 정수로 변환."""
    if isinstance(value, (int, float)):
        candidate_int = int(value)
        return candidate_int if candidate_int > 0 else 0
    if value in (None, ""):
        return 0
    return parse_created_date(str(value))


# ── 보조 추출(fallback) ──────────────────────────────────────────────────────

def extract_created_date_from_document_text(document: "DoclingDocument") -> int:
    """metadata 추출이 비어있을 때 문서 본문에서 작성/기준일을 휴리스틱으로 추출."""
    try:
        raw_text = document.export_to_text() or ""
    except Exception:
        raw_text = ""
    if not raw_text:
        return 0

    prioritized_patterns = (
        r'(?:기준일|작성일|최초\s*작성일|보고자료)[^0-9]{0,20}(\d{4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2})',
        r'(?:기준일|작성일|최초\s*작성일|보고자료)[^0-9]{0,20}(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?)',
        r'(?:기준일|작성일|최초\s*작성일|보고자료)[^0-9]{0,20}(\d{4}\s*[./-]\s*\d{1,2})',
        r'(?:기준일|작성일|최초\s*작성일|보고자료)[^0-9]{0,20}(\d{4}\s*년\s*\d{1,2}\s*월)',
    )
    for pattern in prioritized_patterns:
        m = re.search(pattern, raw_text)
        if not m:
            continue
        parsed = parse_created_date(m.group(1))
        if parsed:
            return parsed

    # 키워드 기반으로 찾지 못하면 문서 최초 날짜를 fallback 으로 사용.
    fallback_match = re.search(
        r'(\d{4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2}|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?)',
        raw_text,
    )
    if fallback_match:
        parsed = parse_created_date(fallback_match.group(1))
        if parsed:
            return parsed
    return 0


# ── 레지스트리 ───────────────────────────────────────────────────────────────
# 신규 변환기/보조추출은 함수 작성 후 아래 dict 에 등록만 하면 설정에서 바로 사용 가능.
VALUE_TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "date_int": transform_date_int,
}
FALLBACK_STRATEGIES: dict[str, Callable[["DoclingDocument"], Any]] = {
    "doc_text_scan": extract_created_date_from_document_text,
}


# ── 적용 ─────────────────────────────────────────────────────────────────────

def apply_field_transforms(
    field_transforms: list,
    merged_metadata: dict[str, Any],
    document: "DoclingDocument",
) -> tuple[dict[str, Any], set[str]]:
    """field_transforms 설정에 따라 추출 메타데이터를 typed 벡터 필드로 변환한다.

    Args:
        field_transforms: 변환 spec 의 list (각 spec 은 source/target/type/fallback dict).
        merged_metadata: 문서/컨텍스트 병합 메타데이터.
        document: 본문 휴리스틱(fallback)에 사용할 docling 문서.

    Returns:
        (typed_values, consumed_keys)
        - typed_values: {target_field: 변환값} — global_metadata 로 주입
        - consumed_keys: passthrough 에서 제외할 source/target 키 집합
    """
    typed_values: dict[str, Any] = {}
    consumed_keys: set[str] = set()

    for spec in field_transforms or []:
        if not isinstance(spec, dict):
            continue
        sources = spec.get("source")
        if isinstance(sources, str):
            sources = [sources]
        elif not isinstance(sources, list):
            sources = []
        target = spec.get("target") or (sources[0] if sources else None)
        if not target:
            continue
        consumed_keys.update(sources)
        consumed_keys.add(target)

        # 후보 키를 순서대로 탐색해 첫 non-empty 값 선택
        raw_value = None
        for key in sources:
            candidate = merged_metadata.get(key)
            if candidate not in (None, ""):
                raw_value = candidate
                break

        transform_name = spec.get("type")
        transform_fn = VALUE_TRANSFORMS.get(transform_name) if transform_name else None
        value = transform_fn(raw_value) if transform_fn is not None else raw_value

        # 값이 비어있고 fallback 이 지정된 경우 문서 본문 휴리스틱 적용
        if (value in (None, "") or value == 0):
            fallback_name = spec.get("fallback")
            fallback_fn = FALLBACK_STRATEGIES.get(fallback_name) if fallback_name else None
            if fallback_fn is not None:
                value = fallback_fn(document)

        typed_values[target] = value

    return typed_values, consumed_keys


# ── 문서 메타데이터 추출/직렬화 유틸 ─────────────────────────────────────────────
# compose_vectors 에서 docling 문서의 KeyValueItem 메타데이터를 읽고 벡터 출력용으로
# 직렬화하는 공통 로직. intelligent_processor / convert_processor 가 공유한다.

def normalize_metadata_value(value: Any) -> Any:
    """문자열 값이 JSON 배열/객체 형태면 파싱, 아니면 strip 한 문자열을 반환."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{" and stripped[-1] in "]}":
        try:
            return json.loads(stripped)
        except Exception:
            return stripped
    return stripped


def extract_metadata_from_document(document: "DoclingDocument") -> dict[str, Any]:
    """docling 문서의 KeyValueItem 들에서 key→value 메타데이터를 일반적으로 추출."""
    metadata: dict[str, Any] = {}
    for kv_item in getattr(document, "key_value_items", []) or []:
        graph = getattr(kv_item, "graph", None)
        cells = getattr(graph, "cells", None) or []
        pending_key: Optional[str] = None
        for cell in cells:
            text = str(getattr(cell, "text", "") or "").strip()
            if not text:
                continue
            label_obj = getattr(cell, "label", None)
            label = str(getattr(label_obj, "value", label_obj) or "").strip().lower()
            if label == "key":
                pending_key = text
                continue
            if label == "value" and pending_key:
                metadata[pending_key] = normalize_metadata_value(text)
                pending_key = None
                continue
            # fallback: label 정보가 없거나 예외적인 순서인 경우 순차 pair 처리
            if pending_key is None:
                pending_key = text
            else:
                metadata[pending_key] = normalize_metadata_value(text)
                pending_key = None
    return metadata


def serialize_metadata_value_for_output(value: Any) -> Any:
    """벡터 결과 포맷 일관성을 위해 중첩 metadata는 JSON 문자열로 직렬화."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value
