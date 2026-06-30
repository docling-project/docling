# 엑셀(xlsx/xlsm)·csv 직접 처리 헬퍼 (이슈 #288 / KOBCI-183)
#
# intelligent_processor 가 PDF 변환 없이 xlsx 를 직접 처리하기 위한 헬퍼 모듈.
# facade 가 아니라 converters 하위 모듈이며, intelligent_processor → 이 모듈의 단방향 의존만 갖는다.
#
# 두 가지 처리 방식을 제공한다.
#   - build_docling_document(): docling MsExcelDocumentBackend 로 DoclingDocument 생성(시트=1페이지).
#       → 기존 청킹/벡터 파이프라인을 그대로 태운다. PDF 변환 시 한 행이 페이지 경계로 쪼개지는
#         논리 오류를 원천 제거한다.
#   - build_tabular_vectors(): 데이터 행마다 1청크, 컬럼 헤더→셀 값을 메타데이터로 부여한다.
#       openpyxl 로 병합셀을 unmerge + forward-fill 하여 병합 헤더 유실을 방지한다.
#
# docling 관련 import 는 build_docling_document() 내부에서만 수행한다(로컬 vendored docling import
# 충돌 회피 + tabular 전용 사용 시 docling 미로딩).
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

_log = logging.getLogger(__name__)

XLSX_EXTS = {".xlsx", ".xlsm"}
CSV_EXTS = {".csv"}

# Weaviate property name 제약(GraphQL 표준): /[_A-Za-z][_0-9A-Za-z]*/
# 한글 등 비-ASCII 헤더를 메타데이터 KEY 로 쓰면 grpc 에러가 나므로 키 후보에서 제외한다.
_VALID_KEY_RE = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")

# 예약 필드(컬럼 헤더가 이 이름과 충돌하면 메타 키로 쓰지 않는다)
_RESERVED_FIELDS = {
    "text", "n_char", "n_word", "n_line", "i_page", "e_page",
    "i_chunk_on_page", "n_chunk_of_page", "i_chunk_on_doc", "n_chunk_of_doc",
    "n_page", "reg_date", "chunk_bboxes", "media_files",
}


class GenOSVectorMeta(BaseModel):
    class Config:
        extra = "allow"

    text: Optional[str] = None
    n_char: Optional[int] = None
    n_word: Optional[int] = None
    n_line: Optional[int] = None
    i_page: Optional[int] = None
    e_page: Optional[int] = None
    i_chunk_on_page: Optional[int] = None
    n_chunk_of_page: Optional[int] = None
    i_chunk_on_doc: Optional[int] = None
    n_chunk_of_doc: Optional[int] = None
    n_page: Optional[int] = None
    reg_date: Optional[str] = None
    chunk_bboxes: Optional[str] = None
    media_files: Optional[str] = None


# ------------------------------------------------------------------ #
# 공통 헬퍼                                                            #
# ------------------------------------------------------------------ #
def is_xlsx_like(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in XLSX_EXTS


def is_csv(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() in CSV_EXTS


def _safe_utf8(v: str) -> str:
    return v.encode("utf-8", errors="ignore").decode("utf-8")


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    return _safe_utf8(v) if isinstance(v, str) else str(v)


def _row_to_pipe(values: list) -> str:
    cells = " | ".join(_cell_str(v) for v in values)
    return f"| {cells} |"


def _is_valid_key(key: str) -> bool:
    return bool(key) and key not in _RESERVED_FIELDS and bool(_VALID_KEY_RE.match(key))


# ------------------------------------------------------------------ #
# docling 모드                                                         #
# ------------------------------------------------------------------ #
_docling_converter = None  # lazy singleton


def _get_docling_converter():
    """xlsx/csv 전용 DocumentConverter 를 생성·캐시한다(PDF 컨버터와 분리)."""
    global _docling_converter
    if _docling_converter is None:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import (
            CsvFormatOption,
            DocumentConverter,
            ExcelFormatOption,
        )

        _docling_converter = DocumentConverter(
            allowed_formats=[InputFormat.XLSX, InputFormat.CSV],
            format_options={
                InputFormat.XLSX: ExcelFormatOption(),
                InputFormat.CSV: CsvFormatOption(),
            },
        )
    return _docling_converter


def build_docling_document(file_path: str, *, save_images: bool = False):
    """xlsx/csv 를 docling MsExcel/Csv 백엔드로 변환해 DoclingDocument 를 반환한다.

    SimplePipeline 은 save_images 옵션을 사용하지 않는다(엑셀에 렌더 이미지 없음).
    변환 실패 시 예외를 그대로 올린다(호출부에서 GenosServiceException 으로 매핑).
    """
    converter = _get_docling_converter()
    conv_result = converter.convert(file_path, raises_on_error=True)
    return conv_result.document


# ------------------------------------------------------------------ #
# tabular 모드                                                         #
# ------------------------------------------------------------------ #
def _detect_encoding(file_path: str, fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    try:
        import chardet

        with open(file_path, "rb") as f:
            raw = f.read(10_000)
        return chardet.detect(raw).get("encoding") or "utf-8"
    except Exception:
        return "utf-8"


def _csv_rows(file_path: str, encoding: Optional[str]) -> list[list[str]]:
    import csv as _csv

    enc = _detect_encoding(file_path, encoding)
    rows: list[list[str]] = []
    with open(file_path, "r", encoding=enc, errors="ignore", newline="") as f:
        for row in _csv.reader(f):
            rows.append([_cell_str(c) for c in row])
    return rows


def _sheet_rows_unmerged(ws) -> list[list[str]]:
    """병합셀을 unmerge + forward-fill 한 뒤 2D 문자열 행 목록을 반환한다."""
    for rng in list(ws.merged_cells.ranges):
        top_left = ws.cell(row=rng.min_row, column=rng.min_col).value
        ws.unmerge_cells(str(rng))
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                ws.cell(row=r, column=c).value = top_left

    rows: list[list[str]] = []
    for row in ws.iter_rows(values_only=True):
        rows.append([_cell_str(v) for v in row])
    return rows


def _load_sheets(file_path: str, encoding: Optional[str]) -> dict[str, list[list[str]]]:
    if is_csv(file_path):
        return {"table_1": _csv_rows(file_path, encoding)}

    import openpyxl

    wb = openpyxl.load_workbook(file_path, data_only=True)
    return {name: _sheet_rows_unmerged(wb[name]) for name in wb.sheetnames}


def load_sheets(file_path: str, *, encoding: Optional[str] = None) -> dict[str, list[list[str]]]:
    """xlsx/csv 를 시트명 → 2D 문자열 행 목록으로 로드한다(공개 API).

    xlsx 는 병합셀을 unmerge + forward-fill 하여 병합 헤더/그룹 값 유실을 막는다.
    csv 는 인코딩 자동감지(또는 encoding) 후 단일 시트("table_1")로 반환한다.
    벡터/HTML 등 출력 형태에 독립적인 중립 표현이라 여러 facade 에서 재사용한다.
    """
    return _load_sheets(file_path, encoding)


def _is_empty_row(row: list[str]) -> bool:
    return all(not c.strip() for c in row)


def build_tabular_vectors(
    file_path: str,
    *,
    header_row: int = 0,
    encoding: Optional[str] = None,
    reg_date: Optional[str] = None,
) -> list[GenOSVectorMeta]:
    """데이터 행마다 1청크(GenOSVectorMeta)를 만든다.

    - 병합셀은 unmerge + forward-fill 로 헤더/그룹 값 유실 방지.
    - header_row(0-based) 행을 컬럼 키로, 그 아래 비어있지 않은 행을 데이터 행으로 사용.
    - 컬럼 헤더 중 Weaviate 키 규칙(/[_A-Za-z][_0-9A-Za-z]*/)에 맞는 것만 메타 KEY 로 부여하고,
      그 외(한글 등)는 키에서 제외하되 값은 text 에 그대로 포함해 데이터 손실을 막는다.
    """
    reg_date = reg_date or (datetime.now().isoformat(timespec="seconds") + "Z")
    sheets = _load_sheets(file_path, encoding)

    # (sheet_idx, sheet_name, headers, valid_key_cols, data_rows)
    prepared: list[tuple[int, str, list[str], list[int], list[list[str]]]] = []
    dropped_headers: set[str] = set()

    for sheet_idx, (sheet_name, rows) in enumerate(sheets.items(), start=1):
        if len(rows) <= header_row:
            continue
        headers = rows[header_row]
        data_rows = [r for r in rows[header_row + 1:] if not _is_empty_row(r)]
        if not data_rows:
            continue

        valid_key_cols: list[int] = []
        for col_idx, h in enumerate(headers):
            key = h.strip()
            if _is_valid_key(key):
                valid_key_cols.append(col_idx)
            elif key:
                dropped_headers.add(key)
        prepared.append((sheet_idx, sheet_name, headers, valid_key_cols, data_rows))

    if dropped_headers:
        _log.warning(
            "[xlsx] Weaviate 키 규칙에 맞지 않아 메타 KEY 에서 제외(값은 text 유지): %s",
            sorted(dropped_headers),
        )

    n_chunk_of_doc = sum(len(d) for _, _, _, _, d in prepared)
    if n_chunk_of_doc == 0:
        _log.warning(f"[xlsx] tabular 청크 없음: {file_path}")
        return []

    n_page = len(sheets)
    vectors: list[GenOSVectorMeta] = []
    chunk_doc_idx = 0

    for sheet_idx, sheet_name, headers, valid_key_cols, data_rows in prepared:
        header_line = _row_to_pipe(headers)
        for row_idx, row in enumerate(data_rows):
            text = f"시트명: {sheet_name}\n{header_line}\n{_row_to_pipe(row)}"
            row_meta = {
                headers[c].strip(): (row[c] if c < len(row) else "")
                for c in valid_key_cols
            }
            vectors.append(
                GenOSVectorMeta.model_validate(
                    {
                        **row_meta,
                        "text": text,
                        "n_char": len(text),
                        "n_word": len(text.split()),
                        "n_line": len(text.splitlines()),
                        "i_page": sheet_idx,
                        "e_page": sheet_idx,
                        "i_chunk_on_page": row_idx,
                        "n_chunk_of_page": len(data_rows),
                        "i_chunk_on_doc": chunk_doc_idx,
                        "n_chunk_of_doc": n_chunk_of_doc,
                        "n_page": n_page,
                        "reg_date": reg_date,
                        "chunk_bboxes": ".",
                        "media_files": ".",
                    }
                )
            )
            chunk_doc_idx += 1

    return vectors
