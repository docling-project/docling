# 적재용(지능형) 전처리기 v.2.2.3.1 (2026-07-21 Release)
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

# Genos 웹 UI 환경은 facade 코드를 단일 파일(preprocessor.py)로 처리하므로
# 다른 facade 파일에서 import 가 깨진다. 따라서 convert_to_pdf 는
# attachment_processor / convert_processor 와 동일하게 자체 정의한다.
import shutil
import subprocess
import tempfile
import unicodedata

import httpx


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
    ext = os.path.splitext(file_path)[1].lower()
    is_hwp = ext in (".hwp", ".hwpx")
    if use_pdf_sdk:
        order = ["pdf_sdk", "rhwp", "libreoffice"] if is_hwp else ["pdf_sdk", "libreoffice"]
    else:
        order = ["rhwp", "libreoffice"] if is_hwp else ["libreoffice"]
    return convert_hwp_to_pdf(file_path, order=order)

def _is_pdf(file_path: str) -> bool:
    """파일이 PDF 매직 헤더로 시작하는지 확인 (확장자 무관)."""
    try:
        with open(file_path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _has_any_pdf_converter() -> bool:
    """PDF 변환 backend(pdf_sdk / rhwp / libreoffice) 가 하나라도 가용한지 확인 (이슈 #286).

    빌드 시 INSTALL_LIBREOFFICE / INSTALL_RHWP 를 끄거나 PDF SDK 미포함(standard)이면
    변환 backend 가 0개가 될 수 있다. 이때 비-PDF 입력을 변환 시도하면 무조건 실패하므로,
    호출부에서 "PDF 로 직접 입력" 안내를 주기 위한 판별 헬퍼.
    가용성 판단 자체가 불가하면(import 실패 등) True 를 반환해 기존 동작을 유지한다.
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


# ── 비정상/암호화 파일 사전 감지 (이슈 #278/#307) ─────────────────────────────
# 이 블록은 parser/convert/attachment_processor 에도 복제되어 있다(단일 파일 배포 구조).
# 수정 시 네 파일 동기화 필요.
# 지원 포맷의 매직 헤더(allowlist). 각 값은 아래 공식 출처로 근거 확인 + 실제 샘플로 검증함.
#   - 정본 매직 DB: file/file(libmagic) magic/Magdir — 실제 본 모듈이 쓰는 python-magic의 DB.
#     (PDF=Magdir/pdf "%PDF-", PNG/GIF=Magdir/images, JPEG=Magdir/jpeg 0xffd8ff, ZIP=Magdir/msooxml "PK\3\4")
#   - 포맷 공식 스펙: PDF=ISO 32000(%PDF-), PNG=W3C PNG/RFC2083(89 50 4E 47 0D 0A 1A 0A),
#     ZIP=PKWARE APPNOTE(local file header 0x04034b50), OLE2/CFB=[MS-CFB] §2.2 Header(D0CF11E0A1B11AE1).
# zip(PK)=docx/xlsx/pptx/hwpx, OLE2(d0cf..)=hwp/doc/ppt/xls(레거시).
_KNOWN_MAGIC_PREFIXES = (
    b"%PDF-",                                # pdf
    b"\x89PNG\r\n\x1a\n",                    # png
    b"\xff\xd8\xff",                         # jpeg/jpg
    b"GIF87a", b"GIF89a",                    # gif
    b"BM",                                    # bmp
    b"II*\x00", b"MM\x00*",                  # tiff
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",  # zip 계열(ooxml/hwpx)
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",     # OLE2/CFB(hwp5/doc/ppt/xls)
    b"ID3",                                   # mp3(id3v2)
    b"RIFF",                                  # wav/avi/webp
    b"OggS",                                  # ogg
    b"fLaC",                                  # flac
    b"\x1f\x8b",                             # gzip
    b"7z\xbc\xaf\x27\x1c",                  # 7z
    b"Rar!\x1a\x07",                        # rar
    b"<?xml",                                 # xml
)

# 텍스트로 봐줄 수 없는 제어 바이트(탭/개행/CR/FF 제외). 텍스트 파일엔 거의 없음.
_TEXT_ALLOWED_CTRL = {0x09, 0x0A, 0x0C, 0x0D}


def _looks_like_text(head: bytes) -> bool:
    """csv/txt/json/md/html 등 매직넘버 없는 텍스트 파일인지 휴리스틱 판정.
    NUL 이 있거나 제어문자 비율이 높으면 바이너리(=텍스트 아님)."""
    if not head:
        return False
    # UTF-16/32 텍스트는 NUL 바이트가 흔하므로 BOM 이면 먼저 텍스트로 인정.
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):  # UTF-16 LE/BE (UTF-32 BOM 도 이 prefix로 시작)
        return True
    if b"\x00" in head:
        return False
    ctrl = sum(
        1 for c in head if (c < 0x20 and c not in _TEXT_ALLOWED_CTRL) or c == 0x7F
    )
    return (ctrl / len(head)) < 0.05


def _is_encrypted_pdf(file_path: str) -> bool:
    """PDF /Encrypt(비밀번호/DRM 암호화) 여부. ISO 32000 기준, pypdf is_encrypted 사용."""
    try:
        from pypdf import PdfReader

        return bool(PdfReader(file_path).is_encrypted)
    except Exception:
        return False  # 파싱 실패는 여기서 단정 안 함(후속 단계에서 처리)


def _is_encrypted_office(file_path: str) -> bool:
    """암호화된 OOXML(docx/xlsx/pptx)은 OLE2 컨테이너의 'EncryptedPackage' 스트림으로
    저장된다(MS-OFFCRYPTO). olefile 로 그 스트림 존재를 확인."""
    try:
        import olefile

        if not olefile.isOleFile(file_path):
            return False
        ole = olefile.OleFileIO(file_path)
        try:
            return ole.exists("EncryptedPackage")
        finally:
            ole.close()
    except Exception:
        return False


def _is_protected_hwp(file_path: str) -> bool:
    """암호화/배포용(DRM) HWP 감지. HWP 5.0 'FileHeader' 스트림(OLE2 내, 256B)의
    flags(offset 36, uint32 LE) bit1=password, bit2=distribution(배포용/DRM).
    이런 HWP 는 본문 스트림이 암호화돼 변환기가 정상 처리 못 함. (근거: HWP 5.0 스펙)"""
    try:
        import olefile
        import struct

        if not olefile.isOleFile(file_path):
            return False
        ole = olefile.OleFileIO(file_path)
        try:
            if not ole.exists("FileHeader"):
                return False
            data = ole.openstream("FileHeader").read()
            if len(data) < 40 or data[:17] != b"HWP Document File":
                return False
            flags = struct.unpack("<I", data[36:40])[0]
            return bool(flags & 0x02) or bool(flags & 0x04)  # password or distribution(DRM)
        finally:
            ole.close()
    except Exception:
        return False


def _detect_unsupported_file(file_path: str) -> str | None:
    """입력 파일이 정상 처리 가능한지 판정(이슈 #278). 차단 사유 문자열 또는 정상이면 None.

    근거(공식):
    - 포맷 인식: 매직헤더 allowlist (file/file libmagic 정본 DB + 각 포맷 공식 스펙).
      _KNOWN_MAGIC_PREFIXES 위 주석에 출처 명시. 확장자와 무관하게 실제 바이트로 본다.
    - 암호화 자체는 바이트 패턴으로 못 본다(암호문=고엔트로피 랜덤). 포맷별 구조로 판정:
      PDF=/Encrypt(pypdf is_encrypted, ISO 32000), Office=OLE2의 EncryptedPackage(MS-OFFCRYPTO),
      HWP=FileHeader flags(HWP 5.0 스펙).
    - Fasoo 등 독점 DRM은 표준 감지법이 없다 → 알려진 매직헤더에 안 맞고 텍스트도 아닌
      바이너리(=고엔트로피 garbage)로 걸러낸다.
    """
    try:
        with open(file_path, "rb") as f:
            head = f.read(512)
    except Exception:
        return None  # 읽기 실패는 여기서 판단 안 함(후속 단계에서 처리)
    if not head:
        return "빈 파일"

    is_pdf = head.startswith(b"%PDF-")
    is_ole2 = head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    # ── Layer 1: 알려진 포맷 매직헤더인가 ──
    known = (
        is_pdf
        or is_ole2
        or head[4:8] == b"ftyp"  # mp4/mov/m4a (ISO-BMFF, offset 4)
        or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0)  # mp3 frame sync
        or any(head.startswith(sig) for sig in _KNOWN_MAGIC_PREFIXES)
    )
    if not known:
        if _looks_like_text(head):
            return None  # csv/txt/json/md/html 등 텍스트 파일
        return "지원하지 않거나 손상된 파일(DRM 암호화 등)"

    # ── Layer 2: 알려진 포맷이지만 비밀번호/암호화된 경우 ──
    if is_pdf and _is_encrypted_pdf(file_path):
        return "암호화된 PDF 문서"
    if is_ole2 and _is_encrypted_office(file_path):
        return "암호화된 Office 문서"
    if is_ole2 and _is_protected_hwp(file_path):
        return "암호화/배포용(DRM) HWP 문서"
    return None


# docling imports

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
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
    FormatOption
)
from docling.datamodel.pipeline_options import DataEnrichmentOptions
from docling.prompts.prompt_manager import LLMApiError
from docling.utils.document_enrichment import enrich_document, check_document
from docling.utils.llm_cache import (
    classify_error as _classify_error,
    current_context as _cache_current_context,
    log_summary as _log_cache_summary,
    reset_context as _reset_cache_context,
    resolve_context as _resolve_cache_context,
    set_context as _set_cache_context,
)
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
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

from collections import Counter
import re
import json
import time
import warnings
from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing_extensions import Self

try:
    from genon.preprocessor.facade.enrichment.custom_fields_enricher import CustomFieldsEnricher as _CustomFieldsEnricher
except ImportError:
    _CustomFieldsEnricher = None  # type: ignore[assignment,misc]
try:
    from genon.preprocessor.facade.enrichment.metadata_enricher import MetadataEnricher as _MetadataEnricher
except ImportError:
    _MetadataEnricher = None  # type: ignore[assignment,misc]

from genon.preprocessor.facade.enrichment.enrichment_config import EnrichmentConfig
from genon.preprocessor.facade.enrichment.page_description import (
    PageDescriptionOptions,
    collect_page_texts,
    describe_pages,
)
from genon.preprocessor.facade.enrichment.image_description import (
    ImageDescriptionOptions,
    ImageDescriptionEnricher,
    resolve_runtime_image_options,
)
from genon.preprocessor.facade.enrichment.table_description import (
    TableDescriptionOptions,
    TableDescriptionEnricher,
    TableDescriptionExtractor,
    refined_html_to_format,
    resolve_runtime_table_options,
)
from genon.preprocessor.facade.enrichment.doc_summary import (
    DocSummaryOptions,
    DocSummaryEnricher,
    resolve_runtime_doc_summary_options,
)
from genon.preprocessor.facade.enrichment.field_transforms import (
    DEFAULT_METADATA_FIELD_TRANSFORMS,
    apply_field_transforms,
    extract_metadata_from_document,
    serialize_metadata_value_for_output,
)

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

# HWP/HWPX 품질 복구(선택적). 모듈 로드 실패 시 None → 복구 미적용(기존 동작 유지).
try:
    from genon.preprocessor.converters.hwp_recovery import HwpQualityRecovery
except ImportError:
    HwpQualityRecovery = None


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


_MIN_CHUNK_SIZE = 1024


def _clamp_chunk_size(size: Optional[int]) -> Optional[int]:
    """chunk_size 가 0 초과이면서 _MIN_CHUNK_SIZE 미만이면 _MIN_CHUNK_SIZE 로 보정.
    0(=분할 안 함) 과 None 은 그대로 둔다."""
    if size is not None and 0 < size < _MIN_CHUNK_SIZE:
        _log.info(f"[chunk_size] {size} < {_MIN_CHUNK_SIZE} → {_MIN_CHUNK_SIZE} 로 보정")
        return _MIN_CHUNK_SIZE
    return size


def _parse_optional_float(value: Any, key: str = "") -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(f"[DocumentProcessor] Invalid float value for '{key}': {value!r}. Fallback to default.")
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


def _resolve_chunk_mode(kwargs: dict, yaml_default: str) -> str:
    """청킹 병합 모드 결정. 우선순위: 요청 chunk_mode > yaml > 'split_only'.

    chunk_mode 는 문자열('split_only'|'resize_all') 또는 0/1 플래그를 받는다.
    0(false/no/off) → 'split_only'(구조 경계 보존, 초과 그룹만 분할),
    1(true/yes/on)  → 'resize_all'(인접 섹션을 chunk_size 한도까지 greedy 병합).
    (chunk_mode=0 은 falsy 라 `x or default` 로는 무시되므로 `is not None` 으로 판별한다.)
    """
    raw = kwargs.get("chunk_mode")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in {"split_only", "resize_all"}:
            return s
        if s in {"1", "true", "yes", "on"}:
            return "resize_all"
        if s in {"0", "false", "no", "off"}:
            return "split_only"
    mode = str(yaml_default or "").strip().lower()
    return mode if mode in {"split_only", "resize_all"} else "split_only"


def _copy_enrichment_options(options, **updates):
    """DataEnrichmentOptions 를 얕게 복제하며 지정 필드를 override(원본 불변)."""
    try:
        return options.model_copy(update=updates)
    except AttributeError:
        import copy as _copy
        cloned = _copy.copy(options)
        for key, value in updates.items():
            setattr(cloned, key, value)
        return cloned


# #329: LLM 캐시 / error_policy 컨텍스트 해석은 docling.utils.llm_cache.resolve_context
# (3개 facade 공용, cross-facade import 회피)를 _resolve_cache_context 로 재노출해 사용한다.


def _handle_stage_error(exc: Exception, stage: str) -> None:
    """enrichment 단계 실패 처리(#329).

    - lenient(기본): 기존처럼 warning 후 계속(soft-fail, 하위호환).
    - strict: stage/error_type 를 실어 GenosServiceException 으로 재-raise(Temporal 경로).
    error_policy 는 요청 스코프 CacheContext 에서 읽는다(단일 소스).
    """
    error_type = _classify_error(exc)
    if _cache_current_context().error_policy == "strict":
        raise GenosServiceException(
            "1", f"[{stage}] {exc}", stage=stage, error_type=error_type
        ) from exc
    _log.warning(f"[DocumentProcessor] {stage} enrichment skipped ({error_type}): {exc}")


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


def _resolve_default_intelligent_config_path() -> str:
    base_dir = Path(__file__).resolve().parent
    local_config = (base_dir / "../resource_dev/intelligent_processor_config.yaml").resolve()
    default_config = (base_dir / "../resource/intelligent_processor_config.yaml").resolve()

    if local_config.exists():
        return str(local_config)
    return str(default_config)


# 청킹용 토크나이저 기본 경로 (config 미지정 시 현행 동작 유지)
_DEFAULT_TOKENIZER_LOCAL_PATH = "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
_DEFAULT_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"

# PDF 변환에서 제외(직접 처리)할 엑셀 계열 포맷(이슈 #288).
# PDF 변환 시 한 행이 페이지 경계로 쪼개지는 논리 오류가 생기므로 변환하지 않고 직접 처리한다.
_XLSX_DIRECT_EXTS = {".xlsx", ".xlsm", ".csv"}


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
    # 청킹 모드. "split_only"(기본)=chunk_size 초과 청크만 분할(구조 보존) | "resize_all"=모든 청크를 chunk_size 에 맞게 병합/분할
    chunk_mode: str = "split_only"

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
        for item, level in dl_doc.iterate_items(included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}, traverse_pictures=True):
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
                picture_text = self._extract_picture_annotation_text(item)
                if picture_text and picture_text not in text_parts:
                    text_parts.append(picture_text)

        result_text = self.delim.join(text_parts)
        return result_text

    @staticmethod
    def _extract_picture_annotation_text(item: PictureItem) -> str:
        """PictureItem annotation의 텍스트를 단일 문자열로 추출."""
        texts: list[str] = []
        for annotation in getattr(item, "annotations", []) or []:
            text = str(getattr(annotation, "text", "") or "").strip()
            if text:
                texts.append(text)
        if not texts:
            return ""
        # 동일 annotation 중복 주입 방지
        return "\n".join(dict.fromkeys(texts))

    @staticmethod
    def _resolve_table_format(kwargs: dict) -> str:
        """표 직렬화 형식 결정: table_format(html|markdown) 우선, 없으면 레거시 export_to_html(1/0)."""
        fmt = kwargs.get("table_format")
        if fmt is None:
            return "html" if kwargs.get("export_to_html", 1) == 1 else "markdown"
        fmt = str(fmt).strip().lower()
        return "markdown" if fmt == "markdown" else "html"

    @staticmethod
    def _resolve_compact_tables(kwargs: dict) -> bool:
        """markdown 표를 compact(컬럼 정렬 패딩 제거)로 낼지 결정. 기본 True."""
        return bool(kwargs.get("compact_tables", True))

    def _extract_table_text(self, table_item: TableItem, dl_doc: DoclingDocument, **kwargs) -> str:
        """테이블 청크 텍스트를 만든다.

        표 description(refine/요약) annotation 이 있으면 반영한다:
        - refine ON: 재구성 HTML 로 표 본체 교체
        - 요약 존재: '\\n---\\n[표 설명]\\n<요약>' 을 항상 병기
        annotation 이 없으면(기본) 기존 export 결과를 그대로 반환(회귀 없음).
        """
        refined_html = TableDescriptionExtractor.extract_refined_html(table_item)
        table_summary = TableDescriptionExtractor.extract_summary(table_item)

        # refine 은 항상 HTML 로 재구성 → output table_format 에 맞춰 변환(markdown 등).
        refined = refined_html_to_format(
            refined_html, self._resolve_table_format(kwargs), self._resolve_compact_tables(kwargs))
        base_text = refined or self._compute_table_base_text(table_item, dl_doc, **kwargs)
        if table_summary:
            if base_text:
                return base_text + "\n---\n[표 설명]\n" + table_summary
            return "[표 설명]\n" + table_summary
        return base_text

    def _compute_table_base_text(self, table_item: TableItem, dl_doc: DoclingDocument, **kwargs) -> str:
        """테이블에서 텍스트를 추출하는 일반화된 메서드"""
        try:
            if self._resolve_table_format(kwargs) == "markdown":
                if self._resolve_compact_tables(kwargs):
                    # TableItem.export_to_markdown() 은 compact 옵션이 없어 직접 serializer 구성
                    # (컬럼 정렬 패딩 제거 → 대형 표 markdown 크기 대폭 축소)
                    table_text = MarkdownDocSerializer(
                        doc=dl_doc,
                        params=MarkdownParams(compact_tables=True),
                    ).serialize(item=table_item).text
                else:
                    table_text = table_item.export_to_markdown(dl_doc)
            else:
                table_text = table_item.export_to_html(dl_doc)
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
    def _render_table_row_md(row: list, num_cols: int) -> str:
        """grid 한 행을 markdown 표 행 `| c1 | c2 | ... |` 로 렌더(파이프는 이스케이프).
        markdown 은 colspan/rowspan 미지원이라 num_cols 전 컬럼을 그대로 낸다."""
        cells = []
        for j in range(num_cols):
            text = (row[j].text or "").strip().replace("|", "\\|").replace("\n", " ")
            cells.append(text)
        return "| " + " | ".join(cells) + " |"

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

        # 재구성 HTML(refine)이 있으면 grid/구조가 달라 row 분할이 무의미 → 단일 청크로 둔다.
        if TableDescriptionExtractor.extract_refined_html(table_item):
            return [single]
        # 요약(summary)만 있는 경우: chunk_size 초과 표는 정상적으로 row 분할하고,
        # 요약은 마지막 분할 청크에만 1회 덧붙인다(중복 방지). single 경로는 이미 요약 포함.
        table_summary = TableDescriptionExtractor.extract_summary(table_item)

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

        # 세로 병합(row_span>1)이 데이터 행에 있으면 row 분할이 구조를 깨뜨리므로 분할하지 않는다.
        # (헤더 영역의 세로병합은 헤더 블록이 매 청크에 통째로 반복되므로 무해)
        if any(getattr(c, "row_span", 1) > 1 for r in data_rows for c in r):
            return [single]

        # heading 접두(_generate_section_text_with_heading 과 동일 규칙). xlsx 는 보통 공백.
        merged = {lvl: t for lvl, t in (h_short or {}).items() if t}
        heading = ", ".join(merged[l] for l in sorted(merged)) if merged else ""
        prefix = (heading + ", ") if heading else ""

        # table_format 에 맞춰 헤더/데이터 행을 렌더하고 버킷을 감싼다(html | markdown).
        if self._resolve_table_format(kwargs) == "markdown":
            render_row = self._render_table_row_md
            header_block = [render_row(r, num_cols) for r in header_rows]
            header_block.append("| " + " | ".join(["---"] * num_cols) + " |")

            def wrap(data_rendered: list) -> str:
                return sheet_prefix + prefix + "\n".join(header_block + data_rendered)
        else:
            render_row = self._render_table_row_html
            header_inner = "".join(render_row(r, num_cols) for r in header_rows)

            def wrap(data_rendered: list) -> str:
                return sheet_prefix + prefix + "<table><tbody>" + header_inner + "".join(data_rendered) + "</tbody></table>"

        texts: list[str] = []
        cur: list[str] = []
        for r in data_rows:
            rr = render_row(r, num_cols)
            if cur and self._count_tokens(wrap(cur + [rr])) > self.max_tokens:
                texts.append(wrap(cur))
                cur = [rr]
            else:
                cur.append(rr)
        if cur:
            texts.append(wrap(cur))
        if not texts:
            return [single]
        if table_summary:
            texts[-1] = texts[-1] + "\n---\n[표 설명]\n" + table_summary
        return texts

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
        # 2.5단계: 너무 긴 청크는 분할 (인덱스 꼬임 방지를 위해 새 리스트 사용)
        #   resize_all 전용. split_only 는 구조 그룹핑(4단계) 후 5.5단계에서 분할한다
        #   (여기서 분할하면 같은 섹션 조각들이 4단계에서 다시 병합되어 무의미).
        # ================================================================
        if self.max_tokens > 0 and self.chunk_mode == "resize_all":
            final_sections = []  # 결과를 담을 새 리스트
            for text, items, h_infos, h_short in sections_with_text:
                token_count = self._count_tokens(text)
                if token_count < self.max_tokens:
                    final_sections.append((text, items, h_infos, h_short))
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

                # 분할된 결과들을 새 리스트에 추가
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
                    final_sections.append((new_text, group_items, group_h_infos, group_h_short))

            sections_with_text = final_sections  # 전체 리스트 교체

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

            # split_only: base 섹션 granularity 유지 — 구조 그룹핑 병합 없이 섹션마다 분리(장 단위 병합 방지).
            #   (1·3단계로 만든 섹션을 그대로 두고, 초과분만 5.5단계에서 분할)
            if self.chunk_mode == "split_only" and len(merged_texts) > 0:
                b_new_chunk = True
            # 토큰 수 초과 시 새로운 청크 생성 (resize_all 전용)
            elif self.chunk_mode == "resize_all" and test_tokens > self.max_tokens and len(merged_texts) > 0:
                b_new_chunk = True
            # 현재 섹션헤더 레벨이 더 높으면 새로운 청크 생성 (resize_all 구조 경계)
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
        def _size(g):
            text = "\n".join(g["texts"])
            headings = self._extract_used_headers(g["h_short"]) or []
            header_line = ("HEADER: " + ", ".join(headings) + "\n") if headings else ""
            # char 모드면 문자 수, huggingface 모드면 토큰 수로 산정 (max_tokens 단위와 일치)
            return self._count_tokens(header_line + text)

        if self.max_tokens > 0 and groups and self.chunk_mode == "resize_all":
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
        # 5.5단계: split_only 전용 — chunk_size 초과 그룹만 토큰 기준 균등 분할
        #   (구조 기반 그룹은 유지, 작은 그룹은 병합하지 않고 그대로 둔다)
        # ================================================================
        if self.max_tokens > 0 and groups and self.chunk_mode == "split_only":
            new_groups = []
            for g in groups:
                if _size(g) <= self.max_tokens:
                    new_groups.append(g)
                    continue

                # caption 및 table 내 그림은 같은 조각에 있도록 조정 (2.5단계와 동일 로직)
                items_group = [[(it, inf, sh)] for it, inf, sh in zip(g["items"], g["h_infos"], g["h_short"])]
                items_group = adjust_captions(items_group)
                items_group = adjust_pictures_in_tables(items_group)

                item_token_counts = []
                for grp in items_group:
                    item_token_counts.append(sum(self._count_tokens(get_text_from_item(x[0])) for x in grp))

                for (a, b) in split_items_evenly_by_tokens(item_token_counts, self.max_tokens):
                    gi, gh, gs = [], [], []
                    for idx in range(a, b):
                        for x in items_group[idx]:
                            gi.append(x[0]); gh.append(x[1]); gs.append(x[2])
                    new_text = self._generate_section_text_with_heading(gi, gs, dl_doc, **kwargs)
                    new_groups.append({"texts": [new_text], "items": gi, "h_infos": gh, "h_short": gs})
            groups = new_groups

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
# 민감정보 분류/마스킹(#315)은 facade/guardrail 모듈로 분리 — gr.* 로 사용.
from genon.preprocessor.facade import guardrail as gr


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
    title: str = None
    created_date: int = None
    appendix: str = None ## !! appendix feature (2025-09-30, geonhee kim) !!
    file_path: Optional[str] = None
    guardrail_categories: Optional[list] = None  # #315 민감정보 분류 라벨(부동산/인사/민감 등). 미적용 시 None


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
        self.title: Optional[str] = None
        self.created_date: Optional[int] = None
        self.appendix: Optional[str] = None # !! appendix feature (2025-09-30, geonhee kim) !!
        self.file_path: Optional[str] = None
        self.guardrail_categories: Optional[list] = None  # #315 민감정보 분류 라벨
        self.extra_metadata: dict[str, Any] = {}

    def set_guardrail_categories(self, guardrail_categories: Optional[list]) -> "GenOSVectorMetaBuilder":
        """#315 청크 민감정보 분류 라벨 설정 (부동산/인사/민감 등 리스트, 미적용/없음 시 None)"""
        self.guardrail_categories = guardrail_categories or None
        return self

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
            if hasattr(self, key):
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
            "title": self.title,
            "created_date": self.created_date,
            "appendix": self.appendix or "", # !! appendix feature (2025-09-30, geonhee kim) !!
            "file_path": self.file_path,
            "guardrail_categories": self.guardrail_categories,  # #315 민감정보 분류 라벨
            **self.extra_metadata,
        }
        return GenOSVectorMeta.model_validate(payload)


class DocumentProcessor:

    def __init__(self, config_path: str | None = None):
        '''
        initialize Document Converter (config 기반)

        config_path 가 None 이면 resource_dev/intelligent_processor_config.yaml
        (없으면 resource/intelligent_processor_config.yaml) 을 사용한다.
        GenOS 는 DocumentProcessor() 무인자로 호출하므로 기본 경로 resolve 필수.
        '''
        if config_path is None:
            config_path = _resolve_default_intelligent_config_path()

        cfg = _load_config(config_path)
        self._config_dir = Path(config_path).resolve().parent
        # 런타임 kwargs 기본값(img_desc/chart_desc/chart_detection/doc_summary) 용도
        self._runtime_cfg = _as_dict(cfg.get("runtime"))

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

        # 청킹 모드: "split_only"(기본, chunk_size 초과 청크만 분할) | "resize_all"(모든 청크를 chunk_size 에 맞게 병합/분할)
        self._chunk_mode = str(chunking_cfg.get("chunk_mode", "split_only")).strip().lower()
        if self._chunk_mode not in {"split_only", "resize_all"}:
            _log.warning(f"[DocumentProcessor] Unknown chunking.chunk_mode '{self._chunk_mode}', fallback to 'split_only'.")
            self._chunk_mode = "split_only"

        # xlsx(엑셀) 직접 처리 설정(이슈 #288). formats.xlsx 아래에 둔다(포맷별 옵션 컨테이너).
        #   processing_mode: docling(기본)=MsExcel 백엔드로 DoclingDocument 후 기존 파이프라인 /
        #                    tabular=데이터 행마다 1벡터 + 컬럼 헤더→메타(병합셀 unmerge+ffill)
        #   tabular.{header_row, multi_table}: tabular 모드 전용 세부 옵션
        formats_cfg = _as_dict(cfg.get("formats"))
        xlsx_cfg = _as_dict(formats_cfg.get("xlsx"))
        tabular_cfg = _as_dict(xlsx_cfg.get("tabular"))
        xlsx_mode = str(xlsx_cfg.get("processing_mode", "docling")).strip().lower()
        if xlsx_mode not in {"docling", "tabular"}:
            _log.warning(
                f"[DocumentProcessor] Unknown formats.xlsx.processing_mode '{xlsx_mode}', fallback to 'docling'."
            )
            xlsx_mode = "docling"
        self._xlsx_cfg = {
            "processing_mode": xlsx_mode,
            "header_row": _parse_optional_int(tabular_cfg.get("header_row"), "formats.xlsx.tabular.header_row") or 0,
            "multi_table": bool(_parse_optional_bool(tabular_cfg.get("multi_table"), "formats.xlsx.tabular.multi_table")),
        }

        # 표 텍스트 직렬화 형식(청크 text 내 docling 표 표현). "html"(default) | "markdown".
        output_cfg = _as_dict(cfg.get("output"))
        table_format = str(output_cfg.get("table_format", "html")).strip().lower()
        if table_format not in {"html", "markdown"}:
            _log.warning(
                f"[DocumentProcessor] Unknown output.table_format '{table_format}', fallback to 'html'."
            )
            table_format = "html"
        self._table_format = table_format
        # markdown 표 compact(컬럼 정렬 패딩 제거) 여부. 기본 True. html 포맷엔 무관.
        self._compact_tables = bool(output_cfg.get("compact_tables", True))

        # OCR 엔드포인트는 ocr.paddle.ocr_endpoint 가 정식 위치.
        # 구버전 호환: ocr.ocr_endpoint(상위) / 최상위 ocr_endpoint 도 폴백으로 인식.
        paddle_cfg = _as_dict(ocr_cfg.get("paddle"))
        ocr_ep = (
            paddle_cfg.get("ocr_endpoint")
            or ocr_cfg.get("ocr_endpoint")
            or cfg.get("ocr_endpoint", "http://192.168.73.172:48080/ocr")
        )

        # OCR 수행 모드. "auto"(default)=휴리스틱 기반 재OCR / "force"=무조건 전체 OCR / "disable"=OCR 안 함
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

        # 민감정보 분류/마스킹(#315): GenOS 분류 워크플로우 접속 정보.
        # 기능 on/off 는 요청별 kwargs(guardrail_call), 마스킹 치환은 masking_enabled(config/kwargs).
        gm_cfg = _as_dict(cfg.get("guardrail"))
        self._guardrail_url = str(gm_cfg.get("url") or "").strip()
        self._guardrail_workflow_id = _parse_optional_int(gm_cfg.get("workflow_id"), "guardrail.workflow_id")
        self._guardrail_api_key = str(gm_cfg.get("api_key") or "").strip()
        gm_timeout = _parse_optional_int(gm_cfg.get("timeout"), "guardrail.timeout")
        self._guardrail_timeout = gm_timeout if gm_timeout and gm_timeout > 0 else 60
        self._guardrail_masking_enabled = bool(_parse_optional_bool(gm_cfg.get("masking_enabled"), "guardrail.masking_enabled"))

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

        # PPT 페이지 단위 image description(page-level) 옵션. 기존 PictureItem 단위 설명과 별개로,
        # "페이지 자체"를 렌더링해 설명한 텍스트를 페이지별 TextItem 으로 주입한다(PPT 원본 전용).
        # config 위치: formats.ppt.page_description. 공통 모듈(enrichment/page_description)로 파싱.
        ppt_fmt_cfg = _as_dict(formats_cfg.get("ppt"))
        page_img_cfg = _as_dict(ppt_fmt_cfg.get("page_description"))
        self._page_desc_options = PageDescriptionOptions.from_config(page_img_cfg, self._config_dir)

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
        # 페이지 단위 image description 도 페이지 렌더 이미지를 소스로 하므로 동일하게 강제한다.
        if self.table_image_enabled or self._page_desc_options.enabled:
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
            layout_timeout = 1200
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
        layout_length_fallback = _parse_optional_bool(
            genos_layout_cfg.get("length_fallback_enabled"),
            "layout.genos_layout.length_fallback_enabled",
        )
        if layout_length_fallback is None:
            layout_length_fallback = True
        layout_fallback_dpi = _parse_optional_int(
            genos_layout_cfg.get("fallback_dpi"), "layout.genos_layout.fallback_dpi"
        )
        if layout_fallback_dpi is None or layout_fallback_dpi <= 0:
            layout_fallback_dpi = 200
        layout_table_fallback = _parse_optional_bool(
            genos_layout_cfg.get("table_fallback_enabled"),
            "layout.genos_layout.table_fallback_enabled",
        )
        if layout_table_fallback is None:
            layout_table_fallback = True
        self.pipe_line_options.layout_options.genos_layout_options.model = layout_model
        self.pipe_line_options.layout_options.genos_layout_options.timeout = layout_timeout
        self.pipe_line_options.layout_options.genos_layout_options.retry_count = layout_retry_count
        self.pipe_line_options.layout_options.genos_layout_options.temperature = layout_temperature
        self.pipe_line_options.layout_options.genos_layout_options.top_p = layout_top_p
        self.pipe_line_options.layout_options.genos_layout_options.repetition_penalty = layout_repetition_penalty
        self.pipe_line_options.layout_options.genos_layout_options.length_fallback_enabled = layout_length_fallback
        self.pipe_line_options.layout_options.genos_layout_options.fallback_dpi = layout_fallback_dpi
        self.pipe_line_options.layout_options.genos_layout_options.table_fallback_enabled = layout_table_fallback

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

        # 이미지/차트 description 옵션. chart.enable 이면 변환 단계에서 그림 분류가 필요하므로
        # 컨버터(ocr 포함) 생성 전에 옵션을 결정하고 do_picture_classification 을 켜 둔다.
        self.image_description_options = ImageDescriptionOptions.from_config(
            image_desc_cfg=ec.image_description_cfg,
            fallback_api_url=ec.api_url,
            fallback_api_key=ec.api_key,
            fallback_model=ec.model,
            config_dir=self._config_dir,
        )
        # 런타임 kwargs 오버라이드의 기준(base) 옵션 보관
        self._base_image_description_options = self.image_description_options
        # chart.enable=true 이면 그림 분류를 켠다(런타임 chart_detection=auto 전환 허용).
        # 모델(ds4sd--DocumentFigureClassifier)은 빌드 시 /models 에 포함(docling-tools models download).
        if self.image_description_options.chart_enabled:
            try:
                self.pipe_line_options.do_picture_classification = True
            except Exception as exc:
                _log.warning(
                    f"[DocumentProcessor] do_picture_classification 설정 실패: {exc}"
                )

        # 표 description 옵션. VLM 이 표 영역을 crop 하려면 페이지 이미지가 필요하므로
        # base 옵션이 켜져 있으면 컨버터 생성 전에 generate_page_images 를 강제한다.
        self.table_description_options = TableDescriptionOptions.from_config(
            table_desc_cfg=ec.table_description_cfg,
            fallback_api_url=ec.api_url,
            fallback_api_key=ec.api_key,
            fallback_model=ec.model,
            config_dir=self._config_dir,
        )
        self._base_table_description_options = self.table_description_options
        if self.table_description_options.enabled:
            self.pipe_line_options.generate_page_images = True

        # 문서 본문요약(doc_summary) 옵션. image/table 이 공유하는 {{doc_summary}} 를 1회 계산.
        self.doc_summary_options = DocSummaryOptions.from_config(
            doc_summary_cfg=ec.doc_summary_cfg,
            fallback_api_url=ec.api_url,
            fallback_api_key=ec.api_key,
            fallback_model=ec.model,
            config_dir=self._config_dir,
        )
        self._base_doc_summary_options = self.doc_summary_options

        # ocr 파이프라인 옵션
        self.ocr_pipe_line_options = PdfPipelineOptions()
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True

        # 기본 컨버터들 생성
        self._create_converters()

        # HWP/HWPX 품질 복구(선택적). 모듈 미로드 시 None → 복구 미적용(기존 동작).
        self._hwp_recovery = (
            HwpQualityRecovery(reload_fn=self._load_document) if HwpQualityRecovery else None
        )

        self.image_description_enricher = ImageDescriptionEnricher(
            self.image_description_options
        )
        self.table_description_enricher = TableDescriptionEnricher(
            self.table_description_options
        )
        self.doc_summary_enricher = DocSummaryEnricher(self.doc_summary_options)
        self.custom_fields_enrichers: list = (
            [_CustomFieldsEnricher(**c) for c in ec.custom_fields_cfgs]
            if _CustomFieldsEnricher is not None
            else []
        )
        self.metadata_enricher = (
            _MetadataEnricher(
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
            if _MetadataEnricher is not None and ec.metadata.do_metadata and ec.metadata.has_custom_metadata
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
            # 커스텀 MetadataEnricher가 있으면 docling 내장 metadata 추출을 비활성화한다.
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
        # kwargs에서 save_images 값을 가져와서 옵션 업데이트
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)

        # save_images 옵션이 현재 설정과 다르면 컨버터 재생성
        if (self.simple_pipeline_options.save_images != save_images or
            getattr(self.simple_pipeline_options, 'include_wmf', False) != include_wmf):
            self.simple_pipeline_options.save_images = save_images
            self.simple_pipeline_options.include_wmf = include_wmf
            self._create_converters()

        try:
            conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def load_documents_with_docling_ocr(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        # kwargs에서 save_images 값을 가져와서 옵션 업데이트
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)

        # save_images 옵션이 현재 설정과 다르면 컨버터 재생성
        if (self.simple_pipeline_options.save_images != save_images or
            getattr(self.simple_pipeline_options, 'include_wmf', False) != include_wmf):
            self.simple_pipeline_options.save_images = save_images
            self.simple_pipeline_options.include_wmf = include_wmf
            self._create_converters()

        try:
            conv_result: ConversionResult = self.ocr_converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.ocr_second_converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        # chunk_size 우선순위: kwargs > yaml(chunking.chunk_size) > 0
        chunk_size = _parse_optional_int(kwargs.get('chunk_size'), 'chunk_size')
        if chunk_size is None:
            chunk_size = self._chunk_size
        chunk_size = _clamp_chunk_size(chunk_size)
        # chunk_mode 우선순위: kwargs > yaml(chunking.chunk_mode) > "split_only"
        # chunk_mode(0/1 또는 'split_only'/'resize_all') > yaml > "split_only"
        chunk_mode = _resolve_chunk_mode(kwargs, self._chunk_mode)
        chunker: GenosSmartChunker = GenosSmartChunker(
            max_tokens = chunk_size if chunk_size is not None else 0,
            merge_peers = True,
            tokenizer = self._tokenizer,
            tokenizer_type = self._tokenizer_type,
            chunk_mode = chunk_mode,
        )

        # 표 직렬화 형식(html|markdown)을 청커로 전달(런타임 kwarg 가 있으면 우선).
        kwargs.setdefault("table_format", self._table_format)
        kwargs.setdefault("compact_tables", self._compact_tables)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def split_documents_by_page(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        """PPT 전용 페이지 기반 청킹.

        기본 1 page = 1 chunk. chunk_size(kwargs > yaml) 가 주어지면 연속 페이지를 토큰 기준
        chunk_size 이하가 되도록 greedy 병합한다. 같은 페이지의 native text 와 주입된 page
        description TextItem 은 prov.page_no 로 동일 페이지 청크에 자연히 묶인다.
        """
        chunk_size = _parse_optional_int(kwargs.get('chunk_size'), 'chunk_size')
        if chunk_size is None:
            chunk_size = self._chunk_size
        chunk_size = _clamp_chunk_size(chunk_size)
        # chunk_mode(0/1 또는 'split_only'/'resize_all') > yaml > "split_only"
        chunk_mode = _resolve_chunk_mode(kwargs, self._chunk_mode)
        chunker: GenosSmartChunker = GenosSmartChunker(
            max_tokens=chunk_size if chunk_size is not None else 0,
            merge_peers=True,
            tokenizer=self._tokenizer,
            tokenizer_type=self._tokenizer_type,
            chunk_mode=chunk_mode,
        )
        kwargs.setdefault("table_format", self._table_format)
        kwargs.setdefault("compact_tables", self._compact_tables)

        # 전체 아이템 base chunk(정상 경로와 동일한 아이템 수집/헤더/누락표 복구 재사용)
        base = next(iter(chunker.preprocess(dl_doc=documents, **kwargs)), None)
        if base is None:
            return []
        items = base.meta.doc_items
        header_short = getattr(base, "_header_short_info_list", []) or []

        # prov page_no 로 그룹(아이템 순서 유지). prov 없으면 직전 페이지에 귀속.
        page_items: dict = {}
        page_headers: dict = {}
        last_page = 1
        for idx, it in enumerate(items):
            prov = getattr(it, "prov", None) or []
            pg = prov[0].page_no if prov and getattr(prov[0], "page_no", None) else last_page
            last_page = pg
            page_items.setdefault(pg, []).append(it)
            page_headers.setdefault(pg, []).append(
                header_short[idx] if idx < len(header_short) else {}
            )

        # 페이지별 1 청크 직렬화
        page_chunks: List[DocChunk] = []
        for pg in sorted(page_items.keys()):
            its = page_items[pg]
            text = chunker._generate_section_text_with_heading(
                its, page_headers[pg], documents, **kwargs
            )
            if text and text.strip() and text.strip() != ".":
                page_chunks.append(DocChunk(
                    text=text,
                    meta=DocMeta(doc_items=its, headings=None, captions=None, origin=documents.origin),
                ))

        # chunk_size>0 이면 연속 페이지 greedy 병합 (split_only 는 1 page = 1 chunk 유지)
        if chunk_mode == "resize_all" and chunk_size and chunk_size > 0 and page_chunks:
            merged: List[DocChunk] = [page_chunks[0]]
            for ch in page_chunks[1:]:
                cand_text = merged[-1].text + "\n" + ch.text
                if chunker._count_tokens(cand_text) <= chunk_size:
                    merged[-1] = DocChunk(
                        text=cand_text,
                        meta=DocMeta(
                            doc_items=merged[-1].meta.doc_items + ch.meta.doc_items,
                            headings=None, captions=None, origin=documents.origin,
                        ),
                    )
                else:
                    merged.append(ch)
            page_chunks = merged

        for ch in page_chunks:
            if ch.meta.doc_items and ch.meta.doc_items[0].prov:
                self.page_chunk_counts[ch.meta.doc_items[0].prov[0].page_no] += 1
        _log.info(f"[ppt] page-based chunks: {len(page_chunks)} (chunk_size={chunk_size})")
        return page_chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def enrichment(self, document: DoclingDocument, is_ppt: bool = False, **kwargs: dict) -> DoclingDocument:
        options = self.enrichment_options
        # 런타임 toc(0/1) — config 기본값(do_toc_enrichment)을 요청별로 켜고/끈다.
        # 활성화(0→1)는 TOC endpoint 가 config 에 구성된 경우에만 유효(미구성 시 무시).
        cur_toc = bool(getattr(options, "do_toc_enrichment", False))
        want_toc = bool(_as_int_flag(kwargs.get("toc"), 1 if cur_toc else 0))
        if want_toc != cur_toc:
            if want_toc and not str(getattr(options, "toc_api_base_url", "") or ""):
                _log.warning("[intelligent] toc=1 요청이지만 TOC endpoint 미구성 → 무시")
            else:
                options = _copy_enrichment_options(options, do_toc_enrichment=want_toc)
                _log.info("[intelligent] runtime toc override → %s", want_toc)
        # PPT 는 페이지 기반 1chunk 라 목차 계층이 무의미 → TOC 만 비활성(다른 enrichment 는 유지).
        if is_ppt and getattr(options, "do_toc_enrichment", False):
            options = _copy_enrichment_options(options, do_toc_enrichment=False)
            _log.info("[intelligent] PPT — TOC enrichment skip")
        try:
            # 새로운 enriched result 받기
            document = enrich_document(document, options, **kwargs)
            return document
        except LLMApiError as e:
            # Preserve provider error payload as-is for load status error message.
            # #329: 기존 hard-fail 동작 유지 + stage/error_type 스탬프(4xx→permanent, 5xx→transient).
            raise GenosServiceException(
                "1", e.raw_error_message, stage="enrichment", error_type=_classify_error(e)
            ) from e

    def _normalize_runtime_kwargs(self, kwargs: dict) -> dict:
        """이미지/차트 description 런타임 토글을 정규화한다(전부 0/1 플래그).

        img_desc          : 이미지 description 사용유무          → image_description.enable
        chart_desc        : 차트 description 사용유무            → chart.enable (chart_convert alias)
        chart_detection   : 1=auto(docling 자동판별)/0=all       → chart.detection
        doc_summary       : 문서 본문요약 사용유무               → body_summary.enable
        미지정 kwarg 는 config(runtime 섹션 또는 base 옵션) 기본값을 따른다.
        """
        normalized = dict(kwargs or {})
        runtime = self._runtime_cfg
        base = getattr(self, "_base_image_description_options", None)

        img_default = _as_int_flag(
            runtime.get("img_desc"), 1 if (base and base.enabled) else 0
        )
        chart_default = _as_int_flag(
            runtime.get("chart_desc", runtime.get("chart_convert")),
            1 if (base and base.chart_enabled) else 0,
        )
        detection_default = _as_int_flag(
            runtime.get("chart_detection"),
            1 if (base and base.chart_detection == "auto") else 0,
        )
        dbase = getattr(self, "_base_doc_summary_options", None)
        summary_default = _as_int_flag(
            runtime.get("doc_summary"),
            1 if (dbase and dbase.enabled) else 0,
        )

        normalized["img_desc"] = _as_int_flag(normalized.get("img_desc"), img_default)
        normalized["chart_desc"] = _as_int_flag(
            normalized.get("chart_desc", normalized.get("chart_convert")), chart_default
        )
        normalized["chart_detection"] = _as_int_flag(
            normalized.get("chart_detection"), detection_default
        )
        normalized["doc_summary"] = _as_int_flag(
            normalized.get("doc_summary"), summary_default
        )

        # 표 description 런타임 토글(table_desc→enable, table_refine→refine.enable)
        tbase = getattr(self, "_base_table_description_options", None)
        table_default = _as_int_flag(
            runtime.get("table_desc"), 1 if (tbase and tbase.enabled) else 0
        )
        refine_default = _as_int_flag(
            runtime.get("table_refine"), 1 if (tbase and tbase.refine_enabled) else 0
        )
        normalized["table_desc"] = _as_int_flag(normalized.get("table_desc"), table_default)
        normalized["table_refine"] = _as_int_flag(normalized.get("table_refine"), refine_default)

        # TOC 런타임 토글(toc/toc_on alias) — 기본값은 config 의 do_toc_enrichment.
        toc_default = _as_int_flag(
            runtime.get("toc", runtime.get("toc_on")),
            1 if getattr(self.enrichment_options, "do_toc_enrichment", False) else 0,
        )
        normalized["toc"] = _as_int_flag(
            normalized.get("toc", normalized.get("toc_on")), toc_default
        )
        # merge_sections 별칭은 도입하지 않는다 — 기존 chunk_mode kwarg 가 동일 기능이며
        # split_documents 의 _resolve_chunk_mode() 가 chunk_mode 0/1/문자열을 직접 해석한다.
        return normalized

    def _configure_runtime_image_mode(self, kwargs: dict):
        """정규화된 kwargs 로 image_description_options/enricher 를 재구성한다.

        순수 override 계산은 enrichment.image_description.resolve_runtime_image_options 에 위임.
        """
        doc_summary = _as_int_flag(kwargs.get("doc_summary"), 0)

        # image description 런타임 재구성 (image base 옵션이 있을 때만)
        base = getattr(self, "_base_image_description_options", None)
        if base is not None:
            img_desc = _as_int_flag(kwargs.get("img_desc"), 0)
            chart_desc = _as_int_flag(kwargs.get("chart_desc"), 0)
            chart_detection = _as_int_flag(kwargs.get("chart_detection"), 0)
            self.image_description_options = resolve_runtime_image_options(
                base,
                img_desc=img_desc,
                chart_desc=chart_desc,
                chart_detection=chart_detection,
                classification_available=getattr(
                    self.pipe_line_options, "do_picture_classification", False
                ),
            )
            self.image_description_enricher = ImageDescriptionEnricher(
                self.image_description_options
            )
            _log.info(
                "[runtime_feature] image mode enabled=%s img_desc=%s chart_desc=%s detection=%s",
                self.image_description_options.enabled,
                img_desc,
                chart_desc,
                self.image_description_options.chart_detection,
            )

        # 표 description 런타임 재구성 (image base 유무와 무관하게 독립 실행)
        tbase = getattr(self, "_base_table_description_options", None)
        if tbase is not None:
            table_desc = _as_int_flag(kwargs.get("table_desc"), 0)
            table_refine = _as_int_flag(kwargs.get("table_refine"), 0)
            self.table_description_options = resolve_runtime_table_options(
                tbase,
                table_desc=table_desc,
                table_refine=table_refine,
            )
            self.table_description_enricher = TableDescriptionEnricher(
                self.table_description_options
            )
            _log.info(
                "[runtime_feature] table mode enabled=%s table_desc=%s table_refine=%s",
                self.table_description_options.enabled,
                table_desc,
                table_refine,
            )

        # doc_summary 런타임 재구성(image/table 공통 컨텍스트 제공)
        dbase = getattr(self, "_base_doc_summary_options", None)
        if dbase is not None:
            self.doc_summary_options = resolve_runtime_doc_summary_options(
                dbase, doc_summary=doc_summary
            )
            self.doc_summary_enricher = DocSummaryEnricher(self.doc_summary_options)
            _log.info(
                "[runtime_feature] doc_summary mode enabled=%s doc_summary=%s",
                self.doc_summary_options.enabled,
                doc_summary,
            )

    def _get_or_create_image_description_enricher(self):
        enricher = getattr(self, "image_description_enricher", None)
        if enricher is None:
            # 테스트 등에서 __init__ 우회 시 legacy attribute 기반으로 재구성
            legacy_options = ImageDescriptionOptions.from_legacy_processor(self)
            enricher = ImageDescriptionEnricher(legacy_options)
            self.image_description_enricher = enricher
        return enricher

    def enrich_image_descriptions(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = self._get_or_create_image_description_enricher()
        if enricher is None:
            return document
        return enricher.enrich(document, **kwargs)

    def _get_or_create_doc_summary_enricher(self):
        enricher = getattr(self, "doc_summary_enricher", None)
        if enricher is None:
            base = getattr(self, "_base_doc_summary_options", None)
            enricher = DocSummaryEnricher(base or DocSummaryOptions())
            self.doc_summary_enricher = enricher
        return enricher

    def enrich_doc_summary(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = self._get_or_create_doc_summary_enricher()
        if enricher is None:
            return document
        return enricher.enrich(document, **kwargs)

    def _get_or_create_table_description_enricher(self):
        enricher = getattr(self, "table_description_enricher", None)
        if enricher is None:
            base = getattr(self, "_base_table_description_options", None)
            enricher = TableDescriptionEnricher(base or TableDescriptionOptions())
            self.table_description_enricher = enricher
        return enricher

    def enrich_table_descriptions(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = self._get_or_create_table_description_enricher()
        if enricher is None:
            return document
        return enricher.enrich(document, **kwargs)

    def enrich_page_descriptions(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        """페이지 단위 image description: 각 페이지를 렌더링해 설명한 텍스트를 페이지별
        TextItem 으로 주입한다(기존 PictureItem 단위 설명과 별개, 옵션 default False).
        """
        if not self._page_desc_options.enabled:
            return document

        # 페이지별 native text 수집(설명 주입 전) → 프롬프트({{page_text}})에 반영해 요청
        page_texts = collect_page_texts(document)
        page_descs = describe_pages(document, self._page_desc_options, page_texts=page_texts)
        if not page_descs:
            return document

        for page_no in sorted(page_descs.keys()):
            text = page_descs[page_no].strip()
            if not text:
                continue
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),
                charspan=(0, len(text)),
            )
            document.add_text(label=DocItemLabel.TEXT, text=text, prov=prov)
        _log.info(f"[page_image_description] 페이지 설명 주입: pages={len(page_descs)}")
        return document

    async def enrich_metadata(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = getattr(self, "metadata_enricher", None)
        if enricher is not None:
            document = await enricher.enrich(document, **kwargs)
        return document

    async def enrich_custom_fields(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        for enricher in self.custom_fields_enrichers:
            document = await enricher.enrich(document, **kwargs)
        return document

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request, converted_pdf_path: Optional[str] = None, **kwargs: dict) -> \
            list[dict]:
        title = ""
        _sensitive_infos: list = kwargs.get("_sensitive_infos") or []      # #315 분류 결과
        _gr_masking: bool = bool(kwargs.get("_guardrail_masking", False))   # #315 마스킹 치환 on/off
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

        # kwargs에서 부록 정보 추출 !! appendix feature (2025-09-30, geonhee kim) !!
        appendix_info = kwargs.get('appendix', '')
        appendix_list = []
        if isinstance(appendix_info, str):
            if appendix_info:
                try:
                    parsed = json.loads(appendix_info)
                    if isinstance(parsed, list):
                        appendix_list = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
                    elif isinstance(parsed, str):
                        appendix_list = [parsed.strip()] if parsed.strip() else []
                    else:
                        appendix_list = []
                except json.JSONDecodeError:
                    appendix_list = [appendix_info.strip()] if appendix_info.strip() else []
            else:
                appendix_list = []
        elif isinstance(appendix_info, list):
            appendix_list = appendix_info
        else:
            appendix_list = []

        passthrough_metadata = dict(merged_metadata)
        # GenOSVectorMeta 스키마 예약 필드 + transform 이 소비한 source/target 키는 passthrough 제외.
        reserved_keys = {
            "text", "n_char", "n_word", "n_line", "e_page", "i_page",
            "i_chunk_on_page", "n_chunk_of_page", "i_chunk_on_doc", "n_chunk_of_doc",
            "n_page", "reg_date", "chunk_bboxes", "media_files", "title",
            "created_date", "appendix", "file_path", "metadata", "guardrail_categories",
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
        # 비-PDF 입력이 변환된 경우 vector 의 file_path 를 변환 PDF 경로로 set.
        if converted_pdf_path:
            global_metadata['file_path'] = converted_pdf_path

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
            # header 앞에 헤더 마커 추가 (HEADER: )
            headers_text = "HEADER: " + ", ".join(chunk.meta.headings) + '\n' if chunk.meta.headings else ''
            content = headers_text + chunk.text

            # appendix 추출 !! appendix feature (2025-09-30, geonhee kim) !!
            matched_appendices = self.check_appendix_keywords(content, appendix_list)
            # print(appendix_list, matched_appendices)
            chunk_global_metadata = global_metadata.copy()
            chunk_global_metadata['appendix'] = matched_appendices  # Only matched ones
            ###

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            # #315 가드레일 분류 후처리: quote 매칭 → guardrail_categories 부착(항상) + 마스킹 치환(옵션)
            content, chunk_cats = gr.apply_to_text(content, _sensitive_infos, _gr_masking)

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**chunk_global_metadata) #!! appendix feature (2025-09-30, geonhee kim) !!
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items, include_tables=self.table_image_enabled)
                      .set_guardrail_categories(sorted(chunk_cats) if chunk_cats else None)
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

    def check_empty_text(self, document: DoclingDocument) -> bool:
        """텍스트 클러스터(박스)는 있는데 그 텍스트가 전부 비어 있는 페이지가 있는지 확인.

        length 폴백(layout_only)이나 텍스트레이어 부재 등으로 박스만 있고 텍스트가
        안 채워진 페이지를 잡아 강제 OCR 로 보낸다(이슈 #278 B-2).
        """
        from collections import defaultdict
        page_item_count: dict = defaultdict(int)
        page_text_len: dict = defaultdict(int)
        for item, _level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                page_no = item.prov[0].page_no
                page_item_count[page_no] += 1
                page_text_len[page_no] += len((item.text or "").strip())
        for page_no, n_items in page_item_count.items():
            # 텍스트 아이템이 있는데 그 페이지 텍스트 총량이 0 → 비어있는 페이지
            if n_items > 0 and page_text_len[page_no] == 0:
                _log.info(f"[intelligent] page {page_no} 텍스트가 비어있음 → 강제 OCR 필요")
                return True
        return False

    def check_appendix_keywords(self, content: str, appendix_list: list) -> str: # !! appendix feature (2025-09-30, geonhee kim) !!
        if not content or not appendix_list:
            return ""

        matched_appendices = []

        # 1. Find appendix patterns in content first
        found_patterns = []

        # Complex patterns: 별지/별표/장부 + numbers (with hyphens, Roman numerals)
        # Updated regex to capture full patterns like "별지 제 Ⅰ -1 호 서식" by matching until closing delimiters
        content = re.sub(r"\s+", "", content)
        complex_patterns = re.findall(r'(별지|별표|장부)(?:제)?([^<>()\[\]]+?)(?=(?:호|서식)|[<>\)\]]|$)', content)
        for pattern_type, number in complex_patterns:
            found_patterns.extend([
                f"{pattern_type} {number}",
                f"{pattern_type} 제{number}호",
                f"{pattern_type}{number}",
                f"{pattern_type}제{number}호"
            ])

        # Standalone patterns: (별표), (별지), (장부)
        standalone_patterns = re.findall(r'[\(\[]+(별지|별표|장부)[\)\]]+', content)
        for pattern_type in set(standalone_patterns):
            found_patterns.extend([
                pattern_type,
                f"{pattern_type}",
            ])

        # 2. Check if found patterns match any appendix in the list
        for appendix in appendix_list:
            if not appendix or not isinstance(appendix, str):
                continue

            appendix_clean = appendix.replace('.pdf', '').lower().strip()
            appendix_clean_no_space = re.sub(r"\s+", "", appendix_clean)

            # If any found pattern exists in appendix filename, it's a match
            for pattern in found_patterns:
                pattern_no_space = re.sub(r"\s+", "", pattern).lower()
                if pattern_no_space in appendix_clean_no_space:
                    matched_appendices.append(appendix)
                    break  # Prevent duplicates

        return ', '.join(matched_appendices) if matched_appendices else ""

    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> List[Dict[str, Any]]:
        """
        글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR을 수행합니다.
        Args:
            document: DoclingDocument 객체
            pdf_path: PDF 파일 경로
        Returns:
            OCR이 완료된 문서의 DoclingDocument 객체
        """
        import io
        import base64
        import requests
        from PIL import Image

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
            for table_idx, table_item in enumerate(document.tables):
                if not table_item.data or not table_item.data.table_cells:
                    continue
                if not table_item.prov:
                    continue

                b_ocr = False
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if self.check_glyph_text(cell.text, threshold=self._glyph_table_cell_threshold):
                        b_ocr = True
                        break

                if b_ocr is False:
                    # 글리프 깨진 텍스트가 없는 경우, OCR을 수행하지 않음
                    continue

                # docling 이 이미 렌더해 둔 페이지 이미지(generate_page_images=True)를
                # 재사용해 셀 영역을 crop 한다. PyMuPDF 재렌더(get_pixmap)는 일부 PDF 에서
                # 네이티브 크래시(SIGSEGV, worker code 139)를 유발하므로 사용하지 않는다.
                page_no = table_item.prov[0].page_no
                page = document.pages.get(page_no)
                if page is None or page.size is None or page.image is None:
                    continue
                page_image = page.image.pil_image
                if page_image is None:
                    continue
                W, H = page_image.size

                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    try:
                        if cell.bbox is None:
                            continue

                        # docling 셀 bbox(BOTTOMLEFT) → 페이지 이미지 픽셀 좌표(TOPLEFT)
                        crop = (
                            cell.bbox
                            .to_top_left_origin(page_height=page.size.height)
                            .scale_to_size(old_size=page.size, new_size=page.image.size)
                        )
                        x0, y0, x1, y1 = crop.as_tuple()
                        # 정규화 + 페이지 경계 클램프 + degenerate skip
                        x0, x1 = sorted((x0, x1))
                        y0, y1 = sorted((y0, y1))
                        x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
                        y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
                        if (x1 - x0) < 1 or (y1 - y0) < 1:
                            continue

                        cell_img = page_image.crop((x0, y0, x1, y1))

                        # 아주 작은 셀은 OCR 가독성을 위해 확대(기존 target_height=20, ≤4x)
                        ch = y1 - y0
                        zoom = min(max(20.0 / ch, 1.0), 4.0) if ch > 0 else 1.0
                        if zoom > 1.0:
                            cell_img = cell_img.resize(
                                (max(1, round((x1 - x0) * zoom)), max(1, round(ch * zoom))),
                                Image.LANCZOS,
                            )

                        buf = io.BytesIO()
                        cell_img.save(buf, format="PNG")
                        img_data = buf.getvalue()

                        result = post_ocr_bytes(img_data, timeout=self._table_cell_ocr_timeout)
                        rec_texts, rec_scores, rec_boxes = extract_ocr_fields(result)

                        cell.text = ""
                        for t in rec_texts:
                            if len(cell.text) > 0:
                                cell.text += " "
                            cell.text += t if t else ""
                    except Exception as cell_err:
                        # 한 셀 실패가 나머지 셀/표를 막지 않도록 격리
                        print(f"OCR cell processing failed (table={table_idx}, cell={cell_idx}): {cell_err}")
                        continue
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

    def _convert_to_pdf(self, file_path: str, **kwargs: dict) -> tuple[str, str]:
        """비-PDF 입력을 PDF SDK/LibreOffice 로 변환. (변환된 file_path, converted_pdf_path) 반환."""
        # 변환 backend(pdf_sdk/rhwp/libreoffice)가 전무하면(이슈 #286 — 빌드 시 OFF)
        # 변환 시도 자체가 무의미하므로, PDF 직접 입력을 안내하며 즉시 중단한다.
        if not _has_any_pdf_converter():
            raise GenosServiceException(
                1,
                f"이 전처리기 이미지에는 PDF 변환기(rhwp/LibreOffice/PDF SDK)가 설치되어 "
                f"있지 않아 '{os.path.basename(file_path)}' 를 PDF 로 변환할 수 없습니다. "
                f"PDF 로 변환한 파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 "
                f"빌드하세요 (genon/README.md 참고).",
            )
        _log.info(f"[intelligent] Non-PDF input — auto-converting to PDF: {file_path}")
        use_sdk = kwargs.get('use_pdf_sdk', True)
        converted = convert_to_pdf(file_path, use_pdf_sdk=use_sdk)
        if (not converted or not os.path.exists(converted)) and use_sdk:
            _log.warning(f"[intelligent] SDK conversion failed → fallback to LibreOffice")
            converted = convert_to_pdf(file_path, use_pdf_sdk=False)
        if not converted or not os.path.exists(converted):
            raise GenosServiceException(1, f"PDF 변환 실패: {file_path}")
        _log.info(f"[intelligent] Converted PDF: {converted}")
        return converted, converted

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
            _log.info(f"[intelligent] xlsx tabular 직접 처리: {file_path}")
            vectors = build_tabular_vectors(
                file_path,
                header_row=self._xlsx_cfg["header_row"],
                multi_table=self._xlsx_cfg["multi_table"],
            )
            if not vectors:
                raise GenosServiceException(1, f"chunk length is 0")
            return vectors

        _log.info(f"[intelligent] xlsx docling 직접 처리(PDF 변환 생략): {file_path}")
        try:
            document = build_docling_document(
                file_path, save_images=kwargs.get('save_images', False)
            )
        except Exception as e:
            raise GenosServiceException(
                1, f"xlsx 처리 실패: {os.path.basename(file_path)} ({e})"
            )
        # openpyxl 텍스트라 글리프 깨짐이 없고 렌더 PDF 도 없으므로 테이블셀 재OCR 은 생략.
        # table_as_chunk=True: 시트/표마다 별도 청크로 분리(엑셀은 표 단위가 논리 단위).
        return await self._document_to_vectors(
            document, file_path, request,
            converted_pdf_path=None, ocr_table_cells=False, table_as_chunk=True, **kwargs
        )

    async def _process_pdf(self, request: Request, file_path: str,
                           converted_pdf_path: Optional[str], is_ppt: bool = False,
                           source_file_path: Optional[str] = None, **kwargs: dict):
        """PDF(또는 PDF 로 변환된) 입력을 docling 으로 로딩 후 공유 파이프라인으로 처리."""
        document = self._load_document(file_path, **kwargs)

        # HWP/HWPX 품질 복구(선택적): PDF 변환이 내용을 잃으면(text score 낮음) rhwp 재변환
        # 재시도 → 개선되면 교체. HWPX 는 네이티브 XML 추출로 추가 폴백. hwp_recovery 모듈이
        # 로드된 경우에만 적용하고, 없으면 로딩된 document 를 그대로 통과(기존 동작).
        # source_file_path 는 변환 전 원본(.hwp/.hwpx). 정상 문서는 score≥임계값 이라 미진입.
        if source_file_path is None:
            source_file_path = file_path
        if self._hwp_recovery is not None:
            file_path, converted_pdf_path, document = self._hwp_recovery.recover(
                document, source_file_path, file_path, converted_pdf_path, **kwargs
            )

        return await self._document_to_vectors(
            document, file_path, request,
            converted_pdf_path=converted_pdf_path, ocr_table_cells=True, is_ppt=is_ppt, **kwargs
        )

    def _load_document(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        """ocr_mode 에 따라 docling 문서를 로딩한다.
        "force"=무조건 전체 OCR / "auto"=휴리스틱 기반 재OCR / "disable"=OCR 안 함
        """
        if self.ocr_mode == "force":
            return self.load_documents_with_docling_ocr(file_path, **kwargs)
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        if self.ocr_mode == "auto":
            if not check_document(document, self.enrichment_options) or self.check_glyphs(document) or self.check_empty_text(document):
                # OCR이 필요하다고 판단되면 OCR 수행
                document = self.load_documents_with_docling_ocr(file_path, **kwargs)
        return document

    async def _document_to_vectors(self, document: DoclingDocument, file_path: str,
                                   request: Request, *, converted_pdf_path: Optional[str],
                                   ocr_table_cells: bool, is_ppt: bool = False, **kwargs: dict) -> list:
        """DoclingDocument → enrichment → 청킹 → 벡터 생성(공유 파이프라인).

        ocr_table_cells: 글리프 깨진 테이블 셀 재OCR 수행 여부(xlsx 직접 처리는 False).
        """
        # 글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR 수행 (청크토큰 8k이상 발생 방지)
        if ocr_table_cells and self.ocr_mode != "disable" and self.ocr_endpoint:
            document = self.ocr_all_table_cells(document, file_path)

        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(output_path) / filename  # 빈 output_path 가 절대경로(/filename)로 바뀌는 것 방지
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent

        document = document._with_pictures_refs(image_dir=artifacts_dir, page_no=None, reference_path=reference_path)

        # 표 이미지 저장 옵션이 켜진 경우, picture 와 동일하게 표 영역을 PNG 로 저장하고
        # TableItem.image.uri 를 설정한다(_with_pictures_refs 미러).
        if self.table_image_enabled:
            self._save_table_images(document, image_dir=artifacts_dir, reference_path=reference_path)

        document = self.enrichment(document, is_ppt=is_ppt, **kwargs)

        enrichment_context = kwargs.get("_enrichment_context", {})
        if not isinstance(enrichment_context, dict):
            enrichment_context = {}
        enrichment_kwargs = dict(kwargs)
        enrichment_kwargs["_enrichment_context"] = enrichment_context
        # #329: error_policy=strict 이면 _handle_stage_error 가 GenosServiceException 으로
        # 재-raise(삼키지 않음). lenient(기본)은 기존처럼 warning 후 계속.
        try:
            document = self.enrich_doc_summary(document, **enrichment_kwargs)
        except Exception as exc:
            _handle_stage_error(exc, "doc_summary")
        try:
            document = self.enrich_image_descriptions(document, **enrichment_kwargs)
        except Exception as exc:
            _handle_stage_error(exc, "image_description")
        try:
            document = self.enrich_table_descriptions(document, **enrichment_kwargs)
        except Exception as exc:
            _handle_stage_error(exc, "table_description")
        # 페이지 단위 image description 은 PPT 원본에만 적용(formats.ppt.page_description).
        if is_ppt:
            try:
                document = self.enrich_page_descriptions(document, **enrichment_kwargs)
            except Exception as exc:
                _handle_stage_error(exc, "page_description")
        try:
            document = await self.enrich_metadata(document, **enrichment_kwargs)
        except Exception as exc:
            _handle_stage_error(exc, "metadata")
        try:
            document = await self.enrich_custom_fields(document, **enrichment_kwargs)
        except Exception as exc:
            _handle_stage_error(exc, "custom_fields")

        # 민감정보 분류(#315): 청킹 전, 문서 전체를 분류 워크플로우에 1회 호출 → sensitive_infos.
        # 실제 라벨 부착/마스킹 치환은 청킹 후 compose 에서 quote 매칭으로 수행.
        sensitive_infos: list = []
        if gr.call_enabled(kwargs):
            sensitive_infos = gr.classify_document(
                gr.doc_text(document), self._guardrail_url, self._guardrail_workflow_id,
                self._guardrail_api_key, self._guardrail_timeout,
            )

        has_text_items = False
        for item, _ in document.iterate_items():
            if (isinstance(item, (TextItem, ListItem, CodeItem, SectionHeaderItem)) and item.text and item.text.strip()) or (isinstance(item, TableItem) and item.data and len(item.data.table_cells) == 0):
                has_text_items = True
                break

        if has_text_items:
            # Extract Chunk from DoclingDocument.
            # PPT 는 페이지 기반 청킹(기본 1 page 1 chunk, chunk_size 지정 시 페이지 결합).
            if is_ppt:
                chunks: List[DocChunk] = self.split_documents_by_page(document, **kwargs)
            else:
                chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        else:
            # text가 있는 item이 없을 때 document에 임의의 text item 추가
            # 첫 번째 페이지의 기본 정보 사용 (1-based indexing)
            page_no = 1

            # ProvenanceItem 생성
            prov = ProvenanceItem(
                page_no=page_no,
                bbox=BoundingBox(l=0, t=0, r=1, b=1),  # 최소 bbox
                charspan=(0, 1)
            )

            # document에 temp text item 추가
            document.add_text(
                label=DocItemLabel.TEXT,
                text=".",
                prov=prov
            )

            # split_documents 호출
            if is_ppt:
                chunks: List[DocChunk] = self.split_documents_by_page(document, **kwargs)
            else:
                chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        # await assert_cancelled(request)

        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(
                document, chunks, file_path, request,
                converted_pdf_path=converted_pdf_path,
                _sensitive_infos=sensitive_infos,
                _guardrail_masking=(gr.call_enabled(kwargs) and self._guardrail_masking_enabled),
                **enrichment_kwargs,
            )
        else:
            raise GenosServiceException(1, f"chunk length is 0")

        # 변환된 PDF 를 minio 에 업로드. object key 는 원본 파일명의 stem + ".pdf".
        # (예: 원본 file_name='sample.hwp' → minio key='<doc_id>/sample.pdf')
        # upload_files 가 finally 에서 org_path 를 os.remove 하는데, 변환 PDF 의
        # NFS 원본은 GenOS UI 의 PDF preview 가 직접 참조하므로 보존 필요.
        # → 임시 사본을 만들어 그것만 업로드시키고 NFS 원본은 그대로 둔다.
        if converted_pdf_path and upload_files:
            original_name = kwargs.get('file_name') or os.path.basename(converted_pdf_path)
            pdf_object_name = os.path.splitext(original_name)[0] + '.pdf'
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as _tmp:
                shutil.copy(converted_pdf_path, _tmp.name)
                _tmp_upload_path = _tmp.name
            await upload_files(
                [{'path': _tmp_upload_path, 'name': pdf_object_name}],
                request=request,
            )

        """
        # 미디어 파일 업로드 방법
        media_files = [
            { 'path': '/tmp/graph.jpg', 'name': 'graph.jpg', 'type': 'image' },
            { 'path': '/result/1/graph.jpg', 'name': '1/graph.jpg', 'type': 'image' },
        ]

        # 업로드 요청 시에는 path, name 필요
        file_list = [{k: v for k, v in file.items() if k != 'type'} for file in media_files]
        await upload_files(file_list, request=request)

        # 메타에 저장시에는 name, type 필요
        meta = [{k: v for k, v in file.items() if k != 'path'} for file in media_files]
        vectors[0].media_files = meta
        """

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        runtime_level = kwargs.get('log_level')
        self.setup_logging(runtime_level if runtime_level is not None else self._log_level)

        # 런타임 토글(img_desc/chart_desc/chart_detection/doc_summary)로 이미지·차트 description 재구성
        kwargs = self._normalize_runtime_kwargs(kwargs)
        self._configure_runtime_image_mode(kwargs)

        # #329: LLM 캐시 / error_policy 컨텍스트를 요청 스코프로 설정.
        # ThreadPool 워커 스레드로는 in_current_context 로 전파된다(docling/utils/llm_cache).
        _cache_token = _set_cache_context(_resolve_cache_context(kwargs))
        try:
            _log.info(f"file_path: {file_path}")
            _log.info(f"kwargs: {kwargs}")

            # 비정상 파일 사전 감지(이슈 #278): 지원 포맷 매직헤더에 하나도 안 맞고 텍스트도
            # 아니면(=DRM 암호화/손상 바이너리) 변환 시 garbage PDF → VLM 무한 출력/행을
            # 유발하므로 변환 전에 컷한다. 확장자와 무관하게 실제 헤더로 판정.
            bad_reason = _detect_unsupported_file(file_path)
            if bad_reason:
                _log.warning(
                    f"[intelligent] 비정상 파일 감지({bad_reason}) — 처리 중단: {file_path}"
                )
                raise GenosServiceException(
                    "1", f"{bad_reason} 입니다. 정상 문서로 다시 업로드하세요: {os.path.basename(file_path)}"
                )

            ext = os.path.splitext(file_path)[1].lower()

            # 직접 처리(PDF 변환 없이) 가능한 포맷(이슈 #288): 엑셀 계열(xlsx/xlsm) + csv.
            # csv 는 본질적으로 tabular 이므로 항상 직접 처리한다(PDF 변환 시 행 분할 문제 방지).
            # (.xls/.xlsb 는 openpyxl/docling 미지원 → 아래 PDF 변환 경로로 처리)
            # 이 집합을 변환 가드와 디스패치 양쪽에서 동일하게 써서 "직접 처리 포맷 == 변환 제외 포맷"
            # 불변식을 유지한다.

            # 직접 처리 포맷이 아니고 PDF 도 아니면 PDF 로 변환한다.
            # - auto_convert_to_pdf=True (default): PDF SDK/LibreOffice 로 자동 변환 후 진입
            # - auto_convert_to_pdf=False: 변환 없이 그대로 진행 (변경 전 동작; PDF 가정)
            converted_pdf_path: Optional[str] = None
            # HWP/HWPX 품질 복구가 참조할 변환 전 원본 경로(rhwp 재변환/네이티브 추출용).
            source_file_path = file_path
            if ext not in _XLSX_DIRECT_EXTS and kwargs.get('auto_convert_to_pdf', True) and not _is_pdf(file_path):
                file_path, converted_pdf_path = self._convert_to_pdf(file_path, **kwargs)

            # 포맷별 처리: 직접 처리 가능 포맷은 xlsx 핸들러, 그 외는 PDF(docling) 처리.
            if ext in _XLSX_DIRECT_EXTS:
                return await self._process_xlsx(request, file_path, **kwargs)
            # 원본이 PPT 였는지(변환 전 ext)를 명시 전달 — 페이지 기반 청킹/page description 게이팅용.
            is_ppt = ext in ('.ppt', '.pptx')
            return await self._process_pdf(
                request, file_path, converted_pdf_path, is_ppt=is_ppt,
                source_file_path=source_file_path, **kwargs
            )
        finally:
            _log_cache_summary()
            _reset_cache_context(_cache_token)


class GenosServiceException(Exception):
    # GenOS 와의 의존성 부분 제거를 위해 추가
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None,
                 *, stage: Optional[str] = None, error_type: Optional[str] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}
        # #329: 실패 단계(stage)와 성격(error_type: transient/permanent/timeout).
        self.stage = stage
        self.error_type = error_type

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"


# GenOS 와의 의존성 제거를 위해 추가
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")
