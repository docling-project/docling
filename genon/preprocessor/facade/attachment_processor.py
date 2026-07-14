# 첨부용 전처리기 v.2.2.2 (2026-07-02 Release)
from __future__ import annotations

from collections import defaultdict

import asyncio
import fitz
import json
import math
import os
import pandas as pd
import pydub
import re
import requests
import shutil
import subprocess
import sys
import threading
import uuid
import warnings
import yaml
from datetime import datetime
import logging
from fastapi import Request

_log = logging.getLogger(__name__)


# ── 비정상/암호화 파일 사전 감지 (이슈 #278/#307) ─────────────────────────────
# intelligent_processor.py 의 동일 블록을 복제한 것. facade 는 단일 파일로 배포되므로
# import 공유 대신 복제한다. 수정 시 네 파일(intelligent/parser/convert/attachment) 동기화 필요.
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


from glob import glob
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
from markdown2 import markdown
from pandas import DataFrame
from pathlib import Path
from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from typing import Any, Iterable, Iterator, List, Optional, Union
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
    import chardet
except ImportError:
    raise RuntimeError("Module 'chardet' not imported. Run `pip install chardet`.")
try:
    from weasyprint import HTML
except ImportError:
    print("Warning: WeasyPrint could not be imported. PDF conversion features will be disabled.")
    HTML = None

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
from docling.datamodel.document import ConversionResult, InputDocument
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.document_converter import (
    DocumentConverter, HwpxFormatOption, WordFormatOption, PdfFormatOption,
)
from genon.preprocessor.facade.enrichment.page_description import (
    PageDescriptionOptions,
    describe_pages,
)
from docling_core.transforms.chunker import BaseChunk, BaseChunker, DocChunk, DocMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc import (
    DocItem, DocItemLabel, DoclingDocument,
    PictureItem, SectionHeaderItem, TableItem, TextItem
)
from docling_core.types.doc.document import LevelNumber, ListItem, CodeItem
from docling.backend.genos_msword_backend import GenosMsWordDocumentBackend
from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend
from docling.backend.hwp_backend import HwpDocumentBackend
from docling.backend.xml.hwpx_backend import HwpxDocumentBackend
from docling.exceptions import HwpConversionError

try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None

from pathlib import Path
import os
import subprocess
import tempfile
import shutil
import unicodedata

import logging

for n in ("fontTools", "fontTools.ttLib", "fontTools.ttLib.ttFont"):
    lg = logging.getLogger(n)
    lg.setLevel(logging.CRITICAL)
    lg.propagate = False
    logging.getLogger().setLevel(logging.WARNING)
# pdf 변환 대상 확장자
CONVERTIBLE_EXTENSIONS = ['.hwp', '.txt', '.json', '.md', '.ppt', '.pptx', '.docx']

_DEFAULT_TOKENIZER_LOCAL_PATH = "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
_DEFAULT_TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_HYBRID_MAX_TOKENS = int(1e30)


def _warn_unresolved_placeholders(cfg: dict, config_path: str) -> None:
    """config 에 남아있는 미치환 플레이스홀더(<UPPER_SNAKE>)를 탐지해 경고한다.

    Site 배포 시 Whisper endpoint 등의 치환 누락을 조기에 드러내기 위함.
    fail-fast 하지 않고(기동 보존) WARNING 로그만 남긴다.
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
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _log.warning(f"[DocumentProcessor] Config file not found: {config_path}. Using defaults.")
        return {}
    except Exception as e:
        _log.warning(f"[DocumentProcessor] Failed to load config '{config_path}': {e}. Using defaults.")
        return {}

    if not isinstance(cfg, dict):
        _log.warning(
            f"[DocumentProcessor] Invalid config format in '{config_path}' "
            f"(expected mapping, got {type(cfg).__name__}). Using defaults."
        )
        return {}
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


def _resolve_default_attachment_config_path() -> str:
    base_dir = Path(__file__).resolve().parent
    local_config = (base_dir / "../resource_dev/attachment_processor_config.yaml").resolve()
    default_config = (base_dir / "../resource/attachment_processor_config.yaml").resolve()

    if local_config.exists():
        return str(local_config)
    return str(default_config)


def _resolve_tokenizer(chunking_cfg: dict):
    """chunking config 로부터 토크나이저를 결정한다.

    tokenizer_path 가 실제 존재하면 그 로컬 경로를, 없으면 tokenizer_id(HF) 로 폴백한다
    (외부 네트워크 차단 환경 대비). config 미지정 시 기본값은 현행 하드코딩 값과 동일.
    """
    local = chunking_cfg.get("tokenizer_path") or _DEFAULT_TOKENIZER_LOCAL_PATH
    hf_id = chunking_cfg.get("tokenizer_id") or _DEFAULT_TOKENIZER_ID
    return Path(local) if Path(local).exists() else hf_id


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


def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            _log.warning(f"{package} 패키지가 없습니다. 설치를 시도합니다.")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


# ============================================================================
# 민감정보 분류 (이슈 #315) — GenOS 분류 워크플로우 연동
# 청킹 전 문서 전체를 워크플로우(run/v2)에 1회 보내 sensitive_infos[] 를 받고,
# 청킹 후 각 청크에서 quote_origin 을 매칭해 content_category 라벨을 붙이고(항상),
# 옵션으로 quote_masked 로 치환한다(masking on/off).
# 단일 파일 배포 특성상 이 블록은 각 facade(intelligent/attachment/convert/chunking)에
# 동일하게 복제되어야 한다. 수정 시 함께 동기화한다.
# ----------------------------------------------------------------------------
import re as _gr_re
_GR_WS = _gr_re.compile(r"\s+")


def _gr_classify_document(text: str, url: str, workflow_id, api_key: str, timeout: int = 60) -> list:
    """분류 워크플로우(run/v2)를 문서당 1회 호출 → sensitive_infos[] 반환.
    입력 {"question": <문서 전체>} → 응답 data.sensitive_infos. 실패/미설정 시 [](fail-open)."""
    import requests
    if not url or workflow_id is None or not api_key or not text:
        if not text:
            return []
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


def _gr_find_spans(text: str, quote: str) -> list:
    """청크 text 에서 quote 가 나타나는 모든 (start,end). 1차 정확, 실패 시 공백 무시(fuzzy).
    (LLM quote 의 '5 억에' vs 원문 '5억에' 불일치 대응)"""
    if not quote or not text:
        return []
    spans, start = [], 0
    while True:
        i = text.find(quote, start)
        if i == -1:
            break
        spans.append((i, i + len(quote)))
        start = i + len(quote)
    if spans:
        return spans
    kept = [(j, ch) for j, ch in enumerate(text) if not ch.isspace()]
    if not kept:
        return []
    stripped = "".join(ch for _, ch in kept)
    idx_map = [j for j, _ in kept]
    q = _GR_WS.sub("", quote)
    if not q:
        return []
    s = 0
    while True:
        k = stripped.find(q, s)
        if k == -1:
            break
        spans.append((idx_map[k], idx_map[k + len(q) - 1] + 1))
        s = k + len(q)
    return spans


def _gr_doc_text(document: DoclingDocument) -> str:
    """분류 워크플로우로 보낼 문서 전체 텍스트. 그림만 제외. 표는 마크다운으로 포함한다
    (표 셀 PII 도 분류/마스킹 대상 — 청크도 표를 마크다운으로 담으므로 quote 매칭 가능)."""
    parts = []
    for it, _ in document.iterate_items():
        if isinstance(it, PictureItem):
            continue
        if isinstance(it, TableItem):
            try:
                t = it.export_to_markdown(document)
            except Exception:
                cells = getattr(getattr(it, "data", None), "table_cells", None) or []
                t = " ".join((getattr(c, "text", "") or "") for c in cells)
        else:
            t = getattr(it, "text", None)
        if isinstance(t, str) and t.strip():
            parts.append(t)
    return "\n".join(parts)


def _gr_docs_text(documents) -> str:
    """langchain Document 리스트(page_content)에서 문서 전체 텍스트 결합(#315 fallback 경로)."""
    return "\n".join(
        d.page_content for d in documents
        if isinstance(getattr(d, "page_content", None), str) and d.page_content.strip()
    )


def _gr_apply_to_text(text: str, sensitive_infos: list, masking: bool):
    """청크 text 에 sensitive_infos 적용 → (새 text, 부착할 category 집합).
    category 부착은 항상, quote_masked 치환은 masking=True 일 때만. 매칭 실패는 skip."""
    cats = set()
    repl = []  # (start, end, masked)
    for info in sensitive_infos or []:
        cat = (info or {}).get("category")
        q = (info or {}).get("quote_origin")
        m = (info or {}).get("quote_masked")
        if not cat or not q:
            continue
        spans = _gr_find_spans(text, q)
        if not spans:
            continue
        cats.add(cat)
        if masking and isinstance(m, str) and m != q:
            for st, en in spans:
                repl.append((st, en, m))
    if repl:
        applied = []  # 이미 치환한 구간(겹침 방지)
        for st, en, m in sorted(repl, key=lambda x: -x[0]):
            if any(not (en <= a or st >= b) for a, b in applied):
                continue  # 겹치는 구간은 건너뜀
            text = text[:st] + m + text[en:]
            applied.append((st, en))
    return text, cats


class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str | None = None
    n_char: int | None = None
    n_word: int | None = None
    n_line: int | None = None
    i_page: int | None = None
    e_page: int | None = None
    i_chunk_on_page: int | None = None
    n_chunk_of_page: int | None = None
    i_chunk_on_doc: int | None = None
    n_chunk_of_doc: int | None = None
    n_page: int | None = None
    reg_date: str | None = None
    chunk_bboxes: str | None = None
    media_files: str | None = None
    content_category: Optional[list] = None    # #315 민감정보 분류 라벨(부동산/인사/민감 등). 미적용 시 None


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
        self.content_category: Optional[list] = None   # #315 민감정보 분류 라벨
        # self.title: Optional[str] = None
        # self.created_date: Optional[int] = None

    def set_content_category(self, content_category: Optional[list]) -> "GenOSVectorMetaBuilder":
        """#315 청크 민감정보 분류 라벨 설정 (부동산/인사/민감 등의 list, 미적용 시 None)"""
        self.content_category = content_category or None
        return self

    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련된 데이터를 설정"""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self

    def set_page_info(self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int) -> "GenOSVectorMetaBuilder":
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
                bbox_data = {
                    'l': bbox.l / size.width,
                    't': bbox.t / size.height,
                    'r': bbox.r / size.width,
                    'b': bbox.b / size.height,
                    'coord_origin': bbox.coord_origin.value
                }
                chunk_bboxes.append({
                    'page': page_no,
                    'bbox': bbox_data,
                    'type': type_,
                    'ref': label
                })
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else 0
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self

    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        temp_list = []
        if not doc_items:
            self.media_files = ""
            return self
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
        self.media_files = json.dumps(temp_list)
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        return GenOSVectorMeta(
            text=self.text,
            n_char=self.n_char,
            n_word=self.n_word,
            n_line=self.n_line,
            i_page=self.i_page,
            e_page=self.e_page,
            i_chunk_on_page=self.i_chunk_on_page,
            n_chunk_of_page=self.n_chunk_of_page,
            i_chunk_on_doc=self.i_chunk_on_doc,
            n_chunk_of_doc=self.n_chunk_of_doc,
            n_page=self.n_page,
            reg_date=self.reg_date,
            chunk_bboxes=self.chunk_bboxes,
            media_files=self.media_files,
            content_category=self.content_category,  # #315 민감정보 분류 라벨
        )

class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            with open(self.file_path, 'rb') as f:
                raw = f.read()
            enc = chardet.detect(raw).get('encoding') or ''
            encodings = [enc] if enc and enc.lower() not in ('ascii', 'unknown') else []
            encodings += ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1']

            content = None
            for e in encodings:
                try:
                    content = raw.decode(e)  # 전체 파일로 디코딩
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                content = raw.decode('utf-8', errors='replace')

            # 4) PDF 변환 유지
            html = f"<html><meta charset='utf-8'><body><pre>{content}</pre></body></html>"
            html_path = os.path.join(self.output_dir, 'temp.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            # pdf_path = (self.file_path
            #             .replace('.txt', '.pdf')
            #             .replace('.json', '.pdf'))
            pdf_path = _get_pdf_path(self.file_path)
            if HTML:
                HTML(html_path).write_pdf(pdf_path)
                loader = PyMuPDFLoader(pdf_path)
                return loader.load()
            # PDF가 불가하면 Document 직접 반환 (원형 스키마 유지)
            return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]

        except Exception:
            # 실패 시에도 스키마는 그대로 유지해 반환
            for e in ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1']:
                try:
                    with open(self.file_path, 'r', encoding=e) as f:
                        content = f.read()
                    return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]
                except UnicodeDecodeError:
                    continue
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)


class TabularLoader:
    def __init__(self, file_path: str, ext: str, encoding_detect_sample_bytes: int = 10000):
        packages = ['openpyxl', 'chardet']
        install_packages(packages)

        self.file_path = file_path
        self.encoding_detect_sample_bytes = max(int(encoding_detect_sample_bytes), 1)
        if ext == ".csv":
            # convert_to_pdf(file_path) csv는 Pdf 변환 안 함
            self.data_dict = self.load_csv_documents(file_path)
        elif ext == ".xlsx":
            # convert_to_pdf(file_path) xlsx는 Pdf 변환 안 함
            self.data_dict = self.load_xlsx_documents(file_path)
        else:
            _log.warning(f"Inadequate extension for TabularLoader: {ext}")
            return

    def check_sql_dtypes(self, df):
        df = df.convert_dtypes()
        res = []
        for col in df.columns:
            # col_name = col.strip().replace(' ', '_')
            dtype = str(df.dtypes[col]).lower()

            if 'int' in dtype:
                if '64' in dtype:
                    sql_dtype = 'BIGINT'
                else:
                    sql_dtype = 'INT'
            elif 'float' in dtype:
                sql_dtype = 'FLOAT'
            elif 'bool' in dtype:
                sql_dtype = 'BOOLEAN'
            elif 'date' in dtype:
                sql_dtype = 'DATE'
                df[col] = df[col].astype(str)
            elif 'datetime' in dtype:
                sql_dtype = 'DATETIME'
                df[col] = df[col].astype(str)
            # else:
            #     max_len = df[col].str.len().max().item() + 10
            #     sql_dtype = f'VARCHAR({max_len})'
            else:
                lens = df[col].astype(str).str.len()
                max_len_val = lens.max()
                max_len = int(0 if pd.isna(max_len_val) else max_len_val) + 10
                sql_dtype = f'VARCHAR({max_len})'

            res.append([col, sql_dtype])

        return df, res

    def process_data_rows(self, data: dict):
        """Arg: data (keys: 'sheet_name', 'page_column', 'page_column_type', 'documents')"""

        rows = []
        for doc in data["documents"]:
            row = {}
            if 'int' in data["page_column_type"]:
                row[data["page_column"]] = int(doc.page_content)
            elif 'float' in data["page_column_type"]:
                row[data["page_column"]] = float(doc.page_content)
            elif 'bool' in data["page_column_type"]:
                if doc.page_content.lower() == 'true':
                    row[data["page_column"]] = True
                elif doc.page_content.lower() == 'false':
                    row[data["page_column"]] = False
                else:
                    raise ValueError(f"Invalid boolean string: {doc.page_content}")
            else:
                row[data["page_column"]] = doc.page_content

            row.update(doc.metadata)
            rows.append(row)

        processed_data = {"sheet_name": data["sheet_name"], "data_rows": rows, "data_types": data["dtypes"]}
        return processed_data

    def load_csv_documents(self, file_path: str, **kwargs: dict):
        import chardet

        with open(file_path, "rb") as f:
            raw_file = f.read(self.encoding_detect_sample_bytes)
        enc_type = chardet.detect(raw_file)['encoding']
        df = pd.read_csv(file_path, encoding=enc_type, index_col=False)
        df = df.fillna('null')  # csv 파일에서도 xlsx 파일과 동일하게 null로 채움
        df, dtypes_str = self.check_sql_dtypes(df)

        for i in range(len(df.columns)):
            try:
                col = df.columns[0]
                # col_type = str(type(col))
                col_type = str(df[col].dtype)
                df = df.astype({col: 'str'})
                break
            except:
                raise ValueError(
                    f"Any columns cannot be converted into the string type so that can't load LangChain Documents: {dtypes_str}")

        loader = DataFrameLoader(df, page_content_column=col)
        documents = loader.load()

        data = {
            "sheet_name": "table_1",
            "page_column": col,
            "page_column_type": col_type,
            "documents": documents,
            "dtypes": dtypes_str
        }
        data = self.process_data_rows(data)  # including only one sheet as it's a csv file
        data_dict = {"data": [data]}
        return data_dict

    def load_xlsx_documents(self, file_path: str, **kwargs: dict):
        dfs = pd.read_excel(file_path, sheet_name=None)
        sheets = []
        for sheet_name, df in dfs.items():
            df = df.fillna('null')
            df, dtypes_str = self.check_sql_dtypes(df)

            for i in range(len(df.columns)):
                try:
                    col = df.columns[0]
                    col_type = str(type(col))
                    df = df.astype({col: 'str'})
                    break
                except:
                    raise ValueError(
                        f"Any columns cannot be converted into string type so that can't load LangChain Documents: {dtypes_str}")

            loader = DataFrameLoader(df, page_content_column=col)
            documents = loader.load()

            sheet = {
                "sheet_name": sheet_name,
                "page_column": col,
                "page_column_type": col_type,
                "documents": documents,
                "dtypes": dtypes_str
            }
            sheets.append(sheet)

        data_dict = {"data": []}
        for sheet in sheets:
            data = self.process_data_rows(sheet)
            data_dict["data"].append(data)

        return data_dict

    def return_vectormeta_format(self):
        if not self.data_dict:
            return None

        text = "[DA] " + str(self.data_dict)  # Add a token to indicate this string is for data analysis
        vectors = [GenOSVectorMeta.model_validate({
            'text': text,
            'n_char': 1,
            'n_word': 1,
            'n_line': 1,
            'i_page': 1,
            'e_page': 1,
            'n_page': 1,
            'i_chunk_on_page': 1,
            'n_chunk_of_page': 1,
            'i_chunk_on_doc': 1,
            'reg_date': datetime.now().isoformat(timespec='seconds') + 'Z',
            'chunk_bboxes': ".",
            'media_files': "."
        })]
        return vectors


class AudioLoader:
    def __init__(self,
                 file_path: str,
                 req_url: str,
                 req_data: dict,
                 chunk_sec: int = 29,
                 chunk_overlap_ms: int = 300,
                 tmp_path: str = '.',
                 ):
        self.file_path = file_path
        self.tmp_path = tmp_path
        self.chunk_sec = chunk_sec
        self.chunk_overlap_ms = max(int(chunk_overlap_ms), 0)
        self.req_url = req_url
        self.req_data = req_data

    def split_file_as_chunks(self) -> list:
        audio = pydub.AudioSegment.from_file(self.file_path)
        chunk_len = self.chunk_sec * 1000
        n_chunks = math.ceil(len(audio) / chunk_len)

        for i in range(n_chunks):
            start_ms = i * chunk_len
            overlap_start_ms = start_ms - self.chunk_overlap_ms if start_ms > 0 else start_ms
            end_ms = start_ms + chunk_len
            audio_chunk = audio[overlap_start_ms:end_ms]
            audio_chunk.export(os.path.join(self.tmp_path, "tmp_{}.wav".format(str(i))), format="wav")
        tmp_files = glob(os.path.join(self.tmp_path, "*.wav"))
        return tmp_files

    def transcribe_audio(self, file_path_lst: list):
        transcribed_text_chunks = []

        def _send_request(filepath: str):
            """Send a request to 'whisper' model served"""
            files = {
                'file': (filepath, open(filepath, 'rb'), 'audio/mp3'),
            }

            response = requests.post(self.req_url, data=self.req_data, files=files)
            text = response.json().get('text', ', ')
            transcribed_text_chunks.append({
                'file_name': os.path.basename(filepath),
                'text': text
            })

        # Send parallel requests
        threads = [threading.Thread(target=_send_request, args=(f,)) for f in file_path_lst]
        for t in threads: t.start()
        for t in threads: t.join()

        # Merge transcribed text snippets in order
        transcribed_text_chunks.sort(key=lambda x: x['file_name'])
        transcribed_text = "[AUDIO]" + ' '.join([t['text'] for t in transcribed_text_chunks])
        return transcribed_text

    def return_vectormeta_format(self):
        audio_chunks = self.split_file_as_chunks()
        transcribed_text = self.transcribe_audio(audio_chunks)
        res = [GenOSVectorMeta.model_validate({
            'text': transcribed_text,
            'n_char': 1,
            'n_word': 1,
            'n_line': 1,
            'i_page': 1,
            'e_page': 1,
            'n_page': 1,
            'i_chunk_on_page': 1,
            'n_chunk_of_page': 1,
            'i_chunk_on_doc': 1,
            'reg_date': datetime.now().isoformat(timespec='seconds') + 'Z',
            'chunk_bboxes': ".",
            'media_files': "."
        })]
        return res


### for HWPX from 지능형 전처리기 ###
#  * GenOSVectorMetaBuilder     #
#  * HierarchicalChunker        #
#  * HybridChunker              #
#  * HwpxProcessor              #
#  * GenosServiceException      #

class HierarchicalChunker(BaseChunker):
    r""" Chunker implementation leveraging the document layout.
    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """
    merge_list_items: bool = True

    @classmethod
    def _triplet_serialize(cls, table_df: DataFrame) -> str:
        # copy header as first row and shift all rows by one
        table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
        table_df.index = table_df.index + 1
        table_df = table_df.sort_index()

        rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
        cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

        nrows = table_df.shape[0]
        ncols = table_df.shape[1]
        texts = [
            f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
            for i in range(1, nrows)
            for j in range(1, ncols)
        ]
        output_text = ". ".join(texts)

        return output_text

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.
        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):
                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(
                            item, ListItem
                    ) or (  # TODO remove when all captured as ListItem:
                            isinstance(item, TextItem)
                            and item.label == DocItemLabel.LIST_ITEM
                    ):
                        list_items.append(item)
                        continue
                    elif list_items:  # need to yield
                        yield DocChunk(
                            text=self.delim.join([i.text for i in list_items]),
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset

                if isinstance(item, SectionHeaderItem) or (
                        isinstance(item, TextItem) and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]):
                    level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )
                    heading_by_level[level] = item.text
                    text = ''.join(str(value) for value in heading_by_level.values())

                    # remove headings of higher level as they just went out of scope
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    c = DocChunk(
                        text=text,
                        meta=DocMeta(
                            doc_items=[item],
                            headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                            captions=captions,
                            origin=dl_doc.origin
                        ),
                    )
                    yield c
                    continue

                if isinstance(item, TextItem) or (
                        (not self.merge_list_items) and isinstance(item, ListItem)) or isinstance(item, CodeItem):
                    text = item.text

                elif isinstance(item, TableItem):
                    text = item.export_to_markdown(dl_doc)
                    # dataframe으로 추출할 때 사용되는 코드
                    # if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                    #     # at least two cols needed, as first column contains row headers
                    #     continue
                    # text = self._triplet_serialize(table_df=table_df)
                    captions = [c.text for c in [r.resolve(dl_doc) for r in item.captions]] or None

                elif isinstance(item, PictureItem):
                    text = ''.join(str(value) for value in heading_by_level.values())
                else:
                    continue
                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        if self.merge_list_items and list_items:  # need to yield
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)] or None,
                    origin=dl_doc.origin,
                ),
            )


class HybridChunker(BaseChunker):
    r"""Chunker doing tokenization-aware refinements on top of document layout chunking.
    Args:
        tokenizer: The tokenizer to use; either instantiated object or name or path of
            respective pretrained model
        max_tokens: The maximum number of tokens per chunk. If not set, limit is
            resolved from the tokenizer
        merge_peers: Whether to merge undersized chunks sharing same relevant metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tokenizer: Union[PreTrainedTokenizerBase, str, Path] = (
            Path(_DEFAULT_TOKENIZER_LOCAL_PATH)
            if Path(_DEFAULT_TOKENIZER_LOCAL_PATH).exists()
            else _DEFAULT_TOKENIZER_ID
        )
    max_tokens: int = _DEFAULT_HYBRID_MAX_TOKENS  # type: ignore[assignment]
    merge_peers: bool = True
    # 토큰 수 계산 방식. "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
    tokenizer_type: str = "char"
    _inner_chunker: HierarchicalChunker = HierarchicalChunker()

    @model_validator(mode="after")
    def _patch_tokenizer_and_max_tokens(self) -> Self:
        mode = (self.tokenizer_type or "char").strip().lower()
        if mode not in {"char", "huggingface"}:
            _log.warning(f"[HybridChunker] Unknown tokenizer_type '{mode}', fallback to 'char'.")
            mode = "char"
        self.tokenizer_type = mode
        if mode == "char":
            # 문자 수 기반: HF 토크나이저 로드 불필요 (외부 모델 의존 제거)
            self._tokenizer = None
            if self.max_tokens is None:
                self.max_tokens = _DEFAULT_HYBRID_MAX_TOKENS
        else:
            self._tokenizer = (
                self.tokenizer
                if isinstance(self.tokenizer, PreTrainedTokenizerBase)
                else AutoTokenizer.from_pretrained(self.tokenizer)
            )
            if self.max_tokens is None:
                self.max_tokens = TypeAdapter(PositiveInt).validate_python(
                    self._tokenizer.model_max_length
                )
        return self

    def _count_text_tokens(self, text: Optional[Union[str, list[str]]]):
        if text is None:
            return 0
        elif isinstance(text, list):
            total = 0
            for t in text:
                total += self._count_text_tokens(t)
            return total
        if self._tokenizer is None:   # 문자 수 기반
            return len(text)
        return len(self._tokenizer.tokenize(text))

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _count_chunk_tokens(self, doc_chunk: DocChunk):
        ser_txt = self.serialize(chunk=doc_chunk)
        if self._tokenizer is None:   # 문자 수 기반
            return len(ser_txt)
        return len(self._tokenizer.tokenize(text=ser_txt))

    def _doc_chunk_length(self, doc_chunk: DocChunk):
        text_length = self._count_text_tokens(doc_chunk.text)
        total = self._count_chunk_tokens(doc_chunk=doc_chunk)
        return self._ChunkLengthInfo(
            total_len=total,
            text_len=text_length,
            other_len=total - text_length,
        )

    def _make_chunk_from_doc_items(
            self, doc_chunk: DocChunk, window_start: int, window_end: int
    ):
        doc_items = doc_chunk.meta.doc_items[window_start: window_end + 1]
        meta = DocMeta(
            doc_items=doc_items,
            headings=doc_chunk.meta.headings,
            captions=doc_chunk.meta.captions,
            origin=doc_chunk.meta.origin,
        )
        window_text = (
            doc_chunk.text
            if len(doc_chunk.meta.doc_items) == 1
            else self.delim.join(
                [
                    doc_item.text
                    for doc_item in doc_items
                    if isinstance(doc_item, TextItem)
                ]
            )
        )
        new_chunk = DocChunk(text=window_text, meta=meta)
        return new_chunk

    def _split_by_doc_items(self, doc_chunk: DocChunk) -> list[DocChunk]:
        chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_items = len(doc_chunk.meta.doc_items)
        while window_end < num_items:
            new_chunk = self._make_chunk_from_doc_items(
                doc_chunk=doc_chunk,
                window_start=window_start,
                window_end=window_end,
            )
            if self._count_chunk_tokens(doc_chunk=new_chunk) <= self.max_tokens:
                if window_end < num_items - 1:
                    window_end += 1
                    # 아직 청크에 여유가 있고, 남은 아이템도 있으므로 계속 추가 시도
                    continue
                else:
                    # 현재 윈도우의 모든 아이템이 청크에 들어갔고, 더 이상 아이템이 없음
                    window_end = num_items  # signalizing the last loop
            elif window_start == window_end:
                # 아이템 1개도 청크에 안 들어감 → 단독 청크로 처리, 이후 재분할
                window_end += 1
                window_start = window_end
            else:
                # 마지막 아이템 빼고 청크 생성 → 남은 아이템으로 새 윈도우 시작
                new_chunk = self._make_chunk_from_doc_items(
                    doc_chunk=doc_chunk,
                    window_start=window_start,
                    window_end=window_end - 1,
                )
                window_start = window_end
            chunks.append(new_chunk)
        return chunks

    def _split_using_plain_text(self, doc_chunk: DocChunk) -> list[DocChunk]:
        lengths = self._doc_chunk_length(doc_chunk)
        if lengths.total_len <= self.max_tokens:
            return [doc_chunk]
        else:
            # 헤더/캡션을 제외하고 본문 텍스트에 할당 가능한 토큰 수 계산
            available_length = self.max_tokens - lengths.other_len
            # char 모드는 문자 수 카운터 len 사용
            counter = len if self._tokenizer is None else self._tokenizer
            sem_chunker = semchunk.chunkerify(
                counter, chunk_size=available_length
            )
            if available_length <= 0:
                warnings.warn(
                    f"Headers and captions for this chunk are longer than the total amount of size for the chunk, chunk will be ignored: {doc_chunk.text=}"
                    # noqa
                )
                return []
            text = doc_chunk.text
            segments = sem_chunker.chunk(text)
            chunks = [type(doc_chunk)(text=s, meta=doc_chunk.meta) for s in segments]
            return chunks

    def _merge_chunks_with_matching_metadata(self, chunks: list[DocChunk]):
        output_chunks = []
        window_start = 0
        window_end = 0  # an inclusive index
        num_chunks = len(chunks)

        while window_end < num_chunks:
            chunk = chunks[window_end]
            headings_and_captions = (chunk.meta.headings, chunk.meta.captions)
            ready_to_append = False

            if window_start == window_end:
                current_headings_and_captions = headings_and_captions
                window_end += 1
                first_chunk_of_window = chunk

            else:
                chks = chunks[window_start: window_end + 1]
                doc_items = [it for chk in chks for it in chk.meta.doc_items]
                candidate = DocChunk(
                    text=self.delim.join([chk.text for chk in chks]),
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=current_headings_and_captions[0],
                        captions=current_headings_and_captions[1],
                        origin=chunk.meta.origin,
                    ),
                )

                if (headings_and_captions == current_headings_and_captions
                        and self._count_chunk_tokens(doc_chunk=candidate) <= self.max_tokens
                ):
                    # 토큰 수 여유 있음 → 청크 확장 계속
                    window_end += 1
                    new_chunk = candidate
                else:
                    ready_to_append = True

            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.
                if window_start + 1 == window_end:
                    output_chunks.append(first_chunk_of_window)
                else:
                    output_chunks.append(new_chunk)
                window_start = window_end

        return output_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.
        Args:
            dl_doc (DLDocument): document to chunk
        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        res: Iterable[DocChunk]
        res = self._inner_chunker.chunk(dl_doc=dl_doc, **kwargs)  # type: ignore
        res = [x for c in res for x in self._split_by_doc_items(c)]
        res = [x for c in res for x in self._split_using_plain_text(c)]

        if self.merge_peers:
            res = self._merge_chunks_with_matching_metadata(res)
        return iter(res)


# --- 이슈 #183 / #80 -------------------------------------------------------
# DoclingDocument를 markdown으로 export한 뒤 RecursiveCharacterTextSplitter로 분할.
# 페이지 정보는 export_to_markdown(page_break_placeholder=...)로 삽입한 마커를
# 청크별로 카운트해 복원한다. 한 청크가 여러 페이지에 걸칠 수 있다.
_RECURSIVE_PAGE_BREAK = "<!-- PB -->"


def _char_split_text(text: str, chunk_size=None, chunk_overlap=None) -> list[str]:
    """문자수 기반 청킹 공용 헬퍼 (generic/recursive 경로 공유).

    chunk_size 가 0 이하/None 이면 분할하지 않고 전체를 1청크로 둔다.
    chunk_size > 0 이면 RecursiveCharacterTextSplitter 로 문자 단위 분할한다.
    """
    if not text:
        return []

    cs = int(chunk_size) if chunk_size is not None else 0
    co = max(int(chunk_overlap), 0) if chunk_overlap is not None else 100

    if cs > 0:
        raw_chunks = RecursiveCharacterTextSplitter(
            chunk_size=cs, chunk_overlap=co,
        ).split_text(text)
    else:
        raw_chunks = [text]

    return [c for c in raw_chunks if c]


def _split_with_recursive_chunker(
    document: DoclingDocument,
    chunk_size=None,
    chunk_overlap=None,
) -> List[dict]:
    """Markdown export + 문자수 기반 청킹(_char_split_text)으로 docling 문서를 분할.

    chunk_size 로 문자 분할 (0 이하이면 분할 안 함 = 전체 1청크).

    Returns: list of dict {text, page_no, pages, doc_items}
    """
    md_full = document.export_to_markdown(page_break_placeholder=_RECURSIVE_PAGE_BREAK)
    if not md_full:
        return []

    co = max(int(chunk_overlap), 0) if chunk_overlap is not None else 100
    raw_chunks = _char_split_text(
        md_full,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 페이지별 doc_items 캐시 (반복 조회 방지)
    page_items_cache: dict[int, list] = {}

    def _items_for_page(p: int):
        if p not in page_items_cache:
            page_items_cache[p] = [
                it for it, _ in document.iterate_items(page_no=p)
                if isinstance(it, DocItem)
            ]
        return page_items_cache[p]

    results: list[dict] = []
    cursor = 0
    search_backoff = max(co * 4, 200)
    for raw in raw_chunks:
        pos = md_full.find(raw, max(0, cursor - search_backoff))
        if pos < 0:
            pos = cursor
        end_pos = pos + len(raw)

        start_page = md_full[:pos].count(_RECURSIVE_PAGE_BREAK) + 1
        end_page = md_full[:end_pos].count(_RECURSIVE_PAGE_BREAK) + 1

        text = raw.replace(_RECURSIVE_PAGE_BREAK, "").strip()
        cursor = end_pos
        if not text:
            continue

        doc_items: list = []
        for p in range(start_page, end_page + 1):
            doc_items.extend(_items_for_page(p))

        results.append({
            "text": text,
            "page_no": start_page,
            "pages": list(range(start_page, end_page + 1)),
            "doc_items": doc_items,
        })

    return results


class DocxProcessor:
    def __init__(self, tokenizer=None, guardrail_url="", guardrail_workflow_id=None, guardrail_api_key="", guardrail_timeout=30, guardrail_masking_enabled=False):
        # 청킹용 토크나이저 (config 기반; 미지정 시 현행 기본값)
        self._tokenizer = tokenizer if tokenizer is not None else _resolve_tokenizer({})
        # PII 마스킹(#315) 접속 정보 — DocumentProcessor 가 config 에서 읽어 주입.
        self._guardrail_url = guardrail_url
        self._guardrail_workflow_id = guardrail_workflow_id
        self._guardrail_api_key = guardrail_api_key
        self._guardrail_timeout = guardrail_timeout
        self._guardrail_masking_enabled = guardrail_masking_enabled
        self.page_chunk_counts = defaultdict(int)
        self.pipeline_options = PipelineOptions()
        self.converter = DocumentConverter(
            format_options={
                InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline, backend=GenosMsWordDocumentBackend
                ),
            }
        )

    def get_paths(self, file_path: str):
        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        return artifacts_dir, reference_path

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem) and item.image:
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def split_documents(self, document: DoclingDocument, **kwargs: dict):
        """chunker_type에 따라 HybridChunker 또는 RecursiveCharacterTextSplitter로 분할.

        반환 형식이 chunker_type에 따라 다르다 (DocChunk 리스트 또는 dict 리스트).
        compose_vectors가 동일한 chunker_type 분기로 처리한다.
        """
        # 같은 DocxProcessor 인스턴스가 여러 요청에서 재사용되므로 매 호출마다 초기화
        self.page_chunk_counts = defaultdict(int)
        chunker_type = kwargs.get("chunker_type", "recursive")

        if chunker_type == "recursive":
            recursive_chunk_size = kwargs.get("chunk_size")
            if recursive_chunk_size is None:
                recursive_chunk_size = kwargs.get("recursive_chunk_size")
            recursive_chunk_overlap = kwargs.get("chunk_overlap")
            if recursive_chunk_overlap is None:
                recursive_chunk_overlap = kwargs.get("recursive_chunk_overlap")
            chunks = _split_with_recursive_chunker(
                document,
                chunk_size=recursive_chunk_size,
                chunk_overlap=recursive_chunk_overlap,
            )
            for ch in chunks:
                self.page_chunk_counts[ch["page_no"]] += 1
            return chunks

        # hybrid
        hybrid_chunk_size = _parse_optional_int(kwargs.get("hybrid_chunk_size"), "hybrid_chunk_size")
        if hybrid_chunk_size is None or hybrid_chunk_size <= 0:
            hybrid_chunk_size = _DEFAULT_HYBRID_MAX_TOKENS
        hybrid_merge_peers = _parse_optional_bool(kwargs.get("hybrid_merge_peers"), "hybrid_merge_peers")
        if hybrid_merge_peers is None:
            hybrid_merge_peers = True
        chunker_kwargs = {
            "max_tokens": hybrid_chunk_size,
            "merge_peers": hybrid_merge_peers,
            "tokenizer": self._tokenizer,
            "tokenizer_type": kwargs.get("hybrid_tokenizer_type", "char"),
        }
        chunker = HybridChunker(**chunker_kwargs)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=document, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    async def compose_vectors(self, document: DoclingDocument, chunks, file_path: str, request: Request,
                              **kwargs: dict) -> list[dict]:
        chunker_type = kwargs.get("chunker_type", "recursive")
        _sensitive_infos: list = kwargs.get("_sensitive_infos") or []      # #315 분류 결과
        _gr_masking: bool = bool(kwargs.get("_guardrail_masking", False))   # #315 마스킹 치환 on/off

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            if chunker_type == "recursive":
                chunk_page = chunk["page_no"]
                content = chunk["text"]
                doc_items = chunk["doc_items"]
            else:
                chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
                content = self.safe_join(chunk.meta.headings) + chunk.text
                doc_items = chunk.meta.doc_items

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            # #315 가드레일 분류 후처리: quote 매칭 → content_category 부착(항상) + 마스킹 치환(옵션)
            content, chunk_cats = _gr_apply_to_text(content, _sensitive_infos, _gr_masking)

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(doc_items, document)
                      .set_media_files(doc_items)
                      .set_content_category(sorted(chunk_cats) if chunk_cats else None)
                      ).build()
            vectors.append(vector)

            chunk_index_on_page += 1
            if upload_files:
                file_list = self.get_media_files(doc_items)
                upload_tasks.append(asyncio.create_task(
                    upload_files(file_list, request=request)
                ))

        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)

        artifacts_dir, reference_path = self.get_paths(file_path)
        document = document._with_pictures_refs(image_dir=artifacts_dir, page_no=None, reference_path=reference_path)

        # 민감정보 분류(#315): 청킹 전, 문서 전체를 분류 워크플로우에 1회 호출 → sensitive_infos.
        sensitive_infos: list = []
        if kwargs.get("guardrail_masking", False):
            sensitive_infos = _gr_classify_document(
                _gr_doc_text(document), self._guardrail_url, self._guardrail_workflow_id,
                self._guardrail_api_key, self._guardrail_timeout,
            )

        chunks = self.split_documents(document, **kwargs)
        if len(chunks) == 0:
            raise GenosServiceException(1, "chunk length is 0")
        return await self.compose_vectors(
            document, chunks, file_path, request, _sensitive_infos=sensitive_infos, _guardrail_masking=(kwargs.get("guardrail_masking", False) and self._guardrail_masking_enabled), **kwargs
        )


class HwpProcessor:
    def __init__(self, tokenizer=None, guardrail_url="", guardrail_workflow_id=None, guardrail_api_key="", guardrail_timeout=30, guardrail_masking_enabled=False):
        # 청킹용 토크나이저 (config 기반; 미지정 시 현행 기본값)
        self._tokenizer = tokenizer if tokenizer is not None else _resolve_tokenizer({})
        # PII 마스킹(#315) 접속 정보 — DocumentProcessor 가 config 에서 읽어 주입.
        self._guardrail_url = guardrail_url
        self._guardrail_workflow_id = guardrail_workflow_id
        self._guardrail_api_key = guardrail_api_key
        self._guardrail_timeout = guardrail_timeout
        self._guardrail_masking_enabled = guardrail_masking_enabled

    def get_paths(self, file_path: str):
        """이미지 등 리소스가 저장될 경로 계산 (기존 로직 유지)"""
        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        reference_path = None if artifacts_dir.is_absolute() else artifacts_dir.parent
        return artifacts_dir, reference_path

    def safe_join(self, iterable):
        """청크 내 헤딩들을 텍스트로 합침"""
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ' '.join(map(str, iterable)) + '\n'

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        """SDK 백엔드를 통해 문서를 로드"""
        # 요청마다 독립적인 pipeline_options 생성 (공유 상태 변이 방지) --> save_images, dump_sdk_output
        pipeline_options = PipelineOptions()
        pipeline_options.save_images = kwargs.get('save_images', True)

        use_hwp_sdk = kwargs.get('use_hwp_sdk', True)
        pipeline_options.dump_sdk_output = kwargs.get('dump_sdk_output', False) if use_hwp_sdk else False

        if use_hwp_sdk:
            converter = DocumentConverter(
                format_options={
                    InputFormat.HWP: HwpxFormatOption(
                        pipeline_options=pipeline_options,
                        backend=GenosHwpDocumentBackend
                    ),
                    InputFormat.XML_HWPX: HwpxFormatOption(
                        pipeline_options=pipeline_options,
                        backend=GenosHwpDocumentBackend
                    ),
                }
            )
        else:
            converter = DocumentConverter(
                format_options={
                    InputFormat.HWP: HwpxFormatOption(
                        pipeline_options=pipeline_options,
                        backend=HwpDocumentBackend
                    ),
                    InputFormat.XML_HWPX: HwpxFormatOption(
                        pipeline_options=pipeline_options,
                        backend=HwpxDocumentBackend
                    ),
                }
            )

        conv_result: ConversionResult = converter.convert(Path(file_path).resolve(), raises_on_error=True)
        return conv_result.document

    @staticmethod
    def _hwp_sdk_text_is_empty(document: DoclingDocument) -> bool:
        """GenosHwp SDK 결과 문서에 본문 텍스트가 전혀 없는지 판단(레거시 폴백 트리거용).

        SDK 가 exit 0 으로 "성공"해도 본문을 한 글자도 못 뽑는 경우가 있다(일부 .hwp/.hwpx;
        DRM/암호화 등). 텍스트 run 이 하나도 없으면 True. (convert_processor 와 형평성)
        """
        texts = getattr(document, "texts", None) or []
        return not any((getattr(t, "text", "") or "").strip() for t in texts)

    def split_documents(self, document: DoclingDocument, **kwargs: dict):
        """chunker_type에 따라 HybridChunker 또는 RecursiveCharacterTextSplitter로 분할.

        반환: (chunks, page_chunk_counts). chunks 형식은 chunker_type에 따라 다르다
        (DocChunk 리스트 또는 dict 리스트). compose_vectors가 동일한 chunker_type 분기로 처리한다.
        """
        chunker_type = kwargs.get("chunker_type", "recursive")
        page_chunk_counts: dict[int, int] = defaultdict(int)

        if chunker_type == "recursive":
            recursive_chunk_size = kwargs.get("chunk_size")
            if recursive_chunk_size is None:
                recursive_chunk_size = kwargs.get("recursive_chunk_size")
            recursive_chunk_overlap = kwargs.get("chunk_overlap")
            if recursive_chunk_overlap is None:
                recursive_chunk_overlap = kwargs.get("recursive_chunk_overlap")
            chunks = _split_with_recursive_chunker(
                document,
                chunk_size=recursive_chunk_size,
                chunk_overlap=recursive_chunk_overlap,
            )
            for ch in chunks:
                page_chunk_counts[ch["page_no"]] += 1
            return chunks, page_chunk_counts

        # hybrid
        hybrid_chunk_size = _parse_optional_int(kwargs.get("hybrid_chunk_size"), "hybrid_chunk_size")
        if hybrid_chunk_size is None or hybrid_chunk_size <= 0:
            hybrid_chunk_size = _DEFAULT_HYBRID_MAX_TOKENS
        hybrid_merge_peers = _parse_optional_bool(kwargs.get("hybrid_merge_peers"), "hybrid_merge_peers")
        if hybrid_merge_peers is None:
            hybrid_merge_peers = True
        chunker_kwargs = {
            "max_tokens": hybrid_chunk_size,
            "merge_peers": hybrid_merge_peers,
            "tokenizer": self._tokenizer,
            "tokenizer_type": kwargs.get("hybrid_tokenizer_type", "char"),
        }
        chunker = HybridChunker(**chunker_kwargs)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=document, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks, page_chunk_counts

    async def compose_vectors(self, document: DoclingDocument, chunks, page_chunk_counts: dict[int, int],
                              request: Any, **kwargs: dict) -> list[dict]:
        """빌더를 사용하여 최종 GenOSVectorMeta 리스트 생성"""
        chunker_type = kwargs.get("chunker_type", "recursive")
        _sensitive_infos: list = kwargs.get("_sensitive_infos") or []      # #315 분류 결과
        _gr_masking: bool = bool(kwargs.get("_guardrail_masking", False))   # #315 마스킹 치환 on/off

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []

        for chunk_idx, chunk in enumerate(chunks):
            if chunker_type == "recursive":
                chunk_page = chunk["page_no"]
                content = chunk["text"]
                doc_items = chunk["doc_items"]
            else:
                chunk_page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items[0].prov else 0
                content = self.safe_join(chunk.meta.headings) + chunk.text
                doc_items = chunk.meta.doc_items

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            # #315 가드레일 분류 후처리: quote 매칭 → content_category 부착(항상) + 마스킹 치환(옵션)
            content, chunk_cats = _gr_apply_to_text(content, _sensitive_infos, _gr_masking)

            builder = GenOSVectorMetaBuilder()
            vector_obj = (builder
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(doc_items, document)
                      .set_media_files(doc_items)
                      .set_content_category(sorted(chunk_cats) if chunk_cats else None)
                      ).build()
            vectors.append(vector_obj)
            chunk_index_on_page += 1

        if upload_tasks:
            await asyncio.gather(*upload_tasks)

        return vectors

    async def __call__(self, request: Any, file_path: str, **kwargs: dict):
        """외부에서 호출되는 통합 프로세서 입구"""
        ext = os.path.splitext(file_path)[-1].lower()

        # 1. SDK 백엔드로 문서 변환 (실패 시 폴백)
        document: DoclingDocument = None
        try:
            document = self.load_documents(file_path, **kwargs)
        except Exception as sdk_err:
            _log.warning(f"[HwpProcessor] GenosHwp SDK 변환 실패: {sdk_err}")
            if ext in ('.hwp', '.hwpx'):
                # GenosHwp SDK 실패 시 레거시 백엔드로 폴백 (.hwp → HwpDocumentBackend, .hwpx → HwpxDocumentBackend)
                backend_name = "HwpDocumentBackend" if ext == '.hwp' else "HwpxDocumentBackend"
                try:
                    _log.info(f"[HwpProcessor] {backend_name}로 폴백 시도: {file_path}")
                    kwargs_fallback = dict(kwargs, use_hwp_sdk=False)
                    document = self.load_documents(file_path, **kwargs_fallback)
                    _log.info(f"[HwpProcessor] {backend_name} 폴백 성공")
                except Exception as fallback_err:
                    _log.warning(f"[HwpProcessor] {backend_name} 폴백도 실패: {fallback_err}")
                    raise sdk_err
            else:
                raise

        # 1-b. SDK 가 예외 없이(exit 0) 끝났어도 본문 텍스트가 비어 있으면(빈 doc_items 로
        #      다운스트림이 깨지거나 무의미한 표 청크만 나오는 경우) 레거시 백엔드로 폴백한다.
        #      그래도 본문을 못 얻으면 예외로 올려 DocumentProcessor.__call__ 의 PDF 변환 폴백에
        #      위임한다. (convert_processor 와 형평성 — convert 는 GenosSmartChunker 예외로 잡히지만
        #      attachment 는 recursive splitter 라 예외가 안 나므로 여기서 명시적으로 처리한다.)
        if ext in ('.hwp', '.hwpx') and self._hwp_sdk_text_is_empty(document):
            backend_name = "HwpDocumentBackend" if ext == '.hwp' else "HwpxDocumentBackend"
            _log.warning(f"[HwpProcessor] GenosHwp SDK 결과에 본문 텍스트가 없어 {backend_name} 폴백 시도: {file_path}")
            fallback_doc = None
            try:
                fallback_doc = self.load_documents(file_path, **dict(kwargs, use_hwp_sdk=False))
            except Exception as fallback_err:
                _log.warning(f"[HwpProcessor] {backend_name} 폴백 실패, 상위 PDF 폴백으로 위임: {fallback_err}")
            if fallback_doc is not None and not self._hwp_sdk_text_is_empty(fallback_doc):
                _log.info(f"[HwpProcessor] {backend_name} 폴백 성공(본문 텍스트 확보)")
                document = fallback_doc
            else:
                _log.info(f"[HwpProcessor] {backend_name} 폴백으로도 본문 복구 실패, 상위 PDF 폴백으로 위임")
                raise HwpConversionError(
                    f"HWP/HWPX SDK 결과가 비어 있고 레거시 백엔드로도 본문 복구 실패: {file_path}"
                )

        # 2. 이미지 참조 경로 설정
        artifacts_dir, reference_path = self.get_paths(file_path)
        document = document._with_pictures_refs(
            image_dir=artifacts_dir,
            page_no=None,
            reference_path=reference_path
        )

        # 민감정보 분류(#315): 청킹 전, 문서 전체를 분류 워크플로우에 1회 호출 → sensitive_infos.
        sensitive_infos: list = []
        if kwargs.get("guardrail_masking", False):
            sensitive_infos = _gr_classify_document(
                _gr_doc_text(document), self._guardrail_url, self._guardrail_workflow_id,
                self._guardrail_api_key, self._guardrail_timeout,
            )

        # 3. 청킹 + 4. 벡터화
        chunks, page_chunk_counts = self.split_documents(document, **kwargs)
        if len(chunks) == 0:
            raise GenosServiceException(1, "chunk length is 0")
        return await self.compose_vectors(
            document, chunks, page_chunk_counts, request, _sensitive_infos=sensitive_infos, _guardrail_masking=(kwargs.get("guardrail_masking", False) and self._guardrail_masking_enabled), **kwargs
        )

class GenosServiceException(Exception):
    """GenOS 와의 의존성 부분 제거를 위해 추가"""

    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, errMsg={self.error_msg!r})"


class DocumentProcessor:
    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _resolve_default_attachment_config_path()
        cfg = _load_config(config_path)
        self._config_dir = Path(config_path).resolve().parent

        defaults_cfg = _as_dict(cfg.get("defaults"))
        chunking_cfg = _as_dict(cfg.get("chunking"))
        generic_chunk_cfg = _as_dict(chunking_cfg.get("generic"))
        recursive_chunk_cfg = _as_dict(chunking_cfg.get("recursive"))
        hybrid_chunk_cfg = _as_dict(chunking_cfg.get("hybrid"))
        loaders_cfg = _as_dict(cfg.get("loaders"))
        image_loader_cfg = _as_dict(loaders_cfg.get("image"))
        tabular_loader_cfg = _as_dict(loaders_cfg.get("tabular"))
        whisper_cfg = _as_dict(cfg.get("whisper"))

        # PPT 페이지 단위 설명(page-level image description) 설정.
        # config 위치: formats.ppt.page_description. 공통 모듈(enrichment/page_description)로 파싱.
        formats_cfg = _as_dict(cfg.get("formats"))
        ppt_fmt_cfg = _as_dict(formats_cfg.get("ppt"))
        hwp_fmt_cfg = _as_dict(formats_cfg.get("hwp"))
        ppt_pd_cfg = _as_dict(ppt_fmt_cfg.get("page_description"))
        self._page_desc_options = PageDescriptionOptions.from_config(ppt_pd_cfg, self._config_dir)

        # 청킹용 토크나이저 (chunking config 기반; 미지정 시 현행 기본값)
        self._tokenizer = _resolve_tokenizer(chunking_cfg)

        # 청킹 모드는 chunking.chunker_type 에서 읽는다(구버전 호환: 없으면 defaults.chunker_type).
        chunker_type = str(
            chunking_cfg.get("chunker_type", defaults_cfg.get("chunker_type", "recursive"))
        ).strip().lower()
        if chunker_type not in {"recursive", "hybrid"}:
            _log.warning(
                f"[DocumentProcessor] Unknown defaults.chunker_type '{chunker_type}', fallback to 'recursive'."
            )
            chunker_type = "recursive"

        use_pdf_sdk = _parse_optional_bool(defaults_cfg.get("use_pdf_sdk"), "defaults.use_pdf_sdk")

        # HWP/HWPX 전용 옵션은 formats.hwp 에서 읽는다(구버전 호환: 없으면 defaults 폴백).
        use_hwp_sdk = _parse_optional_bool(hwp_fmt_cfg.get("use_hwp_sdk"), "formats.hwp.use_hwp_sdk")
        if use_hwp_sdk is None:
            use_hwp_sdk = _parse_optional_bool(defaults_cfg.get("use_hwp_sdk"), "defaults.use_hwp_sdk")
        dump_sdk_output = _parse_optional_bool(
            hwp_fmt_cfg.get("dump_sdk_output"), "formats.hwp.dump_sdk_output"
        )
        if dump_sdk_output is None:
            dump_sdk_output = _parse_optional_bool(
                defaults_cfg.get("dump_sdk_output"), "defaults.dump_sdk_output"
            )
        save_images = _parse_optional_bool(hwp_fmt_cfg.get("save_images"), "formats.hwp.save_images")
        if save_images is None:
            save_images = _parse_optional_bool(defaults_cfg.get("save_images"), "defaults.save_images")

        log_level = _parse_optional_int(defaults_cfg.get("log_level"), "defaults.log_level")
        if log_level is None:
            log_level = 4

        # 청크 크기 공통 옵션(chunking.chunk_size). recursive/hybrid 는 chunker_type 으로
        # 택일되므로 값 하나를 활성 모드가 자기 단위(recursive=문자 수 · hybrid=토큰 수)로 해석한다.
        common_chunk_size = _parse_optional_int(chunking_cfg.get("chunk_size"), "chunking.chunk_size")

        # 문자수 기반 통합 청킹 설정. 우선순위: recursive.chunk_size > chunking.chunk_size(공통)
        # > (레거시)chunking.generic.chunk_size > 0.
        recursive_chunk_size = _parse_optional_int(
            recursive_chunk_cfg.get("chunk_size"), "chunking.recursive.chunk_size"
        )
        if recursive_chunk_size is None:
            recursive_chunk_size = common_chunk_size
        if recursive_chunk_size is None:
            recursive_chunk_size = _parse_optional_int(
                generic_chunk_cfg.get("chunk_size"), "chunking.generic.chunk_size"
            )
        if recursive_chunk_size is None or recursive_chunk_size < 0:
            recursive_chunk_size = 0  # 0 = 전체 문서를 1청크로 (문자수 분할 안 함)
        recursive_chunk_overlap = _parse_optional_int(
            recursive_chunk_cfg.get("chunk_overlap", generic_chunk_cfg.get("chunk_overlap")),
            "chunking.recursive.chunk_overlap",
        )
        if recursive_chunk_overlap is None or recursive_chunk_overlap < 0:
            recursive_chunk_overlap = 100

        # hybrid(토큰 수). 우선순위: hybrid.chunk_size > chunking.chunk_size(공통) > 무제한 기본값.
        hybrid_chunk_size = _parse_optional_int(
            hybrid_chunk_cfg.get("chunk_size"), "chunking.hybrid.chunk_size"
        )
        if hybrid_chunk_size is None:
            hybrid_chunk_size = common_chunk_size
        if hybrid_chunk_size is None or hybrid_chunk_size <= 0:
            hybrid_chunk_size = _DEFAULT_HYBRID_MAX_TOKENS
        hybrid_merge_peers = _parse_optional_bool(
            hybrid_chunk_cfg.get("merge_peers"), "chunking.hybrid.merge_peers"
        )
        if hybrid_merge_peers is None:
            hybrid_merge_peers = True
        hybrid_tokenizer_type = str(hybrid_chunk_cfg.get("tokenizer_type", "char")).strip().lower()
        if hybrid_tokenizer_type not in {"char", "huggingface"}:
            _log.warning(
                f"[DocumentProcessor] Unknown chunking.hybrid.tokenizer_type '{hybrid_tokenizer_type}', fallback to 'char'."
            )
            hybrid_tokenizer_type = "char"

        image_ocr_languages = image_loader_cfg.get("ocr_languages", ["kor", "eng"])
        if isinstance(image_ocr_languages, (list, tuple, set)):
            image_ocr_languages = [str(v).strip() for v in image_ocr_languages if str(v).strip()]
        else:
            image_ocr_languages = ["kor", "eng"]
        if not image_ocr_languages:
            image_ocr_languages = ["kor", "eng"]

        tabular_sample_bytes = _parse_optional_int(
            tabular_loader_cfg.get("encoding_detect_sample_bytes"),
            "loaders.tabular.encoding_detect_sample_bytes",
        )
        if tabular_sample_bytes is None or tabular_sample_bytes <= 0:
            tabular_sample_bytes = 10000

        whisper_chunk_sec = _parse_optional_int(whisper_cfg.get("chunk_sec"), "whisper.chunk_sec")
        if whisper_chunk_sec is None or whisper_chunk_sec <= 0:
            whisper_chunk_sec = 29
        whisper_chunk_overlap_ms = _parse_optional_int(
            whisper_cfg.get("chunk_overlap_ms"), "whisper.chunk_overlap_ms"
        )
        if whisper_chunk_overlap_ms is None or whisper_chunk_overlap_ms < 0:
            whisper_chunk_overlap_ms = 300
        whisper_tmp_dir_prefix = str(
            whisper_cfg.get("tmp_dir_prefix", "./tmp_audios_")
        ).strip() or "./tmp_audios_"

        self._default_kwargs = {
            "log_level": log_level,
            "chunker_type": chunker_type,
            "use_pdf_sdk": True if use_pdf_sdk is None else use_pdf_sdk,
            "use_hwp_sdk": True if use_hwp_sdk is None else use_hwp_sdk,
            "dump_sdk_output": False if dump_sdk_output is None else dump_sdk_output,
            "save_images": True if save_images is None else save_images,
            "recursive_chunk_size": recursive_chunk_size,
            "recursive_chunk_overlap": recursive_chunk_overlap,
            "hybrid_chunk_size": hybrid_chunk_size,
            "hybrid_merge_peers": hybrid_merge_peers,
            "hybrid_tokenizer_type": hybrid_tokenizer_type,
            "image_ocr_languages": image_ocr_languages,
            "tabular_encoding_detect_sample_bytes": tabular_sample_bytes,
            "whisper_url": str(
                whisper_cfg.get("url", "http://192.168.74.164:30100/v1/audio/transcriptions")
            ).strip() or "http://192.168.74.164:30100/v1/audio/transcriptions",
            "whisper_model": str(whisper_cfg.get("model", "model")).strip() or "model",
            "whisper_language": str(whisper_cfg.get("language", "ko")).strip() or "ko",
            "whisper_response_format": str(
                whisper_cfg.get("response_format", "json")
            ).strip() or "json",
            "whisper_temperature": str(whisper_cfg.get("temperature", "0")).strip() or "0",
            "whisper_stream": str(whisper_cfg.get("stream", "false")).strip() or "false",
            "whisper_timestamp_granularities": str(
                whisper_cfg.get("timestamp_granularities", "word")
            ).strip() or "word",
            "whisper_chunk_sec": whisper_chunk_sec,
            "whisper_chunk_overlap_ms": whisper_chunk_overlap_ms,
            "whisper_tmp_dir_prefix": whisper_tmp_dir_prefix,
        }

        # 민감정보 분류(#315): GenOS 분류 워크플로우 접속 정보(환경 종속값). on/off 는 요청별 kwargs.
        gm_cfg = _as_dict(cfg.get("guardrail_masking"))
        self._guardrail_url = str(gm_cfg.get("url") or "").strip()
        self._guardrail_workflow_id = _parse_optional_int(gm_cfg.get("workflow_id"), "guardrail_masking.workflow_id")
        self._guardrail_api_key = str(gm_cfg.get("api_key") or "").strip()
        gm_timeout = _parse_optional_int(gm_cfg.get("timeout"), "guardrail_masking.timeout")
        self._guardrail_timeout = gm_timeout if gm_timeout and gm_timeout > 0 else 60
        self._guardrail_masking_enabled = bool(_parse_optional_bool(gm_cfg.get("masking_enabled"), "guardrail_masking.masking_enabled"))

        self.page_chunk_counts = defaultdict(int)
        _gm = dict(
            guardrail_url=self._guardrail_url,
            guardrail_workflow_id=self._guardrail_workflow_id,
            guardrail_api_key=self._guardrail_api_key,
            guardrail_timeout=self._guardrail_timeout,
            guardrail_masking_enabled=self._guardrail_masking_enabled,
        )
        self.hwp_processor = HwpProcessor(tokenizer=self._tokenizer, **_gm)
        self.docx_processor = DocxProcessor(tokenizer=self._tokenizer, **_gm)

    def _merge_runtime_kwargs(self, kwargs: dict) -> dict:
        merged = dict(self._default_kwargs)
        for k, v in kwargs.items():
            if v is not None:
                merged[k] = v
        return merged

    def _get_ppt_pdf_converter(self) -> DocumentConverter:
        """이미지 기반 PPT(→PDF) 파싱용 경량 docling 컨버터(lazy, 캐시).

        첨부용은 dotsocr(genos_layout) 미수행 + do_ocr=False 로 최소 파싱만 수행한다.
        페이지 단위 설명이 켜져 있으면 generate_page_images=True 로 페이지 렌더 이미지를 만든다.
        """
        converter = getattr(self, "_ppt_pdf_converter", None)
        if converter is not None:
            return converter
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = False
        opts.generate_page_images = bool(self._page_desc_options.enabled)
        opts.generate_picture_images = False
        opts.images_scale = self._page_desc_options.images_scale
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        self._ppt_pdf_converter = converter
        return converter

    def _load_ppt_page_documents(self, file_path: str, **kwargs: dict) -> "Optional[list[Document]]":
        """PPT/PPTX → PDF 변환 후 docling 경량 파싱 + 페이지 단위 image description.

        페이지별 Document(metadata['page']=0-based) 리스트를 반환한다. PDF 변환이 불가하면
        None 을 반환해 호출부가 레거시 langchain 경로로 폴백하도록 한다.
        """
        pdf_path = convert_to_pdf(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
        if not pdf_path or not os.path.exists(pdf_path):
            candidate = _get_pdf_path(file_path)
            pdf_path = candidate if os.path.exists(candidate) else None
        if not pdf_path:
            _log.warning(f"[ppt] PDF 변환 실패 — 레거시 경로로 폴백: {os.path.basename(file_path)}")
            return None

        converter = self._get_ppt_pdf_converter()
        document: DoclingDocument = converter.convert(pdf_path, raises_on_error=True).document

        # 페이지별 네이티브 텍스트 수집
        page_text_parts: dict[int, list[str]] = defaultdict(list)
        for item, _ in document.iterate_items():
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            prov = getattr(item, "prov", None) or []
            page_no = prov[0].page_no if prov and getattr(prov[0], "page_no", None) else 1
            page_text_parts[page_no].append(text)
        page_texts: dict[int, str] = {
            pno: "\n".join(parts).strip() for pno, parts in page_text_parts.items()
        }

        # 페이지 단위 image description(옵션). enable=false 면 설명만 skip(파싱은 유지).
        # native text 가 있으면 프롬프트({{page_text}})에 반영해 요청한다.
        page_descs: dict[int, str] = describe_pages(
            document, self._page_desc_options, page_texts=page_texts
        )

        all_pages: set[int] = set()
        if getattr(document, "pages", None):
            all_pages |= set(document.pages.keys())
        all_pages |= set(page_texts.keys()) | set(page_descs.keys())
        if not all_pages:
            all_pages = {1}

        # 같은 페이지의 native text 와 설명을 동일 청크(=동일 Document)로 병합한다.
        documents: list[Document] = []
        for page_no in sorted(all_pages):
            native = page_texts.get(page_no, "").strip()
            desc = page_descs.get(page_no, "").strip()
            if native and desc:
                content = f"{native}\n\n[페이지 이미지 설명]\n{desc}"
            elif desc:
                content = desc
            else:
                content = native
            if not content:
                # 빈 페이지(텍스트/설명 모두 없음) → '.' 폴백으로 Empty document 예외 방지
                content = "."
            documents.append(
                Document(
                    page_content=content,
                    metadata={'source': file_path, 'page': page_no - 1},
                )
            )

        _log.info(
            f"[ppt] page documents 생성: pages={len(documents)}, "
            f"described={len(page_descs)}, description_enabled={self._page_desc_options.enabled}"
        )
        return documents

    def _chunk_ppt_pages(self, documents: "list[Document]", **kwargs: dict) -> "list[Document]":
        """PPT 페이지 Document 를 청크로 구성한다.

        기본: 1 page = 1 chunk. chunk_size(kwargs, 명시된 경우만) 가 주어지면 연속 페이지를
        합친 길이가 chunk_size 이하가 되도록 greedy 병합한다. 병합 청크는 metadata['page']=시작,
        metadata['end_page']=끝(0-based) 을 갖는다.
        """
        self.page_chunk_counts = defaultdict(int)
        if not documents:
            raise Exception('Empty document')

        # 모든 페이지에 추출 가능한 텍스트/설명이 없는 경우(이미지 기반 PPT 등): 페이지별 sentinel('.')
        # 을 이어붙이지 않고, 페이지 전 범위를 span 하는 단일 빈 텍스트('') 청크로 반환한다.
        if all(doc.page_content.strip() in ("", ".") for doc in documents):
            last_page = documents[-1].metadata.get('page', 0)
            self.page_chunk_counts[0] += 1
            return [Document(
                page_content="",
                metadata={
                    'source': documents[0].metadata.get('source'),
                    'page': 0,
                    'end_page': last_page,
                },
            )]

        # chunk_size 우선순위: kwargs['chunk_size'] > chunking.recursive.chunk_size(recursive_chunk_size).
        # 값이 없거나 <=0 이면 1 page = 1 chunk, 있으면 연속 페이지를 그 길이까지 결합.
        chunk_size = _parse_optional_int(kwargs.get('chunk_size'), 'chunk_size')
        if chunk_size is None:
            chunk_size = _parse_optional_int(kwargs.get('recursive_chunk_size'), 'recursive_chunk_size')

        chunks: list[Document] = []
        if chunk_size is None or chunk_size <= 0:
            # 1 page = 1 chunk
            for doc in documents:
                page = doc.metadata.get('page', 0)
                chunks.append(Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, 'end_page': page},
                ))
        else:
            # 연속 페이지 greedy 병합
            cur_parts: list[str] = []
            cur_start: Optional[int] = None
            cur_end: Optional[int] = None
            cur_source = documents[0].metadata.get('source')

            def _flush():
                if cur_parts:
                    chunks.append(Document(
                        page_content="\n\n".join(cur_parts),
                        metadata={'source': cur_source, 'page': cur_start, 'end_page': cur_end},
                    ))

            for doc in documents:
                page = doc.metadata.get('page', 0)
                text = doc.page_content
                if cur_parts and len("\n\n".join(cur_parts + [text])) > chunk_size:
                    _flush()
                    cur_parts = [text]
                    cur_start = page
                    cur_end = page
                else:
                    cur_parts.append(text)
                    if cur_start is None:
                        cur_start = page
                    cur_end = page
            _flush()

        chunks = [c for c in chunks if c.page_content]
        if not chunks:
            raise Exception('Empty document')
        for chunk in chunks:
            self.page_chunk_counts[chunk.metadata.get('page', 0)] += 1
        return chunks

    def get_loader(
        self,
        file_path: str,
        use_pdf_sdk: bool = True,
        image_ocr_languages: Optional[list[str]] = None,
    ):
        ext = os.path.splitext(file_path)[-1].lower()
        real_type = self.get_real_file_type(file_path)

        # 확장자와 실제 파일 타입이 다를 때만 real_type 사용
        if ext != real_type and real_type == 'pdf':
            return PyMuPDFLoader(file_path)
        elif ext != real_type and real_type in ['txt', 'json', 'md']:
            return TextLoader(file_path)
        # 원래 확장자 기반 로직
        elif ext == '.pdf':
            return PyMuPDFLoader(file_path)
        elif ext == '.doc':
            convert_to_pdf(file_path, use_pdf_sdk=use_pdf_sdk)
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            convert_to_pdf(file_path, use_pdf_sdk=use_pdf_sdk)
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            convert_to_pdf(file_path, use_pdf_sdk=use_pdf_sdk)
            languages = image_ocr_languages or ["kor", "eng"]
            if not isinstance(languages, list):
                languages = [str(languages)]
            languages = [str(lang).strip() for lang in languages if str(lang).strip()]
            if not languages:
                languages = ["kor", "eng"]
            # 한국어 OCR 지원을 위한 언어 설정
            return UnstructuredImageLoader(
                file_path,
                languages=languages,  # 한국어 + 영어 OCR
            )
        elif ext in ['.txt', '.json', '.md']:
            return TextLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

    def get_real_file_type(self, file_path: str) -> str:
        """파일 확장자가 아닌 실제 내용으로 파일 타입 판단"""
        with open(file_path, 'rb') as f:
            header = f.read(8)
        if header.startswith(b'%PDF-'):
            return 'pdf'
        elif header.startswith(b'\x89PNG'):
            return 'png'
        elif header.startswith(b'\xff\xd8\xff'):
            return 'jpg'

        # 매직 헤더로 판단할 수 없으면 확장자 사용
        return os.path.splitext(file_path)[-1].lower()

    def convert_md_to_pdf(self, md_path):
        """Markdown 파일을 PDF로 변환"""
        install_packages(['chardet'])
        import chardet

        pdf_path = md_path.replace('.md', '.pdf')
        with open(md_path, 'rb') as f:
            raw_file = f.read()
        candidates = ['utf-8', 'utf-8-sig']
        try:
            det = (chardet.detect(raw_file) or {}).get('encoding') or ''
            # chardet가 ascii/unknown이면 무시. 그 외면 후보에 추가
            if det and det.lower() not in ('ascii', 'unknown'):
                if det.lower() not in [c.lower() for c in candidates]:
                    candidates.append(det)
        except Exception:
            pass
        candidates += ['cp949', 'euc-kr', 'iso-8859-1', 'latin-1']
        md_content = None
        for enc in candidates:
            try:
                md_content = raw_file.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if md_content is None:
            md_content = raw_file.decode('utf-8', errors='replace')

        html_content = markdown(md_content)
        if HTML:
            HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path

    def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
        loader = self.get_loader(
            file_path,
            use_pdf_sdk=kwargs.get('use_pdf_sdk', True),
            image_ocr_languages=kwargs.get("image_ocr_languages"),
        )
        documents = loader.load()

        # 이미지 파일의 경우 텍스트 추출 안되었을 시 기본 텍스트 제공
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            # documents가 없거나, 있어도 모든 page_content가 비어있는 경우
            if not documents or not any(doc.page_content.strip() for doc in documents):
                documents = [Document(page_content=".", metadata={'source': file_path, 'page': 0})]

        return documents

    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        # 문자수 기반 통합 청킹 (chunking.recursive 설정 공유). chunk_size<=0 이면 문서당 1청크.
        chunk_size = kwargs.get('chunk_size')
        if chunk_size is None:
            chunk_size = kwargs.get('recursive_chunk_size', 0)
        chunk_overlap = kwargs.get('chunk_overlap')
        if chunk_overlap is None:
            chunk_overlap = kwargs.get('recursive_chunk_overlap', 100)

        chunks = [
            Document(page_content=part, metadata=dict(doc.metadata))
            for doc in documents
            for part in _char_split_text(
                doc.page_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        ]
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            page = chunk.metadata.get('page', 0)
            self.page_chunk_counts[page] += 1
        return chunks

    def compose_vectors(self, file_path: str, chunks: list[Document], **kwargs: dict) -> list[dict]:
        _sensitive_infos: list = kwargs.get("_sensitive_infos") or []      # #315 분류 결과
        _gr_masking: bool = bool(kwargs.get("_guardrail_masking", False))   # #315 마스킹 치환 on/off
        ext = os.path.splitext(file_path)[-1].lower()
        real_type = self.get_real_file_type(file_path)

        # 확장자와 실제 파일 타입이 다를 때만 real_type 사용
        if ext != real_type and real_type == 'pdf':
            pdf_path = file_path
        elif ext != real_type and real_type in ['txt', 'json', 'md']:
            pdf_path = _get_pdf_path(file_path)
        # 원래 확장자 기반 로직
        elif file_path.endswith('.md'):
            pdf_path = self.convert_md_to_pdf(file_path)
        elif file_path.endswith(('.ppt', '.pptx')):
            pdf_path = _get_pdf_path(file_path)
        else:
            pdf_path = _get_pdf_path(file_path)

        # doc = fitz.open(pdf_path) if (pdf_path and os.path.exists(pdf_path)) else None

        if file_path.endswith(('.ppt', '.pptx')):
            if os.path.exists(pdf_path):
                subprocess.run(["rm", pdf_path], check=True)

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=max([chunk.metadata.get('page', 0) for chunk in chunks]),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )
        current_page = None
        chunk_index_on_page = 0

        vectors = []
        for chunk_idx, chunk in enumerate(chunks):
            page = chunk.metadata.get('page', 1)
            # PPT 페이지 결합 청크는 end_page 로 페이지 범위를 표현(미설정 시 단일 페이지).
            end_page = chunk.metadata.get('end_page', page)
            if ext not in ['.hwpx', '.docx']:
                page += 1
                end_page += 1
            text = chunk.page_content
            # #315 가드레일 분류 후처리: quote 매칭 → content_category 부착(항상) + 마스킹 치환(옵션)
            text, chunk_cats = _gr_apply_to_text(text, _sensitive_infos, _gr_masking)

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            # 첨부용에서는 bbox 정보 추출 X
            # if doc:
            #     fitz_page = doc.load_page(page)
            #     global_metadata['chunk_bboxes'] = json.dumps(merge_overlapping_bboxes([{
            #         'page': page + 1,
            #         'type': 'text',
            #         'bbox': {
            #             'l': rect[0] / fitz_page.rect.width,
            #             't': rect[1] / fitz_page.rect.height,
            #             'r': rect[2] / fitz_page.rect.width,
            #             'b': rect[3] / fitz_page.rect.height,
            #         }
            #     } for rect in fitz_page.search_for(text)], x_tolerance=1 / fitz_page.rect.width,
            #         y_tolerance=1 / fitz_page.rect.height))

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_char': len(text),
                'n_word': len(text.split()),
                'n_line': len(text.splitlines()),
                'i_page': page,
                'e_page': end_page,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                'content_category': sorted(chunk_cats) if chunk_cats else None,  # #315 민감정보 분류 라벨
                **global_metadata
            }))
            chunk_index_on_page += 1

        return vectors

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
        _log.info(f"Setting log level to: {level_name}")

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

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        kwargs = self._merge_runtime_kwargs(kwargs)
        self.setup_logging(kwargs.get('log_level', 4))

        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")

        # 비정상/암호화 파일 사전 감지(이슈 #278/#307): 지원 포맷 매직헤더에 하나도 안 맞고
        # 텍스트도 아니면(=DRM 암호화/손상 바이너리) 파싱/변환 단계의 garbage 처리를 유발하므로
        # 진입부에서 컷한다. 확장자와 무관하게 실제 헤더로 판정.
        bad_reason = _detect_unsupported_file(file_path)
        if bad_reason:
            _log.warning(f"[attachment] 비정상 파일 감지({bad_reason}) — 처리 중단: {file_path}")
            raise GenosServiceException(
                "1", f"{bad_reason} 입니다. 정상 문서로 다시 업로드하세요: {os.path.basename(file_path)}"
            )

        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ('.wav', '.mp3', '.m4a'):
            # TODO(#315): PII 마스킹 미적용(보류) — AudioLoader 는 자체 vector 포맷이라 별도 논의 후 적용.
            # Generate a temporal path saving audio chunks: the audio file is supposed to be splited to several chunks due to limitted length by the model
            file_stem = os.path.basename(file_path).split('.')[0]
            tmp_prefix = str(kwargs.get("whisper_tmp_dir_prefix", "./tmp_audios_"))
            if tmp_prefix.endswith("/"):
                tmp_path = os.path.join(tmp_prefix, file_stem)
            else:
                tmp_path = f"{tmp_prefix}{file_stem}"
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            # Use 'Whisper' model served in-house
            # [!] Modify the request parameters to change a STT model to be used
            loader = AudioLoader(
                file_path=file_path,
                req_url=str(kwargs.get("whisper_url", "")),
                req_data={
                    'model': str(kwargs.get("whisper_model", "model")),
                    'language': str(kwargs.get("whisper_language", "ko")),
                    'response_format': str(kwargs.get("whisper_response_format", "json")),
                    'temperature': str(kwargs.get("whisper_temperature", "0")),
                    'stream': str(kwargs.get("whisper_stream", "false")),
                    'timestamp_granularities[]': str(
                        kwargs.get("whisper_timestamp_granularities", "word")
                    ),
                },
                chunk_sec=int(kwargs.get("whisper_chunk_sec", 29)),
                chunk_overlap_ms=int(kwargs.get("whisper_chunk_overlap_ms", 300)),
                tmp_path=tmp_path
            )
            vectors = loader.return_vectormeta_format()

            # Remove the temporal chunks
            try:
                subprocess.run(['rm', '-r', tmp_path], check=True)
            except:
                pass
            return vectors

        elif ext in ('.csv', '.xlsx'):
            # TODO(#315): PII 마스킹 미적용(보류) — TabularLoader 는 자체 vector 포맷이라 별도 논의 후 적용.
            loader = TabularLoader(
                file_path,
                ext,
                encoding_detect_sample_bytes=int(
                    kwargs.get("tabular_encoding_detect_sample_bytes", 10000)
                ),
            )
            vectors = loader.return_vectormeta_format()
            return vectors

        # [핵심 수정] HWP와 HWPX를 하나의 프로세서로 통합 실행
        elif ext in ('.hwp', '.hwpx'):
            _log.info(f"Processing Korean Document ({ext}) with Unified HwpProcessor")
            try:
                return await self.hwp_processor(request, file_path, **kwargs)
            except Exception as hwp_err:
                # 모든 docling 백엔드 실패 시 LibreOffice PDF 변환으로 최종 폴백
                _log.warning(f"[DocumentProcessor] HWP/HWPX 처리기 전체 실패, PDF 변환 폴백 시도: {hwp_err}")
                converted = convert_to_pdf(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
                if converted:
                    _log.info(f"[DocumentProcessor] PDF 변환 성공: {converted}")
                    documents: list[Document] = self.load_documents(converted, **kwargs)
                    # 민감정보 분류(#315): 청킹 전, 문서 전체를 분류 워크플로우에 1회 호출.
                    sensitive_infos = (_gr_classify_document(
                        _gr_docs_text(documents), self._guardrail_url, self._guardrail_workflow_id,
                        self._guardrail_api_key, self._guardrail_timeout)
                        if kwargs.get("guardrail_masking", False) else [])
                    chunks: list[Document] = self.split_documents(documents, **kwargs)
                    vectors: list[dict] = self.compose_vectors(
                        converted, chunks, _sensitive_infos=sensitive_infos, _guardrail_masking=(kwargs.get("guardrail_masking", False) and self._guardrail_masking_enabled), **kwargs)
                    return vectors
                else:
                    # 이슈 #286 — HWP SDK 도 실패하고 PDF 변환기마저 없으면, 원인을 명확히
                    # 안내한다 (혼란스러운 SDK 에러 대신 PDF 직접 입력/재빌드 안내).
                    if not _has_any_pdf_converter():
                        raise GenosServiceException(
                            1,
                            f"이 전처리기 이미지에는 PDF 변환기(rhwp/LibreOffice/PDF SDK)가 설치되어 "
                            f"있지 않아 '{os.path.basename(file_path)}' 처리에 실패했습니다. "
                            f"PDF 로 변환한 파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 "
                            f"빌드하세요 (genon/README.md 참고).",
                        ) from hwp_err
                    raise hwp_err

        elif ext == '.docx':
            return await self.docx_processor(request, file_path, **kwargs)

        elif ext in ('.ppt', '.pptx'):
            # PPT: PDF 변환 → 경량 docling 파싱 → 페이지 단위 image description(옵션) →
            # 페이지 기반 청킹(기본 1 page 1 chunk, chunk_size 지정 시 페이지 결합).
            # 변환 실패 시에만 레거시 langchain 경로로 폴백한다.
            documents: Optional[list[Document]] = self._load_ppt_page_documents(file_path, **kwargs)
            if documents is None:
                documents = self.load_documents(file_path, **kwargs)
                # 민감정보 분류(#315): 청킹 전 1회 호출.
                sensitive_infos = (_gr_classify_document(
                    _gr_docs_text(documents), self._guardrail_url, self._guardrail_workflow_id,
                    self._guardrail_api_key, self._guardrail_timeout)
                    if kwargs.get("guardrail_masking", False) else [])
                chunks: list[Document] = self.split_documents(documents, **kwargs)
            else:
                # 민감정보 분류(#315): 페이지 결합 청킹 전 1회 호출.
                sensitive_infos = (_gr_classify_document(
                    _gr_docs_text(documents), self._guardrail_url, self._guardrail_workflow_id,
                    self._guardrail_api_key, self._guardrail_timeout)
                    if kwargs.get("guardrail_masking", False) else [])
                chunks = self._chunk_ppt_pages(documents, **kwargs)
            vectors: list[dict] = self.compose_vectors(
                file_path, chunks, _sensitive_infos=sensitive_infos, _guardrail_masking=(kwargs.get("guardrail_masking", False) and self._guardrail_masking_enabled), **kwargs)
            return vectors

        else:
            documents: list[Document] = self.load_documents(file_path, **kwargs)

            # 민감정보 분류(#315): 청킹 전, 문서 전체를 분류 워크플로우에 1회 호출.
            sensitive_infos = (_gr_classify_document(
                _gr_docs_text(documents), self._guardrail_url, self._guardrail_workflow_id,
                self._guardrail_api_key, self._guardrail_timeout)
                if kwargs.get("guardrail_masking", False) else [])

            chunks: list[Document] = self.split_documents(documents, **kwargs)

            vectors: list[dict] = self.compose_vectors(
                file_path, chunks, _sensitive_infos=sensitive_infos, _guardrail_masking=(kwargs.get("guardrail_masking", False) and self._guardrail_masking_enabled), **kwargs)

            return vectors
