# 파싱용 전처리기 v.2.2.0 (2026-06-02 Release)
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import fitz
import pandas as pd
import pydub
import requests
import yaml
from fastapi import Request
from markdown2 import markdown
from pandas import DataFrame
from pydantic import BaseModel
from typing_extensions import Self

from langchain_community.document_loaders import (
    DataFrameLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
    UnstructuredImageLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.genos_hwp_backend import GenosHwpDocumentBackend
from docling.backend.genos_msword_backend import GenosMsWordDocumentBackend
from docling.backend.hwp_backend import HwpDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.xml.hwpx_backend import HwpxDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    DataEnrichmentOptions,
    LayoutModelType,
    PaddleOcrOptions,
    PdfPipelineOptions,
    PipelineOptions,
    TableFormerMode,
    UpstageOcrOptions,
)
from docling.datamodel.settings import settings as docling_settings
from docling.document_converter import (
    DocumentConverter,
    HwpxFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.prompts.prompt_manager import LLMApiError
from docling.utils.api_image_request import api_image_request
from docling.utils.document_enrichment import check_document, enrich_document
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
    DescriptionAnnotation,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.base import CoordOrigin
from docling_core.types.doc.document import CodeItem, ContentLayer, ListItem

try:
    from genon.preprocessor.facade.enrichment.custom_fields_enricher import CustomFieldsEnricher
except ImportError:
    CustomFieldsEnricher = None  # type: ignore[assignment,misc]

try:
    from genon.preprocessor.facade.enrichment.metadata_enricher import MetadataEnricher
except ImportError:
    MetadataEnricher = None  # type: ignore[assignment,misc]

from genon.preprocessor.facade.enrichment.enrichment_config import EnrichmentConfig
from genon.preprocessor.facade.enrichment.prompt_files import read_prompt_file
from genon.preprocessor.facade.enrichment.prompt_template import PromptTemplate

try:
    import chardet
except ImportError:
    raise RuntimeError("Module 'chardet' not imported. Run `pip install chardet`.")

try:
    from weasyprint import HTML
except (ImportError, OSError):
    print("Warning: WeasyPrint could not be imported. PDF conversion features will be disabled.")
    HTML = None

try:
    from genos_utils import upload_files
except ImportError:
    upload_files = None

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


# fontTools 로그 억제
for _n in ("fontTools", "fontTools.ttLib", "fontTools.ttLib.ttFont"):
    _lg = logging.getLogger(_n)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False
    logging.getLogger().setLevel(logging.WARNING)

# PDF 변환 대상 확장자
CONVERTIBLE_EXTENSIONS = ['.hwp', '.txt', '.json', '.md', '.ppt', '.pptx', '.docx']


# ============================================================
# 설정 로딩
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
        _log.warning(
            f"[ImageDescriptionOptions] Invalid bool value for '{key}': {value!r}. Fallback to default."
        )
    return None


def _parse_optional_int(value: Any, key: str = "") -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(
                f"[ImageDescriptionOptions] Invalid int value for '{key}': {value!r}. Fallback to default."
            )
        return None


def _parse_optional_float(value: Any, key: str = "") -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if key:
            _log.warning(
                f"Invalid float value for '{key}': {value!r}. Fallback to default."
            )
        return None




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


def _resolve_default_parser_config_path() -> str:
    base_dir = Path(__file__).resolve().parent
    local_config = (base_dir / "../resource_dev/parser_processor_config.yaml").resolve()
    default_config = (base_dir / "../resource/parser_processor_config.yaml").resolve()

    if local_config.exists():
        return str(local_config)
    return str(default_config)


# ============================================================
# 헬퍼 함수 (from attachment_processor.py)
# ============================================================

def _is_libreoffice_available() -> bool:
    """LibreOffice 가 가용한지 확인 (이슈 #286).

    parser_processor 의 convert_to_pdf 는 soffice(LibreOffice) 단독 구현이라,
    rhwp/pdf_sdk 와 무관하게 LibreOffice 가용성만 따져야 정확하다. 빌드 시
    INSTALL_LIBREOFFICE 를 끄면 False. 가용성 판단 자체가 불가하면(import 실패 등)
    True 를 반환해 기존 동작을 유지한다.
    """
    try:
        from genon.preprocessor.converters.hwp_to_pdf.availability import libreoffice_available
        return bool(libreoffice_available())
    except ImportError:
        # facade 단일 파일 실행 등으로 모듈 import 가 안 되는 경우 → 기존 동작 유지(가용 가정)
        return True
    except Exception as exc:
        # 가용성 probe 자체가 예기치 못하게 실패하면 로그만 남기고 파이프라인은 막지 않는다
        _log.warning(f"[_is_libreoffice_available] LibreOffice 가용성 확인 실패: {exc}")
        return True


def convert_to_pdf(file_path: str) -> str | None:
    """
    LibreOffice로 PDF 변환을 시도한다.
    실패해도 예외를 던지지 않고 None을 반환한다.
    """
    # 이슈 #286 — LibreOffice 가 없으면(이 함수는 soffice 단독 사용) 변환 시도가 무의미하므로,
    # PDF 직접 입력을 안내하는 warning 한 번만 남기고 None 을 반환한다.
    if not _is_libreoffice_available():
        _log.warning(
            "[convert_to_pdf] PDF 변환기(LibreOffice)가 설치되어 있지 않습니다 "
            f"(이슈 #286). '{os.path.basename(file_path)}' 변환을 건너뜁니다. PDF 로 변환된 "
            "파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 빌드하세요 (genon/README.md 참고)."
        )
        return None
    try:
        in_path = Path(file_path).resolve()
        out_dir = in_path.parent
        pdf_path = in_path.with_suffix('.pdf')

        env = os.environ.copy()
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_ALL", "C.UTF-8")

        ext = in_path.suffix.lower()
        if ext in ('.ppt', '.pptx'):
            convert_arg = "pdf:impress_pdf_Export"
        elif ext in ('.doc', '.docx'):
            convert_arg = "pdf:writer_pdf_Export"
        elif ext in ('.xls', '.xlsx', '.csv'):
            convert_arg = "pdf:calc_pdf_Export"
        else:
            convert_arg = "pdf"

        try:
            in_path.name.encode('ascii')
            candidates = [in_path]
            tmp_dir = None
        except UnicodeEncodeError:
            tmp_dir = Path(tempfile.mkdtemp())
            ascii_name = unicodedata.normalize('NFKD', in_path.stem).encode('ascii', 'ignore').decode('ascii') or "file"
            ascii_copy = tmp_dir / f"{ascii_name}{in_path.suffix}"
            shutil.copy2(in_path, ascii_copy)
            candidates = [ascii_copy, in_path]

        for cand in candidates:
            cmd = [
                "soffice", "--headless",
                "--convert-to", convert_arg,
                "--outdir", str(out_dir),
                str(cand)
            ]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode == 0 and pdf_path.exists():
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return str(pdf_path)
            _log.warning(f"[convert_to_pdf] stderr: {proc.stderr.strip()}")

        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    except Exception as e:
        _log.error(f"[convert_to_pdf] error: {e}")
        return None


def _get_pdf_path(file_path: str) -> str:
    """다양한 파일 확장자를 PDF 확장자로 변경하는 공통 함수"""
    p = Path(file_path)
    if p.suffix.lower() in CONVERTIBLE_EXTENSIONS:
        return str(p.with_suffix('.pdf'))
    return file_path

def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            _log.warning(f"{package} 패키지가 없습니다. 설치를 시도합니다.")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


# ============================================================
# TextLoader (from attachment_processor.py)
# ============================================================

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
                    content = raw.decode(e)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                content = raw.decode('utf-8', errors='replace')

            html = f"<html><meta charset='utf-8'><body><pre>{content}</pre></body></html>"
            html_path = os.path.join(self.output_dir, 'temp.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            pdf_path = _get_pdf_path(self.file_path)
            if HTML:
                HTML(html_path).write_pdf(pdf_path)
                loader = PyMuPDFLoader(pdf_path)
                return loader.load()
            return [Document(page_content=content, metadata={'source': self.file_path, 'page': 0})]

        except Exception:
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


# ============================================================
# TabularLoader (from attachment_processor.py)
# ============================================================

class TabularLoader:
    def __init__(self, file_path: str, ext: str):
        packages = ['openpyxl', 'chardet']
        install_packages(packages)

        self.file_path = file_path
        if ext == ".csv":
            self.data_dict = self.load_csv_documents(file_path)
        elif ext == ".xlsx":
            self.data_dict = self.load_xlsx_documents(file_path)
        else:
            _log.warning(f"Inadequate extension for TabularLoader: {ext}")
            return

    def check_sql_dtypes(self, df):
        df = df.convert_dtypes()
        res = []
        for col in df.columns:
            dtype = str(df.dtypes[col]).lower()
            if 'int' in dtype:
                sql_dtype = 'BIGINT' if '64' in dtype else 'INT'
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
            else:
                lens = df[col].astype(str).str.len()
                max_len_val = lens.max()
                max_len = int(0 if pd.isna(max_len_val) else max_len_val) + 10
                sql_dtype = f'VARCHAR({max_len})'
            res.append([col, sql_dtype])
        return df, res

    def process_data_rows(self, data: dict):
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
        return {"sheet_name": data["sheet_name"], "data_rows": rows, "data_types": data["dtypes"]}

    def load_csv_documents(self, file_path: str, **kwargs: dict):
        import chardet as _chardet
        with open(file_path, "rb") as f:
            raw_file = f.read(10000)
        enc_type = _chardet.detect(raw_file)['encoding']
        df = pd.read_csv(file_path, encoding=enc_type, index_col=False)
        df = df.fillna('null')
        df, dtypes_str = self.check_sql_dtypes(df)

        for i in range(len(df.columns)):
            try:
                col = df.columns[0]
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
        return {"data": [self.process_data_rows(data)]}

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
            sheets.append({
                "sheet_name": sheet_name,
                "page_column": col,
                "page_column_type": col_type,
                "documents": documents,
                "dtypes": dtypes_str
            })
        data_dict: dict = {"data": []}
        for sheet in sheets:
            data_dict["data"].append(self.process_data_rows(sheet))
        return data_dict


# ============================================================
# AudioLoader (from attachment_processor.py)
# ============================================================

class AudioLoader:
    def __init__(self, file_path: str, req_url: str, req_data: dict,
                 chunk_sec: int = 29, tmp_path: str = '.', chunk_overlap_ms: int = 300):
        self.file_path = file_path
        self.tmp_path = tmp_path
        self.chunk_sec = chunk_sec
        self.chunk_overlap_ms = chunk_overlap_ms
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
        return glob(os.path.join(self.tmp_path, "*.wav"))

    def transcribe_audio(self, file_path_lst: list):
        transcribed_text_chunks = []

        def _send_request(filepath: str):
            files = {'file': (filepath, open(filepath, 'rb'), 'audio/mp3')}
            response = requests.post(self.req_url, data=self.req_data, files=files)
            text = response.json().get('text', ', ')
            transcribed_text_chunks.append({'file_name': os.path.basename(filepath), 'text': text})

        threads = [threading.Thread(target=_send_request, args=(f,)) for f in file_path_lst]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        transcribed_text_chunks.sort(key=lambda x: x['file_name'])
        return "[AUDIO]" + ' '.join([t['text'] for t in transcribed_text_chunks])


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


class PictureDescriptionExtractor:
    @staticmethod
    def extract(item: PictureItem) -> str:
        for annotation in getattr(item, "annotations", []) or []:
            if not isinstance(annotation, DescriptionAnnotation):
                continue
            text = str(getattr(annotation, "text", "") or "").strip()
            if text:
                return text
        return ""


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


# ============================================================
# IntelligentDocumentProcessor — PDF 전용 (from intelligent_processor.py)
# 파싱에 필요한 메서드만 포함 (청킹/벡터 메서드 제외)
# ============================================================

class IntelligentDocumentProcessor:

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        cfg = _as_dict(config)
        self._config_dir = Path(config_path).resolve().parent if config_path else Path.cwd()
        ocr_cfg = _as_dict(cfg.get("ocr"))
        layout_cfg = _as_dict(cfg.get("layout"))
        pdf_cfg = _as_dict(cfg.get("pdf_pipeline"))
        ec = EnrichmentConfig.from_raw(cfg.get("enrichment"), self._config_dir, parent_cfg=cfg)

        # OCR 엔드포인트는 ocr.paddle.ocr_endpoint 가 정식 위치.
        # 구버전 호환: ocr.ocr_endpoint(상위) / 최상위 ocr_endpoint 도 폴백으로 인식.
        paddle_cfg = _as_dict(ocr_cfg.get("paddle"))
        ocr_ep = (
            paddle_cfg.get("ocr_endpoint")
            or ocr_cfg.get("ocr_endpoint")
            or cfg.get("ocr_endpoint", "")
        )

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
        self._glyph_table_cell_threshold = (
            glyph_cell_th if glyph_cell_th and glyph_cell_th > 0 else 1
        )
        glyph_doc_th = _parse_optional_int(
            glyph_cfg.get("document_threshold"), "ocr.glyph_detection.document_threshold"
        )
        self._glyph_document_threshold = (
            glyph_doc_th if glyph_doc_th and glyph_doc_th > 0 else 10
        )
        raw_ocr_mode = str(ocr_cfg.get("ocr_mode", cfg.get("ocr_mode", "auto"))).lower().strip()
        if raw_ocr_mode not in {"auto", "force", "disable"}:
            _log.warning(f"[IntelligentDocumentProcessor] Unknown ocr_mode '{raw_ocr_mode}', fallback to 'auto'")
            raw_ocr_mode = "auto"
        self.ocr_mode = raw_ocr_mode

        layout_model_type_str = str(
            layout_cfg.get("layout_model_type", cfg.get("layout_model_type", "genos_layout"))
        ).lower().strip()
        if layout_model_type_str == LayoutModelType.DOCLING_LAYOUT.value:
            layout_model_type = LayoutModelType.DOCLING_LAYOUT
        else:
            if layout_model_type_str != LayoutModelType.GENOS_LAYOUT.value:
                _log.warning(
                    f"[IntelligentDocumentProcessor] Unknown layout_model_type '{layout_model_type_str}', "
                    f"fallback to '{LayoutModelType.GENOS_LAYOUT.value}'"
                )
            layout_model_type = LayoutModelType.GENOS_LAYOUT

        genos_layout_cfg = _as_dict(layout_cfg.get("genos_layout"))
        layout_ep = genos_layout_cfg.get("endpoint") or cfg.get("layout_endpoint", "")
        layout_key = genos_layout_cfg.get("api_key") or cfg.get("layout_api_key", "")
        page_batch_size = genos_layout_cfg.get("page_batch_size", cfg.get("page_batch_size", 32))
        max_completion_tokens = _parse_optional_int(
            genos_layout_cfg.get("max_completion_tokens"),
            "layout.genos_layout.max_completion_tokens",
        )
        if max_completion_tokens is None or max_completion_tokens <= 0:
            max_completion_tokens = 16384
        try:
            page_batch_size = int(page_batch_size)
            if page_batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            _log.warning(
                f"[IntelligentDocumentProcessor] Invalid page_batch_size '{page_batch_size}', fallback to 32"
            )
            page_batch_size = 128

        # DotsOCR VLM 호출/생성 파라미터 (yaml 누락·무효 시 기본값 폴백)
        layout_model = genos_layout_cfg.get("model") or "dots-mocr"
        layout_timeout = _parse_optional_int(
            genos_layout_cfg.get("timeout"), "layout.genos_layout.timeout"
        )
        if layout_timeout is None or layout_timeout <= 0:
            layout_timeout = 1200  # 이슈 #278: per-page hang 방지(GenosLayoutOptions 기본과 통일)
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

        ocr_options = self._build_ocr_options(ocr_cfg, paddle_endpoint=ocr_ep)
        if isinstance(ocr_options, UpstageOcrOptions):
            self.ocr_endpoint = ocr_options.api_endpoint
        else:
            self.ocr_endpoint = ocr_ep

        self.page_chunk_counts = defaultdict(int)

        device_str = str(pdf_cfg.get("device", "auto")).lower().strip()
        device = _ACCELERATOR_DEVICE_MAP.get(device_str)
        if device is None:
            _log.warning(
                f"[IntelligentDocumentProcessor] Unknown pdf_pipeline.device '{device_str}', fallback to 'auto'"
            )
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

        table_mode_str = str(pdf_cfg.get("table_structure_mode", "accurate")).lower().strip()
        table_structure_mode = _TABLE_FORMER_MODE_MAP.get(table_mode_str)
        if table_structure_mode is None:
            _log.warning(
                f"[IntelligentDocumentProcessor] Unknown pdf_pipeline.table_structure_mode "
                f"'{table_mode_str}', fallback to 'accurate'"
            )
            table_structure_mode = TableFormerMode.ACCURATE

        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = (
            True if generate_page_images is None else generate_page_images
        )
        self.pipe_line_options.generate_picture_images = (
            True if generate_picture_images is None else generate_picture_images
        )
        self.pipe_line_options.do_ocr = False
        self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.images_scale = images_scale

        self.pipe_line_options.layout_options.layout_model_type = layout_model_type
        self.pipe_line_options.layout_options.genos_layout_options.endpoint = layout_ep
        self.pipe_line_options.layout_options.genos_layout_options.api_key = layout_key
        self.pipe_line_options.layout_options.genos_layout_options.max_completion_tokens = max_completion_tokens
        self.pipe_line_options.layout_options.genos_layout_options.model = layout_model
        self.pipe_line_options.layout_options.genos_layout_options.timeout = layout_timeout
        self.pipe_line_options.layout_options.genos_layout_options.retry_count = layout_retry_count
        self.pipe_line_options.layout_options.genos_layout_options.temperature = layout_temperature
        self.pipe_line_options.layout_options.genos_layout_options.top_p = layout_top_p
        self.pipe_line_options.layout_options.genos_layout_options.repetition_penalty = layout_repetition_penalty

        docling_settings.perf.page_batch_size = page_batch_size

        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = table_structure_mode
        self.pipe_line_options.accelerator_options = accelerator_options

        # docling 모델(TableFormer 등) 로컬 경로. config 에 값이 있을 때만 설정하고,
        # 비어있으면 설정하지 않아 docling 기본 캐시 동작을 그대로 유지(backward compat).
        models_cfg = _as_dict(cfg.get("models"))
        artifacts_path = models_cfg.get("artifacts_path")
        if artifacts_path:
            self.pipe_line_options.artifacts_path = Path(artifacts_path)

        # xlsx(엑셀) 처리 설정(이슈 #288). formats.xlsx 아래에 둔다. 출력은 parse-JSON(시트당 HTML 표).
        #   tabular(기본): openpyxl 로 병합셀 unmerge+forward-fill 후 시트→HTML 표(병합 헤더 보존).
        #   docling: docling MsExcel 백엔드로 DoclingDocument 생성 후 parse-JSON 직렬화.
        #   tabular.{header_row, multi_table}: tabular 모드 전용 세부 옵션
        formats_cfg = _as_dict(cfg.get("formats"))
        xlsx_cfg = _as_dict(formats_cfg.get("xlsx"))
        tabular_cfg = _as_dict(xlsx_cfg.get("tabular"))
        xlsx_mode = str(xlsx_cfg.get("processing_mode", "tabular")).strip().lower()
        if xlsx_mode not in {"docling", "tabular"}:
            _log.warning(
                f"[DocumentProcessor] Unknown formats.xlsx.processing_mode '{xlsx_mode}', fallback to 'tabular'."
            )
            xlsx_mode = "tabular"
        self._xlsx_cfg = {
            "processing_mode": xlsx_mode,
            "header_row": _parse_optional_int(tabular_cfg.get("header_row"), "formats.xlsx.tabular.header_row") or 0,
            "multi_table": bool(_parse_optional_bool(tabular_cfg.get("multi_table"), "formats.xlsx.tabular.multi_table")),
        }

        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False

        # pipe_line_options 의 layout 설정이 deep copy 에 포함되므로 별도 재설정 불필요
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True

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
        self.custom_fields_enrichers: "list[CustomFieldsEnricher]" = (
            [CustomFieldsEnricher(**c) for c in ec.custom_fields_cfgs]
            if CustomFieldsEnricher is not None else []
        )

        # 사용자가 커스텀 metadata 신호(prompt/파일/output_fields/parser)를 하나라도 지정한 경우
        # 커스텀 MetadataEnricher를 사용한다. 지정되지 않으면 docling 내장 enricher가 동작한다
        # (하위 호환). built-in default system prompt 가 이 게이트를 흔들지 않도록
        # system_prompt 유무가 아닌 has_custom_metadata 로 판단한다.
        self.metadata_enricher: "Optional[MetadataEnricher]" = (
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

        self.enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=ec.toc.do_toc,
            toc_doc_type=ec.toc.doc_type,
            # 커스텀 MetadataEnricher가 있으면 docling 내장 비활성화
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
            _log.warning(
                f"[IntelligentDocumentProcessor] Unknown ocr.engine '{ocr_engine}', fallback to 'paddle'"
            )
            ocr_engine = "paddle"

        if ocr_engine == "upstage":
            upstage_cfg = _as_dict(ocr_cfg.get("upstage"))
            upstage_api_key = upstage_cfg.get("api_key", "") or os.getenv("UPSTAGE_API_KEY", "")

            # yaml 의 잘못된 값 (예: timeout: "60s") 으로 startup 이 깨지지 않도록
            # 변환 실패 시 default 로 fallback + warning. 페이지 batch size 등 다른
            # 정수 파싱 패턴과 동일.
            raw_timeout = upstage_cfg.get("timeout", 60)
            try:
                upstage_timeout = int(raw_timeout)
                if upstage_timeout <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                _log.warning(
                    f"[IntelligentDocumentProcessor] Invalid ocr.upstage.timeout '{raw_timeout}', fallback to 60"
                )
                upstage_timeout = 60

            raw_text_score = upstage_cfg.get("text_score", 0.5)
            try:
                upstage_text_score = float(raw_text_score)
            except (TypeError, ValueError):
                _log.warning(
                    f"[IntelligentDocumentProcessor] Invalid ocr.upstage.text_score '{raw_text_score}', fallback to 0.5"
                )
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
                _log.warning(
                    f"[IntelligentDocumentProcessor] Invalid ocr.paddle.lang '{raw_lang}', fallback to ['korean']"
                )
            paddle_lang = ["korean"]

        raw_text_score = paddle_cfg.get("text_score", 0.3)
        try:
            paddle_text_score = float(raw_text_score)
        except (TypeError, ValueError):
            _log.warning(
                f"[IntelligentDocumentProcessor] Invalid ocr.paddle.text_score '{raw_text_score}', fallback to 0.3"
            )
            paddle_text_score = 0.3

        return PaddleOcrOptions(
            force_full_page_ocr=False,
            lang=paddle_lang,
            ocr_endpoint=paddle_endpoint,
            text_score=paddle_text_score,
        )

    def _create_converters(self):
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
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)

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
        save_images = kwargs.get('save_images', True)
        include_wmf = kwargs.get('include_wmf', False)

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

    def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        try:
            document = enrich_document(document, self.enrichment_options, **kwargs)
            return document
        except LLMApiError as e:
            # Preserve provider error payload as-is for load status error message.
            raise GenosServiceException("1", e.raw_error_message) from e

    def _get_or_create_image_description_enricher(self) -> FacadeImageDescriptionEnricher:
        enricher = getattr(self, "image_description_enricher", None)
        if enricher is None:
            # 테스트 등에서 __init__ 우회 시 legacy attribute 기반으로 재구성
            legacy_options = ImageDescriptionOptions.from_legacy_processor(self)
            enricher = FacadeImageDescriptionEnricher(legacy_options)
            self.image_description_enricher = enricher
        return enricher

    def enrich_image_descriptions(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        enricher = self._get_or_create_image_description_enricher()
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

    def check_glyph_text(self, text: str, threshold: int = 1) -> bool:
        if not text:
            return False
        matches = re.findall(r'GLYPH\w*', text)
        if len(matches) >= threshold:
            return True
        return False

    def check_glyphs(self, document: DoclingDocument) -> bool:
        for item, level in document.iterate_items():
            if isinstance(item, TextItem) and hasattr(item, 'prov') and item.prov:
                matches = re.findall(r'GLYPH\w*', item.text)
                if len(matches) > self._glyph_document_threshold:
                    return True
        return False

    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> DoclingDocument:
        """글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR을 수행합니다."""
        import io as _io
        import base64 as _base64
        from PIL import Image as _Image

        def post_ocr_bytes(img_bytes: bytes, timeout=60) -> dict:
            HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
            payload = {"file": _base64.b64encode(img_bytes).decode("ascii"), "fileType": 1, "visualize": False}
            r = requests.post(self.ocr_endpoint, json=payload, headers=HEADERS, timeout=timeout)
            if not r.ok:
                raise RuntimeError(f"OCR HTTP {r.status_code}: {r.text[:500]}")
            return r.json()

        def extract_ocr_fields(resp: dict):
            if resp is None:
                return [], [], []
            if resp.get("errorCode") not in (0, None):
                return [], [], []
            ocr_results = resp.get("result", {}).get("ocrResults", [])
            if not ocr_results:
                return [], [], []
            pruned = ocr_results[0].get("prunedResult", {})
            if not pruned:
                return [], [], []
            rec_texts = pruned.get("rec_texts", [])
            rec_scores = pruned.get("rec_scores", [])
            rec_boxes = pruned.get("rec_boxes", [])
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
                                _Image.LANCZOS,
                            )

                        buf = _io.BytesIO()
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


# ============================================================
# HwpDocumentLoader — HWP/HWPX 전용 (from attachment_processor.py)
# load_documents() 메서드만 포함
# ============================================================

class HwpDocumentLoader:

    def __init__(self):
        pass

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
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


# ============================================================
# DocxDocumentLoader — DOCX 전용 (from attachment_processor.py)
# load_documents() 메서드만 포함
# ============================================================

class DocxDocumentLoader:

    def __init__(self):
        self.pipeline_options = PipelineOptions()
        self.converter = DocumentConverter(
            format_options={
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, backend=GenosMsWordDocumentBackend
                ),
            }
        )

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        return conv_result.document


# ============================================================
# GenericDocumentLoader — 기타 포맷 (from attachment_processor.py)
# load_documents() 메서드만 포함
# ============================================================

class GenericDocumentLoader:

    def __init__(self):
        pass

    def get_real_file_type(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            header = f.read(8)
        if header.startswith(b'%PDF-'):
            return 'pdf'
        elif header.startswith(b'\x89PNG'):
            return 'png'
        elif header.startswith(b'\xff\xd8\xff'):
            return 'jpg'
        return os.path.splitext(file_path)[-1].lower()

    def get_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        real_type = self.get_real_file_type(file_path)

        if ext != real_type and real_type == 'pdf':
            return PyMuPDFLoader(file_path)
        elif ext != real_type and real_type in ['txt', 'json', 'md']:
            return TextLoader(file_path)
        elif ext == '.pdf':
            return PyMuPDFLoader(file_path)
        elif ext == '.doc':
            convert_to_pdf(file_path)
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            convert_to_pdf(file_path)
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            convert_to_pdf(file_path)
            return UnstructuredImageLoader(file_path, languages=["kor", "eng"])
        elif ext in ['.txt', '.json', '.md']:
            return TextLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

    def _load_image_documents_fallback(self, file_path: str) -> list[Document]:
        """UnstructuredImageLoader의 __str__ NoneType 오류를 우회해 이미지 요소를 안전하게 적재."""
        from unstructured.partition.image import partition_image

        elements = partition_image(filename=file_path, languages=["kor", "eng"])
        documents: list[Document] = []

        for element in elements:
            text = getattr(element, "text", "")
            if text is None:
                text = ""
            elif not isinstance(text, str):
                text = str(text)

            metadata: dict[str, Any] = {"source": file_path}
            if hasattr(element, "metadata") and element.metadata is not None:
                try:
                    metadata.update(element.metadata.to_dict())
                except Exception:
                    pass

            if hasattr(element, "category"):
                metadata["category"] = element.category

            if hasattr(element, "to_dict"):
                element_id = element.to_dict().get("element_id")
                if element_id:
                    metadata["element_id"] = element_id

            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def load_documents(self, file_path: str, **kwargs: dict) -> list:
        loader = self.get_loader(file_path)
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            documents = loader.load()
        except TypeError as exc:
            if ext in ['.jpg', '.jpeg', '.png'] and "__str__ returned non-string" in str(exc):
                _log.warning(f"[GenericDocumentLoader] Image loader fallback: {file_path} ({exc})")
                documents = self._load_image_documents_fallback(file_path)
            else:
                raise

        if ext in ['.jpg', '.jpeg', '.png']:
            if not documents or not any((doc.page_content or "").strip() for doc in documents):
                documents = [Document(page_content=".", metadata={'source': file_path, 'page': 0})]

        return documents


# ============================================================
# GenosServiceException
# ============================================================

class GenosServiceException(Exception):
    def __init__(self, error_code: str, error_msg: Optional[str] = None,
                 msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params or {}

    def __repr__(self) -> str:
        return f"GenosServiceException(code={self.code!r}, errMsg={self.error_msg!r})"


# ============================================================
# DocumentProcessor — 메인 클래스
# ============================================================

class DocumentProcessor:
    """
    파싱 단계만 수행하고 결과를 JSON으로 반환하는 파사드.
    청킹/벡터 조합은 수행하지 않음.

    IS_PARSER: main.py 가 이 프로세서가 /parser API 전용임을 식별하는 데 사용.
    """

    IS_PARSER: bool = True

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _resolve_default_parser_config_path()

        cfg = _load_config(config_path)
        self._intel = IntelligentDocumentProcessor(cfg, config_path=config_path)

        # xlsx/csv 처리 설정은 intel 프로세서가 동일 config에서 이미 파싱함 → 재사용
        self._xlsx_cfg = self._intel._xlsx_cfg

        defaults_cfg = _as_dict(cfg.get("defaults"))
        log_level = _parse_optional_int(defaults_cfg.get("log_level"), "defaults.log_level")
        if log_level is None:
            log_level = 4
        self._log_level = log_level

        self._hwp = HwpDocumentLoader()
        self._docx = DocxDocumentLoader()
        self._generic = GenericDocumentLoader()

        # 신/구 설정 스키마 동시 지원
        whisper_cfg = _as_dict(cfg.get("whisper"))
        attach_cfg = _as_dict(cfg.get("attachment"))

        self._whisper_url = whisper_cfg.get("url", attach_cfg.get("whisper_url", ""))
        self._whisper_req_data = {
            "model": whisper_cfg.get("model", attach_cfg.get("whisper_model", "model")),
            "language": whisper_cfg.get("language", attach_cfg.get("whisper_language", "ko")),
            "response_format": whisper_cfg.get(
                "response_format", attach_cfg.get("whisper_response_format", "json")
            ),
            "temperature": whisper_cfg.get("temperature", attach_cfg.get("whisper_temperature", "0")),
            "stream": whisper_cfg.get("stream", attach_cfg.get("whisper_stream", "false")),
            "timestamp_granularities[]": whisper_cfg.get(
                "timestamp_granularities", attach_cfg.get("whisper_timestamp_granularities", "word")
            ),
        }
        try:
            self._whisper_chunk_sec = int(
                whisper_cfg.get("chunk_sec", attach_cfg.get("whisper_chunk_sec", 29))
            )
        except (TypeError, ValueError):
            _log.warning("[DocumentProcessor] Invalid whisper.chunk_sec value, fallback to 29")
            self._whisper_chunk_sec = 29

        try:
            self._whisper_chunk_overlap_ms = int(
                whisper_cfg.get("chunk_overlap_ms", attach_cfg.get("whisper_chunk_overlap_ms", 300))
            )
        except (TypeError, ValueError):
            _log.warning("[DocumentProcessor] Invalid whisper.chunk_overlap_ms value, fallback to 300")
            self._whisper_chunk_overlap_ms = 300

        output_cfg = _as_dict(cfg.get("output"))
        self._output_format = self._normalize_output_format(output_cfg.get("format", "json"))
        self._table_format = self._normalize_table_format(output_cfg.get("table_format", "html"))

    @staticmethod
    def _normalize_output_format(value: Any) -> str:
        fmt = str(value).strip().lower()
        if fmt not in {"json", "html", "markdown", "docling"}:
            _log.warning(f"[DocumentProcessor] Invalid output.format '{value}', fallback to 'json'")
            return "json"
        return fmt

    @staticmethod
    def _normalize_table_format(value: Any) -> str:
        fmt = str(value).strip().lower()
        if fmt not in {"html", "markdown"}:
            _log.warning(f"[DocumentProcessor] Invalid output.table_format '{value}', fallback to 'html'")
            return "html"
        return fmt

    # ------------------------------------------------------------------
    # 포맷별 파싱 메서드
    # ------------------------------------------------------------------

    def _parse_docling(self, file_path: str, **kwargs) -> DoclingDocument:
        """
        intelligent_processor.__call__ 흐름 중 enrichment 까지만 실행.
        load → OCR 검사 → ocr_all_table_cells → enrichment
        """
        ocr_mode = getattr(self._intel, "ocr_mode", "auto")

        if ocr_mode == "force":
            document = self._intel.load_documents_with_docling_ocr(file_path, **kwargs)
        else:
            document = self._intel.load_documents(file_path, **kwargs)
            if ocr_mode == "auto":
                if (not check_document(document, self._intel.enrichment_options)
                        or self._intel.check_glyphs(document)):
                    document = self._intel.load_documents_with_docling_ocr(file_path, **kwargs)

        if ocr_mode != "disable" and self._intel.ocr_endpoint:
            document = self._intel.ocr_all_table_cells(document, file_path)

        # output_path, output_file = os.path.split(file_path)
        # filename, _ = os.path.splitext(output_file)
        # artifacts_dir = Path(f"{output_path}/{filename}")
        # reference_path = None if artifacts_dir.is_absolute() else artifacts_dir.parent

        # document = document._with_pictures_refs(
        #     image_dir=artifacts_dir, page_no=None, reference_path=reference_path
        # )
        document = self._intel.enrichment(document, **kwargs)

        return document

    def _parse_hwp_hwpx(self, file_path: str, **kwargs) -> DoclingDocument:
        """HwpDocumentLoader.load_documents() 만 실행. 실패 시 폴백 적용."""
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            return self._hwp.load_documents(file_path, **kwargs)
        except Exception as sdk_err:
            _log.warning(f"[DocumentProcessor] HWP SDK 실패: {sdk_err}")
            if ext in (".hwp", ".hwpx"):
                try:
                    return self._hwp.load_documents(
                        file_path, **dict(kwargs, use_hwp_sdk=False)
                    )
                except Exception:
                    # 모든 백엔드 실패 시 LibreOffice → PDF → intelligent 경로
                    converted = convert_to_pdf(file_path)
                    if converted:
                        return self._parse_docling(converted, **kwargs)
                    # 이슈 #286 — HWP SDK 도 실패하고 LibreOffice(이 경로의 유일한 변환기)마저
                    # 없으면, 원인을 명확히 안내한다 (혼란스러운 SDK 에러 대신 PDF 직접 입력/재빌드).
                    if not _is_libreoffice_available():
                        raise GenosServiceException(
                            1,
                            f"이 전처리기 이미지에는 PDF 변환기(LibreOffice)가 설치되어 "
                            f"있지 않아 '{os.path.basename(file_path)}' 처리에 실패했습니다. "
                            f"PDF 로 변환한 파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 "
                            f"빌드하세요 (genon/README.md 참고).",
                        ) from sdk_err
                    raise sdk_err
            raise

    def _parse_docx(self, file_path: str, **kwargs) -> DoclingDocument:
        return self._docx.load_documents(file_path, **kwargs)

    def _parse_audio(self, file_path: str, **kwargs) -> str:
        tmp_path = f"./tmp_audios_{os.path.basename(file_path).split('.')[0]}"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        try:
            loader = AudioLoader(
                file_path=file_path,
                req_url=self._whisper_url,
                req_data=self._whisper_req_data,
                chunk_sec=self._whisper_chunk_sec,
                chunk_overlap_ms=self._whisper_chunk_overlap_ms,
                tmp_path=tmp_path,
            )
            audio_chunks = loader.split_file_as_chunks()
            return loader.transcribe_audio(audio_chunks)
        finally:
            try:
                subprocess.run(["rm", "-r", tmp_path], check=True)
            except Exception:
                pass

    def _parse_tabular(self, file_path: str) -> dict:
        """xlsx/csv → {"data":[{"sheet_name","title","data_rows":[{col:val}]}]} (이슈 #288).

        표 감지(멀티헤더 자동 + 1시트 복수표)는 xlsx_processor.load_tables 에 위임한다.
        - 제목행은 title 로(컨텍스트), 계층 헤더는 `상위_하위` flatten, 그 아래 컬럼명행이 leaf.
        - multi_table=True 면 빈 행 기준 복수 표를 표별로 분리.
        헤더명(원본, 한글 가능)을 그대로 key 로 쓴다(HTML 셀 내용 — Weaviate 키 제약 무관).
        """
        from genon.preprocessor.converters.xlsx_processor import load_tables

        tables = load_tables(
            file_path,
            header_row=self._xlsx_cfg["header_row"],
            multi_table=self._xlsx_cfg["multi_table"],
        )
        data: list[dict] = []
        for t in tables:
            headers = t["headers"]
            data_rows = [dict(zip(headers, values)) for values in t["data_rows"]]
            data.append({
                "sheet_name": t["sheet_name"],
                "title": t["title"],
                "data_rows": data_rows,
            })
        return {"data": data}

    def _parse_other(self, file_path: str, **kwargs) -> list:
        return self._generic.load_documents(file_path, **kwargs)

    async def _apply_docling_post_enrichment(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
        """Facade 후처리 enrichment 훅."""
        try:
            document = self._intel.enrich_image_descriptions(document, **kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] facade image enrichment skipped: {exc}")
        try:
            document = await self._intel.enrich_metadata(document, **kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] metadata enrichment skipped: {exc}")
        try:
            document = await self._intel.enrich_custom_fields(document, **kwargs)
        except Exception as exc:
            _log.warning(f"[DocumentProcessor] custom_fields enrichment skipped: {exc}")
        return document

    # ------------------------------------------------------------------
    # 직렬화 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _get_normalized_coords(
        bbox, page_w: float, page_h: float
    ) -> list:
        """BoundingBox → 정규화된 4-코너 좌표 ([top-left, top-right, bottom-right, bottom-left])."""
        if bbox.coord_origin != CoordOrigin.TOPLEFT:
            bbox = bbox.to_top_left_origin(page_h)
        l = round(bbox.l / page_w, 4)
        t = round(bbox.t / page_h, 4)
        r = round(bbox.r / page_w, 4)
        b = round(bbox.b / page_h, 4)
        return [
            {"x": l, "y": t},
            {"x": r, "y": t},
            {"x": r, "y": b},
            {"x": l, "y": b},
        ]

    @staticmethod
    def _item_to_html(item, element_id: int, doc: DoclingDocument) -> str:
        """DocItem → element 수준 HTML 문자열."""
        label_value = item.label.value if hasattr(item.label, "value") else str(item.label)

        if isinstance(item, TableItem):
            return item.export_to_html(doc=doc) or f"<table id='{element_id}'></table>"

        if isinstance(item, PictureItem):
            return f"<figure id='{element_id}'></figure>"

        text = (getattr(item, "text", "") or "").replace("\n", "<br>")

        if label_value == "title":
            return f"<h1 id='{element_id}'>{text}</h1>"

        if label_value == "section_header":
            level = max(1, min(getattr(item, "level", 1), 6))
            return f"<h{level} id='{element_id}'>{text}</h{level}>"

        if label_value == "list_item":
            return f"<p id='{element_id}' data-category='list'>{text}</p>"

        return f"<p id='{element_id}' data-category='{label_value}'>{text}</p>"

    @staticmethod
    def _export_table_content(
        item: TableItem, doc: DoclingDocument, table_format: str = "html"
    ) -> str:
        """TableItem을 지정한 포맷(html/markdown)으로 변환."""
        try:
            if table_format == "markdown":
                text = item.export_to_markdown(doc=doc)
            else:
                text = item.export_to_html(doc=doc)
            if text and text.strip():
                return text
        except Exception:
            pass

        try:
            if item.data and item.data.table_cells:
                parts = []
                for cell in item.data.table_cells:
                    value = getattr(cell, "text", "")
                    if value and str(value).strip():
                        parts.append(str(value).strip())
                if parts:
                    return " ".join(parts)
        except Exception:
            pass

        return getattr(item, "text", "") or ""

    @staticmethod
    def _docling_sheet_prefix(item, doc) -> str:
        """xlsx docling 표의 부모 그룹(name='sheet: X')에서 시트명을 뽑아 '시트명: X\\n' 접두 생성.
        시트 그룹이 없으면 '' 반환(비-xlsx 문서엔 실질 미적용)."""
        try:
            parent = item.parent.resolve(doc) if getattr(item, "parent", None) else None
            name = getattr(parent, "name", None)
        except Exception:
            name = None
        if not name:
            return ""
        if name.startswith("sheet: "):
            name = name[len("sheet: "):]
        name = name.strip()
        return f"시트명: {name}\n" if name else ""

    @staticmethod
    def _docling_to_parse_format(doc: DoclingDocument, table_format: str = "html") -> dict:
        """DoclingDocument → sample_result.json 호환 출력 포맷."""
        elements = []
        element_id = 0
        default_page_no = 1
        try:
            if getattr(doc, "pages", None):
                default_page_no = min(doc.pages.keys())
        except Exception:
            default_page_no = 1

        for item, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            prov_list = getattr(item, "prov", None) or []
            prov = prov_list[0] if len(prov_list) > 0 else None

            page_no = getattr(prov, "page_no", None)
            if not isinstance(page_no, int) or page_no <= 0:
                page_no = default_page_no

            coordinates = []
            if prov is not None:
                try:
                    page_info = doc.pages.get(page_no)
                    if page_info is None or page_info.size is None:
                        raise ValueError("no page size")
                    page_w = page_info.size.width
                    page_h = page_info.size.height
                    coordinates = DocumentProcessor._get_normalized_coords(prov.bbox, page_w, page_h)
                except Exception:
                    coordinates = []

            label_value = item.label.value if hasattr(item.label, "value") else str(item.label)
            # if label_value == "section_header":
            #     level = max(1, min(getattr(item, "level", 1), 6))
            #     category = f"heading{level}"

            # html = DocumentProcessor._item_to_html(item, element_id, doc)
            if isinstance(item, TableItem):
                text = DocumentProcessor._export_table_content(
                    item=item,
                    doc=doc,
                    table_format=table_format,
                )
                # xlsx docling 표면 시트명 접두 추가(비-xlsx 는 "" 라 영향 없음).
                text = DocumentProcessor._docling_sheet_prefix(item, doc) + text
            else:
                text = getattr(item, "text", "") or ""

            element = {
                "category": label_value,
                # "content": {"html": html, "markdown": "", "text": text},
                "content": text,
                "coordinates": coordinates,
                "id": element_id,
                "page": page_no,
            }
            if isinstance(item, PictureItem):
                image_description = PictureDescriptionExtractor.extract(item)
                if image_description:
                    # 최종 소비계층에서 별도 필드 매핑 없이 바로 활용할 수 있도록
                    # picture 의 content 를 이미지 설명 텍스트로 채운다.
                    element["content"] = image_description

            elements.append(element)
            element_id += 1

        # full_html = "\n".join(e["content"]["html"] for e in elements)

        return {
            # "content": {"html": full_html, "markdown": "", "text": ""},
            "elements": elements,
            # "model": "genonai-parser",
            "usage": {"pages": doc.num_pages()},
        }

    @staticmethod
    def _serialize_docling_document(doc: DoclingDocument) -> dict:
        """DoclingDocument를 JSON 직렬화 가능한 dict로 변환."""
        try:
            # pydantic v2 호환 방식 (enum/datetime 등 JSON-safe 변환 포함)
            return doc.model_dump(mode="json")
            # return doc.export_to_dict()
        except Exception:
            try:
                # model_dump가 호환되지 않을 때 문자열 JSON을 다시 dict로 복원
                return json.loads(doc.model_dump_json())
            except Exception:
                # 최후 폴백: docling 기본 export
                return doc.export_to_dict()

    @staticmethod
    def _replace_markdown_tables_with_html(doc: DoclingDocument, markdown_text: str) -> str:
        """Markdown 문자열의 테이블 블록을 순차적으로 HTML 테이블로 치환."""
        if not markdown_text:
            return markdown_text

        out = markdown_text
        for item, _ in doc.iterate_items(
            included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
        ):
            if not isinstance(item, TableItem):
                continue

            try:
                md_table_raw = item.export_to_markdown(doc=doc)
                html_table = item.export_to_html(doc=doc)
            except Exception:
                continue

            if not md_table_raw or not html_table:
                continue

            md_table = md_table_raw.strip()
            if not md_table:
                continue

            idx = out.find(md_table)
            if idx >= 0:
                out = out[:idx] + html_table + out[idx + len(md_table):]
            else:
                idx_raw = out.find(md_table_raw)
                if idx_raw >= 0:
                    out = out[:idx_raw] + html_table + out[idx_raw + len(md_table_raw):]

        return out

    def _docling_to_content(self, doc: DoclingDocument) -> str:
        """DoclingDocument를 output.format에 따라 content 문자열로 변환."""
        output_format = getattr(self, "_output_format", "json")
        table_format = getattr(self, "_table_format", "html")
        layers = {ContentLayer.BODY, ContentLayer.FURNITURE}

        if output_format == "html":
            return doc.export_to_html(included_content_layers=layers)

        if output_format == "markdown":
            markdown_text = doc.export_to_markdown(included_content_layers=layers)
            if table_format == "html":
                return self._replace_markdown_tables_with_html(doc, markdown_text)
            return markdown_text

        return ""

    @staticmethod
    def _normalize_response(result: dict) -> dict:
        """응답에 content / elements / usage 키가 항상 존재하도록 보장."""
        result.setdefault("content", "")
        result.setdefault("elements", [])
        result.setdefault("usage", {"pages": 0})
        return result

    @staticmethod
    def _content_response(content: str, pages: int = 0) -> dict:
        """content 전용 출력 포맷."""
        return {
            "elements": [],
            "usage": {"pages": pages},
            "content": content,
        }

    def _build_docling_response(self, doc: DoclingDocument, clear_coordinates: bool = False) -> dict:
        """Docling 경로의 최종 응답 생성."""
        output_format = getattr(self, "_output_format", "json")
        table_format = getattr(self, "_table_format", "html")

        if output_format == "docling":
            # 복원 가능한 DoclingDocument 원본 JSON(model_dump)을 그대로 반환.
            # DoclingDocument.model_validate(data["document"]) 로 무손실 복원 가능 → Chunk API 입력.
            # clear_coordinates / table_format 은 원본 보존을 위해 docling 포맷에서는 무시한다.
            return {
                "document": self._serialize_docling_document(doc),
                "usage": {"pages": doc.num_pages()},
            }

        if output_format == "json":
            result = self._docling_to_parse_format(doc, table_format=table_format)
            if clear_coordinates:
                for element in result.get("elements", []):
                    element["coordinates"] = []
            return result

        try:
            pages = max(1, int(doc.num_pages()))
        except Exception:
            pages = 0

        content = self._docling_to_content(doc)
        return self._content_response(content, pages=pages)

    @staticmethod
    def _audio_to_parse_format(text: str) -> dict:
        """전사 텍스트 → parse format."""
        return {
            "elements": [
                {
                    "category": "paragraph",
                    "content": text,
                    "coordinates": [],
                    "id": 0,
                    "page": 1,
                }
            ],
            "usage": {"pages": 1},
        }

    @staticmethod
    def _sheet_to_html(sheet: dict) -> str:
        """시트(표) dict → HTML table 문자열(시트명 + 제목 컨텍스트 접두 포함)."""
        name = str(sheet.get("sheet_name", "") or "").strip()
        title = str(sheet.get("title", "") or "").strip()
        prefix = f"시트명: {name}\n" if name else ""
        if title:
            prefix += f"{title}\n"
        data_rows = sheet.get("data_rows", [])
        if not data_rows:
            return f"{prefix}<table></table>"
        cols = list(data_rows[0].keys())
        header = "".join(f"<th>{c}</th>" for c in cols)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>"
            for row in data_rows
        )
        return f"{prefix}<table><tr>{header}</tr>{rows_html}</table>"

    @staticmethod
    def _tabular_to_parse_format(data_dict: dict) -> dict:
        """TabularLoader.data_dict → parse format. 시트 하나당 element 하나."""
        elements = []
        sheets = data_dict.get("data", [])
        for idx, sheet in enumerate(sheets):
            elements.append({
                "category": "table",
                "content": DocumentProcessor._sheet_to_html(sheet),
                "coordinates": [],
                "id": idx,
                "page": idx + 1,
            })
        return {
            "elements": elements,
            "usage": {"pages": len(sheets)},
        }

    @staticmethod
    def _langchain_to_parse_format(docs: list) -> dict:
        """LangChain Document 목록 → parse format."""
        elements = []
        for idx, doc in enumerate(docs):
            page = doc.metadata.get("page", idx)
            if isinstance(page, int):
                page = page + 1  # 0-based → 1-based
            elements.append({
                "category": "paragraph",
                "content": doc.page_content,
                "coordinates": [],
                "id": idx,
                "page": page,
            })
        num_pages = max((e["page"] for e in elements), default=0)
        return {
            "elements": elements,
            "usage": {"pages": num_pages},
        }


    def setup_logging(self, level_num: int):
        def get_level_name(level_num: int) -> str:
            level_map = {5: "DEBUG", 4: "INFO", 3: "WARNING", 2: "ERROR", 1: "CRITICAL", 0: "NOLOG"}
            return level_map.get(level_num, "INFO")
        level_name = get_level_name(level_num)
        print(f"Setting log level to: {level_name}")
        if level_name == "NOLOG" or not hasattr(logging, level_name):
            logging.disable(logging.CRITICAL)
            return
        level = getattr(logging, level_name.upper())
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logging.getLogger().setLevel(level)

    # ------------------------------------------------------------------
    # 메인 진입점
    # ------------------------------------------------------------------

    async def __call__(self, request: Request, file_path: str, **kwargs) -> dict:
        runtime_level = kwargs.get('log_level')
        self.setup_logging(runtime_level if runtime_level is not None else self._log_level)

        ext = os.path.splitext(file_path)[-1].lower()
        _log.info(f"[DocumentProcessor] file_path={file_path}, ext={ext}")

        # 비정상/암호화 파일 사전 감지(이슈 #278/#307): 지원 포맷 매직헤더에 하나도 안 맞고
        # 텍스트도 아니면(=DRM 암호화/손상 바이너리) 파싱/변환 단계의 garbage 처리를 유발하므로
        # 진입부에서 컷한다. 확장자와 무관하게 실제 헤더로 판정.
        bad_reason = _detect_unsupported_file(file_path)
        if bad_reason:
            _log.warning(f"[parser] 비정상 파일 감지({bad_reason}) — 처리 중단: {file_path}")
            raise GenosServiceException(
                "1", f"{bad_reason} 입니다. 정상 문서로 다시 업로드하세요: {os.path.basename(file_path)}"
            )

        if ext in (".wav", ".mp3", ".m4a"):
            text = self._parse_audio(file_path, **kwargs)
            return self._normalize_response(self._audio_to_parse_format(text))

        if ext in (".csv", ".xlsx", ".xlsm"):
            # docling 모드: MsExcel/Csv 백엔드로 DoclingDocument 생성 후 parse-JSON 직렬화.
            if self._xlsx_cfg["processing_mode"] == "docling":
                from genon.preprocessor.converters.xlsx_processor import build_docling_document
                doc = build_docling_document(file_path)
                return self._normalize_response(self._build_docling_response(doc))
            # tabular 모드(기본): openpyxl 병합셀 처리 → 시트당 HTML 표.
            data_dict = self._parse_tabular(file_path)
            return self._normalize_response(self._tabular_to_parse_format(data_dict))

        enrichment_context: dict = {}

        if ext in (".hwp", ".hwpx"):
            doc = self._parse_hwp_hwpx(file_path, **kwargs)
            doc = await self._apply_docling_post_enrichment(doc, _enrichment_context=enrichment_context, **kwargs)
            result = self._build_docling_response(doc)
            if enrichment_context.get("metadata"):
                result["metadata"] = enrichment_context["metadata"]
            return self._normalize_response(result)

        if ext == ".docx":
            doc = self._parse_docx(file_path, **kwargs)
            doc = await self._apply_docling_post_enrichment(doc, _enrichment_context=enrichment_context, **kwargs)
            result = self._build_docling_response(doc, clear_coordinates=True)
            if enrichment_context.get("metadata"):
                result["metadata"] = enrichment_context["metadata"]
            return self._normalize_response(result)

        if ext in (".pdf", ".html", ".htm"):
            doc = self._parse_docling(file_path, _enrichment_context=enrichment_context, **kwargs)
            doc = await self._apply_docling_post_enrichment(doc, _enrichment_context=enrichment_context, **kwargs)
            result = self._build_docling_response(doc)
            if enrichment_context.get("metadata"):
                result["metadata"] = enrichment_context["metadata"]
            return self._normalize_response(result)

        # 기타 포맷: doc, ppt, pptx, txt, json, md, jpg, jpeg, png 등
        docs = self._parse_other(file_path, **kwargs)
        return self._normalize_response(self._langchain_to_parse_format(docs))