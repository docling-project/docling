"""
parser_processor.py — 자체 완결형 파서 파사드.

intelligent_processor.py 와 attachment_processor.py 에서 파싱에 필요한 코드만
복사하여 단일 파일로 구성. 청킹/벡터 조합은 수행하지 않는다.

포맷별 처리 경로:
  .pdf              → IntelligentDocumentProcessor (load → OCR검사 → enrichment)
  .hwp / .hwpx      → HwpDocumentLoader.load_documents()
  .docx             → DocxDocumentLoader.load_documents()
  .wav/.mp3/.m4a    → AudioLoader.transcribe_audio()
  .csv / .xlsx      → TabularLoader.data_dict
  기타              → GenericDocumentLoader.load_documents()  (LangChain Document 목록)
"""

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
import unicodedata
import uuid
import warnings
from collections import defaultdict
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
)
from docling.datamodel.settings import settings as docling_settings
from docling.document_converter import (
    DocumentConverter,
    HwpxFormatOption,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.utils.document_enrichment import check_document, enrich_document
from docling_core.types import DoclingDocument
from docling_core.types.doc import (
    BoundingBox,
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

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: expected mapping, got {type(cfg).__name__}")
    return cfg


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


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

def convert_to_pdf(file_path: str) -> str | None:
    """
    LibreOffice로 PDF 변환을 시도한다.
    실패해도 예외를 던지지 않고 None을 반환한다.

    내부 구현은 `genon.preprocessor.converters.hwp_to_pdf` 모듈에 통합되어 있다.
    기존 호출 동작 보존을 위해 단일 backend만 시도하도록 `disable_fallback=True` 사용.
    """
    from genon.preprocessor.converters.hwp_to_pdf import convert_hwp_to_pdf
    return convert_hwp_to_pdf(file_path, primary="libreoffice", disable_fallback=True)

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
                 chunk_sec: int = 29, tmp_path: str = '.'):
        self.file_path = file_path
        self.tmp_path = tmp_path
        self.chunk_sec = chunk_sec
        self.req_url = req_url
        self.req_data = req_data

    def split_file_as_chunks(self) -> list:
        audio = pydub.AudioSegment.from_file(self.file_path)
        chunk_len = self.chunk_sec * 1000
        n_chunks = math.ceil(len(audio) / chunk_len)
        for i in range(n_chunks):
            start_ms = i * chunk_len
            overlap_start_ms = start_ms - 300 if start_ms > 0 else start_ms
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
# Enrichment 프롬프트 (from intelligent_processor.py)
# ============================================================

_toc_system_prompt = """You are an expert at generating table of contents (목차) from Korean documents. You specialize in regulatory documents, terms of service, contracts, and mixed-format documents that combine formal regulatory structures with general section headers.
""".strip()

_toc_user_prompt = """
Here is the Korean document you need to analyze:

<document>
{raw_text}
</document>

Your task is to extract and organize all structural elements from this document into a hierarchical table of contents. Korean documents often have mixed structures where some sections follow formal regulatory patterns (제x장/절/관/조) while others use general section numbering and headers.

## Analysis Process

Before generating the final table of contents, work through the document systematically in `<analysis>` tags. It's OK for this section to be quite long. Follow these steps:

1. **Document Title Extraction**: Quote the main document title exactly as it appears at the beginning of the document.

2. **Structural Marker Identification**: Scan through the document and quote all the key structural markers you find, such as:
   - Formal regulatory patterns: 제x장, 제x절, 제x관, 제x조
   - General section patterns: numbered headers (1., 2., etc.), lettered headers (가., 나., etc.)
   - Special sections: 부칙, 별지, 별표, etc.

3. **Systematic Section Extraction**: Work through the document from beginning to end, extracting each structural element in order:
   - For each main section, quote the exact title as it appears
   - For each subsection, quote the exact title and note which main section it belongs under
   - For each article/item, quote the exact title and note its parent section
   - Include any appendices, attachments, and addenda

4. **Hierarchy Building**: For each extracted element, explicitly note:
   - What level it should be at (main section, subsection, sub-subsection, etc.)
   - What its parent section is (if any)
   - What numbering it should receive in the final TOC (1., 1.1., 1.1.1., etc.)

5. **Structure Verification**: Review your extracted elements to ensure:
   - All structural elements are captured in document order
   - The hierarchy makes logical sense
   - No elements are duplicated or missed

## Output Requirements

After your analysis, generate the table of contents with this exact format:

```
<toc>
TITLE:<document title>
1. <first main section title>
1.1. <first subsection title>
1.1.1. <first sub-subsection title>
1.2. <second subsection title>
2. <second main section title>
2.1. <subsection under second main section>
3. <third main section title>
</toc>
```

## Formatting Guidelines

- Start with `TITLE:` followed by the document title
- Use hierarchical decimal numbering (1, 1.1, 1.1.1, etc.)
- Follow each number with a space and the original title exactly as it appears
- Maintain the document's logical hierarchy
- Include appendices, attachments, and addenda as separate top-level items
- Extract titles exactly as they appear - do not include explanatory content
- Handle both formal regulatory structures and general section headers
- Wrap the entire table of contents in `<toc></toc>` tags
""".strip()


# ============================================================
# IntelligentDocumentProcessor — PDF 전용 (from intelligent_processor.py)
# 파싱에 필요한 메서드만 포함 (청킹/벡터 메서드 제외)
# ============================================================

class IntelligentDocumentProcessor:

    def __init__(self, config: dict | None = None):
        cfg = _as_dict(config)
        ocr_cfg = _as_dict(cfg.get("ocr"))
        layout_cfg = _as_dict(cfg.get("layout"))
        enrichment_cfg = _as_dict(cfg.get("enrichment"))

        ocr_ep = ocr_cfg.get("ocr_endpoint") or cfg.get("ocr_endpoint", "")
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
        try:
            page_batch_size = int(page_batch_size)
            if page_batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            _log.warning(
                f"[IntelligentDocumentProcessor] Invalid page_batch_size '{page_batch_size}', fallback to 32"
            )
            page_batch_size = 32

        enrichment_url = enrichment_cfg.get("api_url") or cfg.get("enrichment_api_base_url", "")
        enrichment_key = enrichment_cfg.get("api_key") or cfg.get("enrichment_api_key", "")
        enrichment_model = enrichment_cfg.get("model", cfg.get("enrichment_model", "model"))
        do_toc = bool(enrichment_cfg.get("do_toc", cfg.get("do_toc", True)))
        do_metadata = bool(enrichment_cfg.get("do_metadata", cfg.get("do_metadata", True)))

        toc_cfg = _as_dict(enrichment_cfg.get("toc"))
        toc_temperature = toc_cfg.get("temperature", cfg.get("toc_temperature", 0.0))
        toc_top_p = toc_cfg.get("top_p", cfg.get("toc_top_p", 0.00001))
        toc_seed = toc_cfg.get("seed", cfg.get("toc_seed", 33))
        toc_max_tokens = toc_cfg.get("max_tokens", cfg.get("toc_max_tokens", 10000))

        self.ocr_endpoint = ocr_ep
        ocr_options = PaddleOcrOptions(
            force_full_page_ocr=False,
            lang=['korean'],
            ocr_endpoint=ocr_ep,
            text_score=0.3)

        self.page_chunk_counts = defaultdict(int)
        device = AcceleratorDevice.AUTO
        num_threads = 8
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)

        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = True
        self.pipe_line_options.generate_picture_images = True
        self.pipe_line_options.do_ocr = False
        self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.images_scale = 2

        self.pipe_line_options.layout_options.layout_model_type = layout_model_type
        self.pipe_line_options.layout_options.genos_layout_options.endpoint = layout_ep
        self.pipe_line_options.layout_options.genos_layout_options.api_key = layout_key

        docling_settings.perf.page_batch_size = page_batch_size

        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipe_line_options.accelerator_options = accelerator_options

        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False

        # pipe_line_options 의 layout 설정이 deep copy 에 포함되므로 별도 재설정 불필요
        self.ocr_pipe_line_options = self.pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options.model_copy(deep=True)
        self.ocr_pipe_line_options.ocr_options.force_full_page_ocr = True

        self._create_converters()

        self.enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=do_toc,
            toc_doc_type="law",
            extract_metadata=do_metadata,
            toc_api_provider="custom",
            metadata_api_provider="custom",
            toc_api_base_url=enrichment_url,
            metadata_api_base_url=enrichment_url,
            toc_api_key=enrichment_key,
            metadata_api_key=enrichment_key,
            toc_model=enrichment_model,
            metadata_model=enrichment_model,
            toc_temperature=toc_temperature,
            toc_top_p=toc_top_p,
            toc_seed=toc_seed,
            toc_max_tokens=toc_max_tokens,
            toc_system_prompt=_toc_system_prompt,
            toc_user_prompt=_toc_user_prompt,
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
        document = enrich_document(document, self.enrichment_options, **kwargs)
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
                if len(matches) > 10:
                    return True
        return False

    def ocr_all_table_cells(self, document: DoclingDocument, pdf_path) -> DoclingDocument:
        """글리프 깨진 텍스트가 있는 테이블에 대해서만 OCR을 수행합니다."""
        import fitz as _fitz
        import base64 as _base64

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
            doc = _fitz.open(pdf_path)

            for table_idx, table_item in enumerate(document.tables):
                if not table_item.data or not table_item.data.table_cells:
                    continue

                b_ocr = False
                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if self.check_glyph_text(cell.text, threshold=1):
                        b_ocr = True
                        break

                if b_ocr is False:
                    continue

                for cell_idx, cell in enumerate(table_item.data.table_cells):
                    if not table_item.prov:
                        continue

                    page_no = table_item.prov[0].page_no - 1
                    bbox = cell.bbox

                    page = doc.load_page(page_no)
                    cell_bbox = _fitz.Rect(
                        bbox.l, min(bbox.t, bbox.b),
                        bbox.r, max(bbox.t, bbox.b)
                    )

                    bbox_height = cell_bbox.height
                    target_height = 20
                    zoom_factor = target_height / bbox_height if bbox_height > 0 else 1.0
                    zoom_factor = min(zoom_factor, 4.0)
                    zoom_factor = max(zoom_factor, 1)

                    mat = _fitz.Matrix(zoom_factor, zoom_factor)
                    pix = page.get_pixmap(matrix=mat, clip=cell_bbox)
                    img_data = pix.tobytes("png")

                    result = post_ocr_bytes(img_data, timeout=60)
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
        self._intel = IntelligentDocumentProcessor(cfg)

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

        output_cfg = _as_dict(cfg.get("output"))
        self._output_format = self._normalize_output_format(output_cfg.get("format", "json"))
        self._table_format = self._normalize_table_format(output_cfg.get("table_format", "html"))

    @staticmethod
    def _normalize_output_format(value: Any) -> str:
        fmt = str(value).strip().lower()
        if fmt not in {"json", "html", "markdown"}:
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
        ext = os.path.splitext(file_path)[-1].lower()
        loader = TabularLoader(file_path, ext)
        return loader.data_dict

    def _parse_other(self, file_path: str, **kwargs) -> list:
        return self._generic.load_documents(file_path, **kwargs)

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
            else:
                text = getattr(item, "text", "") or ""

            elements.append({
                "category": label_value,
                # "content": {"html": html, "markdown": "", "text": text},
                "content": text,
                "coordinates": coordinates,
                "id": element_id,
                "page": page_no,
            })
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
        """시트 dict → HTML table 문자열."""
        data_rows = sheet.get("data_rows", [])
        if not data_rows:
            return f"<table></table>"
        cols = list(data_rows[0].keys())
        header = "".join(f"<th>{c}</th>" for c in cols)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{row.get(c, '')}</td>" for c in cols) + "</tr>"
            for row in data_rows
        )
        return f"<table><tr>{header}</tr>{rows_html}</table>"

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
        self.setup_logging(kwargs.get('log_level', 4))

        ext = os.path.splitext(file_path)[-1].lower()
        _log.info(f"[DocumentProcessor] file_path={file_path}, ext={ext}")

        if ext in (".wav", ".mp3", ".m4a"):
            text = self._parse_audio(file_path, **kwargs)
            return self._normalize_response(self._audio_to_parse_format(text))

        if ext in (".csv", ".xlsx"):
            data_dict = self._parse_tabular(file_path)
            return self._normalize_response(self._tabular_to_parse_format(data_dict))

        if ext in (".hwp", ".hwpx"):
            doc = self._parse_hwp_hwpx(file_path, **kwargs)
            return self._normalize_response(self._build_docling_response(doc))

        if ext == ".docx":
            doc = self._parse_docx(file_path, **kwargs)
            return self._normalize_response(self._build_docling_response(doc, clear_coordinates=True))

        if ext in (".pdf", ".html", ".htm"):
            doc = self._parse_docling(file_path, **kwargs)
            # result["docling_document"] = self._serialize_docling_document(doc)
            return self._normalize_response(self._build_docling_response(doc))

        # 기타 포맷: doc, ppt, pptx, txt, json, md, jpg, jpeg, png 등
        docs = self._parse_other(file_path, **kwargs)
        return self._normalize_response(self._langchain_to_parse_format(docs))
