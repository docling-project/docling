from __future__ import annotations

from collections import defaultdict

import asyncio
import fitz
import json
import math
import os
import pandas as pd
import pydub
import requests
import shutil
import subprocess
import sys
import threading
import uuid
import warnings
from datetime import datetime
import logging
from fastapi import Request

_log = logging.getLogger(__name__)

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
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.document import ConversionResult, InputDocument
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.document_converter import DocumentConverter, HwpxFormatOption, WordFormatOption
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


def convert_to_pdf(file_path: str, use_pdf_sdk: bool = True) -> str | None:
    """
    PDF 변환을 시도한다. use_pdf_sdk=True면 PDF SDK, False면 LibreOffice.
    실패해도 예외를 던지지 않고 None을 반환한다.
    """
    if use_pdf_sdk:
        return _convert_to_pdf_sdk(file_path)
    return _convert_to_pdf_libreoffice(file_path)


def _convert_to_pdf_sdk(file_path: str) -> str | None:
    sdk_out_dir: Path | None = None
    keep_out_dir = False  # 폴백 경로 리턴 시 임시 디렉토리 보존
    try:
        # 1. PDF_SDK_HOME 환경변수 (도커 환경) → 2. repo_root/pdf_sdk (로컬 실행)
        pdf_sdk_home = os.environ.get(
            "PDF_SDK_HOME",
            str(Path(__file__).resolve().parent.parent.parent.parent / "pdf_sdk"),
        )
        binary = os.path.join(pdf_sdk_home, "pdfConverter")
        fonts_dir = os.path.join(pdf_sdk_home, "fonts")
        moduledata = os.path.join(pdf_sdk_home, "moduledata")
        font_cache = os.path.join(pdf_sdk_home, "font_cache")
        os.makedirs(font_cache, exist_ok=True)

        in_path = Path(file_path).resolve()
        # NFS 쓰기 권한 / 출력 파일명 추측 미스매치 회피를 위해 임시 디렉토리에 출력
        sdk_out_dir = Path(tempfile.mkdtemp(prefix="pdfsdk_out_"))

        _log.info(
            f"[convert_to_pdf:sdk] preflight: "
            f"binary_exists={os.path.exists(binary)} "
            f"binary_x={os.access(binary, os.X_OK)} "
            f"fonts_dir_exists={os.path.exists(fonts_dir)} "
            f"moduledata_exists={os.path.exists(moduledata)} "
            f"input_exists={in_path.exists()} "
            f"input_size={in_path.stat().st_size if in_path.exists() else 'N/A'} "
            f"sdk_out_dir={sdk_out_dir}"
        )

        with tempfile.TemporaryDirectory() as tmp:
            # fontconfig conf 파일을 현재 SDK 경로 기준으로 패치 (원본은 건드리지 않음)
            patched_conf = _patch_fontconfig(fonts_dir, font_cache, tmp)

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{moduledata}:{env.get('LD_LIBRARY_PATH', '')}"
            env["FONTCONFIG_FILE"] = patched_conf
            env["FONTCONFIG_PATH"] = fonts_dir
            env.setdefault("LANG", "C.UTF-8")
            env.setdefault("LC_ALL", "C.UTF-8")

            cmd = [
                binary,
                "-i", str(in_path),
                "-o", str(sdk_out_dir),
                "-t", tmp,
                "-f", fonts_dir,
                "-e", "-1",
                "-p", "1",
            ]
            _log.info(f"[convert_to_pdf:sdk] cmd: {cmd}")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

            produced = sorted(p for p in sdk_out_dir.glob("*") if p.is_file())
            _log.info(
                f"[convert_to_pdf:sdk] returncode={proc.returncode} "
                f"produced_files={[p.name for p in produced]}"
            )
            _log.info(f"[convert_to_pdf:sdk] stdout={proc.stdout!r}")
            _log.info(f"[convert_to_pdf:sdk] stderr={proc.stderr!r}")

            pdf_files = [p for p in produced if p.suffix.lower() == ".pdf"]
            if proc.returncode == 0 and pdf_files:
                produced_pdf = pdf_files[0]
                target_pdf = in_path.with_suffix('.pdf')
                try:
                    shutil.copy2(produced_pdf, target_pdf)
                    _log.info(f"[convert_to_pdf:sdk] success → {target_pdf}")
                    return str(target_pdf)
                except OSError as e:
                    _log.warning(
                        f"[convert_to_pdf:sdk] target copy failed ({e}); "
                        f"using temp path: {produced_pdf}"
                    )
                    keep_out_dir = True
                    return str(produced_pdf)

            _log.warning(
                f"[convert_to_pdf:sdk] FAILED — "
                f"returncode={proc.returncode}, produced={[p.name for p in produced]}"
            )
            return None
    except subprocess.TimeoutExpired as e:
        _log.error(f"[convert_to_pdf:sdk] timeout after {e.timeout}s for {file_path}")
        return None
    except Exception as e:
        _log.error(f"[convert_to_pdf:sdk] error: {e}", exc_info=True)
        return None
    finally:
        if sdk_out_dir is not None and sdk_out_dir.exists() and not keep_out_dir:
            shutil.rmtree(sdk_out_dir, ignore_errors=True)


def _patch_fontconfig(fonts_dir: str, font_cache: str, tmp_dir: str) -> str:
    """원본 fonts_gen.conf의 <dir>/<cachedir> 경로를 현재 SDK 위치 기준으로 치환한 임시 파일 경로 반환."""
    import re
    src = os.path.join(fonts_dir, "fonts_gen.conf")
    dst = os.path.join(tmp_dir, "fonts.conf")
    with open(src, "r", encoding="utf-8") as f:
        conf = f.read()
    conf = re.sub(r"<dir>[^<]*</dir>", f"<dir>{fonts_dir}</dir>", conf, count=1)
    conf = re.sub(r"<cachedir>[^<]*</cachedir>", f"<cachedir>{font_cache}</cachedir>", conf, count=1)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(conf)
    return dst


def _convert_to_pdf_libreoffice(file_path: str) -> str | None:
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
            _log.warning(f"[convert_to_pdf:libreoffice] stderr: {proc.stderr.strip()}")

        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    except Exception as e:
        _log.error(f"[convert_to_pdf:libreoffice] error: {e}")
        return None


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
        # self.title: Optional[str] = None
        # self.created_date: Optional[int] = None

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
    def __init__(self, file_path: str, ext: str):
        packages = ['openpyxl', 'chardet']
        install_packages(packages)

        self.file_path = file_path
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
            raw_file = f.read(10000)
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
                 tmp_path: str = '.',
                 ):
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
            Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2")
            if Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2").exists()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
    max_tokens: int = int(1e30)  # type: ignore[assignment]
    merge_peers: bool = True
    _inner_chunker: HierarchicalChunker = HierarchicalChunker()

    @model_validator(mode="after")
    def _patch_tokenizer_and_max_tokens(self) -> Self:
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
        return len(self._tokenizer.tokenize(text))

    class _ChunkLengthInfo(BaseModel):
        total_len: int
        text_len: int
        other_len: int

    def _count_chunk_tokens(self, doc_chunk: DocChunk):
        ser_txt = self.serialize(chunk=doc_chunk)
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
            sem_chunker = semchunk.chunkerify(
                self._tokenizer, chunk_size=available_length
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
_RECURSIVE_CHUNK_SIZE_CAP = 60000  # 임베딩 입력 한도(~128K 토큰)의 절반 안전 마진


def _resolve_recursive_tokenizer(tokenizer_id=None):
    if tokenizer_id is None:
        local = Path("/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2")
        tokenizer_id = local if local.exists() else "sentence-transformers/all-MiniLM-L6-v2"
    return AutoTokenizer.from_pretrained(tokenizer_id)


def _split_with_recursive_chunker(
    document: DoclingDocument,
    chunk_size=None,
    chunk_overlap=None,
    tokenizer_id=None,
) -> List[dict]:
    """Markdown export + RecursiveCharacterTextSplitter로 docling 문서를 분할.

    1) char 단위로 1차 분할 (chunk_size 기본 8192).
    2) 한 청크가 60,000 토큰을 초과하면 토큰 단위로 강제 재분할 — 임베딩 한도 절대 상한 (이슈 #183).

    Returns: list of dict {text, page_no, pages, doc_items}
    """
    md_full = document.export_to_markdown(page_break_placeholder=_RECURSIVE_PAGE_BREAK)
    if not md_full:
        return []

    cs = max(int(chunk_size), 1) if chunk_size is not None else 8192
    co = max(int(chunk_overlap), 0) if chunk_overlap is not None else 100
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
    )
    raw_chunks = splitter.split_text(md_full)

    # 60K 토큰 절대 상한 — 어떤 chunk_size 설정에서도 초과 청크는 토큰 단위로 강제 재분할
    tokenizer = _resolve_recursive_tokenizer(tokenizer_id)
    token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=_RECURSIVE_CHUNK_SIZE_CAP,
        chunk_overlap=0,
    )
    safe_chunks: list[str] = []
    for raw in raw_chunks:
        if len(tokenizer.tokenize(raw)) <= _RECURSIVE_CHUNK_SIZE_CAP:
            safe_chunks.append(raw)
        else:
            safe_chunks.extend(token_splitter.split_text(raw))
    raw_chunks = safe_chunks

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
    def __init__(self):
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
            chunks = _split_with_recursive_chunker(
                document,
                chunk_size=kwargs.get("chunk_size"),
                chunk_overlap=kwargs.get("chunk_overlap"),
            )
            for ch in chunks:
                self.page_chunk_counts[ch["page_no"]] += 1
            return chunks

        # hybrid
        chunker = HybridChunker(max_tokens=int(1e30), merge_peers=True)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=document, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    async def compose_vectors(self, document: DoclingDocument, chunks, file_path: str, request: Request,
                              **kwargs: dict) -> list[dict]:
        chunker_type = kwargs.get("chunker_type", "recursive")

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

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(doc_items, document)
                      .set_media_files(doc_items)
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

        chunks = self.split_documents(document, **kwargs)
        if len(chunks) == 0:
            raise GenosServiceException(1, "chunk length is 0")
        return await self.compose_vectors(document, chunks, file_path, request, **kwargs)


class HwpProcessor:
    def __init__(self):
        pass

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

    def split_documents(self, document: DoclingDocument, **kwargs: dict):
        """chunker_type에 따라 HybridChunker 또는 RecursiveCharacterTextSplitter로 분할.

        반환: (chunks, page_chunk_counts). chunks 형식은 chunker_type에 따라 다르다
        (DocChunk 리스트 또는 dict 리스트). compose_vectors가 동일한 chunker_type 분기로 처리한다.
        """
        chunker_type = kwargs.get("chunker_type", "recursive")
        page_chunk_counts: dict[int, int] = defaultdict(int)

        if chunker_type == "recursive":
            chunks = _split_with_recursive_chunker(
                document,
                chunk_size=kwargs.get("chunk_size"),
                chunk_overlap=kwargs.get("chunk_overlap"),
            )
            for ch in chunks:
                page_chunk_counts[ch["page_no"]] += 1
            return chunks, page_chunk_counts

        # hybrid
        chunker = HybridChunker(max_tokens=int(1e30), merge_peers=True)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=document, **kwargs))
        for chunk in chunks:
            if chunk.meta.doc_items[0].prov:
                page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks, page_chunk_counts

    async def compose_vectors(self, document: DoclingDocument, chunks, page_chunk_counts: dict[int, int],
                              request: Any, **kwargs: dict) -> list[dict]:
        """빌더를 사용하여 최종 GenOSVectorMeta 리스트 생성"""
        chunker_type = kwargs.get("chunker_type", "recursive")

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

            builder = GenOSVectorMetaBuilder()
            vector_obj = (builder
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(doc_items, document)
                      .set_media_files(doc_items)
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

        # 2. 이미지 참조 경로 설정
        artifacts_dir, reference_path = self.get_paths(file_path)
        document = document._with_pictures_refs(
            image_dir=artifacts_dir,
            page_no=None,
            reference_path=reference_path
        )

        # 3. 청킹 + 4. 벡터화
        chunks, page_chunk_counts = self.split_documents(document, **kwargs)
        if len(chunks) == 0:
            raise GenosServiceException(1, "chunk length is 0")
        return await self.compose_vectors(document, chunks, page_chunk_counts, request, **kwargs)

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
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.hwp_processor = HwpProcessor()
        self.docx_processor = DocxProcessor()

    def get_loader(self, file_path: str, use_pdf_sdk: bool = True):
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
            # 한국어 OCR 지원을 위한 언어 설정
            return UnstructuredImageLoader(
                file_path,
                languages=["kor", "eng"],  # 한국어 + 영어 OCR
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
        loader = self.get_loader(file_path, use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
        documents = loader.load()

        # 이미지 파일의 경우 텍스트 추출 안되었을 시 기본 텍스트 제공
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            # documents가 없거나, 있어도 모든 page_content가 비어있는 경우
            if not documents or not any(doc.page_content.strip() for doc in documents):
                documents = [Document(page_content=".", metadata={'source': file_path, 'page': 0})]

        return documents

    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        chunk_size = kwargs.get('chunk_size', 1000)
        chunk_overlap = kwargs.get('chunk_overlap', 100)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            page = chunk.metadata.get('page', 0)
            self.page_chunk_counts[page] += 1
        return chunks

    def compose_vectors(self, file_path: str, chunks: list[Document], **kwargs: dict) -> list[dict]:
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
            if ext not in ['.hwpx', '.docx']:
                page += 1
            text = chunk.page_content

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
                'e_page': page,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
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
        self.setup_logging(kwargs.get('log_level', 4))

        _log.info(f"file_path: {file_path}")
        _log.info(f"kwargs: {kwargs}")

        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ('.wav', '.mp3', '.m4a'):
            # Generate a temporal path saving audio chunks: the audio file is supposed to be splited to several chunks due to limitted length by the model
            tmp_path = "./tmp_audios_{}".format(os.path.basename(file_path).split('.')[0])
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            # Use 'Whisper' model served in-house
            # [!] Modify the request parameters to change a STT model to be used
            loader = AudioLoader(
                file_path=file_path,
                req_url="http://192.168.74.164:30100/v1/audio/transcriptions",
                req_data={
                    'model': 'model',
                    'language': 'ko',
                    'response_format': 'json',
                    'temperature': '0',
                    'stream': 'false',
                    'timestamp_granularities[]': 'word'
                },
                chunk_sec=29,  # length(sec) of a chunk from the uploaded audio
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
            loader = TabularLoader(file_path, ext)
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
                    chunks: list[Document] = self.split_documents(documents, **kwargs)
                    vectors: list[dict] = self.compose_vectors(converted, chunks, **kwargs)
                    return vectors
                else:
                    raise hwp_err

        elif ext == '.docx':
            return await self.docx_processor(request, file_path, **kwargs)

        else:
            documents: list[Document] = self.load_documents(file_path, **kwargs)

            chunks: list[Document] = self.split_documents(documents, **kwargs)

            vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)

            return vectors