from __future__ import annotations

import os
import sys
import subprocess
import warnings
import shutil
import asyncio
import json
import math
import uuid
import fitz
from collections import defaultdict
from datetime import datetime
from markdown2 import markdown

import pydub
import pandas as pd
from pandas import DataFrame
from glob import glob

from fastapi import Request
import requests
import threading

from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter, model_validator
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union
from typing_extensions import Self

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    # TextLoader,                       # TXT
    PyMuPDFLoader,                    # PDF
    DataFrameLoader,                  # DataFrame
    UnstructuredWordDocumentLoader,   # DOC and DOCX
    UnstructuredPowerPointLoader,     # PPT and PPTX
    UnstructuredImageLoader,          # JPG, PNG
    UnstructuredMarkdownLoader,       # Markdown
    UnstructuredFileLoader,           # Generic fallback
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
    import chardet
except ImportError:
    raise RuntimeError("Module 'chardet' not imported. Run `pip install chardet`.")
try:
    from weasyprint import HTML
except ImportError:
    print("Warning: WeasyPrint could not be imported. PDF conversion features will be disabled.")
    HTML = None

# Genos Packages #
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PipelineOptions
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter, HwpxFormatOption
from docling_core.transforms.chunker import BaseChunk, BaseChunker, DocChunk, DocMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc import (
    DocItem, DocItemLabel, DoclingDocument,
    PictureItem, SectionHeaderItem, TableItem, TextItem
)
from docling_core.types.doc.document import LevelNumber, ListItem, CodeItem
from utils import assert_cancelled
from genos_utils import upload_files, merge_overlapping_bboxes
# import platform


def install_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"[!] {package} 패키지가 없습니다. 설치를 시도합니다.")
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
        self.n_chars: Optional[int] = None
        self.n_words: Optional[int] = None
        self.n_lines: Optional[int] = None
        self.i_page: Optional[int] = None
        self.e_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_pages: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_pages: Optional[int] = None
        self.reg_date: Optional[str] = None
        self.chunk_bboxes: Optional[str] = None
        self.media_files: Optional[str] = None
        # self.title: Optional[str] = None
        # self.created_date: Optional[int] = None

    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련된 데이터를 설정"""
        self.text = text
        self.n_chars = len(text)
        self.n_words = len(text.split())
        self.n_lines = len(text.splitlines())
        return self

    def set_page_info(self, i_page: int, i_chunk_on_page: int, n_chunk_of_page: int) -> "GenOSVectorMetaBuilder":
        """페이지 정보 설정"""
        self.i_page = i_page
        self.i_chunk_on_page = i_chunk_on_page
        self.n_chunk_of_pages = n_chunk_of_page
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
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else None
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self

    def set_media_files(self, doc_items: list) -> "GenOSVectorMetaBuilder":
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
        self.media_files = json.dumps(temp_list)
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        return GenOSVectorMeta(
            text=self.text,
            n_chars=self.n_chars,
            n_words=self.n_words,
            n_lines=self.n_lines,
            i_page=self.i_page,
            e_page=self.e_page,
            i_chunk_on_page=self.i_chunk_on_page,
            n_chunk_of_pages=self.n_chunk_of_pages,
            i_chunk_on_doc=self.i_chunk_on_doc,
            n_chunk_of_doc=self.n_chunk_of_doc,
            n_pages=self.n_pages,
            reg_date=self.reg_date,
            chunk_bboxes=self.chunk_bboxes,
            media_files=self.media_files,
            # title=self.title,
            # created_date=self.created_date,
        )


class HwpLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            subprocess.run(['hwp5html', self.file_path, '--output', self.output_dir], check=True, timeout=600)
            converted_file_path = os.path.join(self.output_dir, 'index.xhtml')
            pdf_save_path = self.file_path.replace('.hwp', '.pdf')
            HTML(converted_file_path).write_pdf(pdf_save_path)
            loader = PyMuPDFLoader(pdf_save_path)
            return loader.load()
        except Exception as e:
            print(f"Failed to convert {self.file_path} to XHTML")
            raise e
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)


class TextLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            with open(self.file_path, 'rb') as f:
                raw_file = f.read(100)
            enc_type = chardet.detect(raw_file)['encoding']
            with open(self.file_path, 'r', encoding=enc_type) as f:
                content = f.read()
            html_content = f"<html><body><pre>{content}</pre></body></html>"
            html_file_path = os.path.join(self.output_dir, 'temp.html')
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            pdf_save_path = self.file_path.replace('.txt', '.pdf').replace('.json', '.pdf')
            HTML(html_file_path).write_pdf(pdf_save_path)
            loader = PyMuPDFLoader(pdf_save_path)
            return loader.load()
        except Exception as e:
            print(f"Failed to convert {self.file_path} to XHTML")
            raise e
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)


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
            print(f"[!] Inadequate extension for TabularLoader: {ext}")
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
            else:
                max_len = df[col].str.len().max().item() + 10
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

        df, dtypes_str = self.check_sql_dtypes(df)

        for i in range(len(df.columns)):
            try:
                col = df.columns[0]
                col_type = str(type(col))
                df = df.astype({col: 'str'})
                break
            except:
                raise ValueError(f"Any columns cannot be converted into the string type so that can't load LangChain Documents: {dtypes_str}")

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
                    raise ValueError(f"Any columns cannot be converted into string type so that can't load LangChain Documents: {dtypes_str}")

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
                 chunk_sec: int=29,
                 tmp_path: str='.',
                 ):
        self.file_path = file_path
        self.tmp_path = tmp_path
        self.chunk_sec = chunk_sec
        self.req_url = req_url
        self.req_data = req_data

    def split_file_as_chunks(self) -> list:
        audio = pydub.AudioSegment.from_file(self.file_path)
        chunk_len = self.chunk_sec * 1000
        n_chunks = math.ceil(len(audio)/chunk_len)

        for i in range(n_chunks):
            start_ms = i*chunk_len
            overlap_start_ms = start_ms-300 if start_ms > 0 else start_ms
            end_ms = start_ms+chunk_len
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

                if isinstance(item, SectionHeaderItem) or (isinstance(item, TextItem) and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]):
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

                if isinstance(item, TextItem) or ((not self.merge_list_items) and isinstance(item, ListItem)) or isinstance(item, CodeItem):
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
    tokenizer: Union[PreTrainedTokenizerBase, str] = (
        "sentence-transformers/all-MiniLM-L6-v2"
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
                    # Still room left to add more to this chunk AND still at least one
                    # item left
                    continue
                else:
                    # All the items in the window fit into the chunk and there are no
                    # other items left
                    window_end = num_items  # signalizing the last loop
            elif window_start == window_end:
                # Only one item in the window and it doesn't fit into the chunk. So
                # we'll just make it a chunk for now and it will get split in the
                # plain text splitter.
                window_end += 1
                window_start = window_end
            else:
                # Multiple items in the window but they don't fit into the chunk.
                # However, the existing items must have fit or we wouldn't have
                # gotten here. So we put everything but the last item into the chunk
                # and then start a new window INCLUDING the current window end.
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
            # How much room is there for text after subtracting out the headers and
            # captions:
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
                    # there is room to include the new chunk so add it to the window and
                    # continue
                    window_end += 1
                    new_chunk = candidate
                else:
                    ready_to_append = True

            if ready_to_append or window_end == num_chunks:
                # no more room OR the start of new metadata.  Either way, end the block
                # and use the current window_end as the start of a new block
                if window_start + 1 == window_end:
                    # just one chunk so use it as is
                    output_chunks.append(first_chunk_of_window)
                else:
                    output_chunks.append(new_chunk)
                # no need to reset window_text, etc. because that will be reset in the
                # next iteration in the if window_start == window_end block
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


class HwpxProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.pipeline_options = PipelineOptions()
        self.pipeline_options.save_images = False
        self.converter = DocumentConverter(
            format_options={
                InputFormat.XML_HWPX: HwpxFormatOption(
                        pipeline_options=self.pipeline_options
                )
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
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        save_images = kwargs.get('save_images', False)

        if self.pipeline_options.save_images != save_images:
            self.pipeline_options.save_images = save_images
            self._create_converters()

        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        chunker = HybridChunker(max_tokens=int(1e30), merge_peers=True)
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request, **kwargs: dict) -> list[dict]:
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
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no
            content = self.safe_join(chunk.meta.headings) + chunk.text

            if chunk_page != current_page:
                current_page = chunk_page
                chunk_index_on_page = 0

            vector = (GenOSVectorMetaBuilder()
                      .set_text(content)
                      .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                      .set_chunk_index(chunk_idx)
                      .set_global_metadata(**global_metadata)
                      .set_chunk_bboxes(chunk.meta.doc_items, document)
                      .set_media_files(chunk.meta.doc_items)
                      ).build()
            vectors.append(vector)

            chunk_index_on_page += 1
            file_list = self.get_media_files(chunk.meta.doc_items)
            upload_tasks.append(asyncio.create_task(
                upload_files(file_list, request=request)
            ))

        if upload_tasks:
            await asyncio.gather(*upload_tasks)
        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        artifacts_dir, reference_path = self.get_paths(file_path)
        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)

        chunks: list[DocChunk] = self.split_documents(document, **kwargs)

        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
        else:
            raise GenosServiceException(1, f"chunk length is 0")
        return vectors


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


async def assert_cancelled(request: Request):
    """GenOS 와의 의존성 제거를 위해 추가"""
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")


class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.hwpx_processor = HwpxProcessor()

    def get_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            return PyMuPDFLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return UnstructuredImageLoader(file_path)
        elif ext in ['.txt', '.json', '.md']:
            return TextLoader(file_path)
        elif ext == '.hwp':
            return HwpLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)

    def convert_to_pdf(self, file_path: str):
        out_path = ""
        try:
            subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', out_path, file_path], check=True)
            pdf_path = os.path.basename(file_path).replace(file_path.split('.')[-1], 'pdf')
            return pdf_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting PPT to PDF: {e}")
            return False

    def convert_md_to_pdf(self, md_path):
        """Markdown 파일을 PDF로 변환"""
        install_packages(['chardet'])
        import chardet

        pdf_path = md_path.replace('.md', '.pdf')
        with open(md_path, 'rb') as f:
            raw_file = f.read(100)
        enc_type = chardet.detect(raw_file)['encoding']
        with open(md_path, 'r', encoding=enc_type) as f:
            md_content = f.read()

        html_content = markdown(md_content)
        HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path

    def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
        loader = self.get_loader(file_path)
        documents = loader.load()
        return documents

    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            page = chunk.metadata.get('page', 0)
            self.page_chunk_counts[page] += 1
        return chunks

    def compose_vectors(self, file_path: str, chunks: list[Document], **kwargs: dict) -> list[dict]:
        if file_path.endswith('.md'):
            pdf_path = self.convert_md_to_pdf(file_path)
        elif file_path.endswith('.ppt'):
            pdf_path = self.convert_to_pdf(file_path)
            if not pdf_path:
                return False
        else:
            pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')

        doc = fitz.open(pdf_path) if os.path.exists(pdf_path) else None

        if file_path.endswith('.ppt'):
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
            page = chunk.metadata.get('page', 0)
            text = chunk.page_content

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            if doc:
                fitz_page = doc.load_page(page)
                global_metadata['chunk_bboxes'] = json.dumps(merge_overlapping_bboxes([{
                    'page': page + 1,
                    'type': 'text',
                    'bbox': {
                        'l': rect[0] / fitz_page.rect.width,
                        't': rect[1] / fitz_page.rect.height,
                        'r': rect[2] / fitz_page.rect.width,
                        'b': rect[3] / fitz_page.rect.height,
                    }
                } for rect in fitz_page.search_for(text)], x_tolerance=1 / fitz_page.rect.width,
                    y_tolerance=1 / fitz_page.rect.height))

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_chars': len(text),
                'n_words': len(text.split()),
                'n_lines': len(text.splitlines()),
                'i_page': page,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                **global_metadata
            }))
            chunk_index_on_page += 1

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
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
            await assert_cancelled(request)

            # Remove the temporal chunks
            try:
                subprocess.run(['rm', '-r', tmp_path], check=True)
            except:
                pass
            await assert_cancelled(request)
            return vectors

        elif ext in ('.csv', '.xlsx'):
            loader = TabularLoader(file_path, ext)
            vectors = loader.return_vectormeta_format()
            await assert_cancelled(request)
            return vectors

        elif ext in ('.hwpx'):
            return await self.hwpx_processor(request, file_path, **kwargs)

        else:
            documents: list[Document] = self.load_documents(file_path, **kwargs)
            await assert_cancelled(request)

            chunks: list[Document] = self.split_documents(documents, **kwargs)
            await assert_cancelled(request)

            vectors: list[dict] = self.compose_vectors(file_path, chunks, **kwargs)
            return vectors
