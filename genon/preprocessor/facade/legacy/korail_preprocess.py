from __future__ import annotations

import json
import os
from pathlib import Path

from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple

from fastapi import Request

# docling imports

from docling.backend.xml.hwpx_backend import HwpxDocumentBackend
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.pipeline.simple_pipeline import SimplePipeline
# from docling.datamodel.document import ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    # EasyOcrOptions,
    # OcrEngine,
    # PdfBackend,
    PdfPipelineOptions,
    TableFormerMode,
    TesseractOcrOptions,
    PipelineOptions
)

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    HwpxFormatOption,
    FormatOption
)
from docling.datamodel.document import ConversionResult
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
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem
)
from collections import Counter
import re
import json
import warnings
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

from genos_utils import upload_files

# ============================================
#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""


class PageBasedChunker(BaseChunker):
    """페이지 단위로 청킹하는 청커 - 1페이지 = 1청크"""
    
    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """각 페이지를 하나의 청크로 생성
        
        Args:
            dl_doc: 청킹할 문서
            
        Yields:
            페이지별 청크
        """
        # 페이지별로 아이템 그룹화
        page_items: Dict[int, List[DocItem]] = defaultdict(list)
        
        # 모든 아이템을 페이지별로 분류
        for item, _ in dl_doc.iterate_items():
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    page_no = prov.page  # 0-based
                    page_items[page_no].append(item)
        
        # 페이지별로 청크 생성
        for page_no in sorted(page_items.keys()):
            items = page_items[page_no]
            
            # 페이지별 청크 생성
            chunk = DocChunk(
                text="",  # 텍스트는 나중에 처리
                meta=DocMeta(
                    doc_items=items,
                    headings=[],  # 페이지 기반이므로 헤딩 구조 없음
                    captions=[],
                    origin=dl_doc.origin,
                )
            )
            
            yield chunk


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


class DocumentProcessor:
    def __init__(self):
        '''
        initialize Document Converter
        '''
        self.page_chunk_counts = defaultdict(int)
        device = AcceleratorDevice.AUTO
        num_threads = 8
        accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
        # PDF 파이프라인 옵션 설정
        self.pipe_line_options = PdfPipelineOptions()
        self.pipe_line_options.generate_page_images = True
        self.pipe_line_options.generate_picture_images = True
        self.pipe_line_options.do_ocr = False
        # self.pipe_line_options.ocr_options.lang = ["ko", 'en']
        # self.pipe_line_options.ocr_options.model_storage_directory = "./.EasyOCR/model"
        # self.pipe_line_options.ocr_options.force_full_page_ocr = True
        # ocr_options = TesseractOcrOptions()
        # ocr_options.lang = ['kor', 'kor_vert', 'eng', 'jpn', 'jpn_vert']
        # ocr_options.path = './.tesseract/tessdata'
        # self.pipe_line_options.ocr_options = ocr_options
        self.pipe_line_options.artifacts_path = Path("/models")  # Path("/nfs-root/aiModel/.cache/huggingface/hub/models--ds4sd--docling-models/snapshots/4659a7d29247f9f7a94102e1f313dad8e8c8f2f6/")
        self.pipe_line_options.do_table_structure = True
        self.pipe_line_options.images_scale = 2
        self.pipe_line_options.table_structure_options.do_cell_matching = True
        self.pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.pipe_line_options.accelerator_options = accelerator_options
        # Simple 파이프라인 옵션을 인스턴스 변수로 저장
        self.simple_pipeline_options = PipelineOptions()
        # 기본 컨버터들 생성
        self._create_converters()
        
    def _create_converters(self):
        """컨버터들을 생성하는 헬퍼 메서드"""
        # HWP와 HWPX 모두 지원하는 통합 컨버터
        self.converter = DocumentConverter(
                format_options={
                    InputFormat.XML_HWPX: HwpxFormatOption(
                        pipeline_options=self.simple_pipeline_options,
                    ),
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipe_line_options,
                        backend=DoclingParseV4DocumentBackend
                    ),
                }
            )
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                ),
            }
        )

    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        try:
            conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        except Exception as e:
            conv_result: ConversionResult = self.second_converter.convert(file_path, raises_on_error=True)
        
        return conv_result.document

    def load_documents(self, file_path: str, **kwargs) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        # PageBasedChunker 사용 - 페이지당 1개 청크
        chunker: PageBasedChunker = PageBasedChunker()
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        
        for chunk in chunks:
            if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
                self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page] += 1
        
        return chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request,
                              **kwargs: dict) -> list[dict]:
        # 전역 메타데이터 설정
        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )
        
        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # 페이지별 청킹이므로 첫 번째 아이템의 페이지 번호 사용
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no
            
            # 청크 텍스트 생성 - 페이지 기반이므로 헤딩 없이 텍스트만
            content = self._generate_chunk_text(chunk)
            
            # 빈 페이지는 건너뛰기
            if not content.strip():
                continue
            
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

    def _generate_chunk_text(self, chunk: DocChunk) -> str:
        """청크에서 텍스트 생성 (페이지 기반이므로 모든 아이템을 텍스트로 처리)"""
        text_parts = []
        
        for item in chunk.meta.doc_items:
            if isinstance(item, (TextItem, ListItem, SectionHeaderItem, CodeItem)):
                if item.text:
                    text_parts.append(item.text)
            elif isinstance(item, TableItem):
                # 테이블을 Markdown으로 변환 - preprocess.py의 _extract_table_text와 동일
                table_text = self._extract_table_text(item)
                if table_text:
                    text_parts.append(table_text)
            elif isinstance(item, PictureItem):
                # 그림은 캡션만 사용
                if item.caption:
                    text_parts.append(item.caption)
        
        return "\n".join(text_parts)

    def _extract_table_text(self, table_item: TableItem) -> str:
        """테이블에서 텍스트를 추출 (preprocess.py와 동일)"""
        try:
            # 먼저 export_to_markdown 시도
            table_text = table_item.export_to_markdown()
            if table_text and table_text.strip():
                return table_text
        except Exception:
            pass
        
        # export_to_markdown 실패 시 테이블 셀에서 직접 추출
        if table_item.data and table_item.data.table_cells:
            rows = defaultdict(list)
            for cell in table_item.data.table_cells:
                rows[cell.row_idx].append((cell.col_idx, cell.text or ""))
            
            if rows:
                # 헤더와 구분선 추가하여 Markdown 테이블 형식으로
                table_lines = []
                
                # 모든 행 처리
                for row_idx in sorted(rows.keys()):
                    cells = sorted(rows[row_idx], key=lambda x: x[0])
                    row_text = "| " + " | ".join([cell[1] for cell in cells]) + " |"
                    table_lines.append(row_text)
                    
                    # 첫 번째 행 다음에 구분선 추가
                    if row_idx == 0 and len(rows) > 1:
                        separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                        table_lines.append(separator)
                
                return "\n".join(table_lines)
        
        return ""

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    def extract_media_files(self, file_path: str, media_files: List[Dict]) -> List[str]:
        temp_list = []
        if media_files:
            try:
                source_directory = os.path.dirname(file_path)
                for media_file in media_files:
                    source_path = os.path.join(source_directory, media_file['name'])
                    temp_list.append(source_path)
            except Exception as e:
                print(f"Error: {e}")
        return temp_list

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        # kwargs['save_images'] = True    # 이미지 처리
        # kwargs['include_wmf'] = True   # wmf 처리
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        output_path, output_file = os.path.split(file_path)
        
        # 이미지 경로 설정 (preprocess.py와 동일)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent
        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)

        # chunking
        await assert_cancelled(request)
        chunks: List[DocChunk] = self.split_documents(document, **kwargs)

        # make vectors
        await assert_cancelled(request)
        vectors: List[Dict[str, Any]] = await self.compose_vectors(document, chunks, file_path, request, **kwargs)

        await assert_cancelled(request)
        return vectors


class GenosServiceException(Exception):
    # GenOS 와의 의존성 부분 제거를 위해 추가
    def __init__(self, error_code: str, error_msg: Optional[str] = None, msg_params: Optional[dict] = None) -> None:
        self.code = 1
        self.error_code = error_code
        self.error_msg = error_msg or "GenOS Service Exception"
        self.msg_params = msg_params
        super().__init__(self.error_msg)


# GenOS 와의 의존성 제거를 위해 추가
async def assert_cancelled(request: Request):
    if await request.is_disconnected():
        raise GenosServiceException(1, f"Cancelled")