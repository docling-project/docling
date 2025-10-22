from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Iterable, Any, List, Dict, Tuple

from fastapi import Request
import shutil
import subprocess
import tempfile
import unicodedata
# docling imports

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
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
    FormatOption,
)
from docling.datamodel.pipeline_options import DataEnrichmentOptions
from docling.utils.document_enrichment import enrich_document
from docling.datamodel.document import ConversionResult
from docling_core.transforms.chunker import (
    BaseChunk,
    BaseChunker,
    DocChunk,
    DocMeta,
)
# from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
# from docling_core.transforms.chunker.hierarchical_chunker import (
#     ChunkingDocSerializer,
#     ChunkingSerializerProvider,
#     DocChunk,
# )
# from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types import DoclingDocument

from pandas import DataFrame
import asyncio
# from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    # DocItem,
    DocumentOrigin,
    LevelNumber,
    ListItem,
    CodeItem,
    ContentLayer
    # SectionHeaderItem,
    # TableItem,
    # TextItem,
)
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc import (
    BoundingBox,
    # CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    # GroupLabel,
    # ImageRef,
    # ProvenanceItem,
    # Size,
    # TableCell,
    # TableData,
    # GroupItem,

    DocItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PageItem
)
from collections import Counter
import re
import warnings
import asyncio

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
CONVERTIBLE_EXTENSIONS = ['.xlsx', '.md', '.docx', '.pptx']
def convert_to_pdf(file_path: str) -> str | None:
    """
    LibreOffice로 PDF 변환을 시도한다.
    실패해도 예외를 던지지 않고 None을 반환한다.
    """
    try:
        in_path = Path(file_path).resolve()
        out_dir = in_path.parent
        pdf_path = in_path.with_suffix('.pdf')

        # headless에서 UTF-8 locale 보장
        env = os.environ.copy()
        env.setdefault("LANG", "C.UTF-8")
        env.setdefault("LC_ALL", "C.UTF-8")

        # 확장자에 따라 필터(특히 .ppt는 impress 필터)
        ext = in_path.suffix.lower()
        if ext in ('.ppt', '.pptx'):
            convert_arg = "pdf:impress_pdf_Export"
        elif ext in ('.doc', '.docx'):
            convert_arg = "pdf:writer_pdf_Export"
        elif ext in ('.xls', '.xlsx', '.csv'):
            convert_arg = "pdf:calc_pdf_Export"
        else:
            convert_arg = "pdf"

        # 비ASCII 파일명 이슈 대비 임시 ASCII 파일명 복사본 시도
        try:
            in_path.name.encode('ascii')
            candidates = [in_path]
            tmp_dir = None
        except UnicodeEncodeError:
            tmp_dir = Path(tempfile.mkdtemp())
            ascii_name = unicodedata.normalize('NFKD', in_path.stem).encode('ascii','ignore').decode('ascii') or "file"
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
                # 성공
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return str(pdf_path)
            # 실패해도 계속 시도 (로그만 찍고 무시)
            print(f"[convert_to_pdf] stderr: {proc.stderr.strip()}")
            print(f"[convert_to_pdf] stdout: {proc.stdout.strip()}")

        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    except Exception as e:
        # 어떤 에러든 삼키고 None 반환
        print(f"[convert_to_pdf] error: {e}")
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

class HierarchicalChunker(BaseChunker):
    """문서 구조와 헤더 계층을 유지하면서 아이템을 순차적으로 처리하는 청커"""

    merge_list_items: bool = True

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
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

                # 더 깊은 레벨의 헤더들 제거
                keys_to_del = [k for k in current_heading_by_level if k > header_level]
                for k in keys_to_del:
                    current_heading_by_level.pop(k, None)

                # 헤더 아이템도 추가 (헤더 자체도 아이템임)
                all_items.append(item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})
                continue

            if (isinstance(item, TextItem) or
                    isinstance(item, ListItem) or
                    isinstance(item, CodeItem) or
                    isinstance(item, TableItem) or
                    isinstance(item, PictureItem)):
                if item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
                    item.text = ""
                all_items.append(item)
                # 현재 아이템의 헤더 정보 저장
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})

        # 마지막 리스트 아이템들 처리
        if list_items:
            for list_item in list_items:
                all_items.append(list_item)
                all_header_info.append({k: v for k, v in current_heading_by_level.items()})

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
        yield chunk


class HybridChunker(BaseChunker):
    """토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 청커"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Union[PreTrainedTokenizerBase, str] = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 1024
    merge_peers: bool = True

    _inner_chunker: BaseChunker = None
    _tokenizer: PreTrainedTokenizerBase = None

    @model_validator(mode="after")
    def _initialize_components(self) -> Self:
        # 토크나이저 초기화
        self._tokenizer = (
            self.tokenizer
            if isinstance(self.tokenizer, PreTrainedTokenizerBase)
            else AutoTokenizer.from_pretrained(self.tokenizer)
        )

        # HierarchicalChunker 초기화
        if self._inner_chunker is None:
            self._inner_chunker = HierarchicalChunker()

        return self

    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산 (안전한 분할 처리)"""
        if not text:
            return 0

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
                                               dl_doc: DoclingDocument) -> str:
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
                            if l <= level:
                                headers_to_add.append(item_headers[l])
                        break

                # 헤더가 있으면 추가
                if headers_to_add:
                    header_text = "\n".join(headers_to_add)
                    text_parts.append(header_text)

                current_section_headers = item_headers.copy()

            # 아이템 텍스트 추가
            if isinstance(item, TableItem):
                table_text = self._extract_table_text(item, dl_doc)
                if table_text:
                    text_parts.append(table_text)
            elif hasattr(item, 'text') and item.text:
                # 타이틀과 섹션 헤더 처리 개선
                is_section_header = (
                        isinstance(item, SectionHeaderItem) or
                        (isinstance(item, TextItem) and
                         item.label in [DocItemLabel.SECTION_HEADER])  # TITLE은 제외
                )

                # 타이틀은 항상 포함, 섹션 헤더는 중복 방지를 위해 스킵
                if not is_section_header:
                    text_parts.append(item.text)
            elif isinstance(item, PictureItem):
                text_parts.append("")  # 이미지는 빈 텍스트

        result_text = self.delim.join(text_parts)
        return result_text

    def _extract_table_text(self, table_item: TableItem, dl_doc: DoclingDocument) -> str:
        """테이블에서 텍스트를 추출하는 일반화된 메서드"""
        try:
            # 먼저 export_to_html 시도
            table_text = table_item.export_to_html(dl_doc)
            if table_text and table_text.strip():
                return table_text
        except Exception:
            pass

        # export_to_html 실패 시 테이블 셀 데이터에서 직접 텍스트 추출
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

    def _extract_used_headers(self, header_info_list: list[dict]) -> Optional[list[str]]:
        """헤더 정보 리스트에서 실제 사용되는 헤더들을 추출"""
        if not header_info_list:
            return None

        # 모든 헤더 정보를 종합하여 사용되는 헤더들 추출
        all_headers = []
        seen_headers = set()
        
        for header_info in header_info_list:
            if header_info:  # dict가 비어있지 않은 경우
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
        # semchunk 사용하여 토큰 제한에 맞게 분할
        chunker = semchunk.chunkerify(self._tokenizer, chunk_size=max_tokens)
        chunks = chunker(table_text)
        return chunks if chunks else [table_text]

    def _split_document_by_tokens(self, doc_chunk: DocChunk, dl_doc: DoclingDocument) -> list[DocChunk]:
        """문서를 토큰 제한에 맞게 분할 (여러 섹션이 하나의 청크에 포함 가능)"""
        items = doc_chunk.meta.doc_items
        header_info_list = getattr(doc_chunk, '_header_info_list', [])  # 각 아이템의 헤더 정보 리스트

        if not items:
            return []

        result_chunks = []
        current_items = []
        current_header_infos = []

        i = 0
        while i < len(items):
            item = items[i]
            header_info = header_info_list[i] if i < len(header_info_list) else {}

            # 테이블 아이템인 경우 특별 처리
            if isinstance(item, TableItem):
                # 현재까지 누적된 아이템들이 있으면 먼저 청크로 생성
                if current_items:
                    chunk_text = self._generate_text_from_items_with_headers(
                        current_items, current_header_infos, dl_doc
                    )
                    tokens = self._count_tokens(chunk_text)

                    # 실제 사용된 헤더들만 추출
                    used_headers = self._extract_used_headers(current_header_infos)
                    result_chunks.append(DocChunk(
                        text=chunk_text,
                        meta=DocMeta(
                            doc_items=current_items.copy(),
                            headings=used_headers,
                            captions=None,
                            origin=doc_chunk.meta.origin,
                        )
                    ))
                    current_items = []
                    current_header_infos = []

                # 테이블과 앞뒤 아이템을 포함한 청크 생성
                table_items = []
                table_header_infos = []

                # 앞 아이템 추가 (가능한 경우)
                # if i > 0 and len(result_chunks) == 0:  # 첫 번째 테이블이고 앞에 아이템이 있는 경우
                #     table_items.append(items[i-1])
                #     prev_header_info = header_info_list[i-1] if i-1 < len(header_info_list) else {}
                #     table_header_infos.append(prev_header_info)

                # 테이블 추가
                table_items.append(item)
                table_header_infos.append(header_info)

                # 뒤 아이템 추가 (가능한 경우)
                # if i + 1 < len(items):
                #     table_items.append(items[i+1])
                #     next_header_info = header_info_list[i+1] if i+1 < len(header_info_list) else {}
                #     table_header_infos.append(next_header_info)
                #     i += 1  # 다음 아이템은 이미 처리했으므로 스킵

                # 테이블 청크 생성 (토큰 제한 확인)
                table_text = self._generate_text_from_items_with_headers(
                    table_items, table_header_infos, dl_doc
                )
                table_tokens = self._count_tokens(table_text)

                # 테이블이 max_tokens를 초과하는 경우, 테이블을 분할
                if table_tokens > self.max_tokens:
                    # 테이블 텍스트만 추출하여 분할
                    table_only_text = self._extract_table_text(item, dl_doc)
                    split_tables = self._split_table_text(table_only_text, 4096)

                    # 분할된 각 테이블에 대해 청크 생성
                    for split_table in split_tables:
                        # 기존 _generate_text_from_items_with_headers 함수 활용
                        full_text = self._generate_text_from_items_with_headers(
                            [item], [header_info], dl_doc
                        )
                        # 원본 테이블 텍스트를 분할된 테이블로 교체
                        full_text = full_text.replace(table_only_text, split_table)

                        # 원래 tableitem에 들어갔어야 할 heading 값 유지
                        used_headers = self._extract_used_headers([header_info])
                        result_chunks.append(DocChunk(
                            text=full_text,
                            meta=DocMeta(
                                doc_items=[item],
                                headings=used_headers,
                                captions=None,
                                origin=doc_chunk.meta.origin,
                            )
                        ))
                else:
                    used_headers = self._extract_used_headers(table_header_infos)
                    result_chunks.append(DocChunk(
                        text=table_text,
                        meta=DocMeta(
                            doc_items=table_items,
                            headings=used_headers,
                            captions=None,
                            origin=doc_chunk.meta.origin,
                        )
                    ))

                i += 1
                continue

            # 일반 아이템 처리 - 토큰 제한 확인
            test_items = current_items + [item]
            test_header_infos = current_header_infos + [header_info]
            test_text = self._generate_text_from_items_with_headers(
                test_items, test_header_infos, dl_doc
            )
            test_tokens = self._count_tokens(test_text)

            if test_tokens <= self.max_tokens:
                current_items.append(item)
                current_header_infos.append(header_info)
            else:
                # 토큰 제한 초과 - 현재까지의 아이템들로 청크 생성
                if current_items:
                    chunk_text = self._generate_text_from_items_with_headers(
                        current_items, current_header_infos, dl_doc
                    )
                    chunk_tokens = self._count_tokens(chunk_text)

                    used_headers = self._extract_used_headers(current_header_infos)
                    result_chunks.append(DocChunk(
                        text=chunk_text,
                        meta=DocMeta(
                            doc_items=current_items.copy(),
                            headings=used_headers,
                            captions=None,
                            origin=doc_chunk.meta.origin,
                        )
                    ))
                    # 새로운 청크 시작
                    current_items = [item]
                    current_header_infos = [header_info]
                else:
                    # 단일 아이템이 토큰 제한을 초과하는 경우
                    single_text = self._generate_text_from_items_with_headers(
                        [item], [header_info], dl_doc
                    )
                    single_tokens = self._count_tokens(single_text)

                    used_headers = self._extract_used_headers([header_info])
                    result_chunks.append(DocChunk(
                        text=single_text,
                        meta=DocMeta(
                            doc_items=[item],
                            headings=used_headers,
                            captions=None,
                            origin=doc_chunk.meta.origin,
                        )
                    ))

            i += 1

        # 마지막 남은 아이템들 처리
        if current_items:
            chunk_text = self._generate_text_from_items_with_headers(
                current_items, current_header_infos, dl_doc
            )
            chunk_tokens = self._count_tokens(chunk_text)

            used_headers = self._extract_used_headers(current_header_infos)
            result_chunks.append(DocChunk(
                text=chunk_text,
                meta=DocMeta(
                    doc_items=current_items,
                    headings=used_headers,
                    captions=None,
                    origin=doc_chunk.meta.origin,
                )
            ))

        # 작은 청크들 병합 처리
        return self._merge_small_chunks(result_chunks, dl_doc)

    def _merge_small_chunks(self, chunks: list[DocChunk], dl_doc: DoclingDocument) -> list[DocChunk]:
        """작은 청크들을 병합하여 토큰 효율성을 높임 (개선된 버전)"""
        if not chunks:
            return chunks

        min_chunk_size = self.max_tokens // 3  # 최소 청크 크기를 더 크게 설정 (2000/3 = 666토큰)
        merged_chunks = []
        current_merge_candidate = None

        for i, chunk in enumerate(chunks):
            chunk_tokens = self._count_tokens(chunk.text)

            # 아주 큰 청크는 분할 필요
            if chunk_tokens > self.max_tokens:
                if current_merge_candidate:
                    merged_chunks.append(current_merge_candidate)
                    current_merge_candidate = None

                # 큰 청크를 분할 (임시로 그대로 추가하되, 경고 표시)
                merged_chunks.append(chunk)
                continue

            # 작은 청크인 경우 병합 대상 (테이블 청크도 포함)
            if chunk_tokens < min_chunk_size:
                if current_merge_candidate is None:
                    current_merge_candidate = chunk
                else:
                    # 병합 시도
                    merged_items = current_merge_candidate.meta.doc_items + chunk.meta.doc_items
                    merged_header_infos = (
                            getattr(current_merge_candidate, '_header_info_list', []) +
                            getattr(chunk, '_header_info_list', [])
                    )

                    merged_text = self._generate_text_from_items_with_headers(
                        merged_items, merged_header_infos, dl_doc
                    )
                    merged_tokens = self._count_tokens(merged_text)

                    if merged_tokens <= self.max_tokens:
                        current_merge_candidate = DocChunk(
                            text=merged_text,
                            meta=DocMeta(
                                doc_items=merged_items,
                                headings=self._extract_used_headers(merged_header_infos),
                                captions=None,
                                origin=chunk.meta.origin,
                            )
                        )
                        current_merge_candidate._header_info_list = merged_header_infos
                    else:
                        merged_chunks.append(current_merge_candidate)
                        current_merge_candidate = chunk
            else:
                if current_merge_candidate:
                    # 이전 병합 후보가 있으면 현재 청크와 병합 시도
                    candidate_tokens = self._count_tokens(current_merge_candidate.text)
                    if candidate_tokens < min_chunk_size:
                        # 현재 청크와 병합 시도
                        merged_items = current_merge_candidate.meta.doc_items + chunk.meta.doc_items
                        merged_header_infos = (
                                getattr(current_merge_candidate, '_header_info_list', []) +
                                getattr(chunk, '_header_info_list', [])
                        )

                        merged_text = self._generate_text_from_items_with_headers(
                            merged_items, merged_header_infos, dl_doc
                        )
                        merged_tokens = self._count_tokens(merged_text)

                        if merged_tokens <= self.max_tokens:
                            merged_chunks.append(DocChunk(
                                text=merged_text,
                                meta=DocMeta(
                                    doc_items=merged_items,
                                    headings=self._extract_used_headers(merged_header_infos),
                                    captions=None,
                                    origin=chunk.meta.origin,
                                )
                            ))
                            current_merge_candidate = None
                            continue

                    # 병합할 수 없으면 후보를 먼저 추가
                    merged_chunks.append(current_merge_candidate)
                    current_merge_candidate = None

                merged_chunks.append(chunk)

        # 마지막 병합 후보 처리
        if current_merge_candidate:
            merged_chunks.append(current_merge_candidate)

        return merged_chunks

    def chunk(self, dl_doc: DoclingDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        """문서를 청킹하여 반환

        Args:
            dl_doc: 청킹할 문서

        Yields:
            토큰 제한에 맞게 분할된 청크들
        """
        doc_chunks = list(self._inner_chunker.chunk(dl_doc=dl_doc, **kwargs))

        if not doc_chunks:
            return iter([])

        doc_chunk = doc_chunks[0]  # HierarchicalChunker는 하나의 청크만 반환

        final_chunks = self._split_document_by_tokens(doc_chunk, dl_doc)

        return iter(final_chunks)


class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'

    text: str = None
    n_char: int = None
    n_word: int = None
    n_line: int = None
    i_page: int = None
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None
    chunk_bboxes: str = None
    media_files: str = None
    created_date: int = None  # YYYYMMDD 형식의 정수
    authors: str = None  # 팀 리스트
    title: str = None  # 문서 제목


class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        self.n_word: Optional[int] = None
        self.n_line: Optional[int] = None
        self.i_page: Optional[int] = None
        self.i_chunk_on_page: Optional[int] = None
        self.n_chunk_of_page: Optional[int] = None
        self.i_chunk_on_doc: Optional[int] = None
        self.n_chunk_of_doc: Optional[int] = None
        self.n_page: Optional[int] = None
        self.reg_date: Optional[str] = None
        self.chunk_bboxes: Optional[str] = None
        self.media_files: Optional[str] = None
        self.created_date: Optional[int] = None
        self.authors: Optional[str] = None  # 팀 리스트
        self.title: Optional[str] = None

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
            i_chunk_on_page=self.i_chunk_on_page,
            n_chunk_of_page=self.n_chunk_of_page,
            i_chunk_on_doc=self.i_chunk_on_doc,
            n_chunk_of_doc=self.n_chunk_of_doc,
            n_page=self.n_page,
            reg_date=self.reg_date,
            chunk_bboxes=self.chunk_bboxes,
            media_files=self.media_files,
            created_date=self.created_date,
            authors=self.authors,  # 팀 리스트
            title=self.title,
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
        pipe_line_options = PdfPipelineOptions()
        pipe_line_options.generate_page_images = True
        pipe_line_options.generate_picture_images = True
        pipe_line_options.do_ocr = False
        # pipe_line_options.ocr_options.lang = ["ko", 'en']
        # pipe_line_options.ocr_options.model_storage_directory = "./.EasyOCR/model"
        # pipe_line_options.ocr_options.force_full_page_ocr = True
        # ocr_options = TesseractOcrOptions()
        # ocr_options.lang = ['kor', 'kor_vert', 'eng', 'jpn', 'jpn_vert']
        # ocr_options.path = './.tesseract/tessdata'
        # pipe_line_options.ocr_options = ocr_options
        # pipe_line_options.artifacts_path = Path("/nfs-root/models/8/12")
        pipe_line_options.do_table_structure = True
        pipe_line_options.images_scale = 2
        pipe_line_options.table_structure_options.do_cell_matching = True
        pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipe_line_options.accelerator_options = accelerator_options

        simple_pipeline_options = PipelineOptions()

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipe_line_options,
                    backend=DoclingParseV4DocumentBackend
                    # backend=PyPdfiumDocumentBackend
                ),
            }
        )

        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipe_line_options,
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

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)

    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
        chunker: HybridChunker = HybridChunker(
            max_tokens=2000,
            merge_peers=True
        )
        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks

    def safe_join(self, iterable):
        if not isinstance(iterable, (list, tuple, set)):
            return ''
        return ''.join(map(str, iterable)) + '\n'

    def parse_created_date(self, date_text: str) -> Optional[int]:
        """
        작성일 텍스트를 파싱하여 YYYYMMDD 형식의 정수로 변환

        Args:
            date_text: 작성일 텍스트 (YYYY-MM 또는 YYYY-MM-DD 형식)

        Returns:
            YYYYMMDD 형식의 정수, 파싱 실패시 None
        """
        if not date_text or not isinstance(date_text, str) or date_text == "None":
            return 0

        # 공백 제거 및 정리
        date_text = date_text.strip()

        # YYYY-MM-DD 형식 매칭
        match_full = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_text)
        if match_full:
            year, month, day = match_full.groups()
            try:
                # 유효한 날짜인지 검증
                datetime(int(year), int(month), int(day))
                return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
            except ValueError:
                pass

        # YYYY-MM 형식 매칭 (일자는 01로 설정)
        match_month = re.match(r'^(\d{4})-(\d{1,2})$', date_text)
        if match_month:
            year, month = match_month.groups()
            try:
                # 유효한 월인지 검증
                datetime(int(year), int(month), 1)
                return int(f"{year}{month.zfill(2)}01")
            except ValueError:
                pass

        # YYYY 형식 매칭 (월일은 0101로 설정)
        match_year = re.match(r'^(\d{4})$', date_text)
        if match_year:
            year = match_year.group(1)
            try:
                datetime(int(year), 1, 1)
                return int(f"{year}0101")
            except ValueError:
                pass

        return 0

    def parse_authors(self, authors_data) -> list[str]:
        """
        작성자 데이터에서 이름만 추출하여 문자열 리스트로 반환

        Args:
            authors_data: 작성자 정보 (딕셔너리 리스트 또는 문자열)

        Returns:
            list[str]: 작성자 이름 리스트
        """
        if not authors_data:
            return []

        # 리스트인 경우
        if isinstance(authors_data, list):
            names = []
            for author in authors_data:
                if isinstance(author, dict):
                    # 딕셔너리에서 "이름" 키 찾기
                    if "이름" in author:
                        name = author["이름"].strip()
                        if name:  # 빈 문자열이 아닌 경우만 추가
                            names.append(name)
                    elif "name" in author:
                        name = author["name"].strip()
                        if name:
                            names.append(name)
                elif isinstance(author, str):
                    # 문자열인 경우 직접 추가
                    name = author.strip()
                    if name:
                        names.append(name)

            # 중복 제거
            return list(set(names))

        # 문자열인 경우 (쉼표로 구분된 이름들)
        elif isinstance(authors_data, str):
            # 여러 구분자로 분리
            separators = [',', ';', '/', '\n', '·', '•']
            for sep in separators:
                if sep in authors_data:
                    names = [name.strip() for name in authors_data.split(sep) if name.strip()]
                    return list(set(names))  # 중복 제거

            # 구분자가 없는 경우 전체를 하나의 이름으로 처리
            name = authors_data.strip()
            return [name] if name else []

        return []

    def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
        # enrichment 옵션 설정
        enrichment_options = DataEnrichmentOptions(
            do_toc_enrichment=True,
            extract_metadata=True,
            toc_api_provider="custom",
            toc_api_base_url="http://llmops-gateway-api-service:8080/serving/13/23/v1/chat/completions",
            metadata_api_base_url="http://llmops-gateway-api-service:8080/serving/13/23/v1/chat/completions",
            toc_api_key="9e32423947fd4a5da07a28962fe88487",
            metadata_api_key="9e32423947fd4a5da07a28962fe88487",
            toc_model="/model/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5",
            metadata_model="/model/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5",
            toc_temperature=0.0,
            toc_top_p=0,
            toc_seed=33,
            toc_max_tokens=1000
        )

        # 새로운 enriched result 받기
        document = enrich_document(document, enrichment_options)
        return document

    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request,
                              **kwargs: dict) -> \
            list[dict]:
        created_date = 0
        authors = ""
        title = ""

        try:
            if (document.key_value_items and
                    len(document.key_value_items) > 0 and
                    hasattr(document.key_value_items[0], 'graph') and
                    hasattr(document.key_value_items[0].graph, 'cells') and
                    len(document.key_value_items[0].graph.cells) > 1):
                # 작성일 추출 (cells[1])
                date_text = document.key_value_items[0].graph.cells[1].text
                created_date = self.parse_created_date(date_text)

        except (AttributeError, IndexError) as e:
            pass

        # kwargs에서 authors 추출 (이름만)
        if "authors" in kwargs:
            authors = json.dumps(self.parse_authors(kwargs["authors"]))

        for item, _ in document.iterate_items():
            if hasattr(item, 'label'):
                if item.label == DocItemLabel.TITLE:
                    title = item.text.strip() if item.text else ""
                    break

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=document.num_pages(),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
            created_date=created_date,
            authors=authors,
            title=title
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []
        upload_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_page = chunk.meta.doc_items[0].prov[0].page_no
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

    def get_media_files(self, doc_items: list):
        temp_list = []
        for item in doc_items:
            if isinstance(item, PictureItem):
                path = str(item.image.uri)
                name = path.rsplit("/", 1)[-1]
                temp_list.append({'path': path, 'name': name})
        return temp_list

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        document: DoclingDocument = self.load_documents(file_path, **kwargs)
        ext = Path(file_path).suffix.lower()
        if ext in ['.pptx', '.docx', '.md']: # pdf 저장 원하는 확장자 추가(pptx, docx, md, xlsx, csv 제공가능)
            convert_to_pdf(file_path)
            pdf_path = _get_pdf_path(file_path)

        output_path, output_file = os.path.split(file_path)
        filename, _ = os.path.splitext(output_file)
        artifacts_dir = Path(f"{output_path}/{filename}")
        if artifacts_dir.is_absolute():
            reference_path = None
        else:
            reference_path = artifacts_dir.parent

        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)

        document = self.enrichment(document, **kwargs)

        # Extract Chunk from DoclingDocument
        chunks: List[DocChunk] = self.split_documents(document, **kwargs)
        # await assert_cancelled(request)

        vectors = []
        if len(chunks) >= 1:
            vectors: list[dict] = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
        else:
            raise GenosServiceException(1, f"chunk length is 0")

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