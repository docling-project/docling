"""
    이 코드는 예시이며, 실제로는 DB의 값을 가져와서 사용한다.
    이 코드를 최초에 DB에 넣어야 한다.
"""

import subprocess
import os
import shutil
import json
import fitz
import uuid

from collections import defaultdict
from datetime import datetime
from fastapi import Request
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader,  # PDF
    UnstructuredWordDocumentLoader,  # DOC, DOCX
    UnstructuredPowerPointLoader,  # PPT, PPTX
    UnstructuredImageLoader,  # JPG, PNG
    UnstructuredFileLoader  # Generic fallback
)

from utils import assert_cancelled
from weasyprint import HTML

from genos_utils import upload_files, merge_overlapping_bboxes
import platform

# pdf 변환 대상 확장자
CONVERTIBLE_EXTENSIONS = ['.hwp', '.txt', '.json', '.md']


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


def get_korean_font():
    """시스템에 따른 한글 폰트 반환"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return ["Apple SD Gothic Neo", "AppleGothic"]
    elif system == "Windows":
        return ["Malgun Gothic", "맑은 고딕"]
    else:  # Linux
        return ["Noto Sans CJK KR", "DejaVu Sans"]


def get_html_content(content: str):
    korean_fonts = get_korean_font()
    font_family = ", ".join([f"'{font}'" for font in korean_fonts])
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: {font_family}, sans-serif;
            font-size: 12px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <pre>{content}</pre>
</body>
</html>"""


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
    chunk_bboxes: str | None = None  # dict 로 할 경우 retrieval 시 nested property 작성이 필요하여 json.dumps 사용
    media_files: str | None = None  # 마찬가지


class HwpLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)

    def load(self):
        try:
            subprocess.run(['hwp5html', self.file_path, '--output', self.output_dir], check=True, timeout=600)

            converted_file_path = os.path.join(self.output_dir, 'index.xhtml')

            pdf_save_path = _get_pdf_path(self.file_path)
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
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            html_content = get_html_content(content)

            html_file_path = os.path.join(self.output_dir, 'temp.html')
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            pdf_save_path = _get_pdf_path(self.file_path)
            HTML(html_file_path).write_pdf(pdf_save_path)

            loader = PyMuPDFLoader(pdf_save_path)
            return loader.load()
        except Exception as e:
            print(f"Failed to convert {self.file_path} to XHTML")
            raise e
        finally:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)


class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)

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
        else:
            return UnstructuredFileLoader(file_path)

    def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
        loader = self.get_loader(file_path)
        documents = loader.load()
        return documents

    def split_documents(self, documents, **kwargs: dict) -> list[Document]:
        splitter_params = {}
        chunk_size = kwargs.get('chunk_size')
        chunk_overlap = kwargs.get('chunk_overlap')

        if chunk_size is not None:
            splitter_params['chunk_size'] = chunk_size

        if chunk_overlap is not None:
            splitter_params['chunk_overlap'] = chunk_overlap

        text_splitter = RecursiveCharacterTextSplitter(**splitter_params)
        chunks = text_splitter.split_documents(documents)
        chunks = [chunk for chunk in chunks if chunk.page_content]
        if not chunks:
            raise Exception('Empty document')

        for chunk in chunks:
            # 'page' 키가 없을 경우 기본값 1로 설정
            page = chunk.metadata.get('page', 1)

            source = chunk.metadata.get('source', '')
            file_ext = os.path.splitext(source)[-1].lower() if source else ''

            if file_ext in ['.jpg', '.jpeg', '.png']:
                # 이미지 파일: 이미 1-based이므로 그대로 사용
                if isinstance(page, int) and page <= 0:
                    page = 1  # 0이거나 음수인 경우에만 1로 설정
            else:
                # 다른 파일들: 0-based를 1-based로 변환
                if isinstance(page, int) and page >= 0:
                    page += 1

            chunk.metadata['page'] = page
            self.page_chunk_counts[page] += 1
        return chunks

    async def _extract_page_images(self, pdf_path: str, request: Request) -> dict[int, list[dict]]:
        if not os.path.exists(pdf_path):
            return {}

        doc = fitz.open(pdf_path)
        file_list: list[dict] = []
        page_meta: dict[int, list[dict]] = defaultdict(list)

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Convert to RGB if needed
                    if pix.n >= 5:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.n == 4:  # RGBA
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    elif pix.n < 3:  # Grayscale
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_name = f"{uuid.uuid4()}.png"
                    img_path = os.path.join("/tmp", img_name)

                    pix.save(img_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    continue
                finally:
                    pix = None  # Free memory

                file_list.append({'path': img_path, 'name': img_name})
                page_meta[page_index + 1].append({'name': img_name, 'type': 'image'})

        if file_list:
            await upload_files(file_list, request=request)

        return page_meta

    def compose_vectors(self, chunks: list[Document], file_path: str, **kwargs: dict) -> list[dict]:
        pdf_path = _get_pdf_path(file_path)
        doc = None
        total_pages = 0

        if os.path.exists(pdf_path):
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

        global_metadata = dict(
            n_chunk_of_doc=len(chunks),
            n_page=max([chunk.metadata['page'] for chunk in chunks]),
            reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
        )

        current_page = None
        chunk_index_on_page = 0
        vectors = []

        for chunk_idx, chunk in enumerate(chunks):
            page = chunk.metadata['page']
            text = chunk.page_content

            if page != current_page:
                current_page = page
                chunk_index_on_page = 0

            chunk_bboxes_data = []
            i_page_value = page  # 디폴트값
            e_page_value = page  # 디폴트값

            if doc and total_pages > 0:
                page_index = page - 1
                if 0 <= page_index < total_pages:
                    fitz_page = doc.load_page(page_index)
                    merged_bboxes = merge_overlapping_bboxes([
                        {
                            'page': page,
                            'type': 'text',
                            'bbox': {
                                'l': rect[0] / fitz_page.rect.width,
                                't': rect[1] / fitz_page.rect.height,
                                'r': rect[2] / fitz_page.rect.width,
                                'b': rect[3] / fitz_page.rect.height,
                            }
                        } for rect in fitz_page.search_for(text)
                    ], x_tolerance=1 / fitz_page.rect.width,
                        y_tolerance=1 / fitz_page.rect.height)

                    chunk_bboxes_data = merged_bboxes
                    global_metadata['chunk_bboxes'] = json.dumps(merged_bboxes)

                    if merged_bboxes:
                        bbox_pages = [bbox.get('page') for bbox in merged_bboxes if bbox.get('page') is not None]
                        if bbox_pages:
                            i_page_value = min(bbox_pages)  # 최소값
                            e_page_value = max(bbox_pages)  # 최대값

            vectors.append(GenOSVectorMeta.model_validate({
                'text': text,
                'n_char': len(text),
                'n_word': len(text.split()),
                'n_line': len(text.splitlines()),
                'i_page': i_page_value,
                'e_page': e_page_value,
                'i_chunk_on_page': chunk_index_on_page,
                'n_chunk_of_page': self.page_chunk_counts[page],
                'i_chunk_on_doc': chunk_idx,
                **global_metadata
            }))
            chunk_index_on_page += 1

        return vectors

    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        documents: list[Document] = self.load_documents(file_path, **kwargs)
        await assert_cancelled(request)

        chunks: list[Document] = self.split_documents(documents, **kwargs)
        await assert_cancelled(request)

        pdf_path = _get_pdf_path(file_path)
        page_image_meta = await self._extract_page_images(pdf_path, request)
        await assert_cancelled(request)

        vectors = self.compose_vectors(chunks, file_path, **kwargs)

        for v in vectors:
            if v.i_page in page_image_meta:
                v.media_files = json.dumps(page_image_meta[v.i_page], ensure_ascii=False)
            else:
                v.media_files = json.dumps([])

        return vectors