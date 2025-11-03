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

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.document_loaders import (
    PyMuPDFLoader,                    # PDF
    UnstructuredWordDocumentLoader,   # DOC and DOCX
    UnstructuredPowerPointLoader,     # PPT and PPTX
    UnstructuredImageLoader,          # JPG, PNG
    TextLoader,                       # Plain Text
    JSONLoader,                       # JSON
    UnstructuredFileLoader            # Generic fallback
)

# from utils import assert_cancelled
# from weasyprint import HTML


# from genos_utils import upload_files


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
    bboxes: str = None


# class HwpLoader:
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
#         os.makedirs(self.output_dir, exist_ok=True)

#     def load(self):
#         try:
#             subprocess.run(['hwp5html', self.file_path, '--output', self.output_dir], check=True, timeout=600)

#             converted_file_path = os.path.join(self.output_dir, 'index.xhtml')

#             pdf_save_path = self.file_path.replace('.hwp', '.pdf')
#             HTML(converted_file_path).write_pdf(pdf_save_path)

#             loader = PyMuPDFLoader(pdf_save_path)
#             return loader.load()
#         except Exception as e:
#             print(f"Failed to convert {self.file_path} to XHTML")
#             raise e
#         finally:
#             if os.path.exists(self.output_dir):
#                 shutil.rmtree(self.output_dir)


# class TextLoader:
#     def __init__(self, file_path: str):
#         self.file_path = file_path
#         self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
#         os.makedirs(self.output_dir, exist_ok=True)

#     def load(self):
#         try:
#             with open(self.file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#             html_content = f"<html><body><pre>{content}</pre></body></html>"
#             html_file_path = os.path.join(self.output_dir, 'temp.html')
#             with open(html_file_path, 'w', encoding='utf-8') as f:
#                 f.write(html_content)
#             pdf_save_path = self.file_path.replace('.txt', '.pdf').replace('.json', '.pdf')
#             HTML(html_file_path).write_pdf(pdf_save_path)

#             loader = PyMuPDFLoader(pdf_save_path)
#             return loader.load()
#         except Exception as e:
#             print(f"Failed to convert {self.file_path} to XHTML")
#             raise e
#         finally:
#             if os.path.exists(self.output_dir):
#                 shutil.rmtree(self.output_dir)


class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)


    def get_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            return PyMuPDFLoader(file_path)
        # elif ext in ['.doc', '.docx']:
        #     return UnstructuredWordDocumentLoader(file_path)
        # elif ext in ['.ppt', '.pptx']:
        #     return UnstructuredPowerPointLoader(file_path)
        # elif ext in ['.jpg', '.jpeg', '.png']:
        #     return UnstructuredImageLoader(file_path)
        # elif ext in ['.txt', '.json']:
        #     return TextLoader(file_path)
        # elif ext == '.hwp':
        #     return HwpLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)


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
            self.page_chunk_counts[chunk.metadata['page']] += 1
        return chunks


    def compose_vectors(self, chunks: list[Document], file_path: str, **kwargs: dict) -> list[dict]:
        pdf_path = file_path.replace('.hwp', '.pdf').replace('.txt', '.pdf').replace('.json', '.pdf')
        if os.path.exists(pdf_path):
            doc = fitz.open(pdf_path)

        global_metadata = dict(
            n_chunk_of_doc = len(chunks),
            n_page = max([chunk.metadata['page'] for chunk in chunks]),
            reg_date = datetime.now().isoformat(timespec='seconds') + 'Z'
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

            if os.path.exists(pdf_path):
                fitz_page = doc.load_page(page)

                global_metadata['bboxes'] = json.dumps([{
                    'p1': { 'x': rect[0]/fitz_page.rect.width, 'y': rect[1]/fitz_page.rect.height },
                    'p2': { 'x': rect[2]/fitz_page.rect.width, 'y': rect[3]/fitz_page.rect.height },
                } for rect in fitz_page.search_for(text)])

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
        documents: list[Document] = self.load_documents(file_path, **kwargs)
        # await assert_cancelled(request)

        chunks: list[Document] = self.split_documents(documents, **kwargs)
        # await assert_cancelled(request)

        vectors: list[dict] = self.compose_vectors(chunks, file_path, **kwargs)

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
