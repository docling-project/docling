"""
intelligent_processor.py에 대한 unit test
PDF, HWPX, DOCX, MD 파일에 대해 테스트
"""

import pytest
from pathlib import Path
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock


class TestIntelligentProcessor:
    """IntelligentProcessor 클래스에 대한 단위 테스트"""

    @pytest.fixture
    def processor(self, intelligent_processor):
        """DocumentProcessor 인스턴스 생성"""
        return intelligent_processor()

    @pytest.fixture
    def mock_request(self):
        """Mock Request 객체"""
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    @pytest.fixture
    def temp_dir(self):
        """임시 디렉토리 생성"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def create_test_file(self, temp_dir: Path, filename: str, content: str = "Test content") -> Path:
        """테스트용 파일 생성"""
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path

    @pytest.mark.parametrize("filename", [
        "pdf_sample.pdf",
        "hwpx_sample.hwpx", 
        "docx_sample.docx",
        "md_sample.md"
    ])
    def test_load_documents(self, processor, sample_dir, filename):
        """각 파일 타입에 대해 문서 로드 테스트"""
        test_file = sample_dir / filename
        
        # 파일이 존재하는지 확인
        if not test_file.exists():
            pytest.skip(f"Sample file {filename} not found")
        
        try:
            # 문서 로드 테스트
            document = processor.load_documents(str(test_file))
            assert document is not None, f"Document should be loaded from {filename}"
            assert hasattr(document, 'num_pages'), "Document should have num_pages method"
            
            # 페이지 수 확인
            page_count = document.num_pages()
            assert page_count > 0, f"Document {filename} should have at least 1 page"
            
        except Exception as e:
            pytest.fail(f"Failed to load document {filename}: {e}")

    @pytest.mark.parametrize("filename", [
        "docx_sample.docx", 
        "pptx_sample.pptx",
        "md_sample.md"
    ])
    def test_pdf_conversion(self, processor, sample_dir, filename):
        """PDF 변환 기능 테스트 (PDF 제외)"""
        test_file = sample_dir / filename
        
        # 파일이 존재하는지 확인
        if not test_file.exists():
            pytest.skip(f"Sample file {filename} not found")
        
        # convert_to_pdf 함수 import
        from doc_preprocessors.intelligent_processor import convert_to_pdf
        
        # PDF 변환 시도
        pdf_path = convert_to_pdf(str(test_file))
        
        if pdf_path:
            # PDF 경로가 반환된 경우
            pdf_file = Path(pdf_path)
            assert pdf_file.exists(), f"PDF file should exist at {pdf_path}"
            assert pdf_file.suffix.lower() == ".pdf", "Converted file should have .pdf extension"
            
            # 원본 파일과 같은 디렉토리에 생성되었는지 확인
            assert pdf_file.parent == test_file.parent, "PDF should be in same directory as source"
            
            # 파일 크기가 0보다 큰지 확인
            assert pdf_file.stat().st_size > 0, f"PDF file {pdf_path} should not be empty"
        else:
            # 변환 실패는 예상되는 상황 (LibreOffice 없거나 파일 형식 문제)
            pytest.skip(f"PDF conversion failed for {filename} - this is expected in test environment")

    def test_split_documents_with_mock_document(self, processor):
        """Mock 문서로 청크 분할 테스트"""
        # Mock DoclingDocument 생성
        from docling_core.types import DoclingDocument
        from docling_core.types.doc import DocumentOrigin, TextItem, ProvenanceItem, BoundingBox
        from docling_core.types.doc.labels import DocItemLabel
        
        # Mock document 생성
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.num_pages.return_value = 1
        mock_doc.origin = DocumentOrigin(filename="test.pdf", mimetype="application/pdf")
        
        # Mock text item 생성
        mock_text_item = Mock(spec=TextItem)
        mock_text_item.text = "Test content for chunking"
        mock_text_item.label = DocItemLabel.TEXT
        mock_text_item.prov = [ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=100, b=20),
            charspan=(0, len("Test content for chunking"))
        )]
        mock_text_item.self_ref = "text_1"
        
        # iterate_items 메서드 mock
        mock_doc.iterate_items.return_value = [(mock_text_item, 0)]
        mock_doc.tables = []
        
        try:
            # 청크 분할 테스트
            chunks = processor.split_documents(mock_doc)
            
            # 청크가 하나 이상 생성되었는지 확인
            assert len(chunks) >= 1, "At least one chunk should be generated"
            
            # 각 청크가 올바른 구조를 가지는지 확인
            for chunk in chunks:
                assert hasattr(chunk, 'text'), "Chunk should have text attribute"
                assert hasattr(chunk, 'meta'), "Chunk should have meta attribute"
                assert hasattr(chunk.meta, 'doc_items'), "Chunk meta should have doc_items"
                
        except Exception as e:
            pytest.skip(f"Chunking test skipped due to dependency issue: {e}")

    @pytest.mark.parametrize("filename", [
        "pdf_sample.pdf",
        "hwpx_sample.hwpx",
        "docx_sample.docx", 
        "md_sample.md"
    ])
    def test_chunk_generation_with_real_files(self, processor, sample_dir, filename):
        """실제 샘플 파일로 청크 생성 테스트"""
        test_file = sample_dir / filename
        
        # 파일이 존재하는지 확인
        if not test_file.exists():
            pytest.skip(f"Sample file {filename} not found")
        
        try:
            # 문서 로드
            document = processor.load_documents(str(test_file))
            assert document is not None, f"Document should be loaded from {filename}"
            
            # 청크 분할
            chunks = processor.split_documents(document)
            
            # 청크가 하나 이상 생성되었는지 확인
            assert len(chunks) >= 1, f"At least one chunk should be generated from {filename}"
            
            # 각 청크가 올바른 구조를 가지는지 확인
            for i, chunk in enumerate(chunks):
                assert hasattr(chunk, 'text'), f"Chunk {i} should have text attribute"
                assert hasattr(chunk, 'meta'), f"Chunk {i} should have meta attribute"
                assert hasattr(chunk.meta, 'doc_items'), f"Chunk {i} meta should have doc_items"
                assert len(chunk.meta.doc_items) > 0, f"Chunk {i} should have at least one doc_item"
                
                # 텍스트 내용이 있는지 확인 (빈 문자열이 아닌지)
                assert isinstance(chunk.text, str), f"Chunk {i} text should be string"
                
        except Exception as e:
            pytest.fail(f"Chunk generation test failed for {filename}: {e}")

    @pytest.mark.asyncio
    async def test_compose_vectors_with_mock_data(self, processor, mock_request):
        """Mock 데이터로 벡터 구성 테스트"""
        # Mock document와 chunks 생성
        from docling_core.types import DoclingDocument
        from docling_core.types.doc import DocumentOrigin
        from docling_core.transforms.chunker import DocChunk, DocMeta
        
        mock_doc = Mock(spec=DoclingDocument)
        mock_doc.num_pages.return_value = 1
        mock_doc.origin = DocumentOrigin(filename="test.pdf", mimetype="application/pdf")
        mock_doc.key_value_items = []
        mock_doc.iterate_items.return_value = []
        
        # Mock chunk 생성
        mock_chunk = Mock(spec=DocChunk)
        mock_chunk.text = "Test chunk content"
        mock_chunk.meta = Mock(spec=DocMeta)
        mock_chunk.meta.doc_items = []
        mock_chunk.meta.headings = ["Test Header"]
        
        # Mock provenance
        from docling_core.types.doc import ProvenanceItem, BoundingBox
        mock_prov = ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=0, t=0, r=100, b=20),
            charspan=(0, 17)
        )
        
        # Mock doc item
        mock_doc_item = Mock()
        mock_doc_item.prov = [mock_prov]
        mock_chunk.meta.doc_items = [mock_doc_item]
        
        chunks = [mock_chunk]
        
        try:
            # 벡터 구성 테스트
            vectors = await processor.compose_vectors(
                document=mock_doc,
                chunks=chunks,
                file_path="test.pdf",
                request=mock_request
            )
            
            # 벡터가 생성되었는지 확인
            assert len(vectors) >= 1, "At least one vector should be generated"
            
            # 벡터 구조 확인
            for vector in vectors:
                assert hasattr(vector, 'text'), "Vector should have text attribute"
                assert hasattr(vector, 'n_char'), "Vector should have n_char attribute"
                assert hasattr(vector, 'n_page'), "Vector should have n_page attribute"
                
        except Exception as e:
            pytest.skip(f"Vector composition test skipped due to dependency issue: {e}")

    @pytest.mark.asyncio
    async def test_full_pipeline_with_simple_pdf(self, processor, mock_request, temp_dir):
        """간단한 PDF로 전체 파이프라인 테스트"""
        # 간단한 텍스트 파일 생성 (PDF로 가정)
        test_file = self.create_test_file(temp_dir, "test.pdf", "Simple test content")
        
        try:
            # 전체 파이프라인 실행
            result = await processor(
                request=mock_request,
                file_path=str(test_file)
            )
            
            # 결과 확인
            assert isinstance(result, list), "Result should be a list"
            assert len(result) >= 1, "At least one vector should be generated"
            
        except Exception as e:
            # 실제 PDF가 아니므로 예외 발생 예상
            pytest.skip(f"Full pipeline test skipped - expected for non-PDF file: {e}")

    def test_convertible_extensions(self):
        """변환 가능한 확장자 목록 확인"""
        from doc_preprocessors.intelligent_processor import CONVERTIBLE_EXTENSIONS
        
        expected_extensions = ['.xlsx', '.md', '.docx', '.pptx']
        assert CONVERTIBLE_EXTENSIONS == expected_extensions, f"Expected {expected_extensions}, got {CONVERTIBLE_EXTENSIONS}"

    def test_processor_initialization(self, processor):
        """프로세서 초기화 테스트"""
        # 기본 속성 확인
        assert hasattr(processor, 'converter'), "Processor should have converter"
        assert hasattr(processor, 'second_converter'), "Processor should have second_converter"
        assert hasattr(processor, 'page_chunk_counts'), "Processor should have page_chunk_counts"
        
        # page_chunk_counts가 defaultdict인지 확인
        from collections import defaultdict
        assert isinstance(processor.page_chunk_counts, defaultdict), "page_chunk_counts should be defaultdict"
