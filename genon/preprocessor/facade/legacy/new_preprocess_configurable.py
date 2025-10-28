"""
Configurable Document Processor
확장자별로 지능형/기본형 처리 선택 가능한 문서 처리기
"""

import os
import sys
from typing import Dict, Any, List, Optional
from fastapi import Request

# Local imports - 로컬 파일이라 상대경로 중요
from utils import assert_cancelled
from genos_utils import upload_files, merge_overlapping_bboxes

# Add path for facade import
sys.path.append('/Users/yangjaeseung/integration/genos_docling/doc_parser')
from docling.facade.document_facade import DocumentFacade
from docling.facade.config import ProcessorMode

# Keep chunkers here for backward compatibility
from new_preprocess import HierarchicalChunker, HybridChunker


class DocumentProcessor:
    """
    Configurable document processor with on/off switch for intelligent mode
    각 확장자별로 지능형/기본형 처리를 선택할 수 있는 프로세서
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: Optional[str] = None):
        """
        Initialize configurable document processor
        
        Args:
            config: Optional configuration dictionary
            config_file: Optional path to JSON configuration file
                        Default: "./processor_config.json" if exists
        """
        # Default configuration
        default_config = {
            'whisper_url': "http://192.168.74.164:30100/v1/audio/transcriptions",
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_tokens': 1024,
        }
        
        # Merge with provided config
        self.config = {**default_config, **(config or {})}
        
        # Check for default config file if not provided
        if config_file is None and os.path.exists("./processor_config.json"):
            config_file = "./processor_config.json"
        
        # Initialize the enhanced facade
        self.facade = DocumentFacade(self.config, config_file)
    
    async def __call__(self, request: Request, file_path: str, **kwargs: dict) -> List[Dict]:
        """
        Process a document file
        
        Args:
            request: FastAPI request object for async operations
            file_path: Path to the file to process
            **kwargs: Additional processing parameters
            
        Returns:
            List of vector dictionaries ready for storage
        """
        # Check cancellation
        await assert_cancelled(request)
        
        # Process using facade
        vectors = await self.facade.process(file_path, request, **kwargs)
        
        # Check cancellation again
        await assert_cancelled(request)
        
        return vectors
    
    # ============================================
    # Configuration Management Methods
    # ============================================
    
    def set_mode(self, extension: str, mode: str):
        """
        Set processing mode for specific extension
        
        Args:
            extension: File extension (e.g., 'pdf' or '.pdf')
            mode: 'basic' or 'intelligent'
        
        Example:
            processor.set_mode('pdf', 'intelligent')  # PDF를 지능형으로
            processor.set_mode('docx', 'basic')       # DOCX를 기본형으로
        """
        mode_enum = ProcessorMode.INTELLIGENT if mode.lower() == 'intelligent' else ProcessorMode.BASIC
        self.facade.set_mode(extension, mode_enum)
    
    def toggle_mode(self, extension: str) -> str:
        """
        Toggle processing mode for extension
        
        Args:
            extension: File extension
            
        Returns:
            New mode as string ('basic' or 'intelligent')
        """
        new_mode = self.facade.toggle_mode(extension)
        return new_mode.value
    
    def set_all_intelligent(self):
        """모든 확장자를 지능형 모드로 설정"""
        self.facade.set_all_intelligent()
        print("✅ All extensions set to INTELLIGENT mode")
    
    def set_all_basic(self):
        """모든 확장자를 기본형 모드로 설정"""
        self.facade.set_all_basic()
        print("✅ All extensions set to BASIC mode")
    
    def enable_intelligent_for(self, extensions: List[str]):
        """
        특정 확장자들만 지능형으로 설정
        
        Args:
            extensions: 확장자 리스트 (e.g., ['pdf', 'hwpx', 'docx'])
        
        Example:
            processor.enable_intelligent_for(['pdf', 'hwpx'])
        """
        for ext in extensions:
            self.set_mode(ext, 'intelligent')
        print(f"✅ Enabled intelligent mode for: {', '.join(extensions)}")
    
    def disable_intelligent_for(self, extensions: List[str]):
        """
        특정 확장자들을 기본형으로 설정
        
        Args:
            extensions: 확장자 리스트
        
        Example:
            processor.disable_intelligent_for(['docx', 'pptx'])
        """
        for ext in extensions:
            self.set_mode(ext, 'basic')
        print(f"✅ Disabled intelligent mode for: {', '.join(extensions)}")
    
    def save_config(self, config_file: str = "processor_config.json"):
        """
        현재 설정을 파일로 저장
        
        Args:
            config_file: 저장할 파일 경로
        """
        self.facade.save_config(config_file)
    
    def load_config(self, config_file: str):
        """
        파일에서 설정 로드
        
        Args:
            config_file: 로드할 파일 경로
        """
        self.facade.load_config(config_file)
    
    def print_status(self):
        """현재 설정 상태 출력"""
        self.facade.print_status()
    
    def get_configuration(self) -> Dict[str, Dict[str, Any]]:
        """현재 설정 정보 반환"""
        return self.facade.get_configuration()
    
    def get_processor_info(self, file_path: str) -> Dict[str, Any]:
        """파일에 사용될 프로세서 정보 반환"""
        return self.facade.get_processor_info(file_path)
    
    def get_supported_extensions(self) -> List[str]:
        """지원되는 확장자 목록 반환"""
        return self.facade.get_supported_extensions()
    
    # ============================================
    # Processor Options Management
    # ============================================
    
    def set_processor_option(self, extension: str, option_path: str, value: Any):
        """
        Set specific processor option
        
        Args:
            extension: File extension (e.g., 'pdf' or '.pdf')
            option_path: Dot-separated path to option (e.g., 'enrichment.toc_max_tokens')
            value: New value for the option
        
        Example:
            processor.set_processor_option('pdf', 'enrichment.toc_max_tokens', 2000)
            processor.set_processor_option('pdf', 'pipeline.do_ocr', True)
            processor.set_processor_option('docx', 'text_splitter.chunk_size', 1500)
        """
        self.facade.set_processor_option(extension, option_path, value)
    
    def set_enrichment_options(self, extension: str, 
                              enabled: bool = None,
                              toc_extraction_mode: str = None,
                              toc_seed: int = None,
                              toc_max_tokens: int = None):
        """
        Set enrichment options for PDF/HWPX processing
        
        Args:
            extension: File extension
            enabled: Enable/disable enrichment
            toc_extraction_mode: TOC extraction mode ('list_items', etc.)
            toc_seed: Random seed for TOC extraction
            toc_max_tokens: Maximum tokens for TOC
        
        Example:
            processor.set_enrichment_options('pdf', enabled=True, toc_max_tokens=2000)
        """
        options = {}
        if enabled is not None:
            options['enabled'] = enabled
        if toc_extraction_mode is not None:
            options['toc_extraction_mode'] = toc_extraction_mode
        if toc_seed is not None:
            options['toc_seed'] = toc_seed
        if toc_max_tokens is not None:
            options['toc_max_tokens'] = toc_max_tokens
        
        self.facade.set_enrichment_options(extension, **options)
    
    def set_pipeline_options(self, extension: str,
                           do_ocr: bool = None,
                           do_table_structure: bool = None,
                           table_structure_options: dict = None):
        """
        Set pipeline options for PDF processing
        
        Args:
            extension: File extension
            do_ocr: Enable/disable OCR
            do_table_structure: Enable/disable table structure extraction
            table_structure_options: Table extraction options
        
        Example:
            processor.set_pipeline_options('pdf', do_ocr=True)
            processor.set_pipeline_options('pdf', 
                do_table_structure=True,
                table_structure_options={'do_cell_matching': True}
            )
        """
        options = {}
        if do_ocr is not None:
            options['do_ocr'] = do_ocr
        if do_table_structure is not None:
            options['do_table_structure'] = do_table_structure
        if table_structure_options is not None:
            options['table_structure_options'] = table_structure_options
        
        self.facade.set_pipeline_options(extension, **options)
    
    def set_chunking_options(self, extension: str,
                           max_tokens: int = None,
                           merge_peers: bool = None,
                           chunk_size: int = None,
                           chunk_overlap: int = None):
        """
        Set chunking options
        
        Args:
            extension: File extension
            max_tokens: Maximum tokens per chunk (for Docling)
            merge_peers: Merge small chunks (for Docling)
            chunk_size: Chunk size (for LangChain)
            chunk_overlap: Overlap between chunks (for LangChain)
        
        Example:
            # For PDF with Docling
            processor.set_chunking_options('pdf', max_tokens=2048, merge_peers=True)
            
            # For DOCX with LangChain
            processor.set_chunking_options('docx', chunk_size=1500, chunk_overlap=300)
        """
        options = {}
        if max_tokens is not None:
            options['max_tokens'] = max_tokens
        if merge_peers is not None:
            options['merge_peers'] = merge_peers
        if chunk_size is not None:
            self.set_processor_option(extension, 'text_splitter.chunk_size', chunk_size)
        if chunk_overlap is not None:
            self.set_processor_option(extension, 'text_splitter.chunk_overlap', chunk_overlap)
        
        if options:
            self.facade.set_chunking_options(extension, **options)
    
    def set_whisper_options(self, extension: str,
                          model: str = None,
                          language: str = None,
                          temperature: float = None,
                          chunk_sec: int = None):
        """
        Set Whisper options for audio processing
        
        Args:
            extension: Audio file extension (mp3, m4a, wav)
            model: Whisper model name
            language: Language code (e.g., 'ko', 'en')
            temperature: Temperature for transcription
            chunk_sec: Chunk size in seconds
        
        Example:
            processor.set_whisper_options('mp3', language='en', temperature=0.2)
        """
        if model is not None:
            self.set_processor_option(extension, 'whisper.model', model)
        if language is not None:
            self.set_processor_option(extension, 'whisper.language', language)
        if temperature is not None:
            self.set_processor_option(extension, 'whisper.temperature', temperature)
        if chunk_sec is not None:
            self.set_processor_option(extension, 'whisper.chunk_sec', chunk_sec)
    
    def disable_enrichment(self, extensions: List[str] = None):
        """
        Disable enrichment for specified extensions
        
        Args:
            extensions: List of extensions, or None for all PDF/HWPX
        
        Example:
            processor.disable_enrichment(['pdf'])  # PDF만 enrichment 비활성화
            processor.disable_enrichment()  # 모든 PDF/HWPX enrichment 비활성화
        """
        if extensions is None:
            extensions = ['pdf', 'hwpx']
        
        for ext in extensions:
            self.set_enrichment_options(ext, enabled=False)
        print(f"✅ Enrichment disabled for: {', '.join(extensions)}")
    
    def enable_ocr(self, extensions: List[str] = None):
        """
        Enable OCR for PDF processing
        
        Args:
            extensions: List of extensions, or None for PDF
        
        Example:
            processor.enable_ocr()  # PDF OCR 활성화
        """
        if extensions is None:
            extensions = ['pdf']
        
        for ext in extensions:
            self.set_pipeline_options(ext, do_ocr=True)
        print(f"✅ OCR enabled for: {', '.join(extensions)}")


# ============================================
# Convenience Functions
# ============================================

def create_intelligent_processor() -> ConfigurableDocumentProcessor:
    """모든 확장자를 지능형으로 설정한 프로세서 생성"""
    processor = ConfigurableDocumentProcessor()
    processor.set_all_intelligent()
    return processor


def create_basic_processor() -> ConfigurableDocumentProcessor:
    """모든 확장자를 기본형으로 설정한 프로세서 생성"""
    processor = ConfigurableDocumentProcessor()
    processor.set_all_basic()
    return processor


def create_hybrid_processor() -> ConfigurableDocumentProcessor:
    """
    하이브리드 프로세서 생성
    - PDF, HWPX: 지능형 (Docling)
    - 나머지: 기본형 (LangChain)
    """
    processor = ConfigurableDocumentProcessor()
    processor.enable_intelligent_for(['pdf', 'hwpx'])
    return processor


# ============================================
# Backward Compatibility
# ============================================

# Alias for backward compatibility
# Alias for backward compatibility
ConfigurableDocumentProcessor = DocumentProcessor