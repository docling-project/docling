# 지능형 문서 전처리기 - 적재용(외부)

외부 PDF 문서를 처리하여 벡터 데이터베이스에 적재하기 위한 전처리기입니다. OCR 자동 판단 기능과 PaddleOCR을 통해 스캔 문서와 디지털 문서를 모두 처리합니다.

## 🔧 공통 컴포넌트

### GenOSVectorMeta
벡터 메타데이터를 정의하는 Pydantic 모델입니다.

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'
    
    text: str = None           # 청크 텍스트
    n_char: int = None         # 문자 수
    n_word: int = None         # 단어 수
    n_line: int = None         # 줄 수
    i_page: int = None         # 페이지 번호
    i_chunk_on_page: int = None    # 페이지 내 청크 인덱스
    n_chunk_of_page: int = None    # 페이지 내 총 청크 수
    i_chunk_on_doc: int = None     # 문서 내 청크 인덱스
    n_chunk_of_doc: int = None     # 문서 내 총 청크 수
    n_page: int = None         # 총 페이지 수
    reg_date: str = None       # 등록일시 (ISO 8601)
    chunk_bboxes: str = None   # JSON 문자열 - 바운딩 박스 정보
    media_files: str = None    # JSON 문자열 - 미디어 파일 정보
    created_date: int = None   # 작성일 (YYYYMMDD 형식) ★BOK 특화
    authors: str = None        # JSON 문자열 - 작성자 리스트
    title: str = None          # 문서 제목
```

### GenOSVectorMetaBuilder
빌더 패턴을 사용하여 메타데이터를 구성합니다.

```python
class GenOSVectorMetaBuilder:
    def set_text(self, text: str) -> "GenOSVectorMetaBuilder":
        """텍스트와 관련 통계 설정"""
        self.text = text
        self.n_char = len(text)
        self.n_word = len(text.split())
        self.n_line = len(text.splitlines())
        return self
    
    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument):
        """청크의 바운딩 박스 정보를 상대 좌표로 저장"""
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                bbox_data = {
                    'l': bbox.l / size.width,   # 왼쪽 (0-1)
                    't': bbox.t / size.height,  # 상단 (0-1)
                    'r': bbox.r / size.width,   # 오른쪽 (0-1)
                    'b': bbox.b / size.height,  # 하단 (0-1)
                    'coord_origin': bbox.coord_origin.value
                }
                chunk_bboxes.append({
                    'page': prov.page_no,
                    'bbox': bbox_data,
                    'type': item.label,
                    'ref': item.self_ref
                })
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self
```

### DocumentProcessor
문서 처리 파이프라인을 관리하는 핵심 클래스입니다.

```python
class DocumentProcessor:
    def __init__(self):
        # 기본 PDF 처리 파이프라인
        pipe_line_options = PdfPipelineOptions()
        pipe_line_options.generate_page_images = True
        pipe_line_options.generate_picture_images = True
        pipe_line_options.do_ocr = False  # 기본적으로 OCR 비활성화
        pipe_line_options.do_table_structure = True
        pipe_line_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        # Primary 컨버터 (DoclingParseV4)
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipe_line_options,
                    backend=DoclingParseV4DocumentBackend
                )
            }
        )
        
        # Fallback 컨버터 (PyPdfium)
        self.second_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipe_line_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )
        
        # OCR 전용 컨버터 (PaddleOCR)
        ocr_options = PaddleOcrOptions(
            force_full_page_ocr=True,
            lang=['korean'],
            text_score=0.3  # 텍스트 신뢰도 임계값
        )
        self.ocr_pipe_line_options = pipe_line_options.model_copy(deep=True)
        self.ocr_pipe_line_options.do_ocr = True
        self.ocr_pipe_line_options.ocr_options = ocr_options
        
        self.ocr_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.ocr_pipe_line_options,
                    backend=DoclingParseV4DocumentBackend
                )
            }
        )
```

### HierarchicalChunker & HybridChunker
문서 구조를 유지하면서 토큰 제한을 고려한 청킹을 수행합니다.

```python
class HybridChunker(BaseChunker):
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 1000  # 적재용(외부)는 1000 토큰 사용
    merge_peers: bool = True
```

## 📂 전처리 흐름

### 1. 문서 로드 및 OCR 자동 판단
```python
def load_documents_with_docling(self, file_path: str, **kwargs) -> DoclingDocument:
    try:
        # 1차: DoclingParseV4로 시도
        conv_result = self.converter.convert(file_path, raises_on_error=True)
    except Exception:
        # 2차: PyPdfium으로 폴백
        conv_result = self.second_converter.convert(file_path, raises_on_error=True)
    
    document = conv_result.document
    
    # OCR 필요성 자동 체크
    if not check_document(document, self.enrichment_options):
        # 텍스트가 부족하다고 판단되면 OCR 수행
        document = self.ocr_converter.convert(file_path, raises_on_error=True).document
    
    return document
```

### 2. 문서 Metadata Enrichment
```python
def enrichment(self, document: DoclingDocument, **kwargs) -> DoclingDocument:
    # LLM을 통한 문서 메타데이터 자동 추출
    # extract_metadata=True로 작성일(created_date) 등의 메타데이터를 추출
    enrichment_options = DataEnrichmentOptions(
        do_toc_enrichment=False,         # 목차 생성 활성화
        extract_metadata=True,           # ★메타데이터 추출 활성화 (작성일 추출)
        metadata_api_provider="custom",  # 메타데이터 API 프로바이더
        metadata_api_base_url="http://llmops-gateway-api-service:8080/serving/13/23/v1/chat/completions",
        metadata_api_key="9e32423947fd4a5da07a28962fe88487",
        metadata_model="/model/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5",
        toc_api_provider="custom",
        toc_api_base_url="http://llmops-gateway-api-service:8080/serving/13/23/v1/chat/completions",
        toc_api_key="9e32423947fd4a5da07a28962fe88487",
        toc_model="/model/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5",
        toc_temperature=0.0,             # 일관성 있는 결과
        toc_top_p=0,
        toc_seed=33,                     # 재현 가능한 결과
        toc_max_tokens=1000
    )
    
    # enrich_document는 문서에서 작성일, 작성자 등의 메타데이터를 LLM으로 추출
    # 추출된 메타데이터는 document.key_value_items에 저장됨
    document = enrich_document(document, enrichment_options)
    return document
```

### 3. 문서 청킹
```python
def split_documents(self, documents: DoclingDocument, **kwargs) -> List[DocChunk]:
    chunker = HybridChunker(
        max_tokens=1000,
        merge_peers=True
    )
    chunks = list(chunker.chunk(dl_doc=documents, **kwargs))
    
    # 페이지별 청크 수 계산
    for chunk in chunks:
        self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
    
    return chunks
```

### 4. 벡터 메타데이터 생성
```python
async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], 
                         file_path: str, request: Request, **kwargs) -> list[dict]:
    # Enrichment에서 추출된 메타데이터(작성일)를 가져옴 ★BOK 특화
    # document.key_value_items에 LLM이 추출한 작성일 정보가 저장됨
    created_date = 0
    if (document.key_value_items and len(document.key_value_items) > 0 and
        hasattr(document.key_value_items[0], 'graph')):
        # metadata enrichment로 추출된 작성일 텍스트 ★BOK 특화
        date_text = document.key_value_items[0].graph.cells[1].text
        created_date = self.parse_created_date(date_text)
    
    # 작성자 정보 파싱
    authors = ""
    if "authors" in kwargs:
        authors = json.dumps(self.parse_authors(kwargs["authors"]))
    
    # 제목 추출
    title = ""
    for item, _ in document.iterate_items():
        if hasattr(item, 'label') and item.label == DocItemLabel.TITLE:
            title = item.text.strip() if item.text else ""
            break
    
    # 글로벌 메타데이터
    global_metadata = dict(
        n_chunk_of_doc=len(chunks),
        n_page=document.num_pages(),
        reg_date=datetime.now().isoformat(timespec='seconds') + 'Z',
        created_date=created_date,  # metadata enrichment로 추출된 작성일 ★BOK 특화
        authors=authors,
        title=title
    )
    
    # 각 청크별 벡터 생성
    vectors = []
    for chunk_idx, chunk in enumerate(chunks):
        vector = (GenOSVectorMetaBuilder()
                  .set_text(content)
                  .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                  .set_chunk_index(chunk_idx)
                  .set_global_metadata(**global_metadata)
                  .set_chunk_bboxes(chunk.meta.doc_items, document)
                  .set_media_files(chunk.meta.doc_items)
                  ).build()
        vectors.append(vector)
    
    return vectors
```

### 5. 작성일 파싱 (BOK 특화)
```python
def parse_created_date(self, date_text: str) -> Optional[int]:
    """작성일을 YYYYMMDD 형식의 정수로 변환 ★BOK 특화"""
    
    if not date_text or date_text == "None":
        return 0
    
    # YYYY-MM-DD 형식
    match_full = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_text)
    if match_full:
        year, month, day = match_full.groups()
        return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
    
    # YYYY-MM 형식 (일자는 01로 설정)
    match_month = re.match(r'^(\d{4})-(\d{1,2})$', date_text)
    if match_month:
        year, month = match_month.groups()
        return int(f"{year}{month.zfill(2)}01")
    
    # YYYY 형식 (월일은 0101로 설정)
    match_year = re.match(r'^(\d{4})$', date_text)
    if match_year:
        year = match_year.group(1)
        return int(f"{year}0101")
    
    return 0
```

### 6. 작성자 정보 파싱
```python
def parse_authors(self, authors_data) -> list[str]:
    """다양한 형식의 작성자 정보를 통일된 리스트로 변환"""
    
    if isinstance(authors_data, list):
        names = []
        for author in authors_data:
            if isinstance(author, dict):
                # "이름" 또는 "name" 키 찾기
                if "이름" in author:
                    names.append(author["이름"].strip())
                elif "name" in author:
                    names.append(author["name"].strip())
            elif isinstance(author, str):
                names.append(author.strip())
        return list(set(names))  # 중복 제거
    
    elif isinstance(authors_data, str):
        # 구분자: , ; / \n · •
        separators = [',', ';', '/', '\n', '·', '•']
        for sep in separators:
            if sep in authors_data:
                names = [name.strip() for name in authors_data.split(sep)]
                return list(set(names))
        return [authors_data.strip()] if authors_data.strip() else []
    
    return []
```

## ✨ 사용자 커스터마이징 포인트

### OCR 옵션 커스터마이징
```python
# 다국어 OCR 설정
ocr_options = PaddleOcrOptions(
    force_full_page_ocr=True,
    lang=['korean', 'en'],     # 한국어와 영어
    text_score=0.5              # 더 엄격한 신뢰도
)

# 일본어 문서 설정
ocr_options = PaddleOcrOptions(
    force_full_page_ocr=True,
    lang=['japan'],
    text_score=0.3
)
```

### OCR 강제 실행
```python
# check_document 건너뛰고 OCR 강제 실행
document = processor.ocr_converter.convert(file_path, raises_on_error=True).document
```

## ✅ 유지보수 팁

### 트러블슈팅
1. **OCR 결과 부정확**: text_score 값 조정
2. **처리 시간 과다**: GPU 사용 확인, 이미지 스케일 감소

### 주요 차이점 (기본 적재용(외부) 대비)
- **OCR 자동 판단**: check_document 함수로 OCR 필요성 자동 감지
- **PaddleOCR 사용**: Tesseract 대신 PaddleOCR 사용 (한국어 성능 우수)

