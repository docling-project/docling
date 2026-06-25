# 지능형 문서 전처리기 - 적재용(내부)

BOK(한국은행) JSON 형식의 내부 문서를 처리하여 벡터 데이터베이스에 적재하기 위한 전처리기입니다. 조직 내부 메타데이터(팀, 부서)를 추출하고 문서 구조를 보존합니다.

> ### 📌 BOK 패치(v2) 안내
>
> 한국은행(BOK) 운영용 최신 facade는 **`BOK_적재용_내부.py`**입니다. 아래 본문은 구버전 코드 흐름을 설명하며, v2 facade는 다음이 다릅니다.
>
> - **설정 위치 이동**: 엔드포인트·모델·thinking 등을 `DataEnrichmentOptions` 인자가 아니라 **모듈 상단 설정 상수**(예: `TOC_*`, `METADATA_*`)로 분리했습니다.
> - **thinking(추론) 모드**: `TOC_THINKING = "auto"` / `TOC_THINKING_DIALECT = "hcx"`, `METADATA_THINKING = "auto"` / `METADATA_THINKING_DIALECT = "hcx"` (`BOK_적재용_내부.py:172-176`). HyperCLOVAX-SEED(hcx) 서빙에서는 `auto`로 두어야 결과가 정상입니다. 동작 매트릭스는 [gitbook_doc/intelligent_processor.md 의 "thinking(추론) 모드"](../gitbook_doc/intelligent_processor.md) 와 동일합니다(`off`→차단, `on`→강제, `auto`→미전송, dialect `hcx`는 `force_reasoning`/`skip_reasoning` 키 사용).
> - **dotsocr 배치 크기**: 내부 facade는 BOK JSON 입력을 직접 받아 레이아웃(dotsocr) 단계를 거치지 않으므로 `LAYOUT_PAGE_BATCH_SIZE` 설정이 없습니다(외부/규정 facade에만 존재).
> - **청킹**: `GenosSmartChunker`(v2) 사용.

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
    created_date: int = None   # 작성일 (YYYYMMDD 형식) ★BOK 전용
    authors_team: str = None   # JSON 문자열 - 팀 리스트 ★BOK 전용
    authors_department: str = None  # JSON 문자열 - 부서 리스트 ★BOK 전용
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
    
    def set_global_metadata(self, **global_metadata) -> "GenOSVectorMetaBuilder":
        """글로벌 메타데이터 병합 (팀, 부서 정보 포함)"""
        for key, value in global_metadata.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument):
        """청크의 바운딩 박스 정보를 상대 좌표로 저장"""
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                bbox_data = {
                    'l': bbox.l / size.width,
                    't': bbox.t / size.height,
                    'r': bbox.r / size.width,
                    'b': bbox.b / size.height,
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
BOK JSON 문서 처리 파이프라인을 관리하는 핵심 클래스입니다.

```python
class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False  # 기본값
        
        # BOK JSON 형식 전용 컨버터
        self.converter = DocumentConverter(
            format_options={
                InputFormat.JSON_DOCLING: BOKJsonFormatOption(
                    pipeline_options=self.simple_pipeline_options,
                )
            }
        )
    
    def _create_converters(self):
        """save_images 옵션 변경 시 컨버터 재생성"""
        self.converter = DocumentConverter(
            format_options={
                InputFormat.JSON_DOCLING: BOKJsonFormatOption(
                    pipeline_options=self.simple_pipeline_options,
                )
            }
        )
```

### HierarchicalChunker & HybridChunker
문서 구조를 유지하면서 토큰 제한을 고려한 청킹을 수행합니다.

```python
class HierarchicalChunker(BaseChunker):
    """문서 구조와 헤더 계층을 유지하면서 아이템을 순차적으로 처리"""
    merge_list_items: bool = True
    
    def chunk(self, dl_doc: DLDocument, **kwargs) -> Iterator[BaseChunk]:
        # 헤더 레벨 관리
        current_heading_by_level: dict[LevelNumber, str] = {}
        
        # 섹션 헤더 처리
        if item.label == DocItemLabel.TITLE:
            header_level = 0
        elif item.label == DocItemLabel.SECTION_HEADER:
            header_level = 1
        else:
            header_level = item.level
        
        # 하위 레벨 헤더 자동 제거
        keys_to_del = [k for k in current_heading_by_level if k > header_level]
        for k in keys_to_del:
            current_heading_by_level.pop(k, None)

class HybridChunker(BaseChunker):
    tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 1000  # BOK 문서도 1000 토큰 사용
    merge_peers: bool = True
```

## 📂 전처리 흐름

### 1. BOK JSON 문서 로드
```python
def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
    save_images = kwargs.get('save_images', False)
    
    # save_images 옵션이 변경되면 컨버터 재생성
    if self.simple_pipeline_options.save_images != save_images:
        self.simple_pipeline_options.save_images = save_images
        self._create_converters()
    
    # BOK JSON 형식으로 변환
    conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
    return conv_result.document
```

### 2. 문서 Metadata Enrichment
```python
def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:
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
def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> List[DocChunk]:
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

### 4. 벡터 메타데이터 생성 (BOK 특화)
```python
async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], 
                         file_path: str, request: Request, **kwargs: dict) -> list[dict]:
    # Enrichment에서 추출된 메타데이터(작성일)를 가져옴
    # document.key_value_items에 LLM이 추출한 작성일 정보가 저장됨
    created_date = 0
    if (document.key_value_items and len(document.key_value_items) > 0 and
        hasattr(document.key_value_items[0], 'graph')):
        # metadata enrichment로 추출된 작성일 텍스트
        date_text = document.key_value_items[0].graph.cells[1].text
        created_date = self.parse_created_date(date_text)
    
    # 팀/부서 정보 추출 ★BOK 전용
    authors_team = ""
    authors_department = ""
    if "authors_team" in kwargs:
        authors_team = json.dumps(kwargs["authors_team"])
    if "authors_department" in kwargs:
        authors_department = json.dumps(kwargs["authors_department"])
    
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
        created_date=created_date,          # ★BOK 전용
        authors_team=authors_team,          # ★BOK 전용
        authors_department=authors_department,  # ★BOK 전용
        title=title
    )
    
    # 각 청크별 벡터 생성
    vectors = []
    for chunk_idx, chunk in enumerate(chunks):
        vector = (GenOSVectorMetaBuilder()
                  .set_text(content)
                  .set_page_info(chunk_page, chunk_index_on_page, self.page_chunk_counts[chunk_page])
                  .set_chunk_index(chunk_idx)
                  .set_global_metadata(**global_metadata)  # 팀/부서 정보 포함
                  .set_chunk_bboxes(chunk.meta.doc_items, document)
                  .set_media_files(chunk.meta.doc_items)
                  ).build()
        vectors.append(vector)
    
    # 미디어 파일 비동기 업로드
    await asyncio.gather(*upload_tasks)
    
    return vectors
```

### 5. 작성일 파싱 (BOK 전용)
```python
def parse_created_date(self, date_text: str) -> Optional[int]:
    """작성일을 YYYYMMDD 형식의 정수로 변환"""
    
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

### 6. kwargs를 임시 JSON 파일로 처리 (BOK 특화)
```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
    # kwargs를 임시 JSON 파일로 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
        json.dump(kwargs, temp_file, ensure_ascii=False, indent=2)
        temp_file_path = temp_file.name
    
    try:
        # BOK JSON 형식으로 로드
        document = self.load_documents(temp_file_path, **kwargs)
        
        # 이미지 참조 설정
        document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)
        
        # Enrichment
        document = self.enrichment(document, **kwargs)
        
        # 청킹 및 벡터 생성
        chunks = self.split_documents(document, **kwargs)
        vectors = await self.compose_vectors(document, chunks, file_path, request, **kwargs)
        
        return vectors
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
```

## ✨ 사용자 커스터마이징 포인트

### 팀/부서 정보 추가
```python
# 팀과 부서 정보를 kwargs로 전달
vectors = await processor(
    request=request,
    file_path="report.json",
    authors_team=["경제분석팀", "금융시장팀"],
    authors_department=["연구부", "조사부"]
)
```

## ✅ 유지보수 팁

### 트러블슈팅
1. **청크 생성 실패**: BOK JSON 형식 검증
2. **메타데이터 누락**: key_value_items 구조 확인
3. **날짜 파싱 오류**: 다양한 날짜 형식 지원

### 주요 차이점 (적재용(외부) 대비)
- **BOK JSON 전용**: PDF가 아닌 JSON_DOCLING 형식 사용
- **kwargs 처리**: 임시 JSON 파일로 변환하여 처리

