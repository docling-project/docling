# 지능형 문서 전처리기 - 첨부용

다양한 파일 형식의 첨부 문서를 처리하는 범용 전처리기입니다. PDF, 오피스 문서, 이미지, 표 데이터 등 광범위한 파일 형식을 지원합니다.

## 🔧 공통 컴포넌트

### GenOSVectorMeta (첨부용 버전)
첨부 문서용 메타데이터 모델입니다.

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'
    
    text: str | None = None
    n_char: int | None = None       # 문자 수
    n_word: int | None = None       # 단어 수
    n_line: int | None = None       # 줄 수
    i_page: int | None = None       # 시작 페이지
    e_page: int | None = None       # 종료 페이지 ★첨부용 전용
    i_chunk_on_page: int | None = None
    n_chunk_of_page: int | None = None    # 페이지 내 총 청크 수
    i_chunk_on_doc: int | None = None
    n_chunk_of_doc: int | None = None
    n_page: int | None = None       # 총 페이지 수
    reg_date: str | None = None
    chunk_bboxes: str | None = None
    media_files: str | None = None
```

### GenOSVectorMetaBuilder (첨부용 버전)
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
        """e_page 필드 자동 계산"""
        chunk_bboxes = []
        for item in doc_items:
            for prov in item.prov:
                # ... 바운딩 박스 계산 ...
                chunk_bboxes.append({
                    'page': page_no,
                    'bbox': bbox_data,
                    'type': type_,
                    'ref': label
                })
        
        # 종료 페이지 자동 설정
        self.e_page = max([bbox['page'] for bbox in chunk_bboxes]) if chunk_bboxes else None
        self.chunk_bboxes = json.dumps(chunk_bboxes)
        return self
```

### DocumentProcessor
다양한 파일 형식을 처리하는 메인 프로세서입니다.

```python
class DocumentProcessor:
    def __init__(self):
        self.page_chunk_counts = defaultdict(int)
        self.hwpx_processor = HwpxProcessor()  # HWPX 전용 프로세서
    
    def get_loader(self, file_path: str):
        """파일 확장자별 적절한 로더 선택"""
        ext = os.path.splitext(file_path)[-1].lower()
        
        if ext == '.pdf':
            return PyMuPDFLoader(file_path)
        elif ext in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['.ppt', '.pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return UnstructuredImageLoader(file_path)
        elif ext in ['.txt', '.json']:
            return TextLoader(file_path)  # 커스텀 로더
        elif ext == '.hwp':
            return HwpLoader(file_path)  # 커스텀 로더
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        else:
            return UnstructuredFileLoader(file_path)  # 범용 폴백
```

### 특수 파일 로더

#### HwpLoader (HWP → PDF 변환)
```python
class HwpLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.output_dir = os.path.join('/tmp', str(uuid.uuid4()))
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load(self):
        # HWP → XHTML 변환
        subprocess.run(['hwp5html', self.file_path, '--output', self.output_dir], 
                      check=True, timeout=600)
        
        # XHTML → PDF 변환
        converted_file_path = os.path.join(self.output_dir, 'index.xhtml')
        pdf_save_path = self.file_path.replace('.hwp', '.pdf')
        HTML(converted_file_path).write_pdf(pdf_save_path)
        
        # PDF에서 텍스트 추출
        loader = PyMuPDFLoader(pdf_save_path)
        return loader.load()
```

#### TextLoader (텍스트 → PDF 변환)
```python
class TextLoader:
    def load(self):
        # 인코딩 자동 감지
        with open(self.file_path, 'rb') as f:
            raw_file = f.read(100)
        enc_type = chardet.detect(raw_file)['encoding']
        
        # HTML로 래핑 (포맷 보존)
        with open(self.file_path, 'r', encoding=enc_type) as f:
            content = f.read()
        html_content = f"<html><body><pre>{content}</pre></body></html>"
        
        # PDF 변환
        HTML(html_file_path).write_pdf(pdf_save_path)
        loader = PyMuPDFLoader(pdf_save_path)
        return loader.load()
```

#### TabularLoader (CSV/XLSX → JSON)
```python
class TabularLoader:
    def check_sql_dtypes(self, df):
        """SQL 데이터 타입 자동 추론"""
        for col in df.columns:
            dtype = str(df.dtypes[col]).lower()
            
            if 'int' in dtype:
                sql_dtype = 'BIGINT' if '64' in dtype else 'INT'
            elif 'float' in dtype:
                sql_dtype = 'FLOAT'
            elif 'bool' in dtype:
                sql_dtype = 'BOOLEAN'
            elif 'date' in dtype:
                sql_dtype = 'DATE'
            elif 'datetime' in dtype:
                sql_dtype = 'DATETIME'
            else:
                max_len = df[col].str.len().max() + 10
                sql_dtype = f'VARCHAR({max_len})'
    
    def load_xlsx_documents(self, file_path: str):
        """다중 시트 처리"""
        dfs = pd.read_excel(file_path, sheet_name=None)
        for sheet_name, df in dfs.items():
            df = df.fillna('null')
            # 각 시트별 처리
    
    def return_vectormeta_format(self):
        """[DA] 접두사로 데이터 분석용 표시"""
        text = "[DA] " + str(self.data_dict)
        return [GenOSVectorMeta.model_validate({
            'text': text,
            # ... 기본 메타데이터 ...
        })]
```

## 📂 전처리 흐름

### 1. 파일 형식별 분기 처리
```python
async def __call__(self, request: Request, file_path: str, **kwargs: dict):
    ext = os.path.splitext(file_path)[-1].lower()
    
    # 표 데이터
    if ext in ('.csv', '.xlsx'):
        loader = TabularLoader(file_path, ext)
        vectors = loader.return_vectormeta_format()
        return vectors
    
    # HWPX
    elif ext in ('.hwpx'):
        return await self.hwpx_processor(request, file_path, **kwargs)
    
    # 일반 문서
    else:
        documents = self.load_documents(file_path, **kwargs)
        chunks = self.split_documents(documents, **kwargs)
        vectors = await self.compose_vectors(chunks, file_path, request, **kwargs)
        return vectors
```

### 2. 일반 문서 로드
```python
def load_documents(self, file_path: str, **kwargs: dict) -> list[Document]:
    loader = self.get_loader(file_path)
    documents = loader.load()
    return documents
```

### 3. 문서 청킹 (LangChain)
```python
def split_documents(self, documents: list[Document], **kwargs: dict) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # 청크 크기
        chunk_overlap=200,     # 오버랩
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # 페이지별 청크 수 계산
    for chunk in chunks:
        page = chunk.metadata.get('page', 0)
        self.page_chunk_counts[page] += 1
    
    return chunks
```

### 4. 벡터 메타데이터 생성 (첨부용)
```python
async def compose_vectors(self, chunks: list[Document], file_path: str, 
                          request: Request, **kwargs: dict) -> list[dict]:
    # 글로벌 메타데이터
    global_metadata = dict(
        n_chunk_of_doc=len(chunks),
        n_page=max([c.metadata.get('page', 0) for c in chunks]) + 1,
        reg_date=datetime.now().isoformat(timespec='seconds') + 'Z'
    )
    
    vectors = []
    for chunk_idx, chunk in enumerate(chunks):
        page = chunk.metadata.get('page', 0)
        
        vector = (GenOSVectorMetaBuilder()
                  .set_text(chunk.page_content)
                  .set_page_info(page, self.page_chunk_counts[page], self.page_chunk_counts[page])
                  .set_chunk_index(chunk_idx)
                  .set_global_metadata(**global_metadata)
                  ).build()
        
        # e_page 설정 (첨부용 전용)
        vector.e_page = page
        vectors.append(vector)
    
    return vectors
```

### 5. HWPX 처리 (Docling)
```python
class HwpxProcessor:
    def __init__(self):
        self.converter = DocumentConverter(
            format_options={
                InputFormat.HWPX: HwpxFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
    
    async def __call__(self, request: Request, file_path: str, **kwargs):
        document = self.load_documents(file_path)
        
        # HybridChunker 사용 (Docling 방식)
        chunker = HybridChunker(max_tokens=2000, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=document))
        
        vectors = await self.compose_vectors(document, chunks, file_path, request)
        return vectors
```

## ✅ 유지보수 팁

### 지원 파일 형식 요약
| 형식 | 로더 | 특이사항 |
|------|------|----------|
| PDF | PyMuPDFLoader | 직접 처리 |
| DOC/DOCX | UnstructuredWordDocumentLoader | 직접 처리 |
| PPT/PPTX | UnstructuredPowerPointLoader | 직접 처리 |
| JPG/PNG | UnstructuredImageLoader | 직접 처리 |
| TXT/JSON | TextLoader | PDF 변환 후 처리 |
| HWP | HwpLoader | XHTML→PDF 변환 |
| HWPX | HwpxProcessor | Docling 네이티브 |
| CSV/XLSX | TabularLoader | [DA] 접두사, SQL 타입 추론 |
| MD | UnstructuredMarkdownLoader | 직접 처리 |

