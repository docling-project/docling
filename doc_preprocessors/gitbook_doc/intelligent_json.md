---
description: >-
  Genos 는 크게 적재용(내부), 적재용(외부), 적재용(규정), 첨부용 4가지 유형의 전처리기 (document parser)를
  지원합니다. 여기서는 적재용(내부) Doc Parser - 의미기반 청킹 전처리기의 코드 원형에 대해서 설명합니다.
icon: forward
---

# 적재용(내부) 문서 전처리기

<figure><img src="../../../../.gitbook/assets/preprocess_code.png" alt=""><figcaption><p>전처리기 상세에서 아래 코드를 확인하실 수 있습니다.</p></figcaption></figure>

***

여기서는 Genos 적재용(내부) 문서 파서의 전처리 파이프라인 내 주요 구성 요소에 대한 코드 중심의 설명을 제공합니다. 코드 조각과 함께 각 부분의 기능을 이해함으로써, 특정 요구 사항 및 문서 유형에 맞게 문서 처리 프로세스를 보다 효과적으로 조정할 수 있습니다.

### 🔧 공통 구성요소

#### `HierarchicalChunker` 및 `HybridChunker`

`HierarchicalChunker`는 문서를 계층적으로 청크로 나누는 역할을 하며, `HybridChunker`는 토큰 제한을 고려하여 섹션별 청크를 분할하고 병합하는 고급 청커입니다.

```python
class HierarchicalChunker(BaseChunker):
    merge_list_items: bool = True

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        # 모든 아이템과 헤더 정보 수집
        all_items = []
        all_header_info = []  # 각 아이템의 헤더 정보
        current_heading_by_level: dict[LevelNumber, str] = {}

        # 모든 아이템 순회하며 헤더 정보 추적
        for item, level in dl_doc.iterate_items():
            # 섹션 헤더 처리
            if isinstance(item, SectionHeaderItem) or (
                isinstance(item, TextItem) and
                item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
            ):
                header_level = (
                    item.level if isinstance(item, SectionHeaderItem)
                    else (0 if item.label == DocItemLabel.TITLE else 1)
                )
                current_heading_by_level[header_level] = item.text

                # ... 헤더 처리 로직

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

        # 청크에 두 가지 헤더 정보 모두 저장
        chunk._header_info_list = all_header_info
        yield chunk

class HybridChunker(BaseChunker):

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
```

#### `GenOSVectorMetaBuilder` 및 `GenOSVectorMeta`

`GenOSVectorMetaBuilder`는 각 청크에 대한 상세 메타데이터 객체인 `GenOSVectorMeta`를 단계적으로 생성하는 역할을 합니다.

**`GenOSVectorMeta` (Pydantic 모델)**

먼저, 최종적으로 생성될 메타데이터의 구조를 정의하는 Pydantic 모델입니다.

Python

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow' # Pydantic v2에서는 extra='allow' 대신 model_config 사용 가능

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
    created_date: int = None  # YYYYMMDD 형식의 정수
    authors_team: str = None      # 팀 리스트
    authors_department: str = None # 부서 리스트
    title: str = None         # 문서 제목
```

**설명:**

* `BaseModel`을 상속받아 Pydantic 모델로 정의됩니다. 이는 데이터 유효성 검사 및 직렬화/역직렬화를 용이하게 합니다.
* `Config.extra = 'allow'`: 모델에 정의되지 않은 추가 필드가 입력 데이터에 존재하더라도 오류를 발생시키지 않고 허용합니다. (Pydantic V2에서는 `model_config = ConfigDict(extra='allow')` 형태로 사용)
* 각 필드는 청크의 메타데이터 항목을 나타냅니다.
  * `text`: 청크의 텍스트 내용.
  * `n_char`, `n_word`, `n_line`: 문자 수, 단어 수, 줄 수.
  * `e_page`, `i_page`, `i_chunk_on_page`, `n_chunk_of_page`: 페이지 내에서의 청크 위치 정보.
  * `i_chunk_on_doc`, `n_chunk_of_doc`: 문서 전체에서의 청크 위치 정보.
  * `n_page`: 문서의 총 페이지 수.
  * `reg_date`: 처리 등록 시간.
  * `bboxes`: 페이지 내 해당 청크의 경계 상자 (JSON 문자열 형태).
  * `chunk_bboxes`: 청크를 구성하는 각 `DocItem`의 상세 경계 상자 정보 리스트.
  * `media_files`: 청크 내 포함된 미디어 파일(이미지) 정보 리스트.
  * **고객사별 필드**: `created_date`, `authors_team`, `authors_department`, `title` 등은 고객사의 특정 요구사항에 따라 추가된 메타데이터 필드입니다.

**사용자 정의 포인트:**

* 고객사에서 필요한 추가적인 메타데이터 항목이 있다면, 이 `GenOSVectorMeta` 모델에 새로운 필드를 추가로 정의할 수 있습니다.
* 필드 타입을 보다 엄격하게 정의하거나 (예: `Optional[str]`), 기본값을 설정하거나, 유효성 검사 로직을 추가할 수 있습니다.

***

**`GenOSVectorMetaBuilder` 클래스 및 주요 메서드**

`GenOSVectorMeta` 객체를 생성하는 빌더 클래스입니다.

Python

```python
class GenOSVectorMetaBuilder:
    def __init__(self):
        """빌더 초기화"""
        self.text: Optional[str] = None
        self.n_char: Optional[int] = None
        # ... (다른 필드들도 초기화) ...
        self.created_date: Optional[int] = None
        self.authors_team: Optional[str] = None      # 팀 리스트
        self.authors_department: Optional[str] = None # 부서 리스트
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
            if hasattr(self, key): # 빌더 내에 해당 속성이 정의되어 있는지 확인
                setattr(self, key, value)
        return self

    def set_chunk_bboxes(self, doc_items: list, document: DoclingDocument) -> "GenOSVectorMetaBuilder":
        self.chunk_bboxes = []
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
            if isinstance(item, PictureItem): # DocItem이 PictureItem인 경우
                path = str(item.image.uri) # 이미지 파일 경로
                name = path.rsplit("/", 1)[-1] # 파일 이름만 추출
                temp_list.append({'name': name, 'type': 'image', 'ref': item.self_ref})
        self.media_files = temp_list
        return self

    def build(self) -> GenOSVectorMeta:
        """설정된 데이터를 사용해 최종적으로 GenOSVectorMeta 객체 생성"""
        return GenOSVectorMeta(
            text=self.text,
            n_char=self.n_char,
            # ... (모든 필드를 GenOSVectorMeta 생성자에 전달) ...
            created_date=self.created_date,
            authors_team= self.authors_team,      # 팀 리스트
            authors_department = self.authors_department, # 부서 리스트
            title=self.title,
        )

```

**설명:**

* **`__init__`**: 빌더 내부의 모든 속성들을 초기화합니다. 이 속성들은 `GenOSVectorMeta`의 필드들과 대부분 일치합니다.
* **`set_text`**: 청크의 텍스트를 설정하고, 문자 수, 단어 수, 줄 수를 계산하여 내부 속성에 저장합니다.
* **`set_page_info`**: 페이지 번호, 페이지 내 청크 인덱스, 페이지 내 총 청크 수를 설정합니다.
* **`set_chunk_index`**: 문서 전체에서의 청크 인덱스를 설정합니다.
* **`set_global_metadata`**: `DocumentProcessor.compose_vectors`에서 전달받은 `global_metadata` 딕셔너리의 값들을 빌더의 해당 속성에 할당합니다. 빌더 내에 `global_metadata`의 키와 동일한 이름의 속성이 있어야 값이 할당됩니다.
* **`set_chunk_bboxes`**: 청크를 구성하는 모든 `DocItem`들의 상세한 경계 상자 정보를 추출하여 리스트로 저장합니다. 각 항목은 페이지 번호, 정규화된 좌표(0\~1 값), `DocItem`의 타입 및 참조 ID를 포함합니다. 정규화된 좌표는 페이지 크기에 상대적인 위치를 나타내므로, 다양한 크기의 페이지에서도 일관되게 위치를 표현할 수 있습니다.
* **`set_media_files`**: 청크 내에 `PictureItem`(이미지)이 포함되어 있다면, 해당 이미지의 파일 이름, 타입("image"), 참조 ID를 추출하여 리스트로 저장합니다.
* **`build`**: 지금까지 `set_...` 메서드들을 통해 빌더 내부에 축적된 모든 속성값들을 사용하여 최종적으로 `GenOSVectorMeta` Pydantic 모델 객체를 생성하고 반환합니다.

**사용자 정의 포인트:**

* `GenOSVectorMeta` 모델에 새로운 필드를 추가했다면, 이 빌더에도 해당 필드를 위한 내부 속성과 `set_...` 메서드를 추가해야 합니다.
* `build` 메서드에서 `GenOSVectorMeta` 객체를 생성할 때 새로 추가된 필드도 인자로 전달하도록 수정해야 합니다.
* 특정 필드값을 설정하기 전에 추가적인 가공 로직(예: 날짜 형식 변환, 특정 코드값 매핑 등)이 필요하다면 해당 `set_...` 메서드 내부에 구현할 수 있습니다.

***

### 📂 공통 전처리 흐름

#### `DocumentProcessor`

`DocumentProcessor` 클래스는 Genos 의 전처리기가 호출되는 관문입니다. 내부 구성을 보면, 문서를 로드, 변환, 분할하고 각 부분에 대한 메타데이터를 구성하는 핵심 요소입니다.

**`__init__` (초기화)**

`DocumentProcessor` 인스턴스가 생성될 때 호출되는 초기화 메서드입니다. 여기서 문서 처리 파이프라인의 주요 설정들이 정의됩니다.

Python

```python
class DocumentProcessor:

    def __init__(self):
        '''
        initialize Document Converter
        '''
        self.page_chunk_counts = defaultdict(int)
        self.simple_pipeline_options = PipelineOptions()
        self.simple_pipeline_options.save_images = False

        # 기본 컨버터들 생성
        self._create_converters()
```

**설명:**

* **`page_chunk_counts`**: 각 페이지에서 생성된 청크의 수를 추적하는 데 사용됩니다.
* **`PipelineOptions`**: 문서 처리 파이프라인의 전반적인 설정을 담당합니다. 문서 유형 및 요구사항에 따라 이 부분을 주로 커스터마이징하게 됩니다.
  * `simple_pipeline_options.save_images = False`: PDF 각 페이지를 이미지로 생성할지 여부입니다.

***

**`_create_converters`**

문서 변환에 사용할 변환기를 설정하는 함수입니다.

Python

```python
    def _create_converters(self):
        """컨버터들을 생성하는 헬퍼 메서드"""
        # HWP와 HWPX 모두 지원하는 통합 컨버터
        self.converter = DocumentConverter(
            format_options={
                InputFormat.JSON_DOCLING: BOKJsonFormatOption(
                  pipeline_options = self.simple_pipeline_options,
                )
              }
            )
```

**설명:**

* **`self.converter` (기본 변환기)**: JSON을 처리하는 Primary 문서 변환기입니다. 이 백엔드는 복잡한 JSON 구조를 효과적으로 처리합니다.

***

**`load_documents_with_docling` 및 `load_documents`**

문서를 실제 로드하고 파싱하는 부분입니다.

Python

```python
    def load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        save_images = kwargs.get('save_images', False)

        # save_images 옵션이 현재 설정과 다르면 컨버터 재생성
        if self.simple_pipeline_options.save_images != save_images:
            self.simple_pipeline_options.save_images = save_images
            self._create_converters()

        conv_result: ConversionResult = self.converter.convert(file_path, raises_on_error=True)
        return conv_result.document

    def load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument:
        return self.load_documents_with_docling(file_path, **kwargs)
```

**설명:**

* `load_documents_with_docling(self, file_path: str, **kwargs: dict) -> DoclingDocument`:
  * 주어진 `file_path`로부터 문서를 로드하고 `DoclingDocument` 객체로 변환하여 반환합니다.
  * 먼저 `self.converter` (기본 변환기: `BOKJsonDocumentBackend`)를 사용하여 문서 변환을 시도합니다.
  * `raises_on_error=True`는 변환 중 오류 발생 시 예외를 발생시키도록 합니다.
  * 성공적으로 변환된 `conv_result.document` (즉, `DoclingDocument` 객체)를 반환합니다.
* `load_documents(self, file_path: str, **kwargs: dict) -> DoclingDocument`:
  * `load_documents_with_docling` 메서드를 호출하여 문서를 로드하는 공개 인터페이스 역할을 합니다.
  * `**kwargs`를 통해 추가적인 파라미터를 내부 메서드로 전달할 수 있는 구조입니다. (현재,`kwargs`가 직접적으로 활용되지는 않고 있습니다.)

**사용자 정의 포인트:**

* 문서 로딩 전 특정 전처리 작업이 필요하거나, PDF 외 다른 포맷에 대해 별도의 로직을 적용하고 싶다면 이 부분을 확장할 수 있습니다.
* `kwargs`를 활용하여 `PipelineOptions`의 일부 값을 동적으로 변경하는 로직을 추가할 수도 있습니다 (예: 특정 문서 유형에 따라 OCR 활성화).

***

**`split_documents`**

로드된 문서를 의미 있는 작은 단위(청크)로 분할합니다.

Python

```python
    def split_documents(self, documents: DoclingDocument, **kwargs: dict) -> list[DocChunk]:
        chunker: HybridChunker = HybridChunker(
            max_tokens=1000,
            merge_peers=True
        )

        chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))
        for chunk in chunks:
            self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1
        return chunks
```

**설명:**

* `chunker: HybridChunker = HybridChunker()`: 문서를 청크로 나누기 위해 `HybridChunker` 인스턴스를 생성합니다. `HybridChunker`는 내부적으로 계층적 구조(Hierarchical)와 의미론적 분할(Semantic, 주석 처리된 `semchunk` 의존성 부분에서 유추)을 결합하여 문서를 분할합니다.&#x20;
  * `max_tokens`는 각 청크의 최대 토큰 수를 제한합니다.
  * `merge_peers`는 인접한 청크들을 병합할지 여부를 결정합니다.
* `chunks: List[DocChunk] = list(chunker.chunk(dl_doc=documents, **kwargs))`: `chunker`의 `chunk` 메서드를 호출하여 `DoclingDocument`를 `DocChunk` 객체들의 리스트로 변환합니다. `**kwargs`는 청킹 과정에 필요한 추가 옵션을 전달하는 데 사용될 수 있습니다 .
* `for chunk in chunks: self.page_chunk_counts[chunk.meta.doc_items[0].prov[0].page_no] += 1`: 각 청크가 어떤 페이지에서 왔는지 파악하여 `self.page_chunk_counts` 딕셔너리에 페이지별 청크 수를 기록합니다. 이는 추후 메타데이터 생성 시 활용됩니다. (`chunk.meta.doc_items[0].prov[0].page_no`는 청크를 구성하는 첫번째 문서 아이템의 첫번째 출처 정보에서 페이지 번호를 가져옵니다.)

**사용자 정의 포인트:**

* `HybridChunker`의 설정 (예: 최대 토큰 수 `max_tokens`, 병합 옵션 `merge_peers`)은 `HybridChunker` 클래스 정의 부분에서 수정할 수 있습니다.

***

**`parse_created_date`**

작성일 텍스트를 파싱하여 YYYYMMDD 형식의 정수로 변환합니다.

Python

```python
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
```

**설명:**

* `date_text`가 유효한 문자열인지 확인합니다. 비어있거나 "None"인 경우 0을 반환합니다.
* 공백을 제거하고 정리합니다.
* `YYYY-MM-DD` 형식과 `YYYY-MM` 형식, `YYYY` 형식에 대해 정규 표현식을 사용하여 매칭을 시도합니다.
* 각 형식에 대해 유효한 날짜인지 검증하고, YYYYMMDD 형식의 정수로 변환하여 반환합니다.
* 모든 매칭이 실패한 경우 0을 반환합니다.
* 작성일이 명확하지 않거나 잘못된 형식인 경우 0으로 처리하여 데이터 일관성을 유지합니다.

***

**`enrichment`**

문서에 대한 추가 정보를 생성하거나 기존 정보를 보강하는 과정을 수행합니다.

Python

```python
    def enrichment(self, document: DoclingDocument, **kwargs: dict) -> DoclingDocument:

        # 새로운 enriched result 받기
        document = enrich_document(document, self.enrichment_options)
        return document
```

**설명:**

* 문서에 대한 추가 정보를 생성하거나 기존 정보를 보강하는 과정을 수행합니다.
* `enrich_document` 함수를 호출하여 문서를 보강하고, 보강된 문서를 반환합니다.
* 문서 보강에 필요한 추가 옵션은 `self.enrichment_options`에서 가져옵니다.

**작성일 메타데이터 추출 프로세스 상세:**

### 1. 프롬프트 구성 방식

메타데이터 추출은 두 가지 프롬프트로 구성됩니다:

* **`metadata_system_prompt`**:
  * 기본값:
    ```
    You are a highly skilled document extraction assistant specializing in Korean financial documents.
    Your task is to carefully analyze the given document and extract the creation date with high accuracy.
    ```
  * 사용자 정의 가능: `enrichment_options.metadata_system_prompt`로 설정

* **`metadata_user_prompt`**:
  * 기본값:
    ```
    Follow these instructions precisely:

    Here is the financial document to analyze:

    <financial_document>
    {document_content}
    </financial_document>

    Process:
    1. Extract and list all date expressions (2024.05.10, 24.05.10, 2024년 5월 10일, 5.10(금), etc.)
    2. Analyze the context, paying attention to:
       - Dates with "작성일", "작성일자", "작성", "게재 확정일"
       - Dates near "보고자료", "회의자료"
       - Dates at the top or beginning of document
    3. Apply selection criteria:
       - Prioritize explicitly labeled creation dates
       - Exclude "기준일", "통계일", "데이터 기준일"
       - Consider "최종수정일" only if no creation date found
    4. Normalize to YYYY-MM-DD format
    5. Output within <date> tags (or <date>없음</date> if not found)
    ```
  * 문서 내용이 `{document_content}` 플레이스홀더에 삽입됨
  * 사용자 정의 가능: `enrichment_options.metadata_user_prompt`로 설정

### 2. created_date로 변환 과정

enrichment를 통해 추출된 날짜는 다음 과정을 거쳐 최종 메타데이터가 됩니다:

**Step 1: KeyValueItem으로 저장**
```python
# 추출된 메타데이터를 GraphCell 구조로 변환
graph_cells = [
    GraphCell(label=GraphCellLabel.KEY, text="작성일"),
    GraphCell(label=GraphCellLabel.VALUE, text="2024-05-10")
]
document.add_key_values(graph=GraphData(cells=graph_cells))
```

**Step 2: compose_vectors에서 추출**
```python
# document.key_value_items[0].graph.cells[1]에서 날짜 텍스트 추출
if document.key_value_items:
    date_text = document.key_value_items[0].graph.cells[1].text
    created_date = self.parse_created_date(date_text)
```

**Step 3: 날짜 형식 정규화**
```python
def parse_created_date(self, date_text: str) -> Optional[int]:
    # "2024-05-10" → 20240510 (YYYYMMDD 정수)
    # "2024-05" → 20240501 (일자를 01로 설정)
    # "2024" → 20240101 (월일을 0101로 설정)

    match_full = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_text)
    if match_full:
        year, month, day = match_full.groups()
        return int(f"{year}{month.zfill(2)}{day.zfill(2)}")
```

**Step 4: 최종 메타데이터 구성**
```python
global_metadata = dict(
    created_date=created_date,  # 20240510 형식의 정수
    title=title,
    authors_team=authors_team,
    authors_department=authors_department
)

# 각 청크의 메타데이터에 포함
vector = GenOSVectorMetaBuilder()
    .set_global_metadata(**global_metadata)
    .build()
```

**사용자 정의 포인트:**

* **커스텀 프롬프트**: 고객사별 문서 형식에 맞춰 `metadata_system_prompt`와 `metadata_user_prompt`를 정의하여 추출 정확도 향상
* **추가 메타데이터**: 작성일 외에도 작성자, 부서, 문서 카테고리 등 다양한 메타데이터를 추출하도록 프롬프트 확장 가능
* **날짜 형식 처리**: `parse_created_date` 함수를 수정하여 다양한 날짜 형식 지원 가능

***

**`compose_vectors`**

분할된 청크들에 대해 메타데이터를 생성하고 최종적인 벡터(딕셔너리 형태) 리스트를 구성합니다. 이 부분이 고객사별 요구사항을 반영하는 부분이며 매우 중요합니다.

Python

```python
    async def compose_vectors(self, document: DoclingDocument, chunks: List[DocChunk], file_path: str, request: Request,
                              **kwargs: dict) -> \
            list[dict]:
        title = ""
        created_date = 0
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

        # kwargs에서 authors_team와 authors_department 추출
        if "authors_team" in kwargs:
            authors_team = json.dumps(kwargs["authors_team"])

        if "authors_department" in kwargs:
            authors_department = json.dumps(kwargs["authors_department"])

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
            authors_team=authors_team,
            authors_department=authors_department,
            title=title
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
```

**설명:**

**한국은행 기록물의 메타데이타 Mapping 예**

* **`global_metadata`**: 문서 전체에 공통적으로 적용될 메타데이터를 딕셔너리로 구성합니다.
  * `n_chunk_of_doc=len(chunks)`: 문서 내 총 청크 수.
  * `n_page=document.num_pages()`: 문서의 총 페이지 수.
  * `reg_date`: 현재 시간을 ISO 형식의 문자열로 등록일로 사용합니다.
  * **고객사별 필드**: `created_date`, `authors_team`, `authors_department`, `title` 등은 enrichment를 통해 값을 가져와 설정됩니다. **이 부분이 고객사 시스템과 연동하여 문서의 고유 메타정보를 주입하는 핵심 지점입니다.**. `authors_team`, `authors_department`은 `**kwargs`로부터 값을 가져와 설정됩니다.
* 루프 (`for chunk_idx, chunk in enumerate(chunks):`): 각 청크를 순회하며 메타데이터를 생성합니다.
  * `chunk_page = chunk.meta.doc_items[0].prov[0].page_no`: 현재 청크의 시작 페이지 번호를 가져옵니다.
  * `content = self.safe_join(chunk.meta.headings) + chunk.text`: 청크의 제목(headings)들과 실제 텍스트(text)를 결합하여 청크의 전체 내용을 구성합니다. `safe_join`은 제목 리스트를 안전하게 문자열로 합치는 유틸리티 함수로 보입니다.
  * **`GenOSVectorMetaBuilder()`**: `GenOSVectorMetaBuilder`를 사용하여 체이닝 방식으로 각 메타데이터 필드를 설정합니다 (상세 내용은 `GenOSVectorMetaBuilder` 섹션 참조).
    * `.set_text(content)`: 청크 내용 설정.
    * `.set_page_info(...)`: 페이지 관련 정보 설정.
    * `.set_chunk_index(chunk_idx)`: 문서 내 청크 인덱스 설정.
    * `.set_global_metadata(**global_metadata)`: 위에서 정의한 `global_metadata`를 전달.
    * `.set_chunk_bboxes(...)`: 청크를 구성하는 세부 항목들의 경계 상자 정보 설정.
    * `.set_media_files(...)`: 청크 내 이미지 파일 정보 설정.
  * `.build()`: 설정된 정보들을 바탕으로 `GenOSVectorMeta` 객체를 생성합니다.
  * `vectors.append(...)`: 생성된 `GenOSVectorMeta` Pydantic 모델 객체를 `vectors` 리스트에 추가합니다.
* 페이지 변경 감지 로직: `current_page`와 `chunk_index_on_page`를 사용하여 페이지가 바뀔 때마다 페이지 내 청크 인덱스를 0으로 초기화합니다.

**사용자 정의 포인트 (매우 중요):**

* **`global_metadata` 확장**: 고객사의 고유한 문서 속성들(예: 작성일, 문서제목 등)을 `global_metadata`에 추가하고, 최종 메타데이터에 포함시킬 수 있습니다.
* `content` 구성 방식 변경: 단순히 제목과 텍스트를 합치는 것 외에, 특정 순서로 재배열하거나 요약 정보를 추가하는 등의 로직을 구현할 수 있습니다.

***

**`__call__`**

`DocumentProcessor` 인스턴스를 GenOS 에서 호출할때의 진입점으로, 함수처럼 호출했을 때 실행되는 메인 로직입니다. 문서 처리의 전체 흐름을 담당합니다.

Python

```python
    async def __call__(self, request: Request, file_path: str, **kwargs: dict):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
            json.dump(kwargs, temp_file, ensure_ascii=False, indent=2)
            temp_file_path = temp_file.name

        try:
            document: DoclingDocument = self.load_documents(temp_file_path, **kwargs)

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

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
```

**설명:**

1. `with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:`: 임시 JSON 파일을 생성하고, 이후에 이 파일에 데이터를 기록합니다.
    * `json.dump(kwargs, temp_file, ensure_ascii=False, indent=2)`: `kwargs` 딕셔너리를 JSON 형식으로 임시 파일에 저장합니다. `ensure_ascii=False`는 한글 등의 비ASCII 문자가 올바르게 저장되도록 합니다.
    * `temp_file_path = temp_file.name`: 생성된 임시 파일의 경로를 변수에 저장합니다.
2. `document: DoclingDocument = self.load_documents(temp_file_path, **kwargs)`: `load_documents` 메서드를 호출하여 입력된 `file_path`의 문서를 로드합니다. `**kwargs`는 사용자가 입력 UI 를 통해서 지정하거나, 혹은 수집기가 수집단에서 지정한 정보를 전달합니다.
3. 아티팩트 경로 설정:
   * `output_path, output_file = os.path.split(file_path)`: 입력 파일 경로에서 디렉토리 경로와 파일 이름을 분리합니다.
   * `filename, _ = os.path.splitext(output_file)`: 파일 이름에서 확장자를 제외한 순수 파일명을 얻습니다.
   * `artifacts_dir = Path(f"{output_path}/{filename}")`: 문서 처리 과정에서 생성되는 이미지 등의 아티팩트(중간 결과물)를 저장할 디렉토리 경로를 구성합니다. 예를 들어, `input/mydoc.pdf` 라면 `input/mydoc/` 와 같은 경로가 됩니다.
   * `reference_path` 설정: `artifacts_dir`이 절대 경로인지 상대 경로인지에 따라 이미지 참조를 위한 기본 경로를 설정합니다.
4. `document = document._with_pictures_refs(image_dir=artifacts_dir, reference_path=reference_path)`:
   * `DoclingDocument` 객체 내의 그림(PictureItem)들이 실제 파일로 저장될 위치(`image_dir`)와 참조 경로(`reference_path`)를 설정하여, 그림 객체가 실제 파일 경로를 참조하도록 업데이트합니다. `PipelineOptions`에서 `generate_picture_images = True`로 설정된 경우, `docling` 라이브러리가 내부적으로 이 경로에 이미지들을 저장하고, 이 메서드를 통해 문서 객체 내의 참조를 업데이트합니다.
5. `document = self.enrichment(document, **kwargs)`: 문서 객체에 추가적인 정보를 주입합니다.
6. `chunks: List[DocChunk] = self.split_documents(document, **kwargs)`: 업데이트된 `document` 객체를 `split_documents` 메서드에 전달하여 청크 리스트를 얻습니다.
    * text가 있는 item이 없을 때 document에 임의의 text item 추가합니다.
7.  `vectors = [] ...`:
   * 만약 생성된 청크가 1개 이상이면 (`len(chunks) >= 1`), `compose_vectors` 메서드를 호출하여 최종 메타데이터 벡터 리스트를 생성합니다.
   * 청크가 하나도 없으면 `GenosServiceException`을 발생시켜 오류 상황임을 알립니다.
8.  `return vectors`: 생성된 벡터 리스트를 반환합니다.

**사용자 정의 포인트:**

* `**kwargs` 활용: `__call__` 메서드에 전달되는 `**kwargs`는 내부적으로 `load_documents`, `split_documents`, `compose_vectors`로 전파될 수 있으므로, 문서 처리 전 과정에 걸쳐 동적인 설정을 주입하는 통로로 활용될 수 있습니다. 예를 들어, API 요청으로부터 특정 파라미터를 받아 `kwargs`로 전달하고, 이 값을 기반으로 `PdfPipelineOptions`의 일부를 변경하거나 `compose_vectors`에서 특정 메타데이터를 추가/제외하는 등의 로직을 구현할 수 있습니다.
* 아티팩트 관리: `artifacts_dir` 생성 규칙이나 `reference_path` 설정 방식을 변경하여, 생성되는 중간 파일들의 저장 위치 및 참조 방식을 조직의 정책에 맞게 수정할 수 있습니다.

### ✨ 사용자 정의 포인트

* **메타 필드 확장**: `GenOSVectorMeta`와 `Builder` 클래스에 필드 추가
* **페이지 처리 로직 수정**: `set_page_info` 파라미터 조정
* **청크 분할 커스터마이징**:  `HybridChunker` 파라미터 기준값 수정
* **문자열 변환**: `NaN`, `None` 등 값은 전처리 단계에서 빈 문자열 처리

### ✅ 유지보수 팁

* Pydantic `extra='allow'` 설정으로 필드 변경이 유연하게 허용됨
* Builder 패턴을 사용하여 필드 설정 오류를 방지하고 유지보수를 단순화

***

이와 같이 코드 조각과 함께 설명을 보면서 `DocumentProcessor`와 `GenOSVectorMetaBuilder`의 작동 방식과 사용자 정의 지점을 파악하시면, 요구사항에 맞게 전처리 파이프라인을 효과적으로 수정하고 확장하실 수 있을 것입니다.
