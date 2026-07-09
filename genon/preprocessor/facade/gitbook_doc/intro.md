# Genos 전처리기 (Genos Doc Parser)
- Genos는 문서의 활용 목적과 요구 사항에 따라 최적화된 4가지 유형의 전처리 파이프라인을 제공합니다.
- 용법: Genos 웹 UI에서 **관리 > 리소스 > 전처리기 > 전처리기 생성**으로 이동하여, 아래 전처리기 파일 중 하나의 코드를 **전처리 코드(facade)** 영역에 그대로 복붙하면 됩니다.

  ![전처리기 등록 방법](./images/pre_register_guide.jpg)

  | 용도 | 파일 경로 |
  |---|---|
  | 첨부용 (채팅 실시간) | `preprocessor/facade/attachment_processor.py` |
  | 변환용 (PDF 표준화) | `preprocessor/facade/convert_processor.py` |
  | 파싱용 (Element 구조화 API) | `preprocessor/facade/parser_processor.py` |
  | 적재용 지능형 (RAG 고품질) | `preprocessor/facade/intelligent_processor.py` |

- 설치(이미지 빌드·배포) 및 Facade·config 구성 절차는 [GenOS v2 Doc Parser 설치 및 Facade 구성 매뉴얼](installation.md)을 참고하세요.

## 1. 첨부용 전처리기 (Attachment Processor)
사용자가 채팅 중 첨부로 업로드하는 파일을 실시간으로 분석하기 위한 경량화 전처리기입니다. 복잡한 구조 분석 과정을 생략하고, **텍스트 추출(Text Extraction)**에 집중하여 즉각적인 응답 속도를 보장합니다.

- **"속도 중심: 다양한 포맷의 텍스트 즉시 추출"**
- 설명: [attachment_processor.md](attachment_processor.md)
- 위치: [preprocessor/facade/attachment_processor.py](https://github.com/genonai/doc_parser/blob/develop/genon/preprocessor/facade/attachment_processor.py)
- 특징
  * **Native 텍스트 추출**: HWP, HWPX, DOCX, XLSX 등 원본 파일의 텍스트를 파싱하여 추출 속도 극대화
  * **멀티미디어 지원**: 오디오 파일(MP3, WAV, M4A)의 음성을 텍스트로 변환(STT)하여 처리
  * **데이터 변환**: CSV, Excel 등의 정형 데이터를 LLM이 이해하기 쉬운 텍스트/JSON 형태로 신속 변환
  * **청커 선택**: HWP/HWPX/DOCX 청킹 방식을 `chunker_type` kwargs로 선택 (`recursive`(기본) / `hybrid`). `recursive`는 docling 문서를 markdown으로 export 후 `RecursiveCharacterTextSplitter`로 분할하며, 임베딩 입력 한도(60,000 토큰)를 절대 상한으로 강제

## 2. 변환용 전처리기 (Convert Processor)
문서의 시각적 형태(Layout)를 유지해야 하거나, 텍스트 추출이 까다로운 레거시 포맷을 처리하기 위한 전처리기입니다. 모든 문서를 **PDF로 우선 변환(Rendering)**하여 포맷의 파편화를 해결합니다.
첨부용 전처리기 대용으로 쓸 수 있도록 고안된 첨부 전처리기의 변형 전처리기 입니다.

**"호환성 중심: PDF 표준화 후 텍스트 추출"**
- 설명: [convert_processor.md](convert_processor.md)
- 위치: [preprocessor/facade/convert_processor.py](https://github.com/genonai/doc_parser/blob/develop/genon/preprocessor/facade/convert_processor.py)
- 특징
  * **PDF 표준화**: PDF 변환 SDK(default) 또는 LibreOffice를 선택적으로 활용하여 PPT, DOCX 등 다양한 문서를 PDF 포맷으로 통일 (kwargs `use_pdf_sdk`로 엔진 선택, 기본값 `True`)
  * **시각적 정합성 유지**: 원본 문서의 폰트, 이미지 배치, 페이지 레이아웃을 그대로 보존
  * **하이브리드 추출**: 변환된 PDF 레이어에서 텍스트와 이미지 정보를 결합하여 안정적인 정보 획득

## 3. 파싱용 전처리기 (Parser Processor)
문서를 파싱하여 **element 단위 구조화 결과**를 반환하는 API 지향 전처리기입니다. 청킹이나 벡터 결합 없이, 원문 구조를 최대한 보존한 파싱 결과가 필요할 때 적합합니다.

**"구조 중심: Element 단위 파싱 결과 반환"**
- 설명: [parser_processor.md](parser_processor.md)
- 위치: [preprocessor/facade/parser_processor.py](https://github.com/genonai/doc_parser/blob/develop/genon/preprocessor/facade/parser_processor.py)
- 특징
  * **Element 기반 출력**: `title`, `paragraph`, `table`, `picture` 등 문서 구조를 `elements` 배열로 반환
  * **다양한 포맷 처리**: PDF/HTML/HWP(HWPX)/DOCX/CSV/XLSX/오디오 및 기타 문서 포맷 파싱 지원
  * **출력 포맷 제어**: `config.yaml`의 `output.format`(`json`/`html`/`markdown`)과 `table_format`으로 응답 형태 제어
  * **Gateway 호출 표준화**: `/preprocessor/{id}/healthcheck`, `/preprocessor/{id}/run` 엔드포인트로 외부 시스템 연동 용이

## 4. 적재용 지능형 전처리기 (Intelligent Processor)
RAG(검색 증강 생성) 시스템의 지식 베이스 구축을 위해 설계된 고성능 전처리기입니다. 단순 텍스트 추출을 넘어, **딥러닝 기반의 Layout 분석**을 통해 문서의 논리적 구조를 정확하게 파악합니다.

**"품질 중심: AI 기반 레이아웃 분석 및 고품질 데이터 적재"**
- 설명: [intelligent_processor.md](intelligent_processor.md)
- 위치: [preprocessor/facade/intelligent_processor.py](https://github.com/genonai/doc_parser/blob/develop/genon/preprocessor/facade/intelligent_processor.py)
- 입력 형식: 기본은 PDF 입력. 비-PDF가 들어오면 진입부에서 자동으로 PDF 변환 SDK(또는 LibreOffice)를 통해 PDF로 변환 후 처리.

### 왜 지능형 처리가 필요한가요?
복잡한 다단 구성, 표, 차트가 포함된 문서는 단순 추출 시 문맥이 파괴될 수 있습니다.
![복잡한 문서 예시](./images/pre_document_complexity.png)

## 핵심 기술

#### 1) Layout Detection (문서 구조 인식)
딥러닝 모델이 문서의 시각적 요소를 분석하여 제목, 본문, 표, 그림, 캡션 등의 역할을 명확히 구분합니다.

![Layout Detection 결과](./images/pre_layout_detection_example.png)
*AI 모델이 문서의 각 요소를 식별하고 구조화하는 과정*

*AI 모델이 문서 요소를 자동 식별 (핵심 11종)
- 1. 구조 및 위계
  - Title: 문서 전체의 메인 제목
  - Section-header: 문서의 장, 절 제목 (Heading)
- 2. 본문 및 컨텐츠
  - Text: 일반적인 본문 문단 (Paragraph)
  - List-item: 글머리 기호나 번호가 붙은 리스트의 개별 항목
- 3. 복합 데이터
  - Picture: 사진, 삽화, 다이어그램 등 이미지 요소
  - Table: 표 데이터
  - Formula: 수학 공식 및 수식
  - Caption: 표, 그림, 수식에 대한 설명 문구
  - Footnote: 페이지 하단의 각주
- 4. 보조 데이터 (Layout 에는 표시되지만, 본문 Chunk 에는 제외됨)
  - Page-header: 페이지 상단 정보 (문서 제목, 장 정보 등)
  - Page-footer: 페이지 하단 정보 (페이지 번호 등)

Layout Detection은 딥러닝 비전 모델을 활용하여 문서의 시각적 구조를 분석합니다. 단순 텍스트 추출을 넘어 제목, 본문, 표, 그림, 캡션 등의 역할을 명확히 구분합니다.

#### 2) TableFormer (표 데이터 구조화)
단순히 텍스트만 긁어오는 것이 아니라, 병합된 셀이나 다중 헤더를 가진 복잡한 표를 분석하여 Markdown 및 HTML 형태로 완벽하게 복원합니다.

![TableFormer 결과](./images/pre_tableformer_result.png)
*복잡한 표가 구조화된 마크다운 데이터로 변환된 결과*

#### 3) 지능형 보정 및 연결
* **Smart OCR**: 문서 전체가 아닌, 인코딩 오류(GLYPH)가 감지된 영역만 선별적으로 OCR을 수행하여 정확도와 효율성 확보
* **Context Enrichment**: LLM을 통해 목차(TOC)를 생성하고, 본문 내 '별지/별표' 참조를 실제 부록 파일과 자동 연결하여 검색 품질 향상

#### 4) LLM을 활용한 문서 문맥 기반 Enrichment
Enrichment는 `config.yaml`의 `enrichment` 항목(list)에 enricher를 나열하여 선택적으로 활성화합니다. 각 항목은 `enable` 플래그와 자체 모델 서빙(`url`/`api_key`/`model`)을 가지며, 현재 지원하는 enricher는 다음과 같습니다.

* **구조화된 목차 생성 (`toc`)**: LLM을 활용하여 문서의 논리적 흐름을 분석하여 문서에 명시적으로 없더라도 문서 전체의 Table of Contents 를 찾아내고 구성합니다.
  * "제1장 > 제1절 > 제1조" 형태의 계층적 목차(Table of Contents)를 생성
* **메타데이터 풍부화 (`metadata`)**: 문서에서 유추 할 수 있는 정보(생성일, 작성자 등)를 추출하여 각 청크에 포함시켜, 검색 시 해당 내용이 문서의 어느 위치(Context)에 해당하는지 파악할 수 있게 합니다.
  * `system_prompt`/`user_prompt`로 추출 항목을 정의하고, `field_transforms`로 추출 키를 벡터 메타 필드(예: `created_date`)로 선언적으로 매핑/변환합니다.
* **이미지 설명 생성 (`image_description`)**: 문서 내 그림을 앞뒤 문맥과 함께 VLM에 전달하여 이미지 설명 텍스트를 생성, 해당 picture 요소의 내용으로 채웁니다. 하위 옵션으로 **차트 처리(`chart`)** — docling 그림 분류로 차트만 골라(또는 전체를) 차트 전용 프롬프트로 설명 — 와 **본문요약(`doc_summary`)** — 문서 전체 요약을 이미지/차트 설명의 공통 컨텍스트(`{{doc_summary}}`)로 주입 — 을 지원합니다. 런타임 kwargs(`img_desc`/`chart_desc`/`chart_detection`/`doc_summary`)로 요청마다 토글할 수 있습니다.
* **커스텀 필드 추출 (`custom_fields`)**: 사용자가 정의한 프롬프트로 임의의 필드를 추출하여 메타데이터에 병합합니다. 복수 지정이 가능하며 외부 `config_file`로 분리할 수도 있습니다.
