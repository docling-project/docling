# 변환용 전처리기 매뉴얼

변환용 전처리기(Convert Processor)는 다양한 문서 포맷을 **PDF로 표준화**한 뒤 레이아웃 분석·표 구조 인식·선택적 OCR·LLM Enrichment 를 거쳐 벡터 청크를 생성하는 전처리기입니다.

> **이 매뉴얼의 전제**: 전처리기의 동작은 **코드 수정이 아니라 `convert_processor_config.yaml` 로 제어**합니다. 과거에는 OCR 주소·모델 서빙 ID·임계값 등을 facade 코드에서 직접 수정했지만, 현재는 모든 주요 옵션이 config YAML 로 분리되어 있습니다. 따라서 운영자는 **YAML 옵션을 튜닝**하는 방식으로 작업하며, 코드 내부는 [부록](#부록-코드-내부-상세)에서 보조적으로만 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [빠른 시작](#2-빠른-시작)
3. [설정 (convert_processor_config.yaml)](#3-설정-convert_processor_configyaml) ★ 핵심
   - 3.1 [전체 스키마](#31-전체-스키마)
   - 3.2 [OCR 설정](#32-ocr-설정)
   - 3.3 [레이아웃 설정](#33-레이아웃-설정)
   - 3.4 [PDF 파이프라인 설정](#34-pdf-파이프라인-설정)
     - [PPT 페이지 설명 (`formats.ppt.page_description`)](#341a-ppt-페이지-설명-formatspptpage_description)
     - 3.4.2 [xlsx(엑셀) 직접 처리 (`formats.xlsx`)](#342-xlsx엑셀-직접-처리-formatsxlsx)
     - 3.4.3 [표 텍스트 형식 (`output.table_format`)](#343-표-텍스트-형식-outputtable_format)
   - 3.5 [Enrichment 설정](#35-enrichment-설정)
   - 3.6 [사이트 적용 시 필수 수정 항목](#36-사이트-적용-시-필수-수정-항목)
   - 3.7 [자주 쓰는 튜닝 시나리오](#37-자주-쓰는-튜닝-시나리오)
4. [처리 동작 개요 (보조)](#4-처리-동작-개요-보조)
5. [출력 데이터 구조](#5-출력-데이터-구조)
6. [예외 / 트러블슈팅](#6-예외--트러블슈팅)
- [부록: 코드 내부 상세](#부록-코드-내부-상세)

---

## 1. 개요

변환용 전처리기는 문서의 시각적 형태(Layout)를 유지해야 하거나, 텍스트 추출이 까다로운 레거시 포맷을 안정적으로 처리하기 위한 전처리기입니다. 모든 문서를 **PDF 로 우선 변환(Rendering)** 하여 포맷 파편화를 해결하고, 그 위에서 레이아웃·표·OCR·Enrichment 를 수행합니다.

### 설계 철학

```
"호환성 중심: PDF 표준화 후 텍스트 추출"
```

PPT·DOCX 등 다양한 포맷을 PDF 로 통일한 뒤, 원본의 폰트·이미지 배치·페이지 레이아웃을 보존하면서 텍스트와 이미지를 결합 추출합니다.

### 대상 포맷

| 입력 | 처리 경로 |
|------|-----------|
| `.pdf` | Docling PDF 파이프라인 (레이아웃 + 표 구조 + 선택적 OCR) |
| `.docx`, `.pptx` | Docling 파이프라인 처리, 이후 PDF 변환(이미지/미리보기용) |
| `.ppt` (레거시) | LangChain `UnstructuredPowerPointLoader` 경로 (PDF 변환 후) |
| 기타 | Docling 경로 시도 |

> Docling 이 직접 지원하지 않는 레거시 `.ppt` 만 LangChain 폴백 경로를 사용합니다.

### 핵심 특징

| 특징 | 설명 | 제어 위치 (config) |
|------|------|--------------------|
| **PDF 표준화** | PDF 변환 SDK 또는 LibreOffice 로 포맷 통일 | (kwargs `use_pdf_sdk`) |
| **레이아웃 분석** | 제목/본문/표/이미지 검출 및 reading order 개선 | `layout` |
| **표 구조 인식** | TableFormer 로 병합 셀·다중 헤더 표 분석 | `pdf_pipeline.table_structure_mode` |
| **Smart OCR** | GLYPH(인코딩 깨짐) 감지 영역만 선별 OCR | `ocr` |
| **LLM Enrichment** | 목차(TOC)·메타데이터·이미지 설명·커스텀 필드 추출 | `enrichment` |

### attachment / intelligent 와의 비교

| 비교 항목 | attachment_processor | convert_processor | intelligent_processor |
|-----------|----------------------|-------------------|------------------------|
| 설계 목표 | 속도 중심(즉각 응답) | **호환성·품질 중심** | 품질 최우선(고정밀) |
| 전처리 파이프라인 | `SimplePipeline`(경량) | **전체 PDF 파이프라인** | 전체 PDF 파이프라인 |
| 청커 | `HybridChunker`(사실상 무제한) | **`GenosSmartChunker`** | 정밀 청커 |
| OCR | 없음 | **GLYPH 감지 시 선택적 OCR** | 적극적 OCR |
| Enrichment | 없음 | **TOC/메타데이터/이미지/커스텀** | 지원 |
| 확장 메타데이터 | 기본(text, page 등) | **created_date, authors, title** | 지원 |
| 오디오/정형 데이터 | 지원(STT, CSV/XLSX) | 미지원(문서 전용) | 미지원 |

---

## 2. 빠른 시작

### 2.1 Genos 등록

1. **Facade 등록**: Genos 관리 UI 의 전처리기(facade) 등록 화면에 `convert_processor.py` 를 등록합니다. Genos 는 `DocumentProcessor()` 를 **무인자**로 호출하므로, 설정 파일 경로는 코드가 자동으로 resolve 합니다.
2. **Resource config 등록**: `convert_processor_config.yaml` 을 리소스로 함께 등록합니다. 이 YAML 이 모든 동작의 단일 제어 지점입니다.

### 2.2 config 로딩 우선순위

`DocumentProcessor.__init__(config_path=None)` 은 `_resolve_default_convert_config_path()` 로 다음 순서로 설정 파일을 찾습니다.

```
1순위: preprocessor/resource_dev/convert_processor_config.yaml   (존재하면 사용 — 개발/사이트 오버라이드용)
2순위: preprocessor/resource/convert_processor_config.yaml       (배포 기본본)
폴백:  파일이 없거나 형식 오류면 → 코드 내장 기본값으로 동작 (경고 로그)
```

> 잘못된 값(타입 오류, 알 수 없는 enum 등)은 경고 로그 후 안전한 기본값으로 폴백하여 startup 견고성을 보장합니다.

### 2.3 사이트별 필수 수정 항목 요약

새 사이트에 배포할 때 최소한 아래 placeholder 를 환경에 맞게 교체해야 합니다 (자세한 내용은 [3.6](#36-사이트-적용-시-필수-수정-항목)).

| placeholder | 위치 | 교체 대상 |
|-------------|------|-----------|
| `<OCR_ENDPOINT>` | `ocr.paddle.ocr_endpoint` | PaddleOCR 서버 주소 |
| `<LAYOUT_SERVING_ID>` | `layout.genos_layout.endpoint` | Genos layout 모델 서빙 ID |
| `<ENRICHMENT_SERVING_ID>` | `enrichment[].toc/metadata.url` | Genos enrichment LLM 서빙 ID |
| `<IMAGE_DESCRIPTION_SERVING_ID>` | `enrichment[].image_description.url` | 이미지 설명 LLM 서빙 ID |

---

## 3. 설정 (convert_processor_config.yaml)

이 전처리기의 동작은 전적으로 이 YAML 로 제어됩니다. 최상위 5개 섹션으로 구성됩니다.

| 섹션 | 역할 |
|------|------|
| `defaults` | 로깅 레벨 등 공통 기본값 |
| `ocr` | OCR 모드·엔진·임계값 |
| `layout` | 레이아웃 분석 모델 선택 및 서빙 설정 |
| `pdf_pipeline` | Docling PDF 파싱 성능/품질 노브 |
| `enrichment` | LLM 기반 보강 작업 목록(list) |

### 3.1 전체 스키마

아래는 캐노니컬 설정(`resource/convert_processor_config.yaml`)의 전체 구조입니다. 프롬프트는 별도 `.md` 파일로 분리되어 `*_file` 키로 참조합니다(상세는 [3.5 Enrichment 설정](#35-enrichment-설정)의 "프롬프트 파일 분리 & 변수 치환" 참고).

```yaml
defaults:
  # 5=DEBUG, 4=INFO, 3=WARNING, 2=ERROR, 1=CRITICAL, 0=NOLOG
  log_level: 4

# 포맷별 처리 옵션(자세한 설명은 3.4.2 참고). xlsx 는 PDF 변환 없이 직접 처리.
formats:
  xlsx:
    processing_mode: "docling"    # "docling"(default) | "tabular"
    tabular:                      # tabular 모드에서만 사용
      header_row: 0               # 0=자동판정 | >0=단일헤더 강제
      multi_table: false          # true 면 1시트 복수표(빈 행 분리) 분리

ocr:
  # OCR 수행 모드. "auto"(default)=휴리스틱 기반 재OCR / "force"=무조건 전체 OCR / "disable"=OCR 안 함
  # (PDF 입력에만 적용. DOCX/기타 포맷은 ocr_mode 무관)
  ocr_mode: "auto"

  # OCR 엔진 선택. "paddle"(default) | "upstage"
  engine: "paddle"

  # 글리프 깨진 테이블 셀 재OCR 시 HTTP timeout(초)
  table_cell_ocr_timeout: 60

  paddle:
    # engine: "paddle" 일 때만 사용. <OCR_ENDPOINT>: PaddleOCR 서버 주소로 변경 필요
    ocr_endpoint: "http://<OCR_ENDPOINT>/ocr"
    text_score: 0.3

  # 글리프 깨짐 기반 auto-OCR 재트리거 임계값
  glyph_detection:
    table_cell_threshold: 1    # 셀 GLYPH 토큰 N개 이상이면 재OCR
    document_threshold: 10      # 문서 GLYPH 토큰 N개 초과면 OCR 경로 재시도

  # engine: "upstage" 일 때만 사용
  upstage:
    api_endpoint: "https://api.upstage.ai/v1/document-digitization"
    api_key: ""               # 비어있으면 UPSTAGE_API_KEY 환경변수에서 fallback
    model: "ocr"
    timeout: 60
    text_score: 0.5

layout:
  layout_model_type: "genos_layout"   # "genos_layout"(default) | "docling_layout"
  genos_layout:
    # <LAYOUT_SERVING_ID>: Genos에 등록한 layout 모델서빙 ID로 변경 필요
    # api_key는 k8s 내부 통신 기반 모델 호출 시 불필요
    endpoint: "http://llmops-gateway-api-service:8080/rep/serving/<LAYOUT_SERVING_ID>/v1/chat/completions"
    api_key: ""
    page_batch_size: 128
    max_completion_tokens: 16384
    model: "dots-mocr"          # 서빙 모델명
    timeout: 3600               # VLM 요청 타임아웃(초)
    retry_count: 2              # 비정상 VLM 응답 재시도 횟수
    temperature: 0.1            # 생성 temperature
    top_p: 0.9                  # 생성 top_p (0<top_p<=1)
    repetition_penalty: 1.15    # >1.0, 토큰 반복(degeneration) 억제 (아래 가이드 참조)

# PDF 파이프라인 (docling PDF 파싱 성능/품질 노브)
pdf_pipeline:
  num_threads: 8                     # accelerator 스레드 수
  device: "auto"                     # "auto"(default) | "cpu" | "cuda" | "mps"
  images_scale: 2                    # 페이지/그림 이미지 렌더 배율
  generate_page_images: true
  generate_picture_images: true
  table_structure_mode: "accurate"   # "accurate"(default) | "fast"

table_image:
  enable: false                      # true 면 모든 표를 이미지로 저장 + media_files 기록

output:
  table_format: "html"              # 청크 text 내 docling 표 직렬화 형식. "html"(default) | "markdown"
  compact_tables: true              # markdown 표 컬럼 정렬 패딩 제거(대형 표 축소). html 포맷엔 무관

# 청킹(GenosSmartChunker) 설정
chunking:
  # 청크 최대 크기. char 모드면 "최대 문자 수", huggingface 모드면 "최대 토큰 수".
  # 우선순위: 호출 kwargs 의 chunk_size > 아래 chunk_size. 0 = 크기 기반 분할 안 함.
  # 0 초과 시 최소 1024 로 보정된다.
  chunk_size: 10000
  # 청킹 모드. split_only(기본) = 구조 기반 청크 유지, chunk_size 초과 청크만 분할(작은 청크 병합 안 함).
  #            resize_all = 모든 청크를 chunk_size 에 맞게 병합/분할. 우선순위: kwargs.chunk_mode > 아래.
  chunk_mode: split_only
  # 토큰 수 계산 방식. "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
  tokenizer_type: "char"
  # 청킹용 토크나이저(huggingface 모드에서만 사용). tokenizer_path 존재 시 그 경로, 없으면 tokenizer_id(HF) 폴백
  tokenizer_path: "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
  tokenizer_id: "sentence-transformers/all-MiniLM-L6-v2"

# enrichment — {이름: {옵션}} 형식의 list.
# 비활성화: ① 항목 삭제  ② 항목 주석 처리  ③ enable: false
# 모든 url 의 <ENRICHMENT_SERVING_ID> 는 Genos에 등록한 모델서빙 ID로 변경 필요.
# api_key 는 k8s 내부 통신 기반 모델 호출 시 불필요.
enrichment:
  - toc:
      enable: true
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      temperature: 0.0
      top_p: 0.00001
      seed: 33
      max_tokens: 10000
      precheck:
        enabled: false
        max_context_tokens: 128000
        completion_reserved_tokens: 12000
      system_prompt_file: prompt_toc_default_system.md
      user_prompt_file: prompt_toc_default_user.md   # 파일 안에서 {{raw_text}} 치환

  - metadata:
      enable: true
      # 커스텀 신호(system_prompt(_file)/user_prompt(_file)/output_fields/parser) 중 하나라도
      # 지정되면 facade custom metadata enricher를 사용하고, 아무 신호도 없으면
      # docling 내장 metadata 추출 경로를 사용한다.
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      max_tokens: 10000
      temperature: 0.0
      timeout: 3600
      pages: [1,4]                          # null = 첫 4페이지
      output_fields: [created_date, authors] # 프롬프트의 JSON 키와 일치해야 함
      parser:
        type: json                          # json(default) | python
      system_prompt_file: prompt_metadata_default_system.md
      user_prompt_file: prompt_metadata_default_user.md   # 파일 안에서 {{raw_text}} 치환
      precheck:
        enabled: false
        max_context_tokens: 128000
        completion_reserved_tokens: 12000

  # doc_summary / image_description / table_description / custom_fields 는 facade 후처리 enricher.
  # 필요 시 enable: true 로 활성화.
  - doc_summary:            # 문서 본문요약 1회 → image/table description 공용 {{doc_summary}}
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      prompt_file: prompt_doc_summary.md   # 파일 안에서 {{full_text}} 치환
      max_chars: 6000

  - image_description:
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<IMAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      concurrency: 16        # 이미지 설명 요청 병렬 수
      before_items: 3
      after_items: 2
      max_context_chars: 1500
      # 파일 안에서 {{before_context}} / {{caption}} / {{after_context}} / {{doc_summary}} 치환
      prompt_template_file: prompt_image_description_default.md
      # 차트 처리
      chart:
        enable: false          # true 면 차트 처리 수행(아니면 일반 image description)
        detection: auto         # auto=docling 자동판별(차트만 차트 프롬프트) | all=모든 이미지를 차트로
        chart_prompt_file: prompt_chart_description_default.md

  - table_description:      # 표 요약(+선택적 refine 구조 재구성). 표 영역을 crop 해 VLM 에 보냄 → 이미지 서빙.
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<IMAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      concurrency: 8         # 표 설명 요청 병렬 수
      before_items: 3
      after_items: 2
      max_context_chars: 1500
      prompt_template_file: prompt_table_description_default.md   # 요약 전용 프롬프트
      refine:
        enable: false        # true 면 재구성 HTML 로 표 본체 교체
        prompt_file: prompt_table_refine_combined.md   # 재구성 HTML + 요약 통합 프롬프트
```

> 위 블록은 `resource/` 기본본 기준입니다. `resource_dev/` 본에는 사이트 운영을 위한 실제 endpoint/key 와 추가 `custom_fields` 항목 예시가 들어 있을 수 있으며, 배포 시 두 본의 placeholder/실값을 환경에 맞게 정리해야 합니다.

---

#### defaults 설정

`defaults` 섹션은 전처리기 공통 기본값을 담습니다. 현재는 로깅 레벨만 노출됩니다. `__init__` 이 이 값을 읽어 `self._log_level` 로 보관하고, 매 실행마다 `setup_logging()` 에 적용합니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `log_level` | `4` (INFO) | 로깅 레벨. `5`=DEBUG, `4`=INFO, `3`=WARNING, `2`=ERROR, `1`=CRITICAL, `0`=NOLOG(전체 비활성화). 누락/오류 시 4 폴백 |

> 실행 시 `params` 로 `log_level` 을 전달하면 그 값이 이 config 기본값보다 **우선**합니다. `params` 에 값이 없을 때만 `defaults.log_level` 이 적용됩니다. (`0`=NOLOG 도 정상 적용되도록 None 여부로 판별합니다.)

---

### 3.2 OCR 설정

`ocr` 섹션은 PDF 입력의 OCR 동작을 제어합니다. 코드에서는 `_build_ocr_options()` 와 `__init__` 이 이 값을 읽습니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `ocr_mode` | `"auto"` | OCR 수행 모드. `auto`/`force`/`disable`. **PDF 입력에만 적용** (DOCX/기타는 무관). 알 수 없는 값은 `auto` 폴백 |
| `engine` | `"paddle"` | OCR 엔진. `paddle` \| `upstage`. 알 수 없는 값은 `paddle` 폴백 |
| `table_cell_ocr_timeout` | `60` | 글리프 깨진 테이블 셀 재OCR HTTP timeout(초). 0 이하/오류 시 60 폴백 |
| `paddle.ocr_endpoint` | `http://<OCR_ENDPOINT>/ocr` | `engine: paddle` 일 때 PaddleOCR 서버 주소. **사이트별 교체 필요**. 구버전 `ocr.ocr_endpoint`(상위) 위치도 호환 인식 |
| `paddle.text_score` | `0.3` | PaddleOCR 텍스트 신뢰도 임계값. 오류 시 0.3 폴백 |
| `glyph_detection.table_cell_threshold` | `1` | 셀 텍스트의 GLYPH 토큰이 **N개 이상**(`>=`)이면 그 표의 셀들을 재OCR. 0 이하 시 1 폴백 |
| `glyph_detection.document_threshold` | `10` | 문서 텍스트 아이템의 GLYPH 토큰이 **N개 초과**(`>`)면 OCR 경로 재시도. 0 이하 시 10 폴백 |
| `upstage.api_endpoint` | `https://api.upstage.ai/v1/document-digitization` | `engine: upstage` 일 때 API 주소 |
| `upstage.api_key` | `""` | 비어있으면 `UPSTAGE_API_KEY` 환경변수에서 fallback |
| `upstage.model` | `"ocr"` | Upstage 모델명 |
| `upstage.timeout` | `60` | Upstage 요청 timeout(초). 0 이하/오류 시 60 폴백 |
| `upstage.text_score` | `0.5` | Upstage 텍스트 신뢰도 임계값. 오류 시 0.5 폴백 |

**`ocr_mode` 의미** (PDF 입력 한정):

| 모드 | 동작 |
|------|------|
| `auto` (기본) | 먼저 일반 추출 후, 문서 품질 검사(`check_document`)에 실패하거나 글리프 휴리스틱(`document_threshold` 초과)이 깨짐을 감지하면 **전체 페이지 OCR** 로 재변환 |
| `force` | 품질과 무관하게 처음부터 **전체 페이지 OCR** |
| `disable` | OCR 을 전혀 수행하지 않음 (테이블 셀 OCR 포함) |

> **선택적 테이블 셀 OCR**: `disable` 이 아니고 `ocr_endpoint` 가 설정되어 있으면, 전체 페이지 OCR 여부와 별개로 **GLYPH 가 감지된 테이블의 셀만** 선별 재OCR 합니다(`ocr_all_table_cells`). 전체 OCR 대비 처리 시간·정확도 모두 유리하며, 깨진 셀로 인한 과대 토큰 청크 발생을 방지합니다.

---

### 3.3 레이아웃 설정

`layout` 섹션은 레이아웃 분석 모델을 선택합니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `layout_model_type` | `"genos_layout"` | `genos_layout` \| `docling_layout` |
| `genos_layout.endpoint` | `.../serving/<LAYOUT_SERVING_ID>/...` | Genos layout 모델 서빙 엔드포인트. **사이트별 교체 필요** |
| `genos_layout.api_key` | `""` | k8s 내부 통신 기반 호출 시 불필요. 외부 호출 정책에 따라 필요할 수 있음 |
| `genos_layout.page_batch_size` | `128` | 레이아웃 모델 페이지 배치 크기(전역 `settings.perf.page_batch_size`). 0 이하/오류 시 128 폴백 |
| `genos_layout.max_completion_tokens` | `16384` | Layout LLM 최대 생성 토큰. 양의 정수, 유효하지 않거나 0 이하이면 16384 폴백 |
| `genos_layout.model` | `"dots-mocr"` | 서빙 모델명. 비어있으면 `dots-mocr` 폴백 |
| `genos_layout.timeout` | `3600` | VLM 요청 HTTP 타임아웃(초). 유효하지 않거나 0 이하이면 3600 폴백 |
| `genos_layout.retry_count` | `2` | 비정상(스키마 불일치 등) VLM 응답 재시도 횟수. 음수/오류 시 2 폴백 |
| `genos_layout.temperature` | `0.1` | 생성 샘플링 temperature. 음수/오류 시 0.1 폴백 |
| `genos_layout.top_p` | `0.9` | nucleus 샘플링 top_p. `0<top_p<=1` 아니면 0.9 폴백 |
| `genos_layout.repetition_penalty` | `1.15` | 토큰 반복(degeneration) 억제. `>0`, 유효하지 않으면 1.15 폴백. **자세한 사용은 아래 가이드 참조** |

- **`genos_layout`**: 외부 서빙형 GenOS 레이아웃 모델. 제목/본문/표/이미지 검출과 reading order 품질 개선을 기대할 수 있으나 별도 서빙 인프라가 필요합니다.
- **`docling_layout`**: Docling 기본 레이아웃 모델. 별도 서빙 인프라가 없는 환경에서 사용합니다.

#### `repetition_penalty` 사용 가이드

**무엇을 막는가** — DotsOCR(VLM)이 OCR 도중 특정 토큰·구절에 갇혀, 같은 내용을 `max_completion_tokens`(기본 16384)에 도달할 때까지 끝없이 반복 생성하는 현상(*degeneration*, 반복 붕괴)을 억제합니다. 증상은 DEBUG 로그(`defaults.log_level: 5`)의 `dotsocr raw response (page=N, ...)` 출력에서 한 `text` 필드가 `"휴식, 휴식, 휴식, ..."`처럼 무한 반복되는 형태로 나타납니다. 이 반복이 **유효한 JSON 문자열 안**에 들어가면 파싱이 성공해 `retry_count` 재시도로도 걸러지지 않으므로, 생성 단계에서 억제하는 `repetition_penalty`가 1차 방어선입니다.

**동작 원리** — 이미 생성된 토큰이 다시 나올 확률(logit)을 나눠 낮춥니다. `1.0`이면 페널티 없음(원본 모델 그대로), `1.0`보다 클수록 반복 억제가 강해집니다.

| 값 | 효과 | 사용 상황 |
|----|------|-----------|
| `1.0` | 억제 없음 | dots.ocr 원본 동작. 반복이 없더라도 비권장 |
| `1.05`~`1.1` | 약한 억제 | 가끔 반복이 보일 때 |
| **`1.15` (기본)** | 표준 억제 | 대부분의 문서에 권장 |
| `1.2`~`1.3` | 강한 억제 | 표/반복 어절이 많아 반복이 끈질긴 문서 |
| `>1.3` | 과도 | 정상 텍스트까지 변형·누락될 위험, 비권장 |

**튜닝 절차**
1. 반복이 보이는 문서로 DEBUG 로그(`defaults.log_level: 5`)를 켜고 실행합니다.
2. `dotsocr raw response`에 무한 반복이 남아 있으면 `repetition_penalty`를 `0.05`씩 상향합니다 (1.15 → 1.2 → 1.25).
3. 반대로 정상 텍스트에서 글자 누락·치환이 생기면 값을 하향합니다.

**주의**
- 한국어처럼 조사·어절이 자연스럽게 반복되는 문서는 과한 값(`>1.3`)에서 정상 반복까지 깎여 품질이 떨어질 수 있습니다.
- `temperature`(기본 0.1)를 약간 올리면(예: 0.2~0.3) 결정성이 낮아져 반복 탈출에 도움이 되기도 하지만 OCR 정확도와 trade-off이므로, `repetition_penalty`를 우선 조정하세요.
- 동일 항목이 여러 번 나오는 표는 과한 penalty가 실제 반복 데이터를 누락시킬 수 있으니 문서별로 결과를 검증하세요.

---

### 3.4 PDF 파이프라인 설정

`pdf_pipeline` 섹션은 Docling PDF 파싱의 성능/품질을 조절합니다. `PdfPipelineOptions` 와 `AcceleratorOptions` 에 주입됩니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `num_threads` | `8` | accelerator 스레드 수. 0 이하/오류 시 8 폴백 |
| `device` | `"auto"` | `auto` \| `cpu` \| `cuda` \| `mps`. 알 수 없는 값은 `auto` 폴백 |
| `images_scale` | `2` | 페이지/그림 이미지 렌더 배율(품질↔메모리). 0 이하/오류 시 2 폴백 |
| `generate_page_images` | `true` | 페이지 이미지 생성 여부. `false` 로 메모리 절감 |
| `generate_picture_images` | `true` | 그림 이미지 생성 여부 |
| `table_structure_mode` | `"accurate"` | `accurate`(품질) \| `fast`(속도). 알 수 없는 값은 `accurate` 폴백 |

> **참고**: 코드는 항상 `do_table_structure=True`, `do_cell_matching=True` 로 표 구조 분석을 켜며, 기본 파이프라인의 `do_ocr=False`(OCR 은 별도 OCR 컨버터에서 수행)로 둡니다. 이 두 값은 config 노브가 아닌 고정 동작입니다.

---

### 3.4.1 표 이미지 설정 (`table_image`)

복잡한 병합/레이아웃 표를 텍스트로 완전히 정규화하기 어려운 경우를 위해, 표를 그림(picture)과 동일하게 **이미지로도 함께 보관**하는 옵션입니다. 검색은 청크 텍스트 기반으로 하고, 답변 단계에서 표 이미지를 활용하는 "검색=청크 / 답변=표 이미지" 하이브리드 전략을 지원합니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `table_image.enable` | `false` | 모든 표 영역을 PNG 로 잘라 저장하고 `media_files` 에 `type='table_image'` 로 기록 |

`enable: true` 면 문서 내 모든 `TableItem` 을 페이지 이미지에서 bbox 로 잘라 artifacts 디렉터리에 `table_*.png` 로 저장하고, 각 표를 `media_files` 에 `{name, type:'table_image', ref}` 로 추가합니다(`ref` 는 `chunk_bboxes` 의 해당 표 엔트리와 동일 → 조인 가능). 청크 내 표 텍스트(MD/HTML)는 그대로 유지됩니다(하이브리드).

> **주의** — 표 이미지는 페이지 이미지를 잘라 만들므로, `enable: true` 이면 `pdf_pipeline.generate_page_images` 가 내부적으로 **강제 `true`** 로 보정됩니다(별도 설정 불필요).

---

### 3.4.1a PPT 페이지 설명 (`formats.ppt.page_description`)

PPT 는 슬라이드가 이미지 위주여서 텍스트 추출만으로는 의미가 소실되기 쉽습니다. 이 옵션은 각 페이지를 **이미지로 렌더링해 VLM(Vision LM)에 보내 페이지 설명을 생성**하고, 그 설명을 해당 페이지 텍스트로 문서에 주입합니다. 기존 `image_description`(페이지 내부 그림 오브젝트 단위)과 달리 **"페이지 자체"를 통째로** 설명합니다.

- **적용 대상**: **`.pptx`** (docling 경로)만 지원합니다. 레거시 **`.ppt`** 는 langchain 경로로 처리되어 페이지 이미지가 없으므로 페이지 설명이 적용되지 않습니다.
- **청킹**: `.pptx` 는 **1 page = 1 chunk**(페이지의 native text + 페이지 설명이 동일 청크). `chunking.chunk_size` 가 지정되면 연속 페이지를 그 크기까지 결합합니다.
- **TOC**: 페이지 단위라 목차 계층이 무의미 → **TOC enrichment 를 자동 skip**합니다.
- `enable: true` 면 `pdf_pipeline.generate_page_images` 가 내부적으로 **강제 `true`** 로 보정됩니다.
- 페이지의 native text 는 프롬프트의 **`{{page_text}}`** 변수로 함께 전달됩니다.

```yaml
formats:
  ppt:
    page_description:
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<PAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      timeout: 360
      concurrency: 16
      prompt_template_file: prompt_page_image_description_default.md
```

| 키 | 의미 | 기본값 |
|----|------|--------|
| `enable` | PPT 페이지 설명 활성화(`.pptx` 만) | `false` |
| `url` / `api_key` / `model` | VLM(chat/completions) 서빙 endpoint / 키 / 모델명 | `<PAGE_DESCRIPTION_SERVING_ID>` / `""` / `model` |
| `timeout` | VLM 요청 타임아웃(초) | `360` |
| `concurrency` | 페이지 설명 병렬 요청 수 | `16` |
| `images_scale` | 페이지 렌더 해상도 배율(클수록 고해상도/느림) | `2.0` |
| `max_image_side` | 전송 전 이미지 최대 변(px)으로 다운스케일. `0`=원본 | `0` |
| `max_tokens` | VLM 출력 토큰 상한(생성 시간↓). `0`=상한 없음 | `0` |
| `params` | 추가 VLM 파라미터(dict). 서버가 `max_completion_tokens` 등 다른 키를 쓸 때 사용 | `{}` |
| `prompt_template_file` | 프롬프트 `.md` 경로(권장). `{{page_text}}` 치환 | — |

> **주의** — 페이지 설명은 **파싱 단계 전용** 기능입니다. Chunk API(`/chunker`)는 파서 출력(JSON)만 받으며 렌더된 페이지 이미지가 없어 페이지 설명을 수행하지 않습니다.

---

### 3.4.2 xlsx(엑셀) 직접 처리 (`formats.xlsx`)

`.xlsx`/`.xlsm`(및 `.csv`)는 **PDF 변환 없이 직접 처리**합니다(이슈 #288). `formats.xlsx.processing_mode` 로 방식을 선택합니다.

| 키 | 값 | 기본 | 설명 |
|----|----|------|------|
| `formats.xlsx.processing_mode` | `docling` \| `tabular` | `docling` | 처리 방식 선택 |
| `formats.xlsx.tabular.header_row` | int | `0` | (tabular) 헤더 행. `0`=자동판정, `>0`=단일 헤더 강제 |
| `formats.xlsx.tabular.multi_table` | bool | `false` | (tabular) 1시트 복수표(빈 행 분리)를 표별로 분리 |

**docling 모드 (기본)** — MsExcel 백엔드로 `DoclingDocument` 를 만들어 기존 청킹/벡터 파이프라인으로 처리합니다(시트=1페이지).
- **표마다 별도 청크**로 분리하고, 표가 `chunking.chunk_size` 를 초과하면 **행(row) 단위로 분할**하며 분할된 각 청크에 **헤더 행을 반복 포함**합니다.
- 각 청크 텍스트 앞에 **`시트명: <시트명>`** 접두를 붙입니다.

**tabular 모드** — 데이터 **행마다 1벡터**를 만들고 각 컬럼 값을 **최상단 스칼라 property** 로 부여해 벡터 DB 컬럼 필터를 지원합니다.
- 병합셀 openpyxl unmerge + forward-fill, **멀티헤더 자동 판정**(전열 병합 제목행은 컨텍스트, 부분 병합 계층 헤더는 `상위_하위` flatten).
- 컬럼 키는 ASCII 헤더는 그대로, 한글 등 비-ASCII 는 **`field_<sha256[:8]>`** alias(파일 간 안정) + 원본명은 **`column_map`**(JSON) 보존.
- `multi_table: true` 면 복수 표를 표별로 분리해 헤더 재판정.

```yaml
formats:
  xlsx:
    processing_mode: "docling"    # 표 단위 청크 + 행 분할(헤더 반복) + 시트명 접두
    # processing_mode: "tabular"  # 행=벡터 + 컬럼 필터 메타(field_/column_map)
    tabular:
      header_row: 0
      multi_table: false
```

> **참고** — 기존에도 convert 는 xlsx 를 docling MsExcel 백엔드로 처리했으나, 위 설정으로 표 단위 청킹·행 분할·시트명 접두가 명시적으로 적용되고 tabular 모드를 선택할 수 있습니다. `.xls`/`.xlsb`(구형/바이너리)는 openpyxl/docling 미지원이라 PDF 변환 경로로 처리됩니다.

---

### 3.4.3 표 텍스트 형식 (`output.table_format`)

청크 `text` 안에 표(TableItem)를 직렬화하는 형식을 선택합니다. docling 모드의 모든 표(PDF/DOCX/HWP/xlsx-docling)에 적용됩니다.

| 키 | 값 | 기본 | 설명 |
|----|----|------|------|
| `output.table_format` | `html` \| `markdown` | `html` | `html`=`<table>…`, `markdown`=`\| c1 \| c2 \|` 파이프 표 |
| `output.compact_tables` | `true` \| `false` | `true` | markdown 표 컬럼 정렬 패딩(공백) 제거 → 대형 표 청크 축소. `html`/tabular 에는 무관 |

- `markdown` 이면 표를 `export_to_markdown` 으로 직렬화하고, xlsx 의 oversized 표 **행 분할 청크**도 markdown 표 행(헤더 반복 + `\| --- \|` 구분선)으로 렌더합니다.
- `compact_tables: true`(기본) 면 whole-table 청크의 markdown 을 **패딩 없는 compact 형식**(`\| c1 \| c2 \|`, 구분선 `\| - \|`)으로 직렬화합니다. 컬럼 정렬용 공백을 없애 대형 표 청크 크기가 크게 줄고, 행 분할 청크(이미 compact)와 형식이 일치합니다. `false` 면 기존 정렬 패딩 형식으로 출력합니다.
- tabular 모드(행=벡터)는 이 옵션들과 무관합니다(자체 파이프 표현 사용).
- 런타임 kwarg 로 `table_format`(또는 레거시 `export_to_html` 1/0)·`compact_tables` 를 주면 config 보다 우선합니다.

```yaml
output:
  table_format: "html"    # 또는 "markdown"
  compact_tables: true     # markdown 표 패딩 제거(기본 true)
```

---

### 3.5 Enrichment 설정

`enrichment` 는 **`{이름: {옵션}}` 형식의 list** 입니다. 항목은 enricher 종류를 나타내며, 같은 종류가 여러 번 올 수 있습니다(`custom_fields`). 파싱은 `EnrichmentConfig.from_raw()` 가 담당합니다.

**비활성화 3가지 방법**:

```
① 항목을 list 에서 삭제
② 항목 전체를 주석 처리
③ enable: false 추가 (설정은 보존, 동작만 끔)  ← 권장
```

> **Format 호환성**: 과거 dict 형식(`enrichment: {do_toc: true, toc: {...}, ...}`, Format A)도 `EnrichmentConfig` 가 여전히 수용하지만, 항목별 enable/url/key 관리가 명확한 **list 형식(Format B)이 권장**입니다.

지원하는 enricher 6종:

| 항목 | 역할 | 처리 주체 |
|------|------|-----------|
| `toc` | 계층적 목차(TOC) 자동 생성 | docling enrichment (`DataEnrichmentOptions`) |
| `metadata` | 작성일·작성자 등 메타데이터 추출 | 커스텀 신호(`system_prompt`/`user_prompt`/`*_file`/`output_fields`/`parser`) 중 하나라도 지정 시 facade custom metadata enricher, 아무 신호 없으면 docling 내장 metadata 경로 |
| `doc_summary` | 문서 본문요약 1회 생성(공용 `{{doc_summary}}` 컨텍스트) | facade 후처리 enricher (`DocSummaryEnricher`) |
| `image_description` | 이미지 설명 생성 | facade 후처리 enricher (`ImageDescriptionEnricher`) |
| `table_description` | 표 요약 + 선택적 구조 재구성(refine) | facade 후처리 enricher (`TableDescriptionEnricher`) |
| `custom_fields` | 사용자 정의 필드 추출 | facade 후처리 enricher (`CustomFieldsEnricher`) |

**공통 항목별 옵션**: 모든 enricher 는 `enable`, `url`, `api_key`, `model` 을 항목별로 가집니다. `url` 의 `<*_SERVING_ID>` 는 사이트별 교체가 필요하고, `api_key` 는 k8s 내부 통신 시 불필요합니다.

#### toc

목차를 LLM 으로 생성합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `temperature` / `top_p` / `seed` | `0.0` / `0.00001` / `33` | 생성 파라미터 |
| `max_tokens` | `10000` | 생성 최대 토큰 |
| `precheck.enabled` | `false` | 컨텍스트 길이 사전 검사 활성화 |
| `precheck.max_context_tokens` | `128000` | 모델 컨텍스트 상한 |
| `precheck.completion_reserved_tokens` | `12000` | 응답용 예약 토큰 |
| `system_prompt_file` / `user_prompt_file` | `.md` 파일 | 프롬프트 파일 경로(권장, config 디렉토리 기준). `user_prompt` 의 `{{raw_text}}` 치환 |
| `system_prompt` / `user_prompt` | (YAML 본문) | inline 프롬프트(`*_file` 미지정 시 fallback). 둘 다 비어있으면 docling 내장 기본 프롬프트 사용 |
| `split.enabled` | `false` | 긴 문서 **분할(Split) TOC 추출**(carry-over refine) 수행 여부(아래 참고) |
| `split.pages_per_chunk` / `split.page_overlap` | `100` / `1` | 청크당 페이지 수 / 청크 경계 중복 페이지 수 |
| `split.carryover_max_tokens` | `1500` | 다음 청크에 주입할 누적 목차(outline) 토큰 상한 |
| `repetition_penalty` | — | 토큰 반복(degeneration) 억제(>1.0). 게이트웨이/vLLM 지원 시에만, 미설정 시 미전송 |
| `thinking` | `off` | 추론(thinking) 모드. `off`(기본, 차단 토큰 전송) \| `on` \| `auto`(미전송, 모델 자동 판단) |
| `thinking_dialect` | `standard` | 추론 토글 키 방언. `standard`(`enable_thinking`) \| `hcx`(`force_reasoning`/`skip_reasoning`) |

##### thinking(추론) 모드

`thinking` / `thinking_dialect` 는 **toc·metadata 양쪽 enricher에 동일하게 적용**됩니다. 모델별로 추론 토글 키 이름이 달라 `thinking_dialect` 로 흡수합니다.

- `off`(기본) → `chat_template_kwargs` 에 `{"enable_thinking": false}`(hcx 면 `{"skip_reasoning": true}`) 전송 — 추론 비활성화, 빠른 응답.
- `on` → `{"enable_thinking": true}`(hcx 면 `{"force_reasoning": true}`) 전송 — 추론 활성화.
- `auto` → `chat_template_kwargs` 미전송 — 모델 기본 동작에 맡김(기존 동작 보존).
- `thinking_dialect`: HyperCLOVAX-SEED-Think 등 **hcx 계열 서빙은 `hcx`**, Qwen3/GLM/DeepSeek 등 그 외는 `standard`(기본).
- 추론이 켜진 응답에 섞인 `<think>...</think>` 블록은 자동 제거되어 본문만 저장됩니다.

##### 분할(Split) TOC 추출 — 긴 문서 대응

기본 TOC 추출은 **문서 전체를 한 번에** LLM에 보냅니다. 문서가 길면 서빙 모델의 `max-model-len` 을 초과해
토큰 에러가 날 수 있습니다. `split.enabled: true` 면 문서를 **페이지 단위 청크**로 나눠 순차 추출하고,
앞 청크까지 누적된 목차를 다음 청크 프롬프트에 컨텍스트로 주입(**carry-over refine**)해 계층/번호 일관성을
유지합니다.

```yaml
- toc:
    enable: true
    # ... url/model/precheck/프롬프트 파일 등 기존 설정
    system_prompt_file: prompt_toc_default_system.md
    user_prompt_file: prompt_toc_default_user.md
    split:
      enabled: false            # true 면 길이와 무관하게 항상 분할 추출
      pages_per_chunk: 100      # 청크당 페이지 수
      page_overlap: 1           # 청크 경계 중복 페이지 수(경계 누락 완화용)
      carryover_max_tokens: 1500
    # repetition_penalty: 1.1   # 반복 억제가 필요할 때만 주석 해제
```

- **OFF(기본)**: 단일 호출. 동작 변화 없음(컨텍스트 초과 시 기존처럼 에러).
- **ON**: 페이지 N개씩 청크화 → 첫 청크는 설정 프롬프트로, 이후 청크는 설정 프롬프트 앞에 누적 목차를 덧붙여
  순차 추출 → 매 스텝 병합(경계 중복 제거·번호 재부여). `<toc>` 블록이 없거나(분석문/절단) 컨텍스트를 초과하는
  청크는 건너뛰어 부분 결과를 보존합니다.
- **통합 프롬프트**: `prompt_toc_default_user.md` 가 `{{prior_toc}}` 자리표시자 + 작업 모드(Operating Mode)로
  첫 추출/이어쓰기를 모두 처리합니다(`<previous_outline>` 비면 전체 추출, 있으면 새 항목만 이어쓰기).
- **주의**: 컨텍스트에 들어가는 문서는 OFF 유지 권장. `page_overlap>0` 은 경계 누락을 줄이나 중복이 일부 남을 수
  있어(중복이 문제면 `0` 권장), 매우 긴 청크는 토큰 소진 시 스킵될 수 있습니다.

#### metadata

문서에서 작성일·작성자 등을 추출합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `max_tokens` / `temperature` / `timeout` | `10000` / `0.0` / `3600` | 생성/요청 파라미터 |
| `pages` | `[1,4]` | 추출 대상 페이지 범위. `null`(또는 빈 값)이면 첫 4페이지 |
| `output_fields` | `[created_date, authors]` | 추출 키 목록. **프롬프트의 JSON 키와 일치해야 함** |
| `parser.type` | `json` | `json`(default) \| `python`. 응답 파싱 방식 |
| `precheck.*` | toc 와 동일 | 컨텍스트 사전 검사 |
| `system_prompt_file` / `user_prompt_file` | `.md` 파일 | 프롬프트 파일 경로(권장). `system_prompt_file` 생략 시 built-in default system prompt |
| `system_prompt` / `user_prompt` | (YAML 본문) | inline 프롬프트(`*_file` 미지정 시 fallback). 커스텀 신호(이들/`*_file`/`output_fields`/`parser`) 중 하나라도 있으면 facade custom enricher, 모두 없으면 docling 내장 |
| `thinking` / `thinking_dialect` | `off` / `standard` | 추론(thinking) 모드 / 방언. toc 와 동일(위 "thinking(추론) 모드" 참고) |

**`field_transforms` — 추출 키를 벡터 메타 필드로 매핑/변환**

LLM 이 `output_fields` 로 뽑아낸 키 이름은 프롬프트가 정하기 나름이라(`created_date`, `작성일`, `doc_date` …) 검색에 쓰이는 최종 벡터 메타 필드 이름과 다를 수 있고, 값도 문자열 그대로라 타입 변환이 필요할 때가 있습니다. `field_transforms` 는 **"추출 결과의 어떤 키를 → 어떤 벡터 필드에 → 어떤 변환을 거쳐 넣을지"** 를 YAML 로 선언하는 다리 역할입니다. 코드 수정 없이 설정만으로 매핑을 바꿀 수 있어, 프롬프트를 고쳐 추출 키 이름이 달라져도 `source` 한 줄만 수정하면 됩니다.

**설정이 비어있으면 기본값**(`DEFAULT_METADATA_FIELD_TRANSFORMS`)이 적용되어 기존 `created_date` 동작을 그대로 재현합니다(하위 호환).

```yaml
metadata:
  field_transforms:
    - source: [created_date, 작성일]   # 추출 결과에서 찾을 후보 키(순서대로 탐색)
      target: created_date            # 값을 넣을 벡터 메타 필드명
      type: date_int                  # 값 변환기 (date_int = YYYYMMDD 정수)
      fallback: doc_text_scan         # 추출이 비면 본문에서 보조 추출
```

각 항목(spec)의 필드:

| 필드 | 필수 | 의미 |
|------|------|------|
| `source` | ✅ | 추출 결과에서 찾을 후보 키. 문자열 1개 또는 목록(목록이면 순서대로 탐색해 **첫 비어있지 않은 값** 사용) |
| `target` | 선택 | 값을 넣을 벡터 메타 필드명. 생략 시 `source` 의 첫 키를 사용 |
| `type` | 선택 | 값 변환기. 현재 지원: `date_int` (날짜 텍스트/정수 → `YYYYMMDD` 정수, 예 `2025-01-15`→`20250115`, `2025-01`→`20250101`, 실패 시 `0`). 생략 시 값을 그대로 사용 |
| `fallback` | 선택 | 추출값이 비어있을 때 쓸 보조 추출. 현재 지원: `doc_text_scan` (본문에서 `기준일`/`작성일`/`최초 작성일`/`보고자료` 키워드 주변 날짜를 휴리스틱 탐색) |

**동작 흐름** (각 spec 마다):

1. `source` 후보 키를 순서대로 보며 첫 비어있지 않은 값을 고릅니다.
2. `type` 가 있으면 그 변환기를 적용합니다 (예: `date_int` → 정수).
3. 값이 비어있거나(또는 `0`) `fallback` 이 지정돼 있으면 본문에서 보조 추출합니다.
4. 결과를 `target` 필드에 넣고, 이때 사용한 `source`/`target` 키는 아래 passthrough 대상에서 제외합니다.
5. 변환 대상이 **아닌** 나머지 추출 키는 그대로 벡터 메타에 통과(passthrough)되며, 중첩 객체는 JSON 문자열로 직렬화됩니다.

**입력 → 출력 예시** (위 기본 설정 기준)

```text
추출 결과(merged metadata):
  { "created_date": "2025-01-15",
    "authors": [{"name": "홍길동"}],
    "department": "IT" }

→ 벡터 메타:
  created_date : 20250115              # date_int 로 정수 변환 (target)
  authors      : "[{\"name\": \"홍길동\"}]"  # passthrough(중첩 객체 → JSON 문자열)
  department   : "IT"                  # passthrough
```

- **키 이름이 달라진 경우**: 프롬프트를 고쳐 추출 키가 `doc_date` 가 되면 `source: [doc_date]` 로만 바꾸면 동일하게 `created_date` 가 채워집니다.
- **fallback 동작**: 추출이 비었을 때 본문에 `"보고자료 2024-01-15 기준"` 이 있으면 `created_date: 20240115` 로 보강됩니다.

convert 의 기본 변환 대상은 `created_date`이며 `authors` 등 나머지는 passthrough 됩니다. 신규 변환기/보조추출은 `field_transforms.py` 의 `VALUE_TRANSFORMS`/`FALLBACK_STRATEGIES` 에 등록하면 설정에서 바로 사용할 수 있습니다.

#### doc_summary

문서 본문을 **요청당 1회** LLM 으로 요약해 공용 `{{doc_summary}}` 컨텍스트로 제공합니다. `image_description`·`table_description` 프롬프트가 이 값을 참조하므로, 과거 `image_description.doc_summary.*` 중첩 설정을 **독립 enricher 항목**으로 승격했습니다(요약 중복 계산 제거). 요약 결과는 출력 metadata 의 `doc_summary` 로도 노출됩니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `enable` | `false` | 본문요약 생성 여부. `true` 또는 런타임 `doc_summary=1` 로 활성화 |
| `prompt_file` | `prompt_doc_summary.md` | 요약 프롬프트 `.md`. 본문 `{{full_text}}` 치환 |
| `max_chars` | `6000` | 요약 입력 본문 최대 길이(문자) |

```yaml
- doc_summary:
    enable: false          # true 또는 런타임 doc_summary=1 로 활성화(image/table 이 공유)
    url: "http://.../serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
    model: "model"
    prompt_file: prompt_doc_summary.md
    max_chars: 6000
```

> `doc_summary` 는 별도 LLM 호출을 유발하므로 기본 off 입니다. `image_description`/`table_description` 이 켜져 있어도 `doc_summary` 가 off 면 `{{doc_summary}}` 는 빈 문자열로 치환됩니다.

#### image_description

이미지 주변 문맥을 참고해 LLM 으로 이미지 설명을 생성합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `concurrency` | `16` | 이미지 설명 요청 병렬 수 |
| `before_items` / `after_items` | `3` / `2` | 앞/뒤 문맥으로 포함할 아이템 수 |
| `max_context_chars` | `1500` | 문맥 최대 문자 수 |
| `prompt_template_file` | `.md` 파일 | 프롬프트 템플릿 파일 경로(권장). `{{before_context}}`/`{{caption}}`/`{{after_context}}`/`{{doc_summary}}` 치환 |
| `prompt_template` | (YAML 본문) | inline 프롬프트(`*_file` 미지정 시 fallback) |
| `chart.enable` | `false` | 차트 처리 활성화(false 면 일반 image description 만) |
| `chart.detection` | `auto` | `auto`=docling 자동판별(차트로 분류된 이미지만 차트 프롬프트) / `all`=모든 이미지를 차트로 처리 |
| `chart.chart_prompt_file` | `prompt_chart_description_default.md` | 차트 전용 프롬프트 `.md` |

> 프롬프트의 `{{doc_summary}}` 컨텍스트는 이제 **독립 `doc_summary` enricher**(위 참조)가 채웁니다(과거 `image_description.doc_summary.*` 중첩 설정은 표준 `- doc_summary:` 항목으로 이동). `chart.enable: true` 면 변환 단계에서 docling 그림 분류가 자동 활성화됩니다. `chart` 는 별도 LLM 호출을 유발하므로 기본 off 입니다.

**런타임 kwargs 오버라이드 (이미지·차트 description)** — 호출 `params` 로 0/1 플래그 전달 시 config 기본값을 덮어씁니다.

| kwargs | 대응 config | 의미 |
|--------|-------------|------|
| `img_desc` | `image_description.enable` | 이미지 description 사용유무 |
| `chart_desc` | `image_description.chart.enable` | 차트 description 사용유무 (별칭 `chart_convert`) |
| `chart_detection` | `image_description.chart.detection` | `1`=auto / `0`=all |
| `doc_summary` | `doc_summary.enable` | 문서 본문요약 사용유무(독립 enricher) |
| `table_desc` | `table_description.enable` | 표 요약 description 사용유무 |
| `table_refine` | `table_description.refine.enable` | 표 구조 재구성(refine) 사용유무 |

> `chart_detection=1`(auto) 은 config `chart.enable: true` 로 그림 분류가 켜져 있어야 하며, 꺼진 경우 `all` 로 강등됩니다(경고).

#### table_description

표(TableItem)마다 앞뒤 문맥·캡션·섹션헤더(+공용 `{{doc_summary}}`)를 참고해 LLM 으로 **표 요약**을 생성하고, 청크의 표 텍스트 뒤에 `\n---\n[표 설명]\n<요약>` 형태로 병기합니다. `refine.enable: true` 면 같은 호출에서 표 구조를 **충실한 HTML 로 재구성**해 표 본체를 그 재구성본으로 교체합니다(요약도 함께 병기).

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `enable` | `false` | 표 요약 생성 여부. `true` 또는 런타임 `table_desc=1` 로 활성화 |
| `concurrency` | `8` | 표 설명 요청 병렬 수 |
| `before_items` / `after_items` | `3` / `2` | 앞/뒤 문맥으로 포함할 아이템 수 |
| `max_context_chars` | `1500` | 문맥 최대 문자 수 |
| `prompt_template_file` | `prompt_table_description_default.md` | 요약 전용 프롬프트 `.md`. `{{before_context}}`/`{{after_context}}`/`{{caption}}`/`{{section_header}}`/`{{doc_summary}}` 치환 |
| `refine.enable` | `false` | 표 구조 재구성(refine) 여부. `true` 또는 런타임 `table_refine=1` 로 활성화 |
| `refine.prompt_file` | `prompt_table_refine_combined.md` | 재구성 HTML + 요약 **통합** 프롬프트(refine 시 요약 프롬프트 대신 사용) |

```yaml
- table_description:
    enable: false          # true 또는 런타임 table_desc=1 로 활성화
    url: "http://.../serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
    model: "model"
    concurrency: 8
    before_items: 3
    after_items: 2
    max_context_chars: 1500
    prompt_template_file: prompt_table_description_default.md
    refine:
      enable: false        # true 또는 런타임 table_refine=1 로 재구성 HTML 로 표 본체 교체
      prompt_file: prompt_table_refine_combined.md
```

- **요약만(refine off)**: 원본 표(html/markdown) 뒤에 `[표 설명]` 요약을 병기합니다.
- **refine on**: 재구성 HTML 로 표 본체를 교체하고 출력 `table_format` 에 맞춰 변환합니다 — `markdown` 이면 `compact_tables` 설정을 반영한 compact 표로 냅니다. refine 표는 구조가 원본 grid 와 달라 **행 분할을 하지 않습니다**(요약도 1회만 포함).
- refine 통합 프롬프트는 `[[[TABLE_HTML]]]` / `[[[TABLE_SUMMARY]]]` 마커로 재구성 HTML 과 요약을 한 응답에 함께 출력하도록 강제합니다.
- `table_description` 은 별도 LLM 호출을 유발하므로 기본 off 입니다.

#### custom_fields (복수 허용)

사용자 정의 필드를 LLM 으로 추출합니다. **여러 항목을 동시에 둘 수 있으며**, 각 항목은 독립 LLM 호출 후 결과를 metadata 에 병합합니다. 설정 방식은 두 가지입니다.

```yaml
# (A) 인라인: 항목 안에 프롬프트(.md 파일 참조)/필드를 직접 작성
- custom_fields:
    enable: true
    url: "http://.../serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
    model: "model"
    pages: [1]
    output_fields: [authors]
    parser: { type: json }     # json | python
    system_prompt_file: prompt_custom_fields_authors_system.md  # 생략 시 built-in default
    user_prompt_file: prompt_custom_fields_authors_user.md      # 파일 안에서 {{raw_text}} 치환

# (B) config_file: 별도 yaml 파일로 분리 (그 yaml 안에서도 *_file 사용 가능)
- custom_fields:
    enable: true
    config_file: custom_field_authors.yaml   # resource_path 는 이 yaml 디렉토리로 자동 주입
```

> `custom_fields` 항목은 `enable: true` 이고 옵션이 비어있지 않을 때만 활성화됩니다. `resource_path` 를 지정하지 않으면 config 파일이 위치한 디렉토리가 자동 주입되어, `config_file` 및 `*_file` 상대 경로 해석에 사용됩니다. system 프롬프트(파일·inline 모두) 미지정 시 built-in default 가 사용됩니다.

#### 프롬프트 파일 분리 & 변수 치환

enrichment 프롬프트는 YAML 안에 inline 으로 박지 않고 **별도 `.md` 파일**로 분리합니다. 운영 시 프롬프트만 교체하기 쉽고, 두 전처리기(parser/intelligent/convert)가 동일 프롬프트를 공유할 때 파일 경로 한 줄만 같게 두면 됩니다.

**프롬프트 파일 지정**

| 키 | 대상 |
|----|------|
| `system_prompt_file` / `user_prompt_file` | toc / metadata / custom_fields |
| `prompt_template_file` | image_description / table_description(요약) |
| `prompt_file` | doc_summary / table_description `refine`(재구성+요약 통합) |

- **경로 규칙:** 상대경로는 **해당 config 파일이 위치한 디렉토리** 기준으로 해석합니다(파일명만 적으면 됨). 절대경로도 허용하지만, 상대경로가 base 디렉토리를 벗어나면(`../…`) 거부합니다.
- **우선순위:** `*_file` > inline(`system_prompt`/`user_prompt`/`prompt_template`) > built-in default. system prompt 는 미지정 시 enricher 별 built-in default 가 채워지므로, **`user_prompt_file` 만 지정**해도 동작합니다(system 은 도메인 안에서 거의 고정이고 자주 바뀌는 것은 user prompt 이기 때문).
- 출하 config 는 `prompt_toc_default_system.md` / `prompt_metadata_default_user.md` 등 중립적 파일명을 사용합니다.

**변수 치환(`{{var}}`)**

프롬프트 본문의 `{{변수}}` 는 런타임 값으로 치환됩니다. JSON 예시(`{"key": "value"}`)의 단일 중괄호와 충돌하지 않도록 **이중 중괄호** 로 통일했습니다.

- **escape:** 리터럴 중괄호가 필요하면 4중 중괄호 `{{{{ ... }}}}` 로 적습니다(→ `{{ ... }}` 로 출력). 치환된 값 안에 `{{...}}` 가 들어와도 다시 치환되지 않습니다(1-pass).
- **단일 중괄호 호환:** 과거 표기 `{raw_text}` 도 값이 주입된 변수에 한해 당분간 치환되지만 deprecation 경고가 남습니다. 신규 프롬프트는 `{{raw_text}}` 를 사용하세요.

**(1) reserved 변수 — 1차 변환 결과 `DoclingDocument` 에서 자동 추출**

문서 단위 (toc / metadata / custom_fields 프롬프트):

| 변수 | 의미 | 추출 소스 | 예시 값 |
|------|------|-----------|---------|
| `{{raw_text}}` | enricher 가 추출한 본문 텍스트(프롬프트의 핵심 입력) | metadata: `pages`(미지정 시 첫 4p)의 markdown — `<!-- image -->` 제거 + 맨 앞에 `filename: …` 주입 / custom_fields: 전체 plain text(또는 `pages` markdown) | `filename: 보고서.pdf\n\n# 1장 총칙 …` |
| `{{full_text}}` | 문서 전체 plain text | `document.export_to_text()` | `보고서\n1장 총칙 …` (전문) |
| `{{filename}}` | 원본 파일명 | `document.origin.filename` | `보고서.pdf` |
| `{{mimetype}}` | MIME 타입 | `document.origin.mimetype` | `application/pdf` |
| `{{page_count}}` | 총 페이지 수 | `document.num_pages()` | `12` |
| `{{table_count}}` | 표 개수 | `len(document.tables)` | `3` |
| `{{picture_count}}` | 이미지 개수 | `len(document.pictures)` | `5` |
| `{{section_headers}}` | 섹션 헤더·제목 목록(줄바꿈 구분) | `texts` 중 label ∈ {section_header, title} | `1장 총칙\n2장 정의 …` |

이미지·표 item 단위 (image_description / table_description 프롬프트 — 아이템마다 값이 달라짐):

| 변수 | 의미 | 추출 소스 |
|------|------|-----------|
| `{{before_context}}` | 이미지 앞 문맥 텍스트(`before_items` 개) | 이미지 직전 텍스트 아이템들 |
| `{{after_context}}` | 이미지 뒤 문맥 텍스트(`after_items` 개) | 이미지 직후 텍스트 아이템들 |
| `{{caption}}` | 이미지 캡션 | `PictureItem.caption_text(document)` |
| `{{section_header}}` | 이미지 직전 섹션 헤더 | 이미지 위쪽에서 가장 가까운 section_header/title |
| `{{doc_summary}}` | 문서 본문요약(image·table description 공용 컨텍스트). 독립 `doc_summary` enricher 가 `enable: true`(또는 `doc_summary=1`)일 때만 채워짐 | 문서 BODY 텍스트 LLM 1회 요약 |

> 값이 없는 reserved 변수는 **빈 문자열**로 치환됩니다. 카운트(`page_count` 등)는 정수지만 문자열로 렌더링됩니다. `raw_text`/`full_text` 는 토큰이 매우 클 수 있으니 보통 둘 중 하나만 사용합니다.

**(2) 사용자 정의 변수 (`variables:`)**

reserved 외에 운영자가 직접 변수를 추가할 수 있습니다. enricher 항목에 `variables:` 블록으로 이름·값을 선언하고 프롬프트에서 `{{이름}}` 으로 참조합니다. 회사명·도메인처럼 프롬프트에 고정 주입할 값에 유용합니다.

```yaml
# config (예: metadata 항목)
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    variables:
      company_name: "GenON"
      domain: "금융"
```

```text
<!-- prompt_metadata_default_user.md 본문 -->
당신은 {{company_name}} 의 {{domain}} 문서 분석가입니다. (총 {{page_count}}페이지)
아래 문서에서 작성일·작성자를 추출하세요.

문서:
---
{{raw_text}}
---
```

- `variables` 의 이름은 strict 검증에서 reserved 와 함께 **허용 목록**에 포함됩니다.
- 이름이 reserved 와 겹치면 `variables` 값이 **우선**합니다(override).

**(3) 미정의 변수 처리 (`template.mode`)**

| mode | 동작 |
|------|------|
| `strict` (기본) | 프롬프트에 (reserved ∪ user-defined)에 없는 `{{foo}}` 가 있으면 **config 로드 시점에 에러**로 즉시 실패(오타·누락 조기 발견) |
| `lenient` | 미정의 변수를 빈 문자열로 치환하고 경고 로그만 남김 |

```yaml
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    template:
      mode: strict   # strict(기본) | lenient
```

**(4) 워크플로우 예시 — 입력 문서 → 렌더링된 프롬프트**

`보고서.pdf`(12페이지, 표 3개)를 metadata enricher 로 처리:

```yaml
# config
- metadata:
    user_prompt_file: prompt_metadata_default_user.md
    variables: { company_name: "GenON" }
```
```text
# prompt_metadata_default_user.md 본문
{{company_name}} 문서 분석 — 파일 {{filename}} ({{page_count}}p, 표 {{table_count}}개)
---
{{raw_text}}
---
```
→ 런타임 치환 후 실제 LLM 에 전달되는 user 메시지:
```text
GenON 문서 분석 — 파일 보고서.pdf (12p, 표 3개)
---
filename: 보고서.pdf

# 1장 총칙 …
---
```

> TOC 프롬프트는 docling 레이어에서 처리되어 `{{raw_text}}` 만 지원하며, 위 reserved 카탈로그·`variables`·`template.mode` 는 facade enricher(metadata / custom_fields / image_description / table_description / doc_summary)에 적용됩니다.

---

### 3.6 사이트 적용 시 필수 수정 항목

배포 시 아래 placeholder 를 환경에 맞는 실제 값으로 교체합니다. **운영 IP/키를 직접 하드코딩하지 말고**, Genos 에 등록한 서빙 ID 기반 endpoint 를 사용합니다.

| placeholder | config 위치 | 교체 값 |
|-------------|-------------|---------|
| `<OCR_ENDPOINT>` | `ocr.paddle.ocr_endpoint` | PaddleOCR 서버 주소 (`engine: paddle` 인 경우만) |
| `<LAYOUT_SERVING_ID>` | `layout.genos_layout.endpoint` | Genos layout 모델 서빙 ID |
| `<ENRICHMENT_SERVING_ID>` | `enrichment[].toc.url`, `enrichment[].metadata.url`, `enrichment[].custom_fields.url` | enrichment LLM 서빙 ID |
| `<IMAGE_DESCRIPTION_SERVING_ID>` | `enrichment[].image_description.url` | 이미지 설명 LLM 서빙 ID |

> `api_key` 는 k8s 내부 통신 기반 모델 호출 시 비워둘 수 있습니다. Upstage OCR 의 `api_key` 는 비워두면 `UPSTAGE_API_KEY` 환경변수에서 읽습니다.

---

### 3.7 자주 쓰는 튜닝 시나리오

| 목표 | 변경 |
|------|------|
| **OCR 강제** (스캔 PDF 등) | `ocr.ocr_mode: "force"` |
| **OCR 완전 비활성** (텍스트 PDF 전용) | `ocr.ocr_mode: "disable"` |
| **OCR 엔진을 Upstage 로** | `ocr.engine: "upstage"` + `ocr.upstage.*` 설정 (`api_key` 비우면 `UPSTAGE_API_KEY` 사용) |
| **글리프 재OCR 민감도 조정** | `ocr.glyph_detection.table_cell_threshold` / `document_threshold` 값 조정 |
| **특정 enricher 끄기/켜기** | 해당 `enrichment[]` 항목의 `enable: true/false` |
| **TOC 만 쓰고 메타데이터는 끄기** | `toc.enable: true`, `metadata.enable: false` |
| **이미지 설명 켜기** | `image_description.enable: true` + `<IMAGE_DESCRIPTION_SERVING_ID>` 교체 |
| **표 분석 속도 우선** | `pdf_pipeline.table_structure_mode: "fast"` |
| **메모리 절감** | `pdf_pipeline.generate_page_images: false`, `images_scale: 1` |
| **레이아웃 인프라 없음** | `layout.layout_model_type: "docling_layout"` |
| **청크 크기 조정** | `chunking.chunk_size` (yaml) 또는 호출 kwargs `chunk_size`. 우선순위 kwargs > yaml > 0 |
| **청크 크기 계산 방식** | `chunking.tokenizer_type`: `char`(기본)=문자 수 기준 / `huggingface`=HF 토큰 수 기준 |

> **청크 크기**: convert 경로의 청크 크기는 `split_documents()` → `GenosSmartChunker(max_tokens=...)` 로 결정됩니다. 값은 호출 kwargs `chunk_size` 가 우선이고, 없으면 yaml `chunking.chunk_size`, 둘 다 없으면 `0`(크기 기반 분할 안 함)입니다. 크기 단위는 `chunking.tokenizer_type` 으로 정해집니다(`char`=문자 수, `huggingface`=HF 토큰 수). char 모드에서는 HF 토크나이저를 로드하지 않습니다.

---

## 4. 처리 동작 개요 (보조)

config 가 각 처리 단계에 어떻게 매핑되는지 요약합니다. 코드 상세는 [부록](#부록-코드-내부-상세) 참고.

### 4.1 아키텍처

```
사용자가 파일 업로드
        │
        ▼
 ┌──────────────────┐
 │ DocumentProcessor│  ◄── 메인 엔트리포인트 (__call__)
 │   (라우터 역할)    │      config 는 __init__ 에서 1회 로드
 └──────┬───────────┘
        │  확장자(ext)에 따라 분기
        │
        ├── .ppt ──────────────► LangChain 경로
        │                        (convert_to_pdf → PowerPointLoader → RecursiveSplitter)
        │
        └── 기타 (.pdf/.docx/.pptx 등) ──► Docling 경로
                                          │
                                          ├─ load_documents (layout: layout.*)
                                          │
                                          ├─ [PDF만] OCR 판단 (ocr.ocr_mode)
                                          │    ├─ force → 전체 OCR
                                          │    ├─ auto  → 품질/글리프 검사 후 필요 시 OCR
                                          │    └─ + 글리프 테이블 셀 선택 OCR
                                          │
                                          ├─ enrichment (enrichment.toc / metadata)
                                          ├─ enrich_image_descriptions (enrichment.image_description)
                                          ├─ enrich_custom_fields (enrichment.custom_fields)
                                          │
                                          ├─ GenosSmartChunker (pdf_pipeline 산출물 청킹)
                                          │
                                          └─ compose_vectors → List[GenOSVectorMeta]
                                               (+ created_date, authors, title, HEADER:)
```

### 4.2 config → 단계 매핑

| config 섹션 | 적용 단계 | 코드 진입점 |
|-------------|-----------|-------------|
| `pdf_pipeline` | PDF 파싱 (스레드/디바이스/이미지/표) | `DocumentProcessor.__init__` → `PdfPipelineOptions` |
| `layout` | 레이아웃 분석 모델 | `__init__` → `layout_options` |
| `ocr` | PDF OCR 모드/엔진/임계값 | `__init__`, `__call__`(ocr_mode 분기), `ocr_all_table_cells` |
| `enrichment` | TOC/메타데이터/이미지/커스텀 보강 | `EnrichmentConfig.from_raw` → `enrichment()`, `enrich_*()` |

### 4.3 GenosSmartChunker 전략

`GenosSmartChunker` 는 **섹션 헤더 기반 의미적 분할 + 토큰 수 기반 크기 제어** 를 결합한 독자 청커입니다. 처리는 5단계로 구성됩니다.

| 단계 | 동작 |
|------|------|
| 1. 섹션 헤더 분할 | 제목/장·절·조 헤더를 만날 때마다 새 섹션 시작 |
| 2. heading 텍스트 생성 | 각 섹션 텍스트 앞에 계층적 헤딩 경로 prepend |
| 2.5. 초과 섹션 분할 | `max_tokens` 초과 섹션을 균등 토큰 분할(bisect). 캡션은 부모에, 표 안 그림(IoS>0.5)은 표에 병합 후 분할 |
| 3. 단독 타이틀 병합 | 30자 이하 단독 헤더는 다음 섹션으로 병합(헤더 레벨이 더 낮지 않을 때) |
| 4. 토큰 기준 최종 병합 | 토큰 여유가 있고 헤더 레벨이 상위가 아니면 인접 섹션 병합 |

> 청크 텍스트 선두에는 짧은 헤더 경로가 `HEADER: ...` 접두어로 부착되어 벡터 검색 시 문맥 파악을 돕습니다. 이 청커는 현재 facade 에 한시적으로 구현되어 있으며, 추후 컨테이너 이미지(라이브러리)에서 제공될 예정입니다.

<a id="41-convert"></a>

### 4.4 convert_to_pdf

`convert_to_pdf(file_path, use_pdf_sdk=True)` 는 다양한 포맷을 PDF 로 변환하는 한 줄 wrapper 로, 실제 로직은 `genon.preprocessor.converters.hwp_to_pdf.convert_hwp_to_pdf()` 에 위임됩니다.

| 입력 | `use_pdf_sdk=True` | `use_pdf_sdk=False` |
|------|--------------------|--------------------|
| `.hwp` / `.hwpx` | `pdf_sdk → libreoffice → rhwp` | `libreoffice → rhwp` |
| 그 외 (`.docx`/`.pptx` 등) | `pdf_sdk → libreoffice` | `libreoffice` |

앞 backend 실패 시 다음으로 자동 fallback 하며, `rhwp` 는 HWP/HWPX 전용·최후순위입니다. 시그니처는 attachment/intelligent processor 와 동일합니다. backend 별 상세 흐름은 [attachment_processor.md#41-convert_to_pdf](attachment_processor.md#41-convert_to_pdf) 참고.

> convert_processor 경로에서는 PDF 변환을 `.docx`/`.pptx`(이미지/미리보기용)와 `.ppt`(LangChain 경로 진입 시)에 사용합니다.

---

## 5. 출력 데이터 구조

각 청크는 하나의 `GenOSVectorMeta` 로 변환되어 벡터 DB 에 저장됩니다.

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'          # 정의되지 않은 추가 필드(custom_fields 등)도 허용

    text: str = None             # 청크 텍스트 (선두에 "HEADER: ..." 접두어 가능)
    n_char: int = None
    n_word: int = None
    n_line: int = None
    i_page: int = None           # 청크 시작 페이지
    e_page: int = None           # 청크 끝 페이지
    i_chunk_on_page: int = None
    n_chunk_of_page: int = None
    i_chunk_on_doc: int = None
    n_chunk_of_doc: int = None
    n_page: int = None
    reg_date: str = None         # 처리 일시 (ISO 8601)
    chunk_bboxes: str = None     # 청크 위치 좌표 (정규화 0~1, JSON 문자열)
    media_files: str = None      # 연관 미디어 파일 (JSON 문자열)
    created_date: int = None     # 확장 필드: 작성일 YYYYMMDD 정수 (예: 20250115)
    authors: str = None          # 확장 필드: 작성자 (JSON 문자열)
    title: str = None            # 확장 필드: 문서 제목
```

**확장 필드 (attachment 대비 추가)**:

| 필드 | 출처 | 설명 |
|------|------|------|
| `created_date` | `enrichment.metadata` + `field_transforms(date_int)` | 작성일 `YYYYMMDD` 정수. 추출 실패 시 본문 휴리스틱(`doc_text_scan`), 그래도 실패 시 0 |
| `authors` | `enrichment.metadata` (passthrough) | 작성자 정보 (JSON 문자열) |
| `title` | 문서 첫 `TITLE` 아이템 | 문서 제목 |
| `HEADER:` (text 선두) | `GenosSmartChunker` 헤더 경로 | 청크가 속한 섹션 헤더 경로 접두어 (예: `HEADER: 제1장 총칙, 제1절 목적`) |

> `extra='allow'` 이므로 `custom_fields` enricher 가 추출한 임의 키도 예약 필드와 충돌하지 않는 한 그대로 벡터에 passthrough 됩니다(중첩 값은 JSON 문자열로 직렬화).

### 5.1 `media_files` 구조

`media_files` 는 청크에 포함된 미디어(그림/표 이미지) 목록을 담은 **JSON 문자열**입니다. 각 항목은 `{name, type, ref}` 형태입니다.

| 키 | 설명 |
|----|------|
| `name` | 저장된 이미지 파일명 (artifacts 디렉터리 기준) |
| `type` | `image` = 그림(`PictureItem`), `table_image` = 표 이미지(`table_image.enable: true` 일 때만) |
| `ref` | docling `self_ref`. **같은 `ref` 가 `chunk_bboxes` 에도 존재**하므로 bbox·이미지·청크를 조인할 수 있습니다 |

```json
[
  {"name": "image_000001_ab12.png", "type": "image",       "ref": "#/pictures/0"},
  {"name": "table_000000_cd34.png", "type": "table_image", "ref": "#/tables/3"}
]
```

> `type='table_image'` 항목은 `table_image.enable: true` 인 경우에만 생성됩니다(기본 false → 그림만 기록되어 기존 동작과 동일).

---

## 6. 예외 / 트러블슈팅

| 증상 | 원인 / 점검 | config 키 |
|------|-------------|-----------|
| OCR 이 동작하지 않음 | `ocr_mode: disable` 이거나 `ocr_endpoint` 미설정 | `ocr.ocr_mode`, `ocr.paddle.ocr_endpoint` |
| 표/본문에 `GLYPH...` 잔존 | 글리프 임계값이 높아 재OCR 미트리거, 또는 OCR 서버 응답 실패 | `ocr.glyph_detection.*`, `ocr.paddle.ocr_endpoint` |
| OCR 이 너무 자주/과하게 발생 | `auto` 모드 + 낮은 임계값 | `ocr.glyph_detection.document_threshold` 상향 또는 `ocr_mode: disable` |
| 레이아웃 모델 호출 실패 | `<LAYOUT_SERVING_ID>` 미교체, 서빙 미가용 | `layout.genos_layout.endpoint` / `layout_model_type: docling_layout` 로 전환 |
| TOC/메타데이터 비어있음 | 해당 항목 `enable: false` 또는 `<ENRICHMENT_SERVING_ID>` 미교체 | `enrichment[].enable`, `enrichment[].url` |
| LLM provider 오류로 적재 실패 | enrichment 호출 중 `LLMApiError` → `GenosServiceException` | enrichment url/key/서빙 상태 점검 |
| `created_date` 가 0 | 추출/본문 스캔 모두 실패 또는 날짜 포맷 미인식 | `enrichment.metadata.field_transforms`, 프롬프트 `output_fields` |
| 메모리 부족(OOM) | 이미지 생성·배율 과다 | `pdf_pipeline.generate_page_images: false`, `images_scale` 하향 |
| 표 분석 느림 | accurate 모드 | `pdf_pipeline.table_structure_mode: "fast"` |
| 청크가 너무 큼/작음 | `chunking.chunk_size`(yaml) 또는 kwargs `chunk_size` | 우선순위 kwargs > yaml > 0. 단위는 `chunking.tokenizer_type`(char=문자/huggingface=토큰) |
| `chunk length is 0` | 청크 미생성(빈 문서/파싱 실패) | 입력 파일·OCR·레이아웃 점검 |

---

## 부록: 코드 내부 상세

> 이 부록은 고급 독자를 위한 보조 설명입니다. 일반 운영은 §3 의 config 튜닝으로 충분합니다.

### A. config 로딩 (`__init__`)

`DocumentProcessor.__init__(config_path=None)` 흐름:

1. `config_path is None` → `_resolve_default_convert_config_path()` 로 `resource_dev` → `resource` 순 resolve.
2. `_load_config()` 가 YAML 을 dict 로 로드(매핑 아니면 `ValueError`).
3. `ocr` / `layout` / `pdf_pipeline` 섹션을 각각 파싱. enum/정수/불리언 값은 `_parse_optional_*` + 맵(`_ACCELERATOR_DEVICE_MAP`, `_TABLE_FORMER_MODE_MAP`)으로 변환하며 잘못된 값은 경고 후 폴백.
4. `EnrichmentConfig.from_raw(cfg.get("enrichment"), config_dir, parent_cfg=cfg)` 로 enrichment 파싱 → `DataEnrichmentOptions`(toc/metadata) 및 facade enricher(image_description/custom_fields) 구성.
5. `_metadata_field_transforms = ec.metadata.field_transforms or DEFAULT_METADATA_FIELD_TRANSFORMS`.

### B. 4개 컨버터 매트릭스 (`_create_converters`)

OCR on/off × 주/보조 백엔드 조합으로 4개 컨버터를 준비, 1순위 실패 시 2순위로 자동 폴백합니다.

| 컨버터 | OCR | 백엔드 | 용도 |
|--------|-----|--------|------|
| `converter` | ❌ | PyPdfium | 일반 PDF 텍스트 추출 (1순위) |
| `second_converter` | ❌ | PyPdfium | `converter` 실패 시 폴백 |
| `ocr_converter` | ✅ 전체 | DoclingParseV4 | OCR 재처리 (1순위) |
| `ocr_second_converter` | ✅ 전체 | PyPdfium | `ocr_converter` 실패 시 폴백 |

OCR 컨버터는 기본 파이프라인을 deep copy 후 `do_ocr=True`, `force_full_page_ocr=True` 로 설정합니다.

### C. GLYPH 감지 / 선택적 OCR

- `check_glyph_text(text, threshold)`: 텍스트에 `GLYPH\w*` 가 `threshold` 개 이상이면 `True`.
- `check_glyphs(document)`: 문서 텍스트 아이템 중 `GLYPH` 토큰이 `_glyph_document_threshold`(config `document_threshold`)를 **초과**하면 `True` → `auto` 모드에서 전체 OCR 트리거.
- `ocr_all_table_cells(document, pdf_path)`: GLYPH 가 있는 표만 대상으로, 셀별 bbox 를 잘라 이미지로 렌더(최소 높이 20px 보장, zoom 1~4 배)한 뒤 `ocr_endpoint` 로 POST, 응답 `rec_texts` 로 셀 텍스트 교체. `table_cell_ocr_timeout` 적용. 예외는 삼켜서 원본 문서 반환.

### D. `__call__` 확장자 분기

```python
ext = Path(file_path).suffix.lower()
if ext == ".ppt":
    # LangChain 경로 (PowerPointLoader → RecursiveCharacterTextSplitter → compose_vectors_langchain)
    #   + _extract_page_images() 로 임베디드 이미지 추출
else:  # .pdf / .docx / .pptx 등 Docling 경로
    if ext == ".pdf":
        if ocr_mode == "force":  document = load_documents_with_docling_ocr(...)
        else:
            document = load_documents(...)
            if ocr_mode == "auto" and (not check_document(...) or check_glyphs(document)):
                document = load_documents_with_docling_ocr(...)
        if ocr_mode != "disable" and ocr_endpoint:
            document = ocr_all_table_cells(document, file_path)
    else:
        document = load_documents(...)

    if ext in (".docx", ".pptx"):  convert_to_pdf(...)   # 미리보기/이미지용
    document = document._with_pictures_refs(...)
    document = enrichment(document, ...)                 # TOC / metadata (docling)
    document = enrich_image_descriptions(document, ...)  # 실패해도 skip
    document = await enrich_custom_fields(document, ...) # 실패해도 skip
    chunks = split_documents(document, ...)              # GenosSmartChunker
    vectors = await compose_vectors(document, chunks, ...)
```

### E. `GenosSmartChunker` 주요 설정값

```python
class GenosSmartChunker(BaseChunker):
    tokenizer = "/models/.../sentence-transformers-all-MiniLM-L6-v2"  # huggingface 모드 토큰 카운팅용 (없으면 HF fallback)
    max_tokens: int = 1024        # 클래스 기본값. split_documents() 에서 chunk_size(kwargs>yaml>0)로 오버라이드
    tokenizer_type: str = "char"  # "char"(기본)=문자 수 기준 | "huggingface"=HF 토큰 수 기준
    merge_peers: bool = True
    merge_list_items: bool = True
```

`split_documents()` 는 청크 크기를 `chunk_size = kwargs.get('chunk_size')` (없으면 yaml `chunking.chunk_size`, 둘 다 없으면 `0`)로 정하고 `GenosSmartChunker(max_tokens=chunk_size, tokenizer_type=<chunking.tokenizer_type>, merge_peers=True)` 로 청커를 만듭니다. 청킹 토크나이저 경로/계산 방식은 yaml `chunking` 섹션(`tokenizer_path`/`tokenizer_id`/`tokenizer_type`)에서 읽습니다. `preprocess()` 로 모든 아이템+헤더를 1개의 DocChunk 로 수집한 뒤 `_split_document_by_tokens()` 의 4단계(+2.5 보정) 파이프라인으로 분할합니다([4.3](#43-genosbucketchunker-전략)).

### F. `compose_vectors` 메타데이터 병합

1. `extract_metadata_from_document(document)` 로 docling KeyValueItem 메타데이터 추출.
2. enrichment 컨텍스트 메타데이터와 병합(`merged_metadata`).
3. `apply_field_transforms(self._metadata_field_transforms, merged_metadata, document)` → typed 값(`created_date` 등)과 consumed_keys 산출.
4. 예약 필드 + consumed_keys 를 제외한 나머지를 passthrough(중첩값은 JSON 직렬화).
5. `title` 은 문서 첫 `DocItemLabel.TITLE` 아이템에서 추출.
6. 청크별 `GenOSVectorMetaBuilder` 로 text(`HEADER:` 접두어 포함)/page/bbox/media + global metadata 조립, 이미지가 있으면 `upload_files` 로 비동기 업로드.

### G. 예외 클래스

- `GenosServiceException(error_code, error_msg)`: Genos 적재 실패 메시지 전달용. enrichment 의 `LLMApiError`(예: `precheck.enabled=true` 상태에서 입력 토큰이 `max_context_tokens - completion_reserved_tokens` 초과)는 provider payload 를 보존해 이 예외로 재던집니다. 청크가 0개면 `GenosServiceException("1", "chunk length is 0")`.
- `assert_cancelled(request)`: 클라이언트 연결이 끊기면 취소 예외를 던지는 유틸(현재 호출부는 주석 처리).

### H. TOC 프롬프트 (config 유래)

TOC/메타데이터 프롬프트는 코드에 하드코딩되지 않고 `enrichment[].toc`(및 `metadata` 항목)의 `system_prompt_file`/`user_prompt_file` 이 가리키는 별도 `.md` 파일에서 읽습니다(파일 미지정 시 inline `system_prompt`/`user_prompt`, 그것도 없으면 내장 기본). user 프롬프트의 `{{raw_text}}` 가 문서 전문으로 치환되며, 출력은 `<toc>TITLE:...\n1. ...\n1.1. ...</toc>` 형식의 계층적 목차입니다.
