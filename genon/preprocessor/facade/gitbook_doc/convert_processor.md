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
| `<OCR_ENDPOINT>` | `ocr.ocr_endpoint` | PaddleOCR 서버 주소 |
| `<LAYOUT_SERVING_ID>` | `layout.genos_layout.endpoint` | Genos layout 모델 서빙 ID |
| `<ENRICHMENT_SERVING_ID>` | `enrichment[].toc/metadata.url` | Genos enrichment LLM 서빙 ID |
| `<IMAGE_DESCRIPTION_SERVING_ID>` | `enrichment[].image_description.url` | 이미지 설명 LLM 서빙 ID |

---

## 3. 설정 (convert_processor_config.yaml)

이 전처리기의 동작은 전적으로 이 YAML 로 제어됩니다. 최상위 4개 섹션으로 구성됩니다.

| 섹션 | 역할 |
|------|------|
| `ocr` | OCR 모드·엔진·임계값 |
| `layout` | 레이아웃 분석 모델 선택 및 서빙 설정 |
| `pdf_pipeline` | Docling PDF 파싱 성능/품질 노브 |
| `enrichment` | LLM 기반 보강 작업 목록(list) |

### 3.1 전체 스키마

아래는 캐노니컬 설정(`resource/convert_processor_config.yaml`)의 전체 구조입니다. 프롬프트 본문 등 긴 문자열은 `| ...` 로 축약했습니다.

```yaml
ocr:
  # OCR 수행 모드. "auto"(default)=휴리스틱 기반 재OCR / "force"=무조건 전체 OCR / "disable"=OCR 안 함
  # (PDF 입력에만 적용. DOCX/기타 포맷은 ocr_mode 무관)
  ocr_mode: "auto"

  # OCR 엔진 선택. "paddle"(default) | "upstage"
  engine: "paddle"

  # engine: "paddle" 일 때만 사용. <OCR_ENDPOINT>: PaddleOCR 서버 주소로 변경 필요
  ocr_endpoint: "http://<OCR_ENDPOINT>/ocr"

  # 글리프 깨진 테이블 셀 재OCR 시 HTTP timeout(초)
  table_cell_ocr_timeout: 60

  paddle:
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
    page_batch_size: 32
    max_completion_tokens: 16384

# PDF 파이프라인 (docling PDF 파싱 성능/품질 노브)
pdf_pipeline:
  num_threads: 8                     # accelerator 스레드 수
  device: "auto"                     # "auto"(default) | "cpu" | "cuda" | "mps"
  images_scale: 2                    # 페이지/그림 이미지 렌더 배율
  generate_page_images: true
  generate_picture_images: true
  table_structure_mode: "accurate"   # "accurate"(default) | "fast"

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
      system_prompt: | ...
      user_prompt: | ...

  - metadata:
      enable: true
      # system_prompt 가 설정되면 facade custom metadata enricher를 사용하고,
      # 비어있으면 docling 내장 metadata 추출 경로를 사용한다.
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
      system_prompt: | ...
      user_prompt: | ...
      precheck:
        enabled: false
        max_context_tokens: 128000
        completion_reserved_tokens: 12000

  # image_description / custom_fields 는 facade 후처리 enricher. 필요 시 enable: true 로 활성화.
  - image_description:
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<IMAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      concurrency: 16        # 이미지 설명 요청 병렬 수
      before_items: 3
      after_items: 2
      max_context_chars: 1500
      prompt_template: | ...
```

> 위 블록은 `resource/` 기본본 기준입니다. `resource_dev/` 본에는 사이트 운영을 위한 실제 endpoint/key 와 추가 `custom_fields` 항목 예시가 들어 있을 수 있으며, 배포 시 두 본의 placeholder/실값을 환경에 맞게 정리해야 합니다.

---

### 3.2 OCR 설정

`ocr` 섹션은 PDF 입력의 OCR 동작을 제어합니다. 코드에서는 `_build_ocr_options()` 와 `__init__` 이 이 값을 읽습니다.

| 키 | 기본값 | 설명 |
|----|--------|------|
| `ocr_mode` | `"auto"` | OCR 수행 모드. `auto`/`force`/`disable`. **PDF 입력에만 적용** (DOCX/기타는 무관). 알 수 없는 값은 `auto` 폴백 |
| `engine` | `"paddle"` | OCR 엔진. `paddle` \| `upstage`. 알 수 없는 값은 `paddle` 폴백 |
| `ocr_endpoint` | `http://<OCR_ENDPOINT>/ocr` | `engine: paddle` 일 때 PaddleOCR 서버 주소. **사이트별 교체 필요** |
| `table_cell_ocr_timeout` | `60` | 글리프 깨진 테이블 셀 재OCR HTTP timeout(초). 0 이하/오류 시 60 폴백 |
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
| `genos_layout.page_batch_size` | `32` | 레이아웃 모델 페이지 배치 크기(전역 `settings.perf.page_batch_size`). 0 이하/오류 시 32 폴백 |
| `genos_layout.max_completion_tokens` | `16384` | Layout LLM 최대 생성 토큰. 양의 정수, 유효하지 않거나 0 이하이면 16384 폴백 |

- **`genos_layout`**: 외부 서빙형 GenOS 레이아웃 모델. 제목/본문/표/이미지 검출과 reading order 품질 개선을 기대할 수 있으나 별도 서빙 인프라가 필요합니다.
- **`docling_layout`**: Docling 기본 레이아웃 모델. 별도 서빙 인프라가 없는 환경에서 사용합니다.

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

### 3.5 Enrichment 설정

`enrichment` 는 **`{이름: {옵션}}` 형식의 list** 입니다. 항목은 enricher 종류를 나타내며, 같은 종류가 여러 번 올 수 있습니다(`custom_fields`). 파싱은 `EnrichmentConfig.from_raw()` 가 담당합니다.

**비활성화 3가지 방법**:

```
① 항목을 list 에서 삭제
② 항목 전체를 주석 처리
③ enable: false 추가 (설정은 보존, 동작만 끔)  ← 권장
```

> **Format 호환성**: 과거 dict 형식(`enrichment: {do_toc: true, toc: {...}, ...}`, Format A)도 `EnrichmentConfig` 가 여전히 수용하지만, 항목별 enable/url/key 관리가 명확한 **list 형식(Format B)이 권장**입니다.

지원하는 enricher 4종:

| 항목 | 역할 | 처리 주체 |
|------|------|-----------|
| `toc` | 계층적 목차(TOC) 자동 생성 | docling enrichment (`DataEnrichmentOptions`) |
| `metadata` | 작성일·작성자 등 메타데이터 추출 | `system_prompt` 설정 시 facade custom metadata enricher, 비어있으면 docling 내장 metadata 경로 |
| `image_description` | 이미지 설명 생성 | facade 후처리 enricher (`FacadeImageDescriptionEnricher`) |
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
| `system_prompt` / `user_prompt` | (YAML 본문) | 비어있으면 docling 내장 기본 프롬프트 사용. `user_prompt` 의 `{{raw_text}}` 가 문서 본문으로 치환 |

#### metadata

문서에서 작성일·작성자 등을 추출합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `max_tokens` / `temperature` / `timeout` | `10000` / `0.0` / `3600` | 생성/요청 파라미터 |
| `pages` | `[1,4]` | 추출 대상 페이지 범위. `null`(또는 빈 값)이면 첫 4페이지 |
| `output_fields` | `[created_date, authors]` | 추출 키 목록. **프롬프트의 JSON 키와 일치해야 함** |
| `parser.type` | `json` | `json`(default) \| `python`. 응답 파싱 방식 |
| `precheck.*` | toc 와 동일 | 컨텍스트 사전 검사 |
| `system_prompt` / `user_prompt` | (YAML 본문) | 설정 시 facade custom metadata enricher 사용, 비어있으면 docling 내장 경로 |

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

#### image_description

이미지 주변 문맥을 참고해 LLM 으로 이미지 설명을 생성합니다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `concurrency` | `16` | 이미지 설명 요청 병렬 수 |
| `before_items` / `after_items` | `3` / `2` | 앞/뒤 문맥으로 포함할 아이템 수 |
| `max_context_chars` | `1500` | 문맥 최대 문자 수 |
| `prompt_template` | (YAML 본문) | `{{before_context}}`/`{{caption}}`/`{{after_context}}` 치환 변수 사용 |

#### custom_fields (복수 허용)

사용자 정의 필드를 LLM 으로 추출합니다. **여러 항목을 동시에 둘 수 있으며**, 각 항목은 독립 LLM 호출 후 결과를 metadata 에 병합합니다. 설정 방식은 두 가지입니다.

```yaml
# (A) 인라인: 항목 안에 프롬프트/필드를 직접 작성
- custom_fields:
    enable: true
    url: "http://.../serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
    model: "model"
    pages: [1]
    output_fields: [authors]
    parser: { type: json }     # json | python
    system_prompt: | ...
    user_prompt: | ...

# (B) config_file: 별도 yaml 파일로 분리
- custom_fields:
    enable: true
    config_file: custom_field_authors.yaml   # resource_path 는 이 yaml 디렉토리로 자동 주입
```

> `custom_fields` 항목은 `enable: true` 이고 옵션이 비어있지 않을 때만 활성화됩니다. `resource_path` 를 지정하지 않으면 config 파일이 위치한 디렉토리가 자동 주입되어, `config_file` 상대 경로 해석에 사용됩니다.

---

### 3.6 사이트 적용 시 필수 수정 항목

배포 시 아래 placeholder 를 환경에 맞는 실제 값으로 교체합니다. **운영 IP/키를 직접 하드코딩하지 말고**, Genos 에 등록한 서빙 ID 기반 endpoint 를 사용합니다.

| placeholder | config 위치 | 교체 값 |
|-------------|-------------|---------|
| `<OCR_ENDPOINT>` | `ocr.ocr_endpoint` | PaddleOCR 서버 주소 (`engine: paddle` 인 경우만) |
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
| **청크 토큰 크기 조정** | 코드의 `GenosSmartChunker` `max_tokens` 조정 (운영 기준 약 2000). config 노브가 아니므로 [부록 E](#부록-코드-내부-상세) 참고 |

> **청크 토큰**: convert 경로의 청크 크기는 `split_documents()` → `GenosSmartChunker(max_tokens=...)` 로 결정됩니다. 현재 facade 구현에서는 호출 kwargs(`max_chunk_size`) 또는 운영 기준 약 2000 토큰을 사용합니다. 향후 config 노브로 노출될 예정이며, 지금은 코드 변경 사항입니다.

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

---

## 6. 예외 / 트러블슈팅

| 증상 | 원인 / 점검 | config 키 |
|------|-------------|-----------|
| OCR 이 동작하지 않음 | `ocr_mode: disable` 이거나 `ocr_endpoint` 미설정 | `ocr.ocr_mode`, `ocr.ocr_endpoint` |
| 표/본문에 `GLYPH...` 잔존 | 글리프 임계값이 높아 재OCR 미트리거, 또는 OCR 서버 응답 실패 | `ocr.glyph_detection.*`, `ocr.ocr_endpoint` |
| OCR 이 너무 자주/과하게 발생 | `auto` 모드 + 낮은 임계값 | `ocr.glyph_detection.document_threshold` 상향 또는 `ocr_mode: disable` |
| 레이아웃 모델 호출 실패 | `<LAYOUT_SERVING_ID>` 미교체, 서빙 미가용 | `layout.genos_layout.endpoint` / `layout_model_type: docling_layout` 로 전환 |
| TOC/메타데이터 비어있음 | 해당 항목 `enable: false` 또는 `<ENRICHMENT_SERVING_ID>` 미교체 | `enrichment[].enable`, `enrichment[].url` |
| LLM provider 오류로 적재 실패 | enrichment 호출 중 `LLMApiError` → `GenosServiceException` | enrichment url/key/서빙 상태 점검 |
| `created_date` 가 0 | 추출/본문 스캔 모두 실패 또는 날짜 포맷 미인식 | `enrichment.metadata.field_transforms`, 프롬프트 `output_fields` |
| 메모리 부족(OOM) | 이미지 생성·배율 과다 | `pdf_pipeline.generate_page_images: false`, `images_scale` 하향 |
| 표 분석 느림 | accurate 모드 | `pdf_pipeline.table_structure_mode: "fast"` |
| 청크가 너무 큼/작음 | `GenosSmartChunker.max_tokens` | (코드 노브, [부록 E](#부록-코드-내부-상세) 참고) |
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
    tokenizer = "/models/.../sentence-transformers-all-MiniLM-L6-v2"  # 토큰 카운팅용 (없으면 HF fallback)
    max_tokens: int = 1024        # 클래스 기본값. split_documents() 호출 시 max_chunk_size kwargs 로 오버라이드(운영 기준 약 2000)
    merge_peers: bool = True
    merge_list_items: bool = True
```

`split_documents()` 는 `GenosSmartChunker(max_tokens=kwargs.get('max_chunk_size', 0), merge_peers=True)` 로 청커를 만들고, 청크별 시작 페이지 카운트를 누적합니다. `preprocess()` 로 모든 아이템+헤더를 1개의 DocChunk 로 수집한 뒤 `_split_document_by_tokens()` 의 4단계(+2.5 보정) 파이프라인으로 분할합니다([4.3](#43-genosbucketchunker-전략)).

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

TOC/메타데이터 프롬프트는 코드에 하드코딩되지 않고 `enrichment[].toc.system_prompt`/`user_prompt`(및 `metadata` 항목)에서 읽습니다. `user_prompt` 의 `{{raw_text}}` 가 문서 전문으로 치환되며, 출력은 `<toc>TITLE:...\n1. ...\n1.1. ...</toc>` 형식의 계층적 목차입니다. 프롬프트를 비우면 docling 내장 기본 프롬프트가 사용됩니다.
