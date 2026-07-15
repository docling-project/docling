# 적재용 지능형 전처리기 매뉴얼

RAG 지식베이스 구축을 위한 **품질 최우선** 전처리기입니다. 딥러닝 레이아웃 분석 + LLM Enrichment + 선택적 OCR 로 문서의 논리적 구조를 최대한 보존해 적재합니다.

> 이 전처리기의 동작은 **`intelligent_processor_config.yaml`** 로 제어합니다. 과거에는 OCR 주소·파이프라인 옵션·Enrichment 설정을 facade 코드(`intelligent_processor.py`)에 직접 수정했지만, 현재는 코드 수정/재빌드 없이 **YAML 옵션 튜닝**만으로 운영 환경을 바꿉니다. 코드 내부 동작이 필요한 경우만 [부록](#부록-코드-내부-상세)을 참고하세요.

---

## 목차

- [적재용 지능형 전처리기 매뉴얼](#적재용-지능형-전처리기-매뉴얼)
  - [목차](#목차)
  - [1. 개요](#1-개요)
    - [설계 철학](#설계-철학)
    - [대상 포맷](#대상-포맷)
    - [핵심 특징](#핵심-특징)
    - [세 전처리기 비교](#세-전처리기-비교)
    - [GPU 필요 여부 (운영 노트)](#gpu-필요-여부-운영-노트)
  - [2. 빠른 시작](#2-빠른-시작)
    - [config 로딩 우선순위](#config-로딩-우선순위)
    - [사이트별 필수 수정 항목 요약](#사이트별-필수-수정-항목-요약)
  - [3. 설정 (`intelligent_processor_config.yaml`)](#3-설정-intelligent_processor_configyaml)
    - [3.1 전체 스키마](#31-전체-스키마)
      - [defaults 설정](#defaults-설정)
    - [3.2 OCR 설정](#32-ocr-설정)
    - [3.3 레이아웃 설정](#33-레이아웃-설정)
      - [`repetition_penalty` 사용 가이드](#repetition_penalty-사용-가이드)
    - [3.4 PDF 파이프라인 설정](#34-pdf-파이프라인-설정)
      - [PPT 페이지 설명 (`formats.ppt.page_description`)](#341a-ppt-페이지-설명-formatspptpage_description)
      - [3.4.2 xlsx(엑셀) 직접 처리 (`formats.xlsx`)](#342-xlsx엑셀-직접-처리-formatsxlsx)
      - [3.4.3 표 텍스트 형식 (`output.table_format`)](#343-표-텍스트-형식-outputtable_format)
    - [3.5 Enrichment 설정](#35-enrichment-설정)
      - [toc](#toc)
      - [metadata](#metadata)
      - [image\_description](#image_description)
      - [custom\_fields](#custom_fields)
      - [프롬프트 파일 분리 \& 변수 치환](#프롬프트-파일-분리--변수-치환)
    - [3.7 청킹 설정](#37-청킹-설정)
    - [3.8 사이트 적용 시 필수 수정 항목](#38-사이트-적용-시-필수-수정-항목)
    - [3.9 자주 쓰는 튜닝 시나리오](#39-자주-쓰는-튜닝-시나리오)
  - [4. 처리 동작 개요 (보조)](#4-처리-동작-개요-보조)
    - [단일 PDF 파이프라인](#단일-pdf-파이프라인)
    - [config → 단계 매핑](#config--단계-매핑)
    - [부록(appendix) 자동 연결](#부록appendix-자동-연결)
    - [`HEADER:` 접두어](#header-접두어)
    - [빈 문서 처리](#빈-문서-처리)
  - [5. 출력 데이터 구조](#5-출력-데이터-구조)
  - [6. 예외 / 트러블슈팅](#6-예외--트러블슈팅)
  - [부록: 코드 내부 상세](#부록-코드-내부-상세)
    - [A.1 초기화 (`__init__` / `_load_config`)](#a1-초기화-__init__--_load_config)
    - [A.2 동적 이미지 옵션](#a2-동적-이미지-옵션)
    - [A.3 청커 (`GenosSmartChunker`)](#a3-청커-genossmartchunker)
    - [A.4 enrichment / compose\_vectors](#a4-enrichment--compose_vectors)
    - [A.5 진입점 (`__call__`)](#a5-진입점-__call__)
    - [A.6 호출 예시](#a6-호출-예시)

---

## 1. 개요

### 설계 철학

```
"품질 중심: AI 기반 레이아웃 분석 및 고품질 데이터 적재"
```

단순 다단·표·차트가 섞인 문서를 단순 추출하면 문맥이 파괴됩니다(2단 레이아웃에서 문장이 끊기거나, 병합 셀의 행/열 관계가 소실되거나, 캡션이 그림과 분리됨). 지능형 전처리기는 이를 **딥러닝 레이아웃 분석 + 구조적 청킹**으로 해결합니다.

### 대상 포맷

- **PDF 우선** — 입력이 PDF 면 그대로 단일 Docling 파이프라인으로 처리합니다.
- **비-PDF 자동 변환** — HWP / DOCX / PPTX / 이미지 등 비-PDF 입력은 `__call__` 진입부에서 `_is_pdf()`(매직 헤더 `%PDF-`) 검사 후 PDF 로 자동 변환(`auto_convert_to_pdf=True` 기본)하여 동일 파이프라인에 진입합니다.
- **xlsx/csv 직접 처리** — `.xlsx`/`.xlsm`/`.csv` 는 PDF 변환 없이 직접 처리합니다(행 분할 버그 방지). 자세한 내용은 [3.4.2](#342-xlsx엑셀-직접-처리-formatsxlsx) 참고.

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **AI 레이아웃 분석** | 딥러닝 모델이 제목/본문/표/그림/캡션 등 요소를 자동 식별 (`layout` 섹션) |
| **TableFormer** | 병합 셀·다중 헤더 등 복잡한 표를 마크다운으로 복원 (`pdf_pipeline.table_structure_mode`) |
| **Smart OCR** | GLYPH(인코딩 깨짐)가 감지된 영역만 선별적 OCR (`ocr` 섹션) |
| **LLM Enrichment** | 목차(TOC) 자동 생성·메타데이터 추출·이미지 설명·커스텀 필드 (`enrichment` 섹션) |
| **부록 자동 연결** | 본문의 '별지/별표' 참조를 실제 부록 파일과 자동 매칭 (intelligent 고유) |
| **섹션 기반 순수 분할** | 토큰 제한 없이 문서 구조(섹션 헤더)를 100% 존중하는 청킹 (`max_tokens=0`) |

### 세 전처리기 비교

| 비교 항목 | attachment | convert | **intelligent** |
|-----------|-----------|---------|-----------------|
| **설계 목표** | 속도 중심 | 호환성 중심 | **품질 중심** |
| **대상 포맷** | 다양 (PDF, HWP, 오디오, CSV 등) | 다양 (PPT, DOCX→PDF 변환) | **PDF 우선 (비-PDF 자동 변환)** |
| **Docling 파이프라인** | SimplePipeline (경량) | 전체 PDF 파이프라인 | **전체 PDF 파이프라인** |
| **청커** | HybridChunker (∞ 토큰) | GenosSmartChunker (2000 토큰) | **GenosSmartChunker (0=무제한)** |
| **청킹 전략** | 레이아웃 병합 | 섹션+토큰 병합 | **순수 섹션 분할 (병합 없음)** |
| **OCR** | 없음 | 선택적 (PaddleOCR/Upstage) | **선택적 (PaddleOCR/Upstage)** |
| **Enrichment** | 없음 | LLM TOC + 메타데이터 | **LLM TOC + 메타데이터 + 이미지설명** |
| **부록 연결** | ❌ | ❌ | **✅ 별지/별표 자동 매칭** |
| **빈 문서 처리** | 예외 발생 | 예외 발생 | **더미 텍스트 삽입** |
| **고유 출력 필드** | — | `authors` | **`appendix`, `file_path`** |

### GPU 필요 여부 (운영 노트)

기본 config 에서는 **layout 분석 / OCR 을 외부 API endpoint 로 호출**하므로 preprocessor 컨테이너의 GPU 의존도가 낮습니다.

- **layout** — `layout_model_type: genos_layout` + `genos_layout.endpoint` 로 별도 서빙(예: DotsOCR) 호출. 내재 layout 모델 미사용.
- **OCR** — `ocr.paddle.ocr_endpoint` 로 별도 OCR 서빙(PaddleOCR) 호출. 내재 OCR 모델 미사용.
- **TableFormer** (`do_table_structure=True`) — 내재 모델이지만 `device` 가 cuda 미발견 시 CPU 로 fallback.

즉 기본 config 그대로면 GenOS UI 의 GPU 할당량을 **0** 으로 두어도 정상 동작합니다(TableFormer 는 CPU fallback). 반대로 `layout_model_type: docling_layout` 으로 내재 layout 모델을 쓰거나 TableFormer 추론을 GPU 가속(`pdf_pipeline.device: cuda`)하려면 GenOS UI 에서 GPU 할당량을 **1 이상**으로 지정합니다.

---

## 2. 빠른 시작

1. **facade 등록** — `intelligent_processor.py` 를 Genos UI 의 전처리기 facade 로 등록합니다. GenOS 는 `DocumentProcessor()` 를 **무인자**로 호출합니다.
2. **config 등록** — `intelligent_processor_config.yaml` 을 resource 디렉터리에 둡니다.
3. **필수 항목 치환** — placeholder([§3.8](#38-사이트-적용-시-필수-수정-항목))를 사이트 값으로 바꿉니다.

### config 로딩 우선순위

`config_path` 인자가 없으면 `_resolve_default_intelligent_config_path()` 가 다음 순서로 탐색합니다.

```
1. resource_dev/intelligent_processor_config.yaml   (존재하면 우선 — 사이트 전용 override)
2. resource/intelligent_processor_config.yaml       (기본 / 캐노니컬 스키마)
3. (둘 다 없으면) 코드 기본값
```

> 사이트 전용 설정은 `resource_dev` 에 두면 기본값을 덮어씁니다. 캐노니컬 스키마는 `preprocessor/resource/intelligent_processor_config.yaml` 입니다.

### 사이트별 필수 수정 항목 요약

| 항목 | YAML 키 | placeholder |
|------|---------|-------------|
| OCR 서버 | `ocr.paddle.ocr_endpoint` | `<OCR_ENDPOINT>` |
| 레이아웃 서빙 | `layout.genos_layout.endpoint` | `<LAYOUT_SERVING_ID>` |
| Enrichment 서빙(toc/metadata) | `enrichment[].url` | `<ENRICHMENT_SERVING_ID>` |
| 이미지 설명 서빙 | `image_description.url` | `<IMAGE_DESCRIPTION_SERVING_ID>` |

---

## 3. 설정 (`intelligent_processor_config.yaml`)

이 전처리기 운영의 **중심**입니다. `__init__()` 이 시작 시 이 YAML 을 읽어(`_load_config` → `yaml.safe_load`) 각 섹션(`defaults` / `ocr` / `layout` / `pdf_pipeline` / `enrichment`)을 파싱하고 파이프라인을 구성합니다. 잘못된 값(타입 오류, 알 수 없는 enum 등)은 경고 로그 후 안전한 기본값으로 폴백합니다.

### 3.1 전체 스키마

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
  ocr_mode: "auto"            # "auto"(default) | "force" | "disable"
  engine: "paddle"            # "paddle"(default) | "upstage"
  table_cell_ocr_timeout: 60  # 글리프 깨진 셀 재OCR HTTP timeout(초)
  paddle:
    ocr_endpoint: "http://<OCR_ENDPOINT>/ocr"   # engine=paddle 일 때만 사용
    text_score: 0.3
  glyph_detection:
    table_cell_threshold: 1   # 셀 GLYPH 토큰 N개 이상이면 재OCR
    document_threshold: 10    # 문서 GLYPH 토큰 N개 초과면 OCR 경로 재시도
  upstage:                    # engine=upstage 일 때만 사용
    api_endpoint: "https://api.upstage.ai/v1/document-digitization"
    api_key: ""               # 비어있으면 UPSTAGE_API_KEY 환경변수 fallback
    model: "ocr"
    timeout: 60
    text_score: 0.5

layout:
  layout_model_type: "genos_layout"   # "genos_layout"(default) | "docling_layout"
  genos_layout:
    endpoint: "http://llmops-gateway-api-service:8080/rep/serving/<LAYOUT_SERVING_ID>/v1/chat/completions"
    api_key: ""               # k8s 내부 통신 시 불필요
    page_batch_size: 128
    max_completion_tokens: 16384
    model: "dots-mocr"          # 서빙 모델명
    timeout: 3600               # VLM 요청 타임아웃(초)
    retry_count: 2              # 비정상 VLM 응답 재시도 횟수
    temperature: 0.1            # 생성 temperature
    top_p: 0.9                  # 생성 top_p (0<top_p<=1)
    repetition_penalty: 1.15    # >1.0, 토큰 반복(degeneration) 억제 (아래 가이드 참조)

pdf_pipeline:
  num_threads: 8              # accelerator 스레드 수
  device: "auto"              # "auto"(default) | "cpu" | "cuda" | "mps"
  images_scale: 2             # 페이지/그림 이미지 렌더 배율
  generate_page_images: true
  generate_picture_images: true   # false 면 이미지 설명 enrichment 비활성화됨
  table_structure_mode: "accurate" # "accurate"(default) | "fast"

table_image:
  enable: false                  # true 면 모든 표를 이미지로 저장 + media_files 기록

output:
  table_format: "html"           # 청크 text 내 docling 표 직렬화 형식. "html"(default) | "markdown"
  compact_tables: true           # markdown 표 컬럼 정렬 패딩 제거(대형 표 축소). html 포맷엔 무관

# 청킹(GenosSmartChunker) 설정
chunking:
  # 청크 최대 크기. char 모드면 "최대 문자 수", huggingface 모드면 "최대 토큰 수".
  # 우선순위: 호출 kwargs 의 chunk_size > 아래 chunk_size. 0 = 크기 기반 분할 안 함(순수 섹션 분할).
  # 0 초과 시 최소 1024 로 보정된다(그보다 작은 값은 문맥이 과도하게 잘림).
  chunk_size: 10000
  # 청킹 모드. split_only(기본) = 구조 기반 청크를 유지하고 chunk_size 를 초과하는 청크만 분할(작은 청크는 병합하지 않음).
  #            resize_all = 모든 청크를 chunk_size 에 맞게 병합/분할(작은 청크도 인접 청크와 합쳐 채움).
  # 우선순위: 호출 kwargs 의 chunk_mode > 아래 chunk_mode.
  chunk_mode: split_only
  # 토큰 수 계산 방식. "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
  tokenizer_type: "char"
  # 청킹용 토크나이저(huggingface 모드에서만 사용). tokenizer_path 존재 시 그 경로, 없으면 tokenizer_id(HF) 폴백
  tokenizer_path: "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
  tokenizer_id: "sentence-transformers/all-MiniLM-L6-v2"

# enrichment: {이름: {옵션}} 형식의 list.
# 비활성화: ① 항목 삭제  ② 항목 주석 처리  ③ enable: false
# url 의 <ENRICHMENT_SERVING_ID> 는 Genos에 등록한 모델서빙 ID로 변경. api_key 는 k8s 내부 통신 시 불필요.
# 프롬프트는 별도 .md 파일로 분리(*_file). 경로는 이 config 와 같은 디렉토리 기준(파일명만).
#   상세는 3.5 의 "프롬프트 파일 분리 & 변수 치환" 참고.
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
        enabled: true
        max_context_tokens: 128000
        completion_reserved_tokens: 12000
      system_prompt_file: prompt_toc_default_system.md
      user_prompt_file: prompt_toc_default_user.md   # 파일 안에서 {{raw_text}} 치환
  - metadata:
      enable: true
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      max_tokens: 10000
      temperature: 0.0
      timeout: 3600
      pages: [1, 4]             # null = 첫 4페이지
      output_fields: [created_date, authors]   # 프롬프트의 JSON 키와 일치해야 함
      parser:
        type: json              # json(default) | python
      system_prompt_file: prompt_metadata_default_system.md
      user_prompt_file: prompt_metadata_default_user.md   # 파일 안에서 {{raw_text}} 치환
      precheck:
        enabled: true
        max_context_tokens: 128000
        completion_reserved_tokens: 12000
  - doc_summary:            # 문서 본문요약 1회 → image/table description 공용 {{doc_summary}}
      enable: false
      url: "http://llmops-gateway-api-service:8080/rep/serving/<ENRICHMENT_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      prompt_file: prompt_doc_summary.md   # {{full_text}} 기반 문서 요약
      max_chars: 6000
  - image_description:
      enable: true
      url: "http://llmops-gateway-api-service:8080/rep/serving/<IMAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      concurrency: 16           # 이미지 설명 요청 병렬 수
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
      concurrency: 8            # 표 설명 요청 병렬 수
      before_items: 3
      after_items: 2
      max_context_chars: 1500
      prompt_template_file: prompt_table_description_default.md   # 요약 전용 프롬프트
      refine:
        enable: false          # true 면 재구성 HTML 로 표 본체 교체
        prompt_file: prompt_table_refine_combined.md   # 재구성 HTML + 요약 통합 프롬프트
```

> 위는 캐노니컬 스키마를 그대로 반영한 것입니다(프롬프트 본문은 `...` 로 축약). `output` / `whisper` 섹션은 [§3.6](#36-출력--whisper-설정) 참조.

#### defaults 설정

`defaults` 섹션은 전처리기 공통 기본값을 담습니다. 현재는 로깅 레벨만 노출됩니다. `__init__` 이 이 값을 읽어 `self._log_level` 로 보관하고, 매 실행마다 `setup_logging()` 에 적용합니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `defaults.log_level` | 로깅 레벨. `5`=DEBUG / `4`=INFO / `3`=WARNING / `2`=ERROR / `1`=CRITICAL / `0`=NOLOG(전체 비활성화). 누락/오류 시 4 폴백 | `4` (INFO) |

> 실행 시 `params` 로 `log_level` 을 전달하면 그 값이 이 config 기본값보다 **우선**합니다. `params` 에 값이 없을 때만 `defaults.log_level` 이 적용됩니다. (`0`=NOLOG 도 정상 적용되도록 None 여부로 판별합니다.)

### 3.2 OCR 설정

`ocr` 섹션은 `_build_ocr_options()` 로 파싱되어 OCR 엔진·임계값을 결정합니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `ocr.ocr_mode` | `auto`=글리프 휴리스틱 기반 재OCR / `force`=무조건 전체 OCR / `disable`=OCR 안 함 | `auto` |
| `ocr.engine` | OCR 엔진 (`paddle` \| `upstage`). 알 수 없는 값은 `paddle` 폴백 | `paddle` |
| `ocr.paddle.ocr_endpoint` | PaddleOCR 서버 주소 (`<OCR_ENDPOINT>` 치환). engine=paddle 일 때만 사용. 구버전 `ocr.ocr_endpoint`(상위) 위치도 호환 인식 | — |
| `ocr.table_cell_ocr_timeout` | 글리프 깨진 테이블 셀 재OCR HTTP timeout(초) | 60 |
| `ocr.paddle.text_score` | PaddleOCR 텍스트 신뢰도 임계값 | 0.3 |
| `ocr.glyph_detection.table_cell_threshold` | 셀 GLYPH 토큰 N개 **이상**이면 재OCR (`>=`) | 1 |
| `ocr.glyph_detection.document_threshold` | 문서 GLYPH 토큰 N개 **초과**면 OCR 경로 재시도 (`>`) | 10 |

**`engine: upstage`** 선택 시 `ocr.upstage` 하위 키가 사용됩니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `ocr.upstage.api_endpoint` | Upstage 문서 디지털화 API | `https://api.upstage.ai/v1/document-digitization` |
| `ocr.upstage.api_key` | 비어있으면 `UPSTAGE_API_KEY` 환경변수에서 fallback | `""` |
| `ocr.upstage.model` | 모델명 | `ocr` |
| `ocr.upstage.timeout` | 요청 timeout(초). 0 이하/비정상 값은 60 폴백 | 60 |
| `ocr.upstage.text_score` | 텍스트 신뢰도 임계값. 비정상 값은 0.5 폴백 | 0.5 |

### 3.3 레이아웃 설정

| 키 | 의미 | 기본값 |
|----|------|--------|
| `layout.layout_model_type` | `genos_layout`=외부 서빙형 / `docling_layout`=Docling 내재 모델 | `genos_layout` |
| `layout.genos_layout.endpoint` | GenOS layout 모델 서빙 주소 (`<LAYOUT_SERVING_ID>` 치환) | — |
| `layout.genos_layout.api_key` | k8s 내부 통신 시 불필요 | `""` |
| `layout.genos_layout.page_batch_size` | layout 추론 페이지 배치 크기 (전역 `settings.perf` 에 반영) | 128 |
| `layout.genos_layout.max_completion_tokens` | layout LLM 최대 생성 토큰. 양의 정수, 유효하지 않거나 0 이하이면 16384 폴백 | 16384 |
| `layout.genos_layout.model` | 서빙 모델명. 비어있으면 `dots-mocr` 폴백 | `"dots-mocr"` |
| `layout.genos_layout.timeout` | VLM 요청 HTTP 타임아웃(초). 유효하지 않거나 0 이하이면 3600 폴백 | 3600 |
| `layout.genos_layout.retry_count` | 비정상(스키마 불일치 등) VLM 응답 재시도 횟수. 음수/오류 시 2 폴백 | 2 |
| `layout.genos_layout.temperature` | 생성 샘플링 temperature. 음수/오류 시 0.1 폴백 | 0.1 |
| `layout.genos_layout.top_p` | nucleus 샘플링 top_p. `0<top_p<=1` 아니면 0.9 폴백 | 0.9 |
| `layout.genos_layout.repetition_penalty` | 토큰 반복(degeneration) 억제. `>0`, 유효하지 않으면 1.15 폴백. **자세한 사용은 아래 가이드 참조** | 1.15 |

- `genos_layout` — 제목/본문/표/이미지 검출과 reading order 품질 개선을 기대할 수 있으나 별도 서빙 인프라가 필요합니다.
- `docling_layout` — 별도 서빙 없이 동작. 서빙 환경이 없을 때 사용.

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

### 3.4 PDF 파이프라인 설정

docling PDF 파싱 성능/품질 노브입니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `pdf_pipeline.num_threads` | accelerator 스레드 수 | 8 |
| `pdf_pipeline.device` | `auto` \| `cpu` \| `cuda` \| `mps`. 알 수 없는 값은 `auto` 폴백 | `auto` |
| `pdf_pipeline.images_scale` | 페이지/그림 이미지 렌더 배율 | 2 |
| `pdf_pipeline.generate_page_images` | 페이지 이미지 생성 | true |
| `pdf_pipeline.generate_picture_images` | 그림 이미지 생성 | true |
| `pdf_pipeline.table_structure_mode` | `accurate`(품질) \| `fast`(속도) → TableFormer 모드. 알 수 없는 값은 `accurate` 폴백 | accurate |

> **주의** — `generate_picture_images: false` 면 그림 이미지가 생성되지 않아 **이미지 설명 enrichment(`image_description`)가 비활성화**됩니다. 이미지 설명이 필요하면 이 값을 `true` 로 유지하세요.

### 3.4.1 표 이미지 설정 (`table_image`)

복잡한 병합/레이아웃 표를 텍스트로 완전히 정규화하기 어려운 경우를 위해, 표를 그림(picture)과 동일하게 **이미지로도 함께 보관**하는 옵션입니다. 검색은 기존처럼 청크 텍스트 기반으로 하고, 답변 단계에서 표 이미지를 활용하는 "검색=청크 / 답변=표 이미지" 하이브리드 전략을 지원합니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `table_image.enable` | 모든 표 영역을 PNG 로 잘라 저장하고 `media_files` 에 `type='table_image'` 로 기록 | false |

`enable: true` 면:
- 문서 내 모든 `TableItem` 을 페이지 이미지에서 bbox 로 잘라 artifacts 디렉터리에 `table_*.png` 로 저장합니다.
- 각 표는 `media_files` 에 `{name, type:'table_image', ref}` 로 추가됩니다(`ref` 는 `chunk_bboxes` 의 해당 표 엔트리와 동일 → 조인 가능).
- 청크 내 표 텍스트(MD/HTML)는 **그대로 유지**됩니다(하이브리드).

> **주의** — 표 이미지는 페이지 이미지를 잘라 만들므로, `enable: true` 이면 `pdf_pipeline.generate_page_images` 가 내부적으로 **강제 `true`** 로 보정됩니다(별도 설정 불필요).

### 3.4.1a PPT 페이지 설명 (`formats.ppt.page_description`)

PPT(`.ppt`/`.pptx`)는 슬라이드가 이미지 위주여서 텍스트 추출만으로는 의미가 소실되기 쉽습니다. 이 옵션은 각 페이지를 **이미지로 렌더링해 VLM(Vision LM)에 보내 페이지 설명을 생성**하고, 그 설명을 해당 페이지 텍스트로 문서에 주입합니다. 기존 `image_description`(페이지 내부 그림 오브젝트 단위)과 달리 **"페이지 자체"를 통째로** 설명합니다.

- **적용 대상**: PPT(`.ppt`/`.pptx`) 원본만. 적재용은 PPT→PDF→docling 변환 후 각 페이지를 설명합니다.
- **청킹**: PPT 는 **1 page = 1 chunk**(페이지의 native text + 페이지 설명이 동일 청크). `chunking.chunk_size` 가 지정되면 연속 페이지를 그 크기까지 결합합니다.
- **TOC**: PPT 는 페이지 단위라 목차 계층이 무의미 → **TOC enrichment 를 자동 skip**합니다.
- `enable: true` 면 `pdf_pipeline.generate_page_images` 가 내부적으로 **강제 `true`** 로 보정됩니다.
- 페이지의 native text 는 프롬프트의 **`{{page_text}}`** 변수로 함께 전달되어 설명 품질을 높입니다.

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
| `enable` | PPT 페이지 설명 활성화 | `false` |
| `url` / `api_key` / `model` | VLM(chat/completions) 서빙 endpoint / 키 / 모델명 | `<PAGE_DESCRIPTION_SERVING_ID>` / `""` / `model` |
| `timeout` | VLM 요청 타임아웃(초) | `360` |
| `concurrency` | 페이지 설명 병렬 요청 수 | `16` |
| `images_scale` | 페이지 렌더 해상도 배율(클수록 고해상도/느림) | `2.0` |
| `max_image_side` | 전송 전 이미지 최대 변(px)으로 다운스케일. `0`=원본 | `0` |
| `max_tokens` | VLM 출력 토큰 상한(생성 시간↓). `0`=상한 없음 | `0` |
| `params` | 추가 VLM 파라미터(dict). 서버가 `max_completion_tokens` 등 다른 키를 쓸 때 사용 | `{}` |
| `prompt_template_file` | 프롬프트 `.md` 경로(권장). `{{page_text}}` 치환 | — |

> **주의** — 페이지 설명은 **파싱 단계 전용** 기능입니다. Chunk API(`/chunker`)는 파서 출력(JSON)만 받으며 렌더된 페이지 이미지가 없어 페이지 설명을 수행하지 않습니다.

### 3.4.2 xlsx(엑셀) 직접 처리 (`formats.xlsx`)

`.xlsx`/`.xlsm`(및 `.csv`)는 **PDF 변환 없이 직접 처리**합니다(이슈 #288). 기존에는 xlsx 를 LibreOffice 로 PDF 변환 후 파싱했는데, 엑셀 페이지 레이아웃에 따라 **한 행의 데이터가 여러 페이지로 쪼개지는** 논리 오류가 있었습니다. 직접 처리는 이를 원천 차단합니다. `.csv` 도 본질적으로 tabular 이므로 항상 직접 처리됩니다(PDF 변환 안 함).

처리 방식은 `formats.xlsx.processing_mode` 로 선택합니다.

| 키 | 값 | 기본 | 설명 |
|----|----|------|------|
| `formats.xlsx.processing_mode` | `docling` \| `tabular` | `docling` | 처리 방식 선택 |
| `formats.xlsx.tabular.header_row` | int | `0` | (tabular) 헤더 행. `0`=자동판정, `>0`=해당 행을 단일 헤더로 강제 |
| `formats.xlsx.tabular.multi_table` | bool | `false` | (tabular) 한 시트에 표가 여러 개(빈 행으로 분리)면 표별로 분리 |

**docling 모드 (기본)** — MsExcel 백엔드로 `DoclingDocument` 를 만들어 **기존 청킹/벡터 파이프라인**을 그대로 태웁니다(시트=1페이지).
- **표마다 별도 청크**로 분리합니다.
- 표가 `chunking.chunk_size` 를 초과하면 **행(row) 단위로 분할**하고, 분할된 각 청크에 **헤더 행을 반복 포함**합니다(제목행 + 컬럼명행 자동 판정).
- 각 청크 텍스트 앞에 **`시트명: <시트명>`** 접두를 붙입니다.

**tabular 모드** — 데이터 **행마다 1벡터**를 만들고, 각 컬럼 값을 **최상단 스칼라 property** 로 부여해 벡터 DB 에서 컬럼 단위 `where` 필터가 가능하게 합니다.
- 병합셀은 openpyxl 로 **unmerge + forward-fill** 하여 병합 헤더 유실을 방지합니다.
- **멀티헤더 자동 판정**: 전열 병합 제목행은 컨텍스트로만(키 제외), 부분 병합 계층 헤더는 `상위_하위` 로 flatten, 그 아래 컬럼명행을 헤더로 사용합니다.
- **컬럼 키 규칙**: Weaviate property 명 규칙(`/[_A-Za-z][_0-9A-Za-z]*/`)상 ASCII 헤더는 그대로 키로 쓰고, 한글 등 비-ASCII 헤더는 **`field_<sha256[:8]>`** 로 alias 합니다(같은 헤더 텍스트는 파일이 달라도 동일 키 → 컬렉션 전체 필터 안정). 원본 헤더명은 **`column_map`**(JSON 문자열) 메타에 보존합니다.
- `multi_table: true` 면 한 시트의 복수 표(빈 행 분리)를 표별로 분리해 각각 헤더를 재판정합니다.

```yaml
formats:
  xlsx:
    processing_mode: "docling"    # 표 단위 청크 + 행 분할(헤더 반복) + 시트명 접두
    # processing_mode: "tabular"  # 행=벡터 + 컬럼 필터 메타(field_/column_map)
    tabular:
      header_row: 0
      multi_table: false
```

> **참고** — `.xls`/`.xlsb`(구형/바이너리 엑셀)는 openpyxl/docling 이 못 읽으므로 직접 처리 대상이 아니며, 기존대로 PDF 변환 경로로 처리됩니다.

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

### 3.5 Enrichment 설정

`enrichment` 는 `{이름: {옵션}}` 형식의 **list** 입니다. 각 항목은 `enable` 플래그와 항목별 `url` / `api_key` / `model` 을 가집니다. `EnrichmentConfig.from_raw()` 가 파싱하며, 결과로 docling `DataEnrichmentOptions`(toc + 내장 metadata)와 facade enricher 들(metadata / doc_summary / image_description / table_description / custom_fields)이 구성됩니다.

**비활성화 3가지 방법**: ① 항목 자체 삭제 ② 항목 주석 처리 ③ `enable: false`.

| 항목 | 역할 | 처리 주체 |
|------|------|-----------|
| `toc` | 계층적 목차(TOC) 자동 생성 | docling enrichment (`DataEnrichmentOptions`) |
| `metadata` | 작성일 등 메타데이터 추출 | 커스텀 신호(`system_prompt`/`user_prompt`/`*_file`/`output_fields`/`parser`) 중 하나라도 지정 시 facade custom metadata enricher (이때 docling 내장 metadata 추출 비활성), 아무 신호 없으면 docling 내장 경로 |
| `doc_summary` | 문서 본문요약 1회 생성(공용 `{{doc_summary}}` 컨텍스트) | facade 후처리 (`DocSummaryEnricher`) |
| `image_description` | 그림에 대한 LLM 설명 생성 | facade 후처리 (`generate_picture_images=false` 면 비활성) |
| `table_description` | 표 요약 + 선택적 구조 재구성(refine) | facade 후처리 (`TableDescriptionEnricher`) |
| `custom_fields` | 사용자 정의 추출 필드 (복수 가능) | facade 후처리 (`resource_path` 자동 주입) |

#### toc

| 키 | 의미 | 기본값 |
|----|------|--------|
| `url` / `api_key` / `model` | 서빙 endpoint / 키 / 모델명 | `<ENRICHMENT_SERVING_ID>` / `""` / `model` |
| `temperature` / `top_p` / `seed` | 샘플링 파라미터 | 0.0 / 0.00001 / 33 |
| `max_tokens` | 생성 최대 토큰 | 10000 |
| `precheck.enabled` | 입력 토큰 사전 추정 차단 | true |
| `precheck.max_context_tokens` / `precheck.completion_reserved_tokens` | 컨텍스트 한도 / 예약 토큰 | 128000 / 12000 |
| `system_prompt_file` / `user_prompt_file` | 프롬프트 `.md` 파일 경로(권장, config 디렉토리 기준). `user_prompt` 의 `{{raw_text}}` 치환 | — |
| `system_prompt` / `user_prompt` | inline 프롬프트(`*_file` 미지정 시 fallback) | — |
| `split.enabled` | 긴 문서 **분할(Split) TOC 추출** 수행 여부(아래 참고) | false |
| `split.pages_per_chunk` / `split.page_overlap` | 청크당 페이지 수 / 청크 경계 중복 페이지 수 | 100 / 1 |
| `split.carryover_max_tokens` | 다음 청크에 주입할 누적 목차(outline) 토큰 상한 | 1500 |
| `repetition_penalty` | 토큰 반복(degeneration) 억제(>1.0). 게이트웨이/vLLM 지원 시에만 사용, 미설정 시 미전송 | — |
| `thinking` | 추론(thinking) 모드. `off`(기본, 차단 토큰 전송) \| `on` \| `auto`(미전송, 모델 자동 판단) | off |
| `thinking_dialect` | 추론 토글 키 방언. `standard`(`enable_thinking`) \| `hcx`(`force_reasoning`/`skip_reasoning`) | standard |

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

**동작**
- **OFF(기본)**: 단일 호출. 동작 변화 없음(컨텍스트 초과 시 기존처럼 에러).
- **ON**: 페이지 N개씩 청크화 → 첫 청크는 설정 프롬프트로, 이후 청크는 설정 프롬프트 앞에 누적 목차를 덧붙여
  순차 추출 → 매 스텝 병합(경계 중복 제거·번호 재부여). 응답에 `<toc>` 블록이 없거나(분석문/절단) 컨텍스트를
  초과하는 청크는 건너뛰어 부분 결과를 보존합니다.

**통합 프롬프트(`prompt_toc_default_user.md`)** — 하나의 프롬프트가 첫 추출/이어쓰기를 모두 처리합니다.
- `{{prior_toc}}`(누적 목차) 자리표시자 + **작업 모드(Operating Mode)** 분기를 가집니다: `<previous_outline>`
  이 비면 전체 추출(분석 후 `<toc>`, `TITLE:` 포함), 내용이 있으면 이어쓰기(분석 출력 금지·`<toc>`만·새 항목만,
  이미 누적된 항목/부모 섹션은 반복 금지하되 미추출 하위 조항은 모두 추출).
- 커스텀 프롬프트에 `{{prior_toc}}` 가 없으면 코드가 누적 목차 블록을 앞에 자동으로 덧붙입니다.

**권장 / 주의**
- 컨텍스트에 들어가는 문서는 분할이 불필요하므로 기본값(OFF) 유지를 권장합니다. 긴 문서가 토큰 초과로 실패할 때
  켜세요.
- `page_overlap>0` 은 경계 항목 누락을 줄이지만, 모델 재추출이 경계에서 깔끔히 정렬되지 않으면 중복이 일부 남을
  수 있습니다(중복이 문제면 `0` 권장).
- 분할은 LLM의 지시 준수에 의존합니다. 매우 긴 청크에서 모델이 장황해지면 `<toc>` 전에 토큰이 소진될 수 있으며,
  이때 해당 청크는 건너뜁니다. `repetition_penalty`(지원 시) 활성화가 도움이 됩니다.

#### metadata

| 키 | 의미 | 기본값 |
|----|------|--------|
| `url` / `api_key` / `model` | 서빙 endpoint / 키 / 모델명 | `<ENRICHMENT_SERVING_ID>` / `""` / `model` |
| `max_tokens` / `temperature` / `timeout` | 생성 토큰 / 샘플링 / timeout(초) | 10000 / 0.0 / 3600 |
| `pages` | 메타데이터 추출 대상 페이지 범위. `null`/빈 값이면 첫 4페이지 | `[1, 4]` |
| `output_fields` | 추출 필드 목록. **프롬프트의 JSON 키와 일치해야 함** | `[created_date, authors]` |
| `parser.type` | 응답 파서 (`json` \| `python`) | json |
| `precheck.*` | toc 와 동일 | — |
| `system_prompt_file` / `user_prompt_file` | 프롬프트 `.md` 파일 경로(권장). `system_prompt_file` 생략 시 built-in default | — |
| `system_prompt` / `user_prompt` | inline 프롬프트(`*_file` 미지정 시 fallback). `{{raw_text}}` 치환 | — |
| `field_transforms` | 추출 키 → 벡터 필드 변환 매핑 (아래) | 미지정 시 기본값 |
| `thinking` / `thinking_dialect` | 추론(thinking) 모드 / 방언. toc 와 동일(위 "thinking(추론) 모드" 참고) | off / standard |

**`field_transforms` (선언적 메타데이터 매핑)**

LLM 이 `output_fields` 로 추출한 키 이름은 프롬프트가 정하기 나름이고(`created_date`, `작성일`, `doc_date` …) 값도 문자열이라, 검색에 쓰이는 최종 벡터 메타 필드와 이름·타입이 다를 수 있습니다. `field_transforms` 는 **"추출 결과의 어떤 키를 → 어떤 벡터 필드에 → 어떤 변환을 거쳐 넣을지"** 를 YAML 로 선언하는 다리입니다. 프롬프트를 바꿔 추출 키 이름이 달라져도 `source` 만 고치면 되고, 코드 없이 설정만으로 매핑을 변경할 수 있습니다. 비워두면 `DEFAULT_METADATA_FIELD_TRANSFORMS` 가 적용되어 기존 `created_date` 동작이 보존됩니다.

```yaml
field_transforms:
  - source: [created_date, 작성일]   # 추출 결과에서 순서대로 탐색할 후보 키
    target: created_date            # 값을 넣을 벡터 메타 필드 (생략 시 source 첫 키)
    type: date_int                  # 값 변환기: 날짜 텍스트 → YYYYMMDD 정수
    fallback: doc_text_scan         # 추출이 비면 본문 휴리스틱으로 보조 추출
```

각 항목(spec)의 필드:

| 필드 | 필수 | 의미 |
|------|------|------|
| `source` | ✅ | 추출 결과에서 찾을 후보 키. 문자열 1개 또는 목록(목록이면 순서대로 탐색해 **첫 비어있지 않은 값** 사용) |
| `target` | 선택 | 값을 넣을 벡터 메타 필드명. 생략 시 `source` 첫 키 |
| `type` | 선택 | 값 변환기. 현재 지원: `date_int` (`"2024-01-15"`→`20240115`, `YYYY-MM`→일자 `01`, `YYYY`→`0101` 보정, 실패 시 `0`). 생략 시 값을 그대로 사용 |
| `fallback` | 선택 | 추출값이 비었을 때 쓸 보조 추출. 현재 지원: `doc_text_scan` (본문에서 `작성일`/`기준일`/`최초 작성일`/`보고자료` 키워드 주변 날짜 휴리스틱 탐색) |

**동작 흐름** (각 spec 마다): ① `source` 후보 키를 순서대로 보며 첫 비어있지 않은 값 선택 → ② `type` 변환기 적용 → ③ 값이 비면 `fallback` 으로 본문에서 보조 추출 → ④ 결과를 `target` 에 주입(사용한 키는 passthrough 제외) → ⑤ 변환 대상이 아닌 나머지 추출 키는 그대로 벡터 메타에 통과(passthrough), 중첩 객체는 JSON 문자열로 직렬화.

**입력 → 출력 예시** (위 기본 설정 기준)

```text
추출 결과:  { "created_date": "2025-01-15", "department": "IT" }
→ 벡터 메타: created_date = 20250115 (date_int 변환),  department = "IT" (passthrough)
```

- **키 이름 변경**: 프롬프트가 `doc_date` 로 추출하면 `source: [doc_date]` 로만 바꿔도 `created_date` 가 동일하게 채워집니다.
- **fallback**: 추출이 비고 본문에 `"보고자료 2024-01-15 기준"` 이 있으면 `created_date: 20240115` 로 보강됩니다.
- intelligent 의 기본 변환은 `created_date` 입니다 (convert 는 `created_date` + `authors`).
- 신규 변환기/보조추출은 `field_transforms.py` 의 `VALUE_TRANSFORMS` / `FALLBACK_STRATEGIES` 에 등록하면 YAML 에서 바로 사용 가능합니다.

#### doc_summary

문서 본문을 **요청당 1회** LLM 으로 요약해 공용 `{{doc_summary}}` 컨텍스트로 제공합니다. `image_description`·`table_description` 프롬프트가 이 값을 참조하므로, 과거 `image_description.doc_summary.*` 중첩 설정을 **독립 enricher 항목**으로 승격했습니다(요약 중복 계산 제거). 요약 결과는 출력 metadata 의 `doc_summary` 로도 노출됩니다.

| 키 | 의미 | 기본값 |
|----|------|--------|
| `enable` | 본문요약 생성 여부(`true` 또는 런타임 `doc_summary=1` 로 활성화) | `false` |
| `url` / `api_key` / `model` | 서빙 endpoint / 키 / 모델명 | `<ENRICHMENT_SERVING_ID>` / `""` / `model` |
| `prompt_file` | 요약 프롬프트 `.md`(`{{full_text}}` 치환) | `prompt_doc_summary.md` |
| `max_chars` | 요약 입력으로 넣을 본문 최대 글자 수 | 6000 |

> `doc_summary` 는 별도 LLM 호출을 유발하므로 기본 off 입니다. `image_description`/`table_description` 이 켜져 있어도 `doc_summary` 가 off 면 `{{doc_summary}}` 는 빈 문자열로 치환됩니다.

#### image_description

| 키 | 의미 | 기본값 |
|----|------|--------|
| `url` / `api_key` / `model` | 서빙 endpoint / 키 / 모델명 | `<IMAGE_DESCRIPTION_SERVING_ID>` / `""` / `model` |
| `concurrency` | 이미지 설명 요청 병렬 수 | 16 |
| `before_items` / `after_items` | 캡션 앞/뒤 문맥으로 포함할 아이템 수 | 3 / 2 |
| `max_context_chars` | 문맥 최대 글자 수 | 1500 |
| `prompt_template_file` | 프롬프트 `.md` 파일 경로(권장). `{{before_context}}` / `{{caption}}` / `{{after_context}}` / `{{doc_summary}}` 치환 | — |
| `prompt_template` | inline 프롬프트(`*_file` 미지정 시 fallback) | — |
| `chart.enable` | 차트 처리 활성화(false 면 일반 image description 만) | `false` |
| `chart.detection` | `auto`=docling 자동판별(차트로 분류된 이미지만 차트 프롬프트) / `all`=모든 이미지를 차트로 처리 | `auto` |
| `chart.chart_prompt_file` | 차트 전용 프롬프트 `.md`(`{{doc_summary}}` 등 동일 변수) | `prompt_chart_description_default.md` |

> `pdf_pipeline.generate_picture_images: false` 면 이 항목은 enable 이어도 동작하지 않습니다.
> 프롬프트의 `{{doc_summary}}` 컨텍스트는 이제 **독립 `doc_summary` enricher**(위 참조)가 채웁니다(과거 `image_description.doc_summary.*` 중첩 설정은 표준 `- doc_summary:` 항목으로 이동). `chart.enable: true` 면 변환 단계에서 docling 그림 분류(`do_picture_classification`)가 자동 활성화됩니다(모델 `ds4sd--DocumentFigureClassifier` 는 빌드 시 `/models` 에 포함). `chart` 는 별도 LLM 호출을 유발하므로 기본 off 입니다.

**런타임 kwargs 오버라이드 (이미지·차트 description)**

호출 시 `params`(kwargs)로 아래 0/1 플래그를 주면 해당 호출에 한해 config 기본값을 덮어씁니다(미지정 시 config 값 유지).

| kwargs | 대응 config | 의미 |
|--------|-------------|------|
| `img_desc` | `image_description.enable` | 이미지 description 사용유무 |
| `chart_desc` | `image_description.chart.enable` | 차트 description 사용유무 (레거시 별칭 `chart_convert`) |
| `chart_detection` | `image_description.chart.detection` | `1`=auto(docling 자동판별) / `0`=all(모든 이미지를 차트로) |
| `doc_summary` | `doc_summary.enable` | 문서 본문요약 사용유무(독립 enricher) |
| `table_desc` | `table_description.enable` | 표 요약 description 사용유무 |
| `table_refine` | `table_description.refine.enable` | 표 구조 재구성(refine) 사용유무 |

> `chart_detection=1`(auto) 은 변환 단계 그림 분류가 켜져 있어야 동작합니다. config `chart.enable: false` 로 분류가 꺼진 배포에서는 auto 요청이 자동으로 `all` 로 강등됩니다(경고 로그). 런타임 auto 를 쓰려면 config 에서 `chart.enable: true` 로 두세요.

#### table_description

표(TableItem)마다 앞뒤 문맥·캡션·섹션헤더(+공용 `{{doc_summary}}`)를 참고해 LLM 으로 **표 요약**을 생성하고, 청크의 표 텍스트 뒤에 `\n---\n[표 설명]\n<요약>` 형태로 병기합니다. `refine.enable: true` 면 같은 호출에서 표 구조를 **충실한 HTML 로 재구성**해 표 본체를 그 재구성본으로 교체합니다(요약도 함께 병기).

| 키 | 의미 | 기본값 |
|----|------|--------|
| `enable` | 표 요약 생성 여부(`true` 또는 런타임 `table_desc=1` 로 활성화) | `false` |
| `url` / `api_key` / `model` | 서빙 endpoint / 키 / 모델명 | `<ENRICHMENT_SERVING_ID>` / `""` / `model` |
| `concurrency` | 표 설명 요청 병렬 수 | 8 |
| `before_items` / `after_items` | 앞/뒤 문맥으로 포함할 아이템 수 | 3 / 2 |
| `max_context_chars` | 문맥 최대 글자 수 | 1500 |
| `prompt_template_file` | 요약 전용 프롬프트 `.md`. `{{before_context}}` / `{{after_context}}` / `{{caption}}` / `{{section_header}}` / `{{doc_summary}}` 치환 | `prompt_table_description_default.md` |
| `refine.enable` | 표 구조 재구성(refine) 여부(`true` 또는 런타임 `table_refine=1` 로 활성화) | `false` |
| `refine.prompt_file` | 재구성 HTML + 요약 **통합** 프롬프트(refine 시 요약 프롬프트 대신 사용) | `prompt_table_refine_combined.md` |

- **요약만(refine off)**: 원본 표(html/markdown) 뒤에 `[표 설명]` 요약을 병기합니다.
- **refine on**: 재구성 HTML 로 표 본체를 교체하고 출력 `table_format` 에 맞춰 변환합니다 — `markdown` 이면 `compact_tables` 설정을 반영한 compact 표로 냅니다. refine 표는 구조가 원본 grid 와 달라 **행 분할을 하지 않습니다**(요약도 1회만 포함).
- refine 통합 프롬프트는 `[[[TABLE_HTML]]]` / `[[[TABLE_SUMMARY]]]` 마커로 재구성 HTML 과 요약을 한 응답에 함께 출력하도록 강제합니다.
- `table_description` 은 별도 LLM 호출을 유발하므로 기본 off 입니다.

#### custom_fields

사용자 정의 추출 필드입니다. **복수 항목**을 list 에 나열할 수 있으며, 각 항목에 `resource_path` 가 없으면 config 디렉터리가 자동 주입됩니다. 프롬프트는 `system_prompt_file`/`user_prompt_file`(또는 inline `system_prompt`/`user_prompt`)로 지정하며, system 미지정 시 built-in default 가 사용됩니다. 외부 정의 파일을 쓰는 경우 `config_file` 키로 지정합니다.

#### 프롬프트 파일 분리 & 변수 치환

enrichment 프롬프트는 YAML 안에 inline 으로 박지 않고 **별도 `.md` 파일**로 분리합니다. 운영 시 프롬프트만 교체하기 쉽고, 두 전처리기(parser/intelligent/convert)가 동일 프롬프트를 공유할 때 파일 경로 한 줄만 같게 두면 됩니다.

**프롬프트 파일 지정**

| 키 | 대상 |
|----|------|
| `system_prompt_file` / `user_prompt_file` | toc / metadata / custom_fields |
| `prompt_template_file` | image_description |

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
| `{{before_context}}` | 아이템 앞 문맥 텍스트(`before_items` 개) | 이미지/표 직전 텍스트 아이템들 |
| `{{after_context}}` | 아이템 뒤 문맥 텍스트(`after_items` 개) | 이미지/표 직후 텍스트 아이템들 |
| `{{caption}}` | 이미지/표 캡션 | `PictureItem`/`TableItem.caption_text(document)` |
| `{{section_header}}` | 아이템 직전 섹션 헤더 | 위쪽에서 가장 가까운 section_header/title |
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

### 3.7 청킹 설정

intelligent 는 `GenosSmartChunker` 로 섹션 분할을 수행합니다. 청크 크기는 yaml `chunking.chunk_size`(또는 호출 kwargs `chunk_size`)로 지정하며, `chunk_size=0` 이면 **순수 섹션 분할**("크기 기반 병합·분할을 하지 않고 각 섹션을 독립 청크로 유지")이 됩니다.

```
Bucket = 바구니(하나의 청크), max_tokens = 바구니 크기, 섹션 = 담을 물건
규칙: 담은 합이 max_tokens 초과 시 새 바구니. 단 바구니가 비어있으면 무조건 1개는 담음(at least 1).
      상위 레벨 헤더를 만나면 토큰과 무관하게 새 바구니.
max_tokens=0 → 어떤 섹션이든 담으면 즉시 0 초과 → 각 섹션이 독립 청크 (빈 바구니는 안 만들어짐)
```

**동일 문서 비교 (요약)** — 섹션 A(500), B(300), C(200), D(400), E(1800), F(600) 토큰, A~C 는 제1장 / D~F 는 제2장:

```
max_tokens=2000 (convert):          max_tokens=0 (intelligent):
  [A+B+C] [D] [E] [F]                 [A] [B] [C] [D] [E] [F]
  4개 청크, 작은 조항 병합               6개 청크, 모든 섹션 독립
```

- `max_tokens=0`: 법률/규정/약관 등 **조항별 독립성**이 중요한 문서, 정밀 질의("제5조 설명해줘")에 최적. (intelligent 기본)
- `max_tokens=2000`: 문맥 연속성이 중요하고 청크 수를 줄이고 싶을 때. (convert 기본)

> `max_tokens=0` 이면 "초과 청크 균등 분할"(캡션/표내그림 조정 포함)도 스킵됩니다 — 아무리 긴 섹션도 자르지 않아 표·조항이 중간에 끊기지 않습니다.

**값 변경 (YAML 노브)** — 청크 크기는 yaml `chunking.chunk_size` 또는 호출 kwargs `chunk_size` 로 지정합니다(우선순위 kwargs > yaml > 0). 크기 단위는 `chunking.tokenizer_type` 으로 정해집니다(`char`=문자 수 기준(기본, HF 토크나이저 미로드) / `huggingface`=HF 토큰 수 기준).

```yaml
chunking:
  chunk_size: 0           # 0=순수 섹션 분할 / 512·1024·2000 등=크기 기반 병합
  tokenizer_type: "char"  # char=문자 수 기준 | huggingface=HF 토큰 수 기준
```

| `chunk_size` | 동작 | 권장 |
|----|------|------|
| `0` | 각 섹션 독립 청크 (순수 섹션 분할) | 법률/규정, 정밀 검색 RAG |
| `512` | 작은 섹션 병합 | 짧은 조항 多 |
| `1024` | 중간 병합 | 일반 RAG |
| `2000` | 큰 단위 병합 | 문맥 연속성 중시 |

### 3.8 사이트 적용 시 필수 수정 항목

운영 환경에 맞춰 아래 placeholder 를 실제 값으로 치환합니다. (IP/키 하드코딩 금지 — 서빙 ID 형식 권장)

| placeholder | 위치 | 설명 |
|-------------|------|------|
| `<OCR_ENDPOINT>` | `ocr.paddle.ocr_endpoint` | PaddleOCR 서버 주소 (engine=paddle) |
| `<LAYOUT_SERVING_ID>` | `layout.genos_layout.endpoint` | Genos 등록 layout 모델 서빙 ID |
| `<ENRICHMENT_SERVING_ID>` | `enrichment[].toc.url`, `enrichment[].metadata.url` | TOC/메타데이터 LLM 서빙 ID |
| `<IMAGE_DESCRIPTION_SERVING_ID>` | `enrichment[].image_description.url` | 이미지 설명 LLM 서빙 ID |

> k8s 내부 통신 기반 호출 시 각 `api_key` 는 비워둘 수 있습니다.

### 3.9 자주 쓰는 튜닝 시나리오

| 목표 | 변경 |
|------|------|
| GPU 없이 운영 | `layout.layout_model_type: genos_layout` + `pdf_pipeline.device: cpu`(또는 auto). 내재 layout 미사용 |
| OCR 강제 / 비활성 | `ocr.ocr_mode: force` / `disable` |
| Upstage OCR 사용 | `ocr.engine: upstage` + `ocr.upstage.*` 설정 (api_key 또는 `UPSTAGE_API_KEY`) |
| 처리 속도 우선 | `pdf_pipeline.table_structure_mode: fast`, `images_scale: 1` |
| 이미지 설명 끄기 | `image_description` 항목 삭제/`enable: false` 또는 `generate_picture_images: false` |
| 목차/메타데이터 끄기 | `toc` / `metadata` 항목 삭제·주석·`enable: false` |
| 토큰 초과 문서 차단 완화 | `precheck.enabled: false` 또는 `max_context_tokens` 상향 |
| 작성일 외 추가 메타 필드 | `metadata.output_fields` 추가 + 프롬프트 JSON 키 일치 (+ 필요 시 `field_transforms`) |

---

## 4. 처리 동작 개요 (보조)

이 절은 YAML 설정이 실제 어떤 단계에 매핑되는지 개념적으로 설명합니다(코드 상세는 [부록](#부록-코드-내부-상세)).

### 단일 PDF 파이프라인

```
입력 파일
  │  (비-PDF면 __call__ 진입부에서 PDF 자동 변환: auto_convert_to_pdf=True)
  ▼
① load_documents (Docling)         ← layout, pdf_pipeline 섹션
  ├─ Layout Detection (genos_layout / docling_layout)
  └─ TableFormer (accurate/fast)
  ▼
② 품질 검사 / GLYPH 감지            ← ocr 섹션 (ocr_mode, glyph_detection)
  └─ 필요 시 전체 페이지 OCR 재처리
  ▼
③ ocr_all_table_cells              ← GLYPH 셀만 선별 OCR
  ▼
④ enrichment                       ← enrichment 섹션 (toc/metadata/image_description/custom_fields)
  ▼
⑤ 빈 문서 검사                      ← 텍스트 없으면 더미 "." 삽입
  ▼
⑥ GenosSmartChunker(max_tokens=0) ← 순수 섹션 분할
  ▼
⑦ compose_vectors                  ← HEADER: 접두어, appendix 매칭, created_date/title, file_path
  ▼
List[GenOSVectorMeta]
```

### config → 단계 매핑

| YAML 섹션 | 영향 단계 |
|-----------|-----------|
| `layout`, `pdf_pipeline` | ① 로딩/레이아웃/표 구조 |
| `ocr` | ②③ 품질검사·OCR |
| `enrichment` | ④ TOC/메타데이터/이미지설명/커스텀필드 |
| (코드 `max_tokens`) | ⑥ 청킹 |

### 부록(appendix) 자동 연결

intelligent 고유 기능입니다. 본문에서 "별지/별표/장부" 참조를 탐지해 입력으로 받은 부록 파일 목록(`kwargs['appendix']`)과 매칭하고, 청크별로 `appendix` 필드에 매칭 파일명을 채웁니다.

```
"...세부 사항은 별지 제1호 서식에 따른다..."  +  appendix_list=["별지 제1호 서식.pdf", ...]
   → 공백 제거 → 복합/독립 패턴 탐색 → 파일명 대조 → appendix = "별지 제1호 서식.pdf"
   (여러 개 매칭 시 쉼표 연결, 없으면 "")
```

### `HEADER:` 접두어

각 청크 텍스트 앞에 계층적 헤더 경로를 붙여 검색 문맥을 강화합니다.

```
HEADER: 제1장 총칙, 제1절 목적
제1조(목적) 이 법은 국민의 기본적 권리를 보장하고...
```

### 빈 문서 처리

텍스트 아이템이 하나도 없으면(이미지만 있는 스캔본 등) 예외 대신 더미 텍스트 `"."` 를 삽입해 최소 1개 청크를 보장합니다(`attachment`/`convert` 는 예외 발생). 이미지 메타데이터(`media_files`) 기반 후속 처리를 가능하게 합니다.

---

## 5. 출력 데이터 구조

출력은 `List[GenOSVectorMeta]` 입니다. 공통 필드 + intelligent 고유 필드를 가집니다.

```python
class GenOSVectorMeta(BaseModel):
    class Config:
        extra = 'allow'        # 추출 메타데이터 passthrough 허용

    # 공통 필드
    text: str; n_char: int; n_word: int; n_line: int
    i_page: int; e_page: int
    i_chunk_on_page: int; n_chunk_of_page: int
    i_chunk_on_doc: int; n_chunk_of_doc: int; n_page: int
    reg_date: str; chunk_bboxes: str; media_files: str

    # intelligent 고유/확장 필드
    title: str = None              # 문서 제목
    created_date: int = None       # 작성일 (YYYYMMDD 정수, field_transforms 결과)
    appendix: str = None           # 매칭된 부록 파일명 (없으면 "")
    file_path: str = None          # 비-PDF 자동 변환 시 변환된 PDF 로컬 경로
```

| 필드 | attachment | convert | intelligent | 설명 |
|------|:--:|:--:|:--:|------|
| 공통 (text, page 등) | ✅ | ✅ | ✅ | 기본 청크/페이지 메타 |
| `title` | ❌ | ✅ | ✅ | 문서 제목 |
| `created_date` | ❌ | ✅ | ✅ | 작성일 (YYYYMMDD) |
| `authors` | ❌ | ✅ | ❌ | 작성자 (intelligent 미주입) |
| **`appendix`** | ❌ | ❌ | **✅** | 매칭된 부록 파일명 (청크별) |
| **`file_path`** | ❌ | ❌ | **✅** | 변환된 PDF 경로 (비-PDF 입력 시) |

> `output_fields` 등으로 추출된 그 밖의 메타데이터 키는 `field_transforms` 가 소비하지 않은 경우 `extra='allow'` 에 의해 그대로 벡터 메타에 passthrough 됩니다(중첩 객체는 JSON 문자열로 직렬화).

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

| 증상 | 원인 | 조치 |
|------|------|------|
| `chunk length is 0` | 분할 결과 청크 0개 | 빈 문서 폴백이 더미 텍스트를 삽입하므로 통상 발생 안 함. 발생 시 입력/변환 결과 확인 |
| Enrichment 토큰 초과 | `precheck.enabled=true` + 입력 토큰 추정치가 `max_context_tokens - completion_reserved_tokens` 초과 | 문서 분할/축소, `max_context_tokens` 상향, 또는 해당 항목 `precheck.enabled: false` |
| 이미지 설명이 안 나옴 | `generate_picture_images: false` 또는 `image_description.enable: false` | `pdf_pipeline.generate_picture_images: true` + 항목 enable |
| 표가 깨짐 | 글리프 깨짐 미감지 | `ocr.ocr_mode: force`, `glyph_detection.table_cell_threshold` 하향 |
| layout/OCR 호출 실패 | endpoint/serving ID 오설정 | `<...>` placeholder 치환 확인, k8s 통신 시 api_key 공란 확인 |
| 설정값 무시됨 | `resource_dev` 가 `resource` 를 override | 로딩 우선순위([§2](#2-빠른-시작)) 확인 — 사이트 값은 `resource_dev` 에 |

**토큰 초과 예외 페이로드** — `LLMApiError` → `GenosServiceException("1", <JSON>)` 으로 전파. JSON: `object=error`, `type=BadRequestError`, `param=prompt`, `code=400`, `message="프롬프트 입력 토큰 (N) 초과 하였습니다. (128000 - reserved 12000)."`.

---

## 부록: 코드 내부 상세

YAML 만으로 운영되지만, 코드 내부 동작이 필요한 고급 사용자를 위한 요약입니다.

### A.1 초기화 (`__init__` / `_load_config`)

```python
def __init__(self, config_path: Optional[str] = None):
    if config_path is None:
        config_path = _resolve_default_intelligent_config_path()  # resource_dev → resource
    cfg = _load_config(config_path)                 # yaml.safe_load (mapping 아니면 ValueError)
    self._config_dir = Path(config_path).resolve().parent

    ocr_cfg, layout_cfg, pdf_cfg = _as_dict(cfg.get("ocr")), _as_dict(cfg.get("layout")), _as_dict(cfg.get("pdf_pipeline"))
    ec = EnrichmentConfig.from_raw(cfg.get("enrichment"), self._config_dir, parent_cfg=cfg)
```

- OCR: `ocr_mode`(auto/force/disable, 비정상값 auto), `_build_ocr_options()`(paddle/upstage), glyph 임계값, `table_cell_ocr_timeout`.
- layout: `layout_options.layout_model_type` = `GENOS_LAYOUT`, `genos_layout_options.endpoint/api_key`, `settings.perf.page_batch_size`.
- pdf_pipeline: `PdfPipelineOptions` 에 `generate_page_images / generate_picture_images / images_scale / do_table_structure=True / table_structure_options.mode / accelerator_options(num_threads, device)` 주입. `do_ocr=False`.
- enrichment: `ec` 로부터 `DataEnrichmentOptions`(toc + 내장 metadata) + `_MetadataEnricher` + `_CustomFieldsEnricher[]` + `ImageDescriptionEnricher` + `_metadata_field_transforms` 구성.
- 컨버터 4종(`converter`/`second_converter`/`ocr_converter`/`ocr_second_converter`) — OCR×백엔드 매트릭스, 주 실패 시 보조 폴백.

### A.2 동적 이미지 옵션

`load_documents_with_docling()` 은 intelligent 고유로 `save_images`(기본 `True`), `include_wmf`(기본 `False`) kwargs 를 받아 값이 바뀌면 `_create_converters()` 로 컨버터를 재생성합니다. `include_wmf` 는 일부 한국어 공문서의 WMF 벡터 그래픽을 추출 대상에 포함합니다.

### A.3 청커 (`GenosSmartChunker`)

`convert_processor` 와 동일 코드(facade 한시 구현, 추후 라이브러리 제공 예정). `max_tokens=0` 효과는 [§3.7](#37-청킹-설정) 참조.

- `preprocess()` — `iterate_items()` 로 아이템 + `heading_by_level` 스냅샷을 3개 병렬 리스트(`all_items` / `all_header_info`(`item.text`) / `all_header_short_info`(`item.orig`))로 수집, 누락 테이블 복구 후 1개의 DocChunk yield.
- `_split_document_by_tokens()` — 1단계 섹션 헤더 분할 → 2단계 heading 텍스트 생성 → 2.5단계 초과 분할(**max_tokens=0 시 스킵**) → 3단계 단독 타이틀(≤30자) 병합 → 4단계 토큰 병합(**max_tokens=0 시 사실상 모두 독립**).

### A.4 enrichment / compose_vectors

```python
def enrichment(self, document, **kwargs):
    try:
        return enrich_document(document, self.enrichment_options, **kwargs)
    except LLMApiError as e:
        raise GenosServiceException("1", e.raw_error_message) from e
```

- 별도 enricher 호출: `enrich_metadata()`(`_MetadataEnricher`, `system_prompt` 설정 시 custom 추출), `enrich_image_descriptions()`(`generate_picture_images=False` 면 비활성), `enrich_custom_fields()`.
- `compose_vectors(document, chunks, file_path, request, converted_pdf_path=None, ...)`:
  - 청크마다 `global_metadata.copy()` 후 `check_appendix_keywords()` 결과를 `appendix` 에 설정.
  - `extract_metadata_from_document()` + context 병합 → `apply_field_transforms(self._metadata_field_transforms, ...)` 로 `created_date` 등 typed 필드 생성(`authors` 미주입).
  - `converted_pdf_path` 전달 시 각 벡터의 `file_path` 에 반영, 변환 PDF 를 minio 업로드(object key = 원본 stem + `.pdf`).

### A.5 진입점 (`__call__`)

```python
async def __call__(self, request, file_path, **kwargs):
    # 비-PDF + auto_convert_to_pdf=True(기본) → convert_to_pdf(use_pdf_sdk=kwargs.get('use_pdf_sdk', True))
    document = self.load_documents(file_path, **kwargs)
    if not check_document(document, self.enrichment_options) or self.check_glyphs(document):
        document = self.load_documents_with_docling_ocr(file_path, **kwargs)
    document = self.ocr_all_table_cells(document, file_path)
    document = document._with_pictures_refs(...)
    document = self.enrichment(document, **kwargs)
    # 빈 문서면 더미 "." 삽입
    chunks = self.split_documents(document, **kwargs)   # GenosSmartChunker(max_tokens=chunking.chunk_size, 기본 0)
    if len(chunks) < 1: raise GenosServiceException(1, "chunk length is 0")
    return await self.compose_vectors(document, chunks, file_path, request, converted_pdf_path=..., **kwargs)
```

- `auto_convert_to_pdf` (기본 `True`): 비-PDF 입력 자동 변환. `False` 면 변환 생략(PDF 가정, 변경 전 동작).
- `use_pdf_sdk` (기본 `True`): 변환 엔진 — `True`=PDF SDK, `False`=LibreOffice. `convert_to_pdf()` 는 세 facade 공통 wrapper 로 `converters/hwp_to_pdf/` 에 위임.

### A.6 호출 예시

```python
processor = DocumentProcessor()                 # 무인자 → 기본 config 경로 resolve
vectors = await processor(
    request=request,
    file_path="/path/to/document.pdf",
    save_images=True,                            # 기본 True
    include_wmf=False,
    appendix='["별지 제1호.pdf", "별표.pdf"]',     # 부록 자동 연결 대상
)
# v.text / v.title / v.created_date / v.appendix / v.chunk_bboxes / v.media_files / v.file_path
```

> 이 전처리기는 가장 많은 AI 단계를 거치므로 처리 시간이 세 전처리기 중 가장 길 수 있습니다. 실시간 채팅 첨부는 `attachment_processor`, 빠른 PDF 적재는 `convert_processor`, **최고 품질 RAG 적재**는 이 `intelligent_processor` 를 사용하세요.
