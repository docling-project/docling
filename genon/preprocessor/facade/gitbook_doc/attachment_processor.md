# 첨부용 전처리기 매뉴얼

채팅 중 첨부 파일을 실시간으로 분석하기 위한 속도 중심 경량 전처리기입니다.

> **이 문서의 핵심**: 전처리기의 동작은 코드 수정이 아니라 **`attachment_processor_config.yaml` 옵션으로 제어**합니다. 대부분의 튜닝은 이 설정 파일만 편집하면 됩니다. 코드 내부는 [부록](#부록-코드-내부-상세)에서 보조 참고용으로 다룹니다.

---

## 목차

1. [개요](#1-개요)
2. [빠른 시작](#2-빠른-시작)
3. [설정 (attachment_processor_config.yaml)](#3-설정-attachment_processor_configyaml)
   - 3.1 [전체 스키마](#31-전체-스키마)
   - 3.2 [섹션별 상세](#32-섹션별-상세)
   - 3.3 [런타임 kwargs override](#33-런타임-kwargs-override)
   - 3.4 [자주 쓰는 튜닝 시나리오](#34-자주-쓰는-튜닝-시나리오)
4. [처리 동작 개요 (보조)](#4-처리-동작-개요-보조)
   - 4.1 [convert_to_pdf() — PDF 변환](#41-convert_to_pdf--pdf-변환)
   - 4.2 [HWP 폴백 체인](#42-hwp-폴백-체인)
   - 4.3 [주요 클래스 요약](#43-주요-클래스-요약)
5. [출력 데이터 구조](#5-출력-데이터-구조)
6. [예외 / 트러블슈팅](#6-예외--트러블슈팅)
- [부록: 코드 내부 상세](#부록-코드-내부-상세)

---

## 1. 개요

첨부용 전처리기는 사용자가 채팅 중 업로드하는 파일을 **실시간**으로 분석합니다. AI 기반 레이아웃 분석(Layout Detection)을 생략하고 **텍스트 추출**에 집중하여 즉각적인 응답 속도를 보장합니다.

### 설계 철학

```
"속도 중심: 다양한 포맷의 텍스트 즉시 추출"
```

### 대상 포맷

PDF, HWP/HWPX, DOCX/DOC, PPT/PPTX, CSV/XLSX, 이미지(JPG/PNG), 텍스트(TXT/JSON/MD), HTML, 오디오(MP3/WAV/M4A).

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **Native 텍스트 추출** | HWP, HWPX, DOCX 등 원본 파일의 텍스트를 직접 파싱 |
| **멀티미디어 지원** | MP3, WAV, M4A 등 오디오 파일의 음성→텍스트 변환(STT) |
| **데이터 변환** | CSV, Excel 등 정형 데이터를 LLM이 이해하기 쉬운 형태로 변환 |
| **설정 기반 제어** | 청킹/SDK/STT 등 동작을 config yaml로 조정 (코드 수정 불필요) |

> 고품질 구조 분석이 필요하면 `intelligent_processor`, PDF 표준화가 필요하면 `convert_processor`를 사용하세요. 첨부용에는 enrichment/layout/OCR 파이프라인이 없습니다.

---

## 2. 빠른 시작

### 등록 흐름

1. **Genos UI**: 전처리기 facade(`attachment_processor.py`)를 등록합니다.
2. **resource**: 같은 위치에 `attachment_processor_config.yaml`을 등록합니다.
3. 전처리기는 무인자(`DocumentProcessor()`)로 생성되며, 생성 시 config yaml을 읽어 동작 기본값을 구성합니다.

### config 로딩 우선순위

`DocumentProcessor()`가 무인자로 생성될 때 설정 파일을 다음 우선순위로 찾습니다.

```
1순위) resource_dev/attachment_processor_config.yaml   (있으면 우선 사용 — 개발/사이트 오버라이드)
2순위) resource/attachment_processor_config.yaml       (공개 기본본)
3순위) 내장 기본값                                       (파일 없음/형식 오류 시, 경고 로그 후 동작)
```

파일이 없거나 형식이 잘못돼도 예외를 던지지 않고 경고 로그를 남긴 뒤 내장 기본값으로 동작합니다.

### 기본값 사용 vs 사이트별 변경

- **기본값 그대로 사용 가능**: 청킹 크기, SDK 사용 여부, OCR 언어 등 대부분의 항목은 기본값으로 일반 문서를 처리할 수 있습니다.
- **사이트별로 반드시 바꿔야 하는 항목**: `whisper.url` 의 `<WHISPER_ENDPOINT>` 는 placeholder이므로, 오디오 STT를 사용하려면 실제 음성인식 서버 주소로 변경해야 합니다.

---

## 3. 설정 (attachment_processor_config.yaml)

전처리기 동작을 제어하는 중심 파일입니다. 아래 스키마와 표를 기준으로 옵션을 조정합니다.

### 3.1 전체 스키마

```yaml
# attachment_processor 기본 설정
# DocumentProcessor() 무인자 호출 시 이 파일을 우선 로드한다.

defaults:
  # 5=DEBUG, 4=INFO, 3=WARNING, 2=ERROR, 1=CRITICAL, 0=NOLOG
  log_level: 4

  # 입력 포맷 자동 PDF 변환 엔진 선택 (ppt/txt/md/img→pdf, hwp 폴백 등 전 변환 경로 공용)
  use_pdf_sdk: true

# 포맷별 옵션 컨테이너
formats:
  # HWP/HWPX: SDK 백엔드 파싱 옵션
  hwp:
    # SDK 백엔드 사용 여부 (false=레거시 backend 로 폴백)
    use_hwp_sdk: true
    # use_hwp_sdk=true 일 때만 유효한 SDK 출력 덤프(디버그)
    dump_sdk_output: false
    # 이미지 저장 여부
    save_images: true

  # PPT(ppt/pptx) 페이지 설명(VLM). 상세는 3.2 formats.ppt.page_description 참고
  ppt:
    page_description:
      enable: false
      # ... (아래 formats.ppt 소절 참고)

chunking:
  # 청킹 모드: "recursive"(문자수 기반, 기본) | "hybrid"(layout 구조 기반; hwp/hwpx/docx 에만 적용)
  chunker_type: "recursive"

  # 청크 크기(공통): recursive 모드=문자 수 · hybrid 모드=토큰 수. 0=크기 기반 분할 안 함(전체 1청크)
  chunk_size: 1000000

  # 문자수 기반 청킹 (전 포맷 공용: pdf/txt/md/img/ppt/hwp/hwpx/docx). chunker_type: recursive 일 때.
  recursive:
    chunk_overlap: 100

  # layout 구조 기반 청킹 (선택 모드). chunker_type: hybrid 일 때 hwp/hwpx/docx 에만 적용.
  # (토크나이저는 아래 chunking.tokenizer_path/tokenizer_id 를 공용으로 사용)
  hybrid:
    # 토큰 수 계산 방식. "char"(default)=문자 수 기준 | "huggingface"=HF 토크나이저 기준
    tokenizer_type: "char"
    merge_peers: true

  # 청킹용 토크나이저 기본값(hybrid huggingface 모드 등). tokenizer_path 가 실제 존재하면 그 경로,
  # 없으면 tokenizer_id(HF) 로 폴백 (외부 네트워크 차단 환경 대비)
  tokenizer_path: "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
  tokenizer_id: "sentence-transformers/all-MiniLM-L6-v2"

loaders:
  image:
    # 이미지 OCR 언어
    ocr_languages: ["kor", "eng"]

  tabular:
    # CSV 인코딩 감지 시 샘플링 바이트 수
    encoding_detect_sample_bytes: 10000

# 음성 파일(.wav/.mp3/.m4a) 처리 기본값
# <WHISPER_ENDPOINT>: 음성인식 모델 서버 주소로 변경 필요
whisper:
  url: "http://<WHISPER_ENDPOINT>/v1/audio/transcriptions"
  model: "model"
  language: "ko"
  response_format: "json"
  temperature: "0"
  stream: "false"
  timestamp_granularities: "word"
  chunk_sec: 29
  chunk_overlap_ms: 300
  # 끝에 / 를 주면 디렉터리로 간주해 내부에 파일명 기준 하위 폴더 생성
  tmp_dir_prefix: "./tmp_audios_"
```

### 3.2 섹션별 상세

모든 기본값과 동작은 코드(`DocumentProcessor.__init__`)에서 검증되었습니다. 값이 누락되거나 무효하면 표의 기본값으로 폴백합니다.

#### defaults (전역)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `log_level` | `4` (INFO) | 로깅 레벨. 5=DEBUG, 4=INFO, 3=WARNING, 2=ERROR, 1=CRITICAL, 0=NOLOG |
| `use_pdf_sdk` | `true` | 입력 포맷 자동 PDF 변환 시 SDK 엔진 사용 여부 (PPT/DOC/이미지/HWP 폴백 변환에 영향, 전 변환 경로 공용) |

#### formats.hwp (HWP/HWPX 전용)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `use_hwp_sdk` | `true` | hwp/hwpx 처리 시 기본 SDK 백엔드(`GenosHwpDocumentBackend`) 사용 여부. `false`면 레거시 backend로 폴백 |
| `dump_sdk_output` | `false` | SDK 원본 출력 덤프 여부(디버그). `use_hwp_sdk=true` 일 때만 유효 |
| `save_images` | `true` | hwp/hwpx 처리 시 추출 이미지 저장 여부 |

> 구버전 config 호환: `formats.hwp` 가 없으면 예전 위치(`defaults.use_hwp_sdk` 등)를 폴백으로 읽습니다.

#### chunking

| 키 | 기본값 | 설명 |
|----|--------|------|
| `chunker_type` | `"recursive"` | 청킹 모드. `recursive`(문자수 기반, 전 포맷 공용) 또는 `hybrid`(layout 기반, hwp/hwpx/docx 에만 적용). 그 외 값은 `recursive`로 폴백. 구버전 호환: 없으면 `defaults.chunker_type` 폴백 |
| `chunk_size` | `1000000` | **공통 청크 크기.** recursive 모드=문자 수, hybrid 모드=토큰 수로 해석. `0`(또는 음수)=크기 기반 분할 안 함 → 전체 문서를 1청크로 둠. recursive/hybrid 는 `chunker_type`으로 택일되므로 값 하나를 활성 모드가 자기 단위로 사용 |
| `tokenizer_path` | `/models/...all-MiniLM-L6-v2` | 청킹용 토크나이저 로컬 경로(hybrid `huggingface` 모드에서 사용). 경로가 실제 존재하면 그 경로 사용 |
| `tokenizer_id` | `sentence-transformers/all-MiniLM-L6-v2` | `tokenizer_path` 가 없을 때 폴백할 HF ID (외부 네트워크 차단 환경 대비) |

> `chunk_size`/`chunker_type`/`tokenizer_*` 는 하위 블록이 아닌 `chunking` 공통 레벨에 둡니다. per-block 로 `chunking.recursive.chunk_size` / `chunking.hybrid.chunk_size` 를 지정하면 해당 모드에 한해 공통값을 덮어씁니다(선택).

#### chunking.recursive (문자수 기반, 전 포맷 공용 — 기본 모드)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `chunk_overlap` | `100` | 청크 간 오버랩(문자 단위). 음수면 100으로 폴백 |

> 청크 크기는 공통 `chunking.chunk_size`(문자 수)로 지정합니다. `0`이면 문자 분할 없이 전체를 1청크로 둡니다.

#### chunking.hybrid (layout 기반 — hwp/hwpx/docx 전용 선택 모드)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `tokenizer_type` | `"char"` | 토큰 수 계산 방식. `char`(기본)=문자 수 기준(HF 토크나이저 미로드, 외부 모델 의존 없음) / `huggingface`=HF 토크나이저 기준. 그 외 값은 `char`로 폴백 |
| `merge_peers` | `true` | 같은 제목/캡션을 가진 작은 청크를 크기 제한 내에서 병합 |

> hybrid 의 청크 크기(토큰 수)도 공통 `chunking.chunk_size` 로 지정합니다. 기본 `1000000`은 사실상 제한이 없어 hybrid를 레이아웃 기반 병합 도구로만 쓰게 됩니다. `huggingface` 모드 토크나이저는 위 `chunking.tokenizer_path/tokenizer_id` 를 공용으로 사용합니다.

#### loaders.image

| 키 | 기본값 | 설명 |
|----|--------|------|
| `ocr_languages` | `["kor", "eng"]` | 이미지(JPG/PNG) OCR 언어 목록. 비거나 무효하면 `["kor","eng"]`로 폴백 |

#### loaders.tabular

| 키 | 기본값 | 설명 |
|----|--------|------|
| `encoding_detect_sample_bytes` | `10000` | CSV 인코딩 감지 시 chardet에 넘길 샘플 바이트 수. 0 이하면 10000으로 폴백 |

#### whisper (오디오 STT)

| 키 | 기본값 | 설명 |
|----|--------|------|
| `url` | `"http://<WHISPER_ENDPOINT>/v1/audio/transcriptions"` | STT 서버 엔드포인트. **placeholder이므로 사이트별로 실제 주소로 변경 필요** |
| `model` | `"model"` | STT 요청 모델명 |
| `language` | `"ko"` | 전사 언어 |
| `response_format` | `"json"` | 응답 포맷 |
| `temperature` | `"0"` | 디코딩 temperature |
| `stream` | `"false"` | 스트리밍 여부 |
| `timestamp_granularities` | `"word"` | 타임스탬프 단위 (요청 시 `timestamp_granularities[]`로 전달) |
| `chunk_sec` | `29` | 오디오 분할 단위(초). 0 이하면 29로 폴백 |
| `chunk_overlap_ms` | `300` | 청크 간 오버랩(ms). 단어 잘림 방지. 음수면 300으로 폴백 |
| `tmp_dir_prefix` | `"./tmp_audios_"` | 임시 분할 파일 저장 prefix. 끝에 `/`를 주면 디렉터리로 간주해 파일명 기준 하위 폴더 생성 |

#### chunker_type: recursive vs hybrid

- **recursive (기본, 전 포맷 공용)**: 문자 수 기반 청킹입니다. hwp/hwpx/docx 는 `DoclingDocument`를 markdown으로 export한 뒤, pdf/txt/md/img 는 로드된 텍스트를 대상으로 `RecursiveCharacterTextSplitter`로 **문자 단위** 분할합니다(공통 `chunk_size` 문자 수). 단락/헤딩/표 경계의 줄바꿈이 보존되어 문장 중간에서 잘리는 현상이 적고, 페이지 단위 정확도를 가집니다. 대부분의 경우 권장됩니다.
- **`chunk_size=0` → 전체 1청크**: 공통 `chunk_size` 가 `0`(또는 음수)이면 문자 분할을 하지 않고 문서 전체를 한 청크로 둡니다(PPT 는 예외적으로 "1 page = 1 chunk" 유지). 검색 정밀도를 위해 문서를 잘게 나누려면 `chunk_size` 를 적정 문자 수(예: 2000)로 지정하세요.
- **hybrid (hwp/hwpx/docx 전용)**: 레이아웃 계층 구조를 유지하며 청크 크기 기반으로 청크를 조절합니다. layout 파싱이 되는 hwp/hwpx/docx 에만 적용되고, pdf/txt/md/img/ppt 는 `chunker_type: hybrid` 로 두어도 각자 문자수/페이지 기반으로 동작합니다. 크기 계산 방식은 `chunking.hybrid.tokenizer_type`(기본 `char`=문자 수 / `huggingface`=HF 토큰 수)으로 선택하며, 크기는 공통 `chunk_size`(토큰 수)로 지정합니다. 기본값(`1000000`)은 사실상 제한이 없어 주로 레이아웃 기반 병합 도구로 동작합니다. 청크 단위 bbox 정확도가 필요할 때 사용합니다.

> **임베딩 토큰 상한 주의**: 이전 버전의 60K 토큰 강제 재분할(안전망)은 제거되었습니다. `chunk_size` 가 매우 크거나 `0`(전체 1청크)이면 한 청크가 임베딩 모델 입력 한도를 초과할 수 있습니다. 대용량/토큰과다 문서는 `chunk_size` 를 적정 값으로 지정해 방지하세요.

#### formats.ppt.page_description (PPT 페이지 설명 — 속도 최적화)

PPT(`.ppt`/`.pptx`)를 **PDF→경량 docling 파싱** 후 각 페이지를 **이미지로 렌더링해 VLM 으로 설명**하고, 페이지의 native text 와 **동일 청크**로 구성합니다. 첨부용은 즉시 응답이 목적이라 **속도 우선**으로 튜닝합니다(이미지·출력 축소, 병렬↑, 간결 프롬프트).

- **청킹**: **1 page = 1 chunk**. 공통 `chunking.chunk_size` 가 `>0` 이면 연속 페이지를 그 크기(문자 수)까지 결합합니다(`0`=페이지별 1청크 고정). bbox 는 첨부용 원칙대로 미추출.
- **enable=false** 여도 PPT 는 docling 경량 파싱으로 처리되며, 설명 생성만 생략됩니다(텍스트/설명이 모두 없는 페이지는 `.` 로 채워 Empty 예외 방지).
- native text 는 프롬프트의 **`{{page_text}}`** 변수로 전달됩니다.

```yaml
formats:
  ppt:
    page_description:
      enable: true
      url: "http://.../rep/serving/<PAGE_DESCRIPTION_SERVING_ID>/v1/chat/completions"
      api_key: ""
      model: "model"
      timeout: 360
      concurrency: 32          # 병렬↑ (VLM 서버 처리량 한도 내)
      images_scale: 1.0        # 렌더 배율↓ (payload·image token↓)
      max_image_side: 1536     # 전송 전 이미지 최대 변(px) 캡. 0=원본
      max_tokens: 512          # 출력 토큰 상한(생성 시간↓). 0=상한 없음
      # params: { max_completion_tokens: 512 }   # 서버가 다른 키를 쓰면 여기로
      prompt_template_file: prompt_page_image_description_fast.md   # 간결(2~3문장) 프롬프트
```

| 키 | 의미 | 기본값(첨부 권장) |
|----|------|--------|
| `enable` | PPT 페이지 설명 활성화 | `false`(운영 시 `true`) |
| `url` / `api_key` / `model` | VLM 서빙 endpoint / 키 / 모델명 | `<PAGE_DESCRIPTION_SERVING_ID>` / `""` / `model` |
| `timeout` | VLM 요청 타임아웃(초) | `360` |
| `concurrency` | 페이지 설명 병렬 요청 수 | `32` |
| `images_scale` | 페이지 렌더 배율(작을수록 빠름) | `1.0` |
| `max_image_side` | 전송 전 이미지 최대 변(px). `0`=원본 | `1536` |
| `max_tokens` | VLM 출력 토큰 상한. `0`=상한 없음 | `512` |
| `params` | 추가 VLM 파라미터(dict) | `{}` |
| `prompt_template_file` | 프롬프트 `.md` 경로. `{{page_text}}` 치환 | `prompt_page_image_description_fast.md` |

> **속도 방법 요약**: (1) `concurrency`↑ 로 페이지 병렬 처리, (2) `images_scale`↓ + `max_image_side` 로 payload·image token↓, (3) `max_tokens` + 간결 프롬프트로 생성 시간↓.

### 3.3 런타임 kwargs override

config yaml의 기본값은 `DocumentProcessor.__init__`에서 `self._default_kwargs`로 로드됩니다. 호출 시 `params`(kwargs)로 동일 항목을 넘기면 **해당 호출에 한해 기본값을 덮어씁니다**. 병합 규칙은 `_merge_runtime_kwargs`로, **런타임 값 중 `None`이 아닌 것만** 기본값을 덮어씁니다.

자주 쓰는 override kwargs:

| kwargs | 대응 config | 비고 |
|--------|-------------|------|
| `log_level` | `defaults.log_level` | |
| `chunker_type` | `chunking.chunker_type` | |
| `use_pdf_sdk` | `defaults.use_pdf_sdk` | |
| `use_hwp_sdk` | `formats.hwp.use_hwp_sdk` | |
| `chunk_size` | `chunking.chunk_size` (공통) | 활성 모드 단위(recursive=문자 수 / hybrid=토큰 수). 미지정 시 config 기본값 사용 |
| `chunk_overlap` | `chunking.recursive.chunk_overlap` | 미지정 시 config 기본값 사용 |

> `chunk_size`/`chunk_overlap`은 모든 문자수 청킹 경로(pdf/txt/md/img/ppt + hwp/hwpx/docx recursive)에 공통 적용되고, 미지정 시 `chunking.chunk_size` / `chunking.recursive.chunk_overlap` 기본값으로 폴백합니다.

### 3.4 자주 쓰는 튜닝 시나리오

**① 청크 크기 조정** — 더 작은 청크로 검색 정밀도를 높이고 싶을 때 (공통 `chunk_size`):

```yaml
chunking:
  chunk_size: 2000        # 문자 수 (recursive 모드). 0=전체 문서를 1청크
  recursive:
    chunk_overlap: 200
```

**② hybrid 청커 사용** — hwp/hwpx/docx 에서 청크 단위 bbox 정확도가 필요할 때:

```yaml
chunking:
  chunker_type: "hybrid"
  chunk_size: 1000        # hybrid 모드에서는 토큰 수. 실제 분할 제한을 두려면 작은 값 지정
  hybrid:
    tokenizer_type: "char"  # char=문자 수 기준 | huggingface=HF 토큰 수 기준
    merge_peers: true
```

**③ whisper STT 서버 지정** — 오디오 처리를 활성화할 때:

```yaml
whisper:
  url: "http://10.0.0.5:30100/v1/audio/transcriptions"
  language: "ko"
```

**④ HWP SDK 비활성화** — 엔터프라이즈 SDK 없이 레거시 백엔드로 처리할 때:

```yaml
formats:
  hwp:
    use_hwp_sdk: false
```

---

## 4. 처리 동작 개요 (보조)

설정만으로 충분한 경우 이 섹션은 건너뛰어도 됩니다. 동작 원리를 이해하고 싶을 때 참고하세요.

### 확장자별 분기 아키텍처

```
사용자가 파일 업로드
        │
        ▼
 ┌──────────────────┐
 │ DocumentProcessor│  ◄── 메인 엔트리포인트 (__call__)
 │   (라우터 역할)   │      진입 시 _merge_runtime_kwargs로 config+kwargs 병합
 └──────┬───────────┘
        │  확장자(ext)에 따라 분기
        │
        ├── .wav/.mp3/.m4a ──────► AudioLoader ──► STT(whisper.*) ──► GenOSVectorMeta
        │
        ├── .csv/.xlsx ──────────► TabularLoader ──► DataFrame 파싱 ──► GenOSVectorMeta
        │
        ├── .hwp/.hwpx ─────────► HwpProcessor ──┬──► recursive (기본)
        │                         (폴백 체인 4.2)  └──► hybrid (chunker_type='hybrid')
        │
        ├── .docx ───────────────► DocxProcessor ──┬──► recursive (기본)
        │                                          └──► hybrid
        │
        └── 기타 (.pdf, .ppt,    ► get_loader() ──► LangChain Loader
            .doc, .jpg, .txt,                          │
            .json, .md, .html 등)                      ▼
                                    문자수 청킹(recursive, 공통 chunk_size)
                                    (PPT 는 페이지 기반: 1 page=1 chunk)
                                                       │
                                                       ▼
                                              ┌─────────────────────────────┐
                                              │  List[GenOSVectorMeta]       │
                                              └─────────────────────────────┘
```

<a id="41-convert"></a>
### 4.1 convert_to_pdf() — PDF 변환

PPT/DOC/이미지/HWP 등을 PDF로 변환합니다. 실패해도 예외 없이 `None`을 반환하는 방어적 설계이며, 실제 변환 로직은 `genon.preprocessor.converters.hwp_to_pdf` 모듈로 일원화되어 있습니다. 변환 chain은 입력 확장자와 `use_pdf_sdk`(config `defaults.use_pdf_sdk`)로 결정되며, 앞 backend 실패 시 다음으로 자동 fallback합니다.

| 입력 | `use_pdf_sdk=true` | `use_pdf_sdk=false` |
|------|--------------------|---------------------|
| `.hwp` / `.hwpx` | `pdf_sdk → libreoffice → rhwp` | `libreoffice → rhwp` |
| 그 외 (`.docx`/`.pptx`/이미지 등) | `pdf_sdk → libreoffice` | `libreoffice` |

- `rhwp`는 HWP/HWPX 전용이라 비-HWP chain에는 포함되지 않으며, 안정성 검증 전까지 최후순위 fallback으로만 둡니다.
- backend별 가용성: `pdf_sdk`(엔터프라이즈 자산, 실행권한 시 활성), `libreoffice`(`soffice` 존재 여부), `rhwp`(`RHWP_BIN` 바이너리).

### 4.2 HWP 폴백 체인

HWP/HWPX 파일은 변환 실패 시 단계적으로 폴백합니다.

```
.hwp / .hwpx 입력
        │
        ├─[use_hwp_sdk=true]──► ① GenosHwpDocumentBackend ──── 성공 ──► 청킹(recursive/hybrid)
        │                                   │ 실패
        │                                   ▼
        └─[use_hwp_sdk=false]─► ② .hwp  → HwpDocumentBackend ─ 성공 ──► 청킹(recursive/hybrid)
                                   .hwpx → HwpxDocumentBackend
                                           │ 실패 (전체 docling 백엔드 실패)
                                           ▼
                                ③ PDF 변환 (HWP: pdf_sdk→libreoffice→rhwp) ─ 성공 ──► 일반 분기 처리
                                           │ 실패
                                           ▼
                                        에러 반환
```

<a id="82-hwpprocessor"></a>
### 4.3 주요 클래스 요약

| 클래스/함수 | 역할 | config 연관 |
|-------------|------|-------------|
| `DocumentProcessor` | 메인 엔트리포인트. config 로딩 + 확장자 라우팅 + generic 분기 처리 | 전체 |
| `HwpProcessor` | hwp/hwpx 통합 처리 (SDK 백엔드 + 폴백) | `formats.hwp.{use_hwp_sdk,dump_sdk_output,save_images}`, `chunking.*` |
| `DocxProcessor` | docx Docling 파싱 + 청킹 | `chunking.*` |
| `AudioLoader` | 오디오 분할 + Whisper STT 병렬 호출 | `whisper.*` |
| `TabularLoader` | CSV/XLSX → DataFrame 파싱, `[DA]` 마커 부착 | `loaders.tabular.encoding_detect_sample_bytes` |
| `TextLoader` | txt/json/md 인코딩 자동 감지 로드 | — |
| `HierarchicalChunker` / `HybridChunker` | 레이아웃 기반/토큰 기반 청킹 | `chunking.chunk_size`, `chunking.hybrid.*` |
| `_char_split_text` / `_split_with_recursive_chunker` | 문자수 청킹 공용 헬퍼(전 포맷) / recursive 분기(markdown export + 문자 분할) | `chunking.chunk_size`, `chunking.recursive.chunk_overlap` |

---

## 5. 출력 데이터 구조

모든 처리 경로는 청크당 하나의 `GenOSVectorMeta` 객체를 생성하며, 결과는 `List[GenOSVectorMeta]`입니다. 이 객체가 벡터 DB에 저장됩니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `text` | str | 청크의 실제 텍스트 내용 |
| `n_char` | int | 텍스트의 문자 수 |
| `n_word` | int | 텍스트의 단어 수 |
| `n_line` | int | 텍스트의 줄 수 |
| `i_page` | int | 청크가 시작하는 페이지 번호 |
| `e_page` | int | 청크가 끝나는 페이지 번호 |
| `i_chunk_on_page` | int | 해당 페이지 내 청크 순서 (0부터) |
| `n_chunk_of_page` | int | 해당 페이지의 전체 청크 수 |
| `i_chunk_on_doc` | int | 문서 전체에서의 청크 순서 (0부터) |
| `n_chunk_of_doc` | int | 문서 전체의 청크 수 |
| `n_page` | int | 문서의 전체 페이지 수 |
| `reg_date` | str | 처리 일시 (ISO 8601) |
| `chunk_bboxes` | str | 청크 위치 정보 (정규화된 bbox, JSON 문자열) |
| `media_files` | str | 연관 미디어 파일 정보 (JSON 문자열) |
| `pii_status` | str | PII 마스킹 상태(청크별): `none`/`masked`/`exposed`/`unknown`. `guardrail_masking` on 일 때만 채워짐 (아래 PII 마스킹 절) |

> `extra = 'allow'`로 정의되어 추가 필드도 허용됩니다. 일반 분기(PDF/TXT 등)와 CSV/오디오 경로는 bbox/media 추출을 생략(속도 우선)하며, `chunk_bboxes`/`media_files`는 빈 값/플레이스홀더로 채워집니다. HWP/DOCX 경로는 doc_items 기반으로 실제 bbox/media를 채웁니다.

### PII 마스킹 (개인정보 비식별화, `guardrail_masking`)

문서 텍스트의 개인정보를 GenOS Guardrail 에 위임해 마스킹합니다(청킹 직전 문서당 1회 호출, 청크별 `pii_status` 기록).

- **켜기**: 요청 kwargs `guardrail_masking: true` (기본 `false`). yaml 아님, 업로드 건별 제어.
- **접속 정보 (yaml)**:
  ```yaml
  guardrail_masking:
    url: ""           # 예: "http://<내부 gateway>/api/gateway"
    guardrail_id:     # 가드레일 인스턴스 ID(단일)
    timeout: 30
  ```
- **적용 범위 (attachment)**:
  - **docx / hwp·hwpx** (docling 경로): 아이템 단위 마스킹 + 아이템→청크 `pii_status` 집계.
  - **그 외 langchain 경로**(PDF/TXT/이미지 등, PPT·HWP PDF 폴백 포함): 페이지 단위 마스킹 + **페이지 단위** `pii_status`(같은 페이지의 여러 청크는 같은 상태를 공유 — 과다표시 가능, 안전 방향).
  - **오디오(wav/mp3/m4a)·tabular(csv/xlsx)**: 현재 **미적용(보류)** — 자체 vector 포맷이라 별도 논의 예정.
- **출력**: `pii_status` = `none`/`masked`/`exposed`/`unknown` (기능 off 면 필드 없음).
- **실패 시**: fail-open — 원문 통과 + warning + `pii_status=unknown`.
- **운영 주의**: 마스킹용 가드레일에는 "마스킹 처리하여 제공" 필터만 구성하세요.

> 동작·응답 형태·`pii_status` 값 정의·한계의 **상세 설명은 지능형 전처리기 매뉴얼의 "PII 마스킹" 절**을 참고하세요. 4개 전처리기 공통 동작입니다.

---

## 6. 예외 / 트러블슈팅

| 증상 | 원인 | 대응 |
|------|------|------|
| `chunk length is 0` (`GenosServiceException`) | HWP/DOCX에서 추출된 청크가 0개 | 원본 파일이 비었거나 파싱 실패. 다른 백엔드(`use_hwp_sdk` 변경)나 PDF 폴백 확인 |
| 오디오 전사 결과 비어있음 | `whisper.url`이 placeholder(`<WHISPER_ENDPOINT>`) | config에서 실제 STT 서버 주소로 변경 |
| HWP 변환 실패 후 에러 | SDK/레거시 백엔드/PDF 변환 모두 실패 | `formats.hwp.use_hwp_sdk`, `defaults.use_pdf_sdk` 조정. LibreOffice(`soffice`) 설치 확인 |
| 청크가 비정상적으로 큼 / 임베딩 입력 초과 | 기본 `chunk_size` 가 크거나 `0`(전체 1청크)이라 한 청크가 임베딩 토큰 한도 초과 (60K 토큰 안전망 제거됨) | `chunking.chunk_size` 를 적정 문자 수(예: 2000~8000)로 지정 |
| 한글 텍스트 깨짐 (CSV/TXT) | 인코딩 감지 실패 | TXT는 다단 인코딩 폴백 사용. CSV는 `encoding_detect_sample_bytes` 증대 |
| 로그가 너무 많음/적음 | `log_level` 설정 | config `defaults.log_level` 조정 (4=INFO 기본) |
| config가 적용 안 됨 | `resource_dev` 파일이 우선 로드됨 | 로딩 우선순위([2장](#2-빠른-시작)) 확인. 의도치 않은 `resource_dev` 파일 제거 |

---

## 부록: 코드 내부 상세

> 보조 참고용입니다. 일상적인 운영/튜닝에는 [3장 설정](#3-설정-attachment_processor_configyaml)만으로 충분합니다. 아래는 고급 사용자를 위한 내부 동작 요약입니다.

### A. 설정 로딩 (`DocumentProcessor.__init__`)

생성 시 `_resolve_default_attachment_config_path()`로 config 경로를 결정(`resource_dev` → `resource`)한 뒤 `_load_config()`로 읽습니다. 각 섹션(`defaults`/`formats`/`chunking`/`loaders`/`whisper`)은 `_parse_optional_bool/_parse_optional_int` 등으로 검증되어 `self._default_kwargs`에 평탄화됩니다. `chunker_type`·공통 `chunk_size` 는 `chunking` 에서, HWP 옵션은 `formats.hwp` 에서 읽습니다(각각 구버전 위치 `defaults.*` 폴백 지원). 공통 `chunk_size` 는 `recursive_chunk_size`/`hybrid_chunk_size` 양쪽 소스로 쓰이고, 문자 오버랩은 `recursive_chunk_overlap` 로 저장됩니다. 무효 값은 표의 기본값으로 폴백하고 경고 로그를 남깁니다.

### B. 라우팅 (`DocumentProcessor.__call__`)

`_merge_runtime_kwargs(kwargs)`로 config 기본값과 런타임 kwargs를 병합(None 아닌 값만 override)한 뒤 `setup_logging(log_level)`을 호출합니다. 이후 확장자로 분기합니다.

- 오디오: `whisper_tmp_dir_prefix`로 임시 폴더 생성 → `AudioLoader` → 처리 후 임시 폴더 삭제
- 정형: `tabular_encoding_detect_sample_bytes`를 넘겨 `TabularLoader` 호출
- hwp/hwpx: `HwpProcessor` 호출, 실패 시 `convert_to_pdf` 후 일반 분기로 최종 폴백
- docx: `DocxProcessor` 호출
- 기타: `get_loader` → `split_documents`(generic) → `compose_vectors`

### C. 로더 선택 (`get_loader` / `get_real_file_type`)

`get_real_file_type`가 매직 바이트(`%PDF-`, `\x89PNG`, `\xff\xd8\xff`)로 실제 타입을 판정해, 확장자가 조작/손실된 경우에도 올바른 로더를 선택합니다.

| 확장자 | 로더 | 비고 |
|--------|------|------|
| `.pdf` | `PyMuPDFLoader` | 빠른 텍스트 추출 |
| `.doc` | `UnstructuredWordDocumentLoader` | PDF 변환 후 처리 |
| `.ppt`/`.pptx` | `UnstructuredPowerPointLoader` | PDF 변환 후 처리 |
| `.jpg`/`.jpeg`/`.png` | `UnstructuredImageLoader` | OCR (`loaders.image.ocr_languages`) |
| `.txt`/`.json` | `TextLoader` | 인코딩 자동 감지 |
| `.md` | `TextLoader`(우선) / `UnstructuredMarkdownLoader` | |
| 기타/`.html` | `UnstructuredFileLoader` | 범용 폴백 |

### D. 청킹 내부

- **문자수 청킹(`_char_split_text`)**: 공용 헬퍼. `chunk_size>0` 이면 `RecursiveCharacterTextSplitter`로 문자 단위 분할, `<=0` 이면 분할 없이 전체를 1청크로 둡니다. pdf/txt/md/img 는 로드된 텍스트에, hwp/hwpx/docx(recursive)는 markdown export 에 적용됩니다. (별도 토큰 상한 후처리는 없음 — 60K cap 제거됨)
- **recursive (`_split_with_recursive_chunker`)**: `export_to_markdown(page_break_placeholder="<!-- PB -->")`로 단일 markdown 생성 → `_char_split_text`(문자 단위) → placeholder 카운트로 페이지 매핑 복원. 반환은 `list[dict]{text, page_no, pages, doc_items}`. 한 청크가 여러 페이지에 걸칠 수 있어 페이지 단위 정확도를 가집니다.
- **hybrid (`HybridChunker`)**: `HierarchicalChunker`로 레이아웃 청크 생성 → `_split_by_doc_items`(슬라이딩 윈도우) → `_split_using_plain_text`(semchunk) → `merge_peers=true`면 동일 메타데이터 청크 병합. 청크 단위 bbox 정확도.
- 두 분기는 `compose_vectors`에서 `chunker_type`으로 다시 분기해 처리됩니다 (recursive=dict, hybrid=DocChunk).

### E. 출력 조립 (`GenOSVectorMetaBuilder`)

빌더 패턴으로 메타데이터를 조립합니다: `set_text`(n_char/word/line 자동 계산) → `set_page_info` → `set_chunk_index` → `set_global_metadata` → `set_chunk_bboxes`(0~1 정규화 좌표) → `set_media_files` → `build`. 일반 분기(`DocumentProcessor.compose_vectors`)는 빌더 대신 dict를 직접 구성하며 bbox/media를 생략합니다.

### F. HWP 수식(LaTeX) 추출

`GenosHwpDocumentBackend`는 SDK가 emit하는 수식을 `DocItemLabel.FORMULA` 노드로 변환합니다(별도 옵션 불필요). block 수식은 `$$...$$`, 표 셀 내 inline 수식은 `<math>...</math>`로 출력됩니다. SDK 출력의 base64 줄바꿈은 stream 파싱으로, 표 셀 임베드 `<latex value="..."/>`의 미escape `"`는 정규식 정규화로 대응합니다.

---

> **참고**: 이 전처리기는 속도를 최우선으로 설계되어 AI 기반 레이아웃 분석이나 TableFormer 같은 고급 기능은 포함하지 않습니다. 고품질 구조 분석은 `intelligent_processor`, PDF 표준화는 `convert_processor`를 사용하세요.
