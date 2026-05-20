# GenOS Document Intelligence 전처리 시스템 (개발중인 Facade 포맷)

## 📋 개요

GenOS DI(Document Intelligence)는 다양한 형식의 문서를 벡터 데이터베이스에 저장하기 위해 전처리하는 시스템입니다.
각 문서 타입과 요구사항에 따라 **지능형(Intelligent)** 또는 **기본형(Basic)** 처리 방식을 동적으로 선택할 수 있습니다.

## 🏗️ 시스템 구조

```
genos_di/
├── 핵심 전처리 파일
│   └── new_preprocess_configurable.py # 설정 가능한 전처리기 (확장자별 on/off)
│
├── 유틸리티
│   ├── genos_utils.py                 # 파일 업로드, bbox 병합 등
│   └── utils.py                       # 비동기 처리 유틸리티
│
├── 설정 파일
    └── processor_config.json          # 확장자별 처리 모드 및 기본값 설정

```

### Facade 패턴 구조 (docling/facade/)
```
docling/facade/
├── document_facade.py                 # 메인 Facade (설정 가능)
├── config/
│   └── processor_config.py            # 처리 모드 설정 관리
└── processors/
    ├── docling_processor.py           # PDF/HWPX 지능형 처리
    ├── audio_processor.py             # 오디오 처리 (Whisper)
    ├── tabular_processor.py           # CSV/XLSX 처리
    ├── langchain_processor.py         # 일반 문서 처리
    └── processor_factory.py           # 동적 프로세서 생성
```

## 🚀 DocumentProcessor 사용법

### 기본 사용법

```python
from new_preprocess_configurable import DocumentProcessor

# 프로세서 생성 (processor_config.json의 기본값 자동 로드)
processor = DocumentProcessor()

# 확장자별 모드 설정
processor.set_mode('pdf', 'intelligent')   # PDF를 지능형으로
processor.set_mode('docx', 'basic')        # DOCX를 기본형으로

# 문서 처리
vectors = await processor(request, 'document.pdf')
```

### 일괄 설정

```python
# 모든 확장자를 지능형으로
processor.set_all_intelligent()

# 모든 확장자를 기본형으로
processor.set_all_basic()

# 특정 확장자들만 지능형으로
processor.enable_intelligent_for(['pdf', 'hwpx', 'docx'])

# 특정 확장자들을 기본형으로
processor.disable_intelligent_for(['pptx', 'xlsx'])
```

### 설정 저장/로드

```python
# 현재 설정 저장
processor.save_config('my_config.json')

# 설정 파일 로드
new_processor = DocumentProcessor(config_file='my_config.json')
```

### 사전 설정 프로세서

```python
from new_preprocess_configurable import (
    create_intelligent_processor,  # 모두 지능형
    create_basic_processor,        # 모두 기본형
    create_hybrid_processor        # PDF/HWPX만 지능형 (권장)
)

# 하이브리드 프로세서 (추천)
processor = create_hybrid_processor()
```

## 🔧 세부 옵션 설정 및 기본값

### Enrichment 옵션 (PDF/HWPX) - 지능형 모드
```python
# 기본값이 이미 설정되어 있음
processor.set_enrichment_options('pdf',
    enabled=True,                      # 기본값: True
    do_toc_enrichment=True,           # 기본값: True
    extract_metadata=True,             # 기본값: True
    toc_extraction_mode='list_items',  # 기본값: 'list_items'
    toc_seed=33,                       # 기본값: 33
    toc_max_tokens=1000,               # 기본값: 1000
    toc_temperature=0.0,               # 기본값: 0.0
    toc_top_p=0                        # 기본값: 0
)

# API 설정 (필요시 변경)
processor.set_enrichment_options('pdf',
    toc_api_base_url="http://llmops-gateway-api-service:8080/serving/364/1073/v1/chat/completions",
    toc_api_key="your_api_key"
)

# Enrichment 비활성화
processor.disable_enrichment(['pdf', 'hwpx'])
```

### Pipeline 옵션 (PDF) - 지능형 모드
```python
# 기본값이 이미 설정되어 있음
processor.set_pipeline_options('pdf',
    do_ocr=False,                      # 기본값: False
    do_table_structure=True,           # 기본값: True
    generate_page_images=True,         # 기본값: True
    generate_picture_images=True,      # 기본값: True
    artifacts_path="/nfs-root/models/223/760",  # 기본값: 모델 경로
    table_structure_options={
        'do_cell_matching': True,      # 기본값: True
        'detect_headers': True          # 기본값: True
    }
)

# OCR 활성화 (필요시)
processor.enable_ocr(['pdf'])
```

### Chunking 옵션
```python
# Docling 기반 청킹 (PDF/HWPX) - 지능형 모드
processor.set_chunking_options('pdf',
    max_tokens=2000,                   # 기본값: 2000
    merge_peers=True,                  # 기본값: True
    tokenizer="/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"  # 기본값
)

# LangChain 기반 청킹 (기타 문서) - 기본형 모드
processor.set_chunking_options('docx',
    chunk_size=1000,                   # 기본값: 1000
    chunk_overlap=200                  # 기본값: 200
)
```

### Whisper 옵션 (오디오)
```python
# 기본값이 이미 설정되어 있음
processor.set_whisper_options('mp3',
    url="http://192.168.74.164:30100/v1/audio/transcriptions",  # 기본값
    model='model',                     # 기본값: 'model'
    language='ko',                     # 기본값: 'ko'
    temperature=0.2,                   # 기본값: 0.2
    chunk_sec=30,                      # 기본값: 30
    response_format='json'             # 기본값: 'json'
)
```

### 특정 옵션 경로 설정
```python
# 개별 옵션 직접 설정
processor.set_processor_option('pdf', 'enrichment.toc_seed', 42)
processor.set_processor_option('pdf', 'pipeline.do_ocr', True)
processor.set_processor_option('pdf', 'chunking.max_tokens', 1536)
```

## 📊 처리 모드 비교

| 모드 | 프로세서 | 장점 | 단점 | 사용 시기 |
|------|----------|------|------|-----------|
| **지능형** | Docling | • AI 기반 구조 분석<br>• 테이블/이미지 처리<br>• Enrichment 지원 | • 처리 시간 김<br>• 리소스 많이 사용 | 고품질 처리 필요시 |
| **기본형** | LangChain | • 빠른 처리<br>• 다양한 형식 지원<br>• 안정적 | • 단순 텍스트 추출<br>• 구조 정보 제한적 | 대량 처리시 |

## 🔧 processor_config.json 완전한 기본값 구조

```json
{
  ".pdf": {
    "mode": "intelligent",
    "processor": "docling",
    "description": "PDF with Docling + Enrichment",
    "options": {
      "enrichment": {
        "enabled": true,
        "do_toc_enrichment": true,
        "extract_metadata": true,
        "toc_extraction_mode": "list_items",
        "toc_seed": 33,
        "toc_max_tokens": 1000,
        "toc_temperature": 0.0,
        "toc_top_p": 0,
        "toc_api_provider": "custom",
        "toc_api_base_url": "http://llmops-gateway-api-service:8080/serving/364/1073/v1/chat/completions",
        "metadata_api_base_url": "http://llmops-gateway-api-service:8080/serving/364/1073/v1/chat/completions",
        "toc_api_key": "a2ffe48f40ab4cf9a0699deac1c0cb76",
        "metadata_api_key": "a2ffe48f40ab4cf9a0699deac1c0cb76",
        "toc_model": "/model/snapshots/9eb2daaa8597bf192a8b0e73f848f3a102794df5"
      },
      "pipeline": {
        "do_ocr": false,
        "do_table_structure": true,
        "generate_page_images": true,
        "generate_picture_images": true,
        "artifacts_path": "/nfs-root/models/223/760",
        "table_structure_options": {
          "do_cell_matching": true,
          "detect_headers": true
        }
      },
      "chunking": {
        "max_tokens": 2000,
        "merge_peers": true,
        "tokenizer": "/models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2"
      }
    }
  },
  ".mp3": {
    "mode": "basic",
    "processor": "audio",
    "description": "Audio transcription with Whisper",
    "options": {
      "whisper": {
        "url": "http://192.168.74.164:30100/v1/audio/transcriptions",
        "model": "model",
        "language": "ko",
        "temperature": 0.2,
        "chunk_sec": 30,
        "response_format": "json"
      },
      "text_splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 200
      }
    }
  },
  ".docx": {
    "mode": "basic",
    "processor": "langchain",
    "description": "Word document with LangChain",
    "options": {
      "text_splitter": {
        "chunk_size": 1000,
        "chunk_overlap": 200
      }
    }
  }
}
```

## 📁 지원 파일 형식

### 문서
- **PDF** (.pdf) - 지능형/기본형 선택 가능
- **HWPX** (.hwpx) - 지능형/기본형 선택 가능
- **Word** (.doc, .docx) - 기본형 (향후 지능형 확장 가능)
- **PowerPoint** (.ppt, .pptx) - 기본형
- **HWP** (.hwp) - 기본형
- **Text** (.txt) - 기본형
- **Markdown** (.md) - 기본형
- **JSON** (.json) - 기본형

### 오디오
- **MP3** (.mp3) - Whisper 전사
- **M4A** (.m4a) - Whisper 전사
- **WAV** (.wav) - Whisper 전사

### 테이블
- **CSV** (.csv) - 테이블 처리
- **Excel** (.xlsx) - 테이블 처리

## 💡 사용 시나리오

### 1. 고품질 문서 처리 (연구/분석)
```python
from new_preprocess_configurable import create_intelligent_processor

processor = create_intelligent_processor()  # 모든 파일 지능형
# 모든 옵션은 기본값 사용
vectors = await processor(request, 'research_paper.pdf')
```

### 2. 대량 문서 처리 (아카이빙)
```python
from new_preprocess_configurable import create_basic_processor

processor = create_basic_processor()  # 모든 파일 기본형
# 빠른 처리를 위한 기본값 사용
vectors = await processor(request, 'document.pdf')
```

### 3. 하이브리드 처리 (균형, 권장)
```python
from new_preprocess_configurable import create_hybrid_processor

processor = create_hybrid_processor()  # PDF/HWPX만 지능형
# 최적의 기본값 조합
vectors = await processor(request, 'document.pdf')
```

### 4. 커스텀 설정
```python
processor = DocumentProcessor()

# 문서 타입별 최적화 (기본값에서 필요한 부분만 변경)
processor.set_mode('pdf', 'intelligent')    # 중요 문서
processor.set_mode('pptx', 'basic')         # 프레젠테이션

# PDF에 대해 특정 옵션만 변경 (나머지는 기본값 유지)
processor.set_enrichment_options('pdf', toc_max_tokens=3000)  # 더 긴 목차
processor.enable_ocr(['pdf'])  # OCR 활성화

processor.save_config('project_config.json')
```

## 🔄 처리 흐름

```mermaid
graph TD
    A[문서 입력] --> B{확장자 확인}
    B --> C{처리 모드}

    C -->|지능형 + PDF/HWPX| D[Docling Processor]
    C -->|기본형 + 문서| E[LangChain Processor]
    C -->|오디오| F[Audio Processor]
    C -->|테이블| G[Tabular Processor]

    D --> H[Enrichment<br/>기본값 적용]
    H --> I[고급 청킹<br/>max_tokens: 2000]

    E --> J[텍스트 추출]
    J --> K[기본 청킹<br/>chunk_size: 1000]

    F --> L[Whisper 전사<br/>ko, 30초 단위]
    G --> M[데이터프레임 변환]

    I --> N[벡터 메타데이터]
    K --> N
    L --> N
    M --> N

    N --> O[Weaviate 저장]
```

## 🔍 메타데이터 필드

모든 처리기는 다음 표준 메타데이터를 생성합니다 (기본값 보장):

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| text | str | "" | 청크 텍스트 |
| n_char | int | 0 | 문자 수 |
| n_word | int | 0 | 단어 수 |
| n_line | int | 0 | 줄 수 |
| i_page | int | 0 | 시작 페이지 |
| e_page | int | 0 | 끝 페이지 |
| i_chunk_on_page | int | 0 | 페이지 내 청크 인덱스 |
| n_chunk_of_page | int | 0 | 페이지 총 청크 수 |
| i_chunk_on_doc | int | 0 | 문서 전체 청크 인덱스 |
| n_chunk_of_doc | int | 0 | 문서 총 청크 수 |
| n_page | int | 0 | 문서 총 페이지 수 |
| reg_date | str | 현재시간 | 등록 일시 |
| chunk_bboxes | str | "[]" | 바운딩 박스 (JSON) |
| media_files | str | "[]" | 미디어 파일 (JSON) |

## 🚨 주의사항

1. **Whisper 서버**: 오디오 처리를 위해 Whisper 서버가 실행 중이어야 함
   - 기본 URL: `http://192.168.74.164:30100/v1/audio/transcriptions`
2. **메모리 관리**: 대용량 문서 처리 시 메모리 사용량 모니터링 필요
3. **파일 권한**: 임시 파일 생성을 위한 쓰기 권한 필요
4. **의존성**: Docling, LangChain, Pandas 등 필수 패키지 설치 필요
5. **기본값 변경**: processor_config.json을 직접 수정하거나 API 사용

## 🧩 HWP → PDF 변환용 rhwp-pdf-api 배포 (이슈 #199)

HWP → PDF 변환 backend 중 `rhwp` 는 OCR / VLM 과 동일하게 **별도 HTTP 서비스**로 호출한다. 호출 client 코드는 [genon/preprocessor/converters/hwp_to_pdf/rhwp.py](../converters/hwp_to_pdf/rhwp.py) 에 이미 들어있고, 서버 측 자산(Dockerfile / k8s 매니페스트)은 [genonai/genos-rhwp](https://github.com/genonai/genos-rhwp) 레포에 있다.

회사 클러스터에 아직 떠 있지 않다면 다음 절차로 직접 배포한다.

### 1. genos-rhwp 클론 + 빌드 사전 준비

```bash
git clone --depth 1 https://github.com/genonai/genos-rhwp.git
cd genos-rhwp
```

원본 `Dockerfile.pdf-api` 그대로는 다음 두 가지가 빠져 빌드되지 않는다 — 임시 사본 `Dockerfile.pdf-api.local` 을 만들어 우회한다.

```bash
cat > Dockerfile.pdf-api.local <<'DOCKERFILE'
FROM rust:latest AS builder
WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
COPY ttfs ./ttfs
COPY saved ./saved
RUN cargo build --release --bin rhwp

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates fontconfig fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/rhwp /usr/local/bin/rhwp
COPY --from=builder /app/ttfs /app/ttfs
EXPOSE 7878
CMD ["rhwp", "serve-pdf", "--host", "0.0.0.0", "--port", "7878"]
DOCKERFILE
```

차이는 두 가지:
- `COPY Cargo.toml Cargo.lock ./` → `COPY Cargo.toml ./` (레포에 `Cargo.lock` 이 커밋되어 있지 않음. cargo 가 fresh resolve)
- `COPY saved ./saved` 추가 (`src/document_core/commands/document.rs` 의 `include_bytes!("../../../saved/blank2010.hwp")` 가 빌드시 필요)

### 2. 이미지 빌드 + 회사 registry push

```bash
# 빌드 (Rust cargo 의존성 컴파일로 5~15분 소요, BuildKit cargo cache mount 권장)
docker build --platform linux/amd64 \
  -f Dockerfile.pdf-api.local \
  -t mncregistry:30500/mnc/rhwp-pdf-api:0.1.0 .

# 회사 registry 로 push
docker push mncregistry:30500/mnc/rhwp-pdf-api:0.1.0
```

태그/registry 는 회사 컨벤션에 맞춰 조정한다.

### 3. k8s 매니페스트 적용

genos-rhwp 레포의 `k8s/rhwp-pdf-api.yaml` 을 사용한다. image 만 위 push 한 태그로 교체:

```bash
sed -i 's|image: rhwp-pdf-api:latest|image: mncregistry:30500/mnc/rhwp-pdf-api:0.1.0|' k8s/rhwp-pdf-api.yaml
kubectl apply -f k8s/rhwp-pdf-api.yaml
kubectl rollout status deploy/rhwp-pdf-api
kubectl get svc rhwp-pdf-api
```

기본 노출 — ClusterIP Service `rhwp-pdf-api:7878`.

### 4. doc_parser 측 endpoint 주입

같은 namespace 면 추가 설정 없음 — Dockerfile 의 `RHWP_PDF_API_URL=http://rhwp-pdf-api:7878` placeholder 가 그대로 동작한다.

다른 namespace 면 doc_parser 의 deploy 매니페스트에서 FQDN 으로 override:

```yaml
env:
  - name: RHWP_PDF_API_URL
    value: http://rhwp-pdf-api.<rhwp-namespace>.svc.cluster.local:7878
```

런타임에 같은 이름의 env 변수를 새로 주입하면 chain config 가 자동으로 반영한다 ([genon/preprocessor/converters/hwp_to_pdf/availability.py](../converters/hwp_to_pdf/availability.py) 의 `rhwp_pdf_api_url()` 참고).

### 5. 동작 검증

서버 단독 검증 (doc_parser pod 안에서):

```bash
kubectl exec -it deploy/doc-parser-preprocessor -- bash
curl -sS -X POST \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/app/sample_files/hwp_sample.hwp \
  -o /tmp/out.pdf \
  http://rhwp-pdf-api:7878/api/convert/hwp-to-pdf
file /tmp/out.pdf  # "PDF document, version 1.7, N pages" 가 나와야 정상
```

doc_parser 코드 경로 검증 (chain 이 rhwp 우선 시도하는지):

```bash
kubectl logs deploy/doc-parser-preprocessor -f | grep '\[hwp_to_pdf'
# HWP 첨부 처리 시 다음과 같은 로그가 보여야 함:
#   [hwp_to_pdf] chain start file=... order=['pdf_sdk', 'rhwp', 'libreoffice']  (enterprise)
#   [hwp_to_pdf] try backend=rhwp file=...
#   [hwp_to_pdf:rhwp] POST http://rhwp-pdf-api:7878/api/convert/hwp-to-pdf (N bytes, ...)
#   [hwp_to_pdf:rhwp] success -> ...pdf
```

### 6. 변환 품질 검증용 HWP 추천

이슈 #199 가 명시한 "표 / 이미지 / 다단 / 머리말꼬리말" 회귀를 일찍 잡으려면 다음 유형 1~2건씩을 `genon/preprocessor/sample_files/` 에 추가하면 [tests/regression/test_hwp_to_pdf_regression.py](../tests/regression/test_hwp_to_pdf_regression.py) 가 backend 별로 자동 회귀를 돌린다.

- 표 위주 — 단순 표 / 병합 셀 / 중첩 표 각각 1건
- 이미지 위주 — PNG 만, WMF/EMF 만, 그리고 둘 혼합 각각 1건 (HWP 의 WMF/EMF 는 rhwp 가 SVG 로 풀어내는 흐름이라 회귀 빈도가 잦음)
- 다단 (2단·3단) — 본문 + 표가 단을 가로지르는 케이스 1건
- 머리말 / 꼬리말 — 페이지 번호 포함 머리말꼬리말 1건
- 각주 / 미주 — 미주 있는 학술/법규 문서 1건
- 수식 (LaTeX) — `<math>` 영역 포함 (이슈 #195 회귀 — `hwp_sample.hwp` 가 이미 일부 케이스 커버)
- HWPX — `.hwpx` 도 1건 이상 (rhwp 는 .hwp 만 받을 가능성이 있어 chain 이 LibreOffice 로 자동 fallback 되는 흐름을 함께 검증)

파일명 규칙: `<유형>_<설명>.hwp` (예: `table_merged_cells.hwp`, `image_wmf.hwpx`).

### 7. 트러블슈팅

- `is_available()=False` 로 rhwp 가 chain 에서 빠짐 → `RHWP_PDF_API_URL` env 가 비어 있음. deploy yaml 의 env 확인.
- HTTP 200 이지만 응답이 PDF 가 아님 → 서버 측 stderr 로그 확인 (`kubectl logs deploy/rhwp-pdf-api`). 입력 HWP 가 손상되었거나 rhwp 가 미지원 요소를 만난 경우 흔하다. 우리 client 가 자동으로 `None` 반환 후 LibreOffice 로 fallback.
- 타임아웃 → 큰 HWP 의 경우 기본 600s 초과 가능. `HWP_TO_PDF_TIMEOUT_SEC` env 로 조정.
