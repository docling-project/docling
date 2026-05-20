## 프로젝트 구조

```
# 최상단은 docling 구조를 그대로 가져감
# doc-parser 관련은 genon 디렉터리에 작성
.
├── build-script # 빌드 위한 스크립트 파일 및 컨피그 파일
├── docling
├── docs
├── genon
│   ├── preprocessor # genos에서 실행 될 전처리기 이미지 및 facade 관련
│   │   ├── configs # gunicorn, supervisor 설정
│   │   ├── docker # 도커파일 위치
│   │   ├── env # 개발 시 설정 파일들
│   │   ├── facade # facade 코드
│   │   │   ├── evaluation
│   │   │   │   └── test_files
│   │   │   │       ├── annotated
│   │   │   │       ├── pdf
│   │   │   │       └── result
│   │   │   ├── gitbook_doc
│   │   │   │   └── images
│   │   │   ├── legacy
│   │   │   └── legal_parser
│   │   │       ├── api
│   │   │       ├── commons
│   │   │       ├── parsers
│   │   │       ├── schemas
│   │   │       └── services
│   │   ├── scripts # 도커 이미지 push 및 디비 등록 관련 스크립트 위치
│   │   ├── resources # 폰트 및 기타 리소스 파일들
│   │   ├── sample_files
│   │   ├── src # 전처리기 API 소스
│   │   │   └── common
│   │   └── tests # genos doc-parser 테스트 소스
│   │       ├── regression
│   │       │   └── baselines
│   │       ├── smoke
│   │       └── unit
│   ├── serving # ocr 및
│   │   └── paddle
│   │       ├── config # ocr, vl paddlex 실행 파일
│   │       ├── docker
│   │       ├── etc
│   │       ├── k8s-manifest
│   │       └── resources
│   └── tools
│       └── genos_tools
│           └── commands
└── tests # docling 리포 test 관련 코드 작성 x
```

## 전처리기 빌드 및 등록

1. `HWP_SDK_TOKEN` 설정 (HWP SDK private 레포 다운로드용 — **두 variant 모두 필수**)
   - 대상 레포: [`HeechanKim-Genon/hwp_sdk`](https://huggingface.co/datasets/HeechanKim-Genon/hwp_sdk) — 이 레포 전용 read 토큰
   - 토큰 값은 [제논 내부 드라이브 (HWP_SDK_TOKEN)](https://drive.google.com/file/d/1e7hIsYaLAVFBwWoe1Oi5OcODWXxuMI_r/view?usp=sharing) 에서 확인
   - `doc_parser/` (레포 최상위 경로) 에서 아래 명령어 실행 (이후 재실행 불필요, Git 미추적):
     ```shell
     echo "HWP_SDK_TOKEN=hf_xxx_your_hwp_sdk_token_here" >> build-script/hf_private_token.env
     ```
   - `doc-parser-build.config` 에 직접 입력하거나 push 하지 말 것 (토큰은 반드시 `hf_private_token.env` 파일에만)

2. `BUILD_VARIANT` 선택 - 무료용(**`opensource`**) / 유료용(**`enterprise`**) 버전 선택
   - **`opensource`** — LibreOffice + rhwp(외부 HTTP API 호출) 만 포함. PDF SDK 자산이 이미지에 일절 들어가지 않음 (다운로드 단계 자체가 없음). 회사 내부 PDF SDK 라이선스가 없는 환경/외부 배포용. **`HWP_SDK_TOKEN` 만 있으면 됨**.
   - **`enterprise`** — 위 + 유료 PDF SDK 포함. HWP → PDF 변환 chain 이 `pdf_sdk → rhwp → libreoffice` 순으로 동작. **`HWP_SDK_TOKEN` 에 더해 `PDF_SDK_TOKEN` 도 필수**.
     - `PDF_SDK_TOKEN` — 대상 레포 [`HeechanKim-Genon/pdf_sdk`](https://huggingface.co/datasets/HeechanKim-Genon/pdf_sdk) 전용 read 토큰
     - 토큰 값은 [제논 내부 드라이브 (PDF_SDK_TOKEN)](https://drive.google.com/file/d/1jF9UXUq91dIw6NRP3hz5CpdVokBDwPF6/view?usp=sharing) 에서 확인
     - `hf_private_token.env` 에 한 줄 더 추가:
       ```shell
       echo "PDF_SDK_TOKEN=hf_yyy_your_pdf_sdk_token_here" >> build-script/hf_private_token.env
       ```
   - 둘 중 하나를 반드시 명시해야 한다. 비워둔 채 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단된다 (의도치 않게 유료 SDK 가 포함될 위험을 막기 위한 안전장치).
   - [`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 `BUILD_VARIANT=` 라인을 위 둘 중 하나로 설정:
     ```bash
     # build-script/doc-parser-build.config
     BUILD_VARIANT=enterprise   # 또는 opensource
     ```
   - 빌드 시 `DOCKERFILE_PATH` 가 자동으로 `genon/preprocessor/docker/Dockerfile.${BUILD_VARIANT}` 로 결정되고, 이미지 태그에도 `-${BUILD_VARIANT}` suffix 가 붙는다 (예: `:1.3.6.3-enterprise`).
   - 두 variant 의 런타임 동작 차이 / chain 우선순위는 [`preprocessor/docker/README.md`](preprocessor/docker/README.md) 참고.

3. build-script 디렉토리 이동

4. [doc-parser-build.config](../build-script/doc-parser-build.config) 기타 변경 사항 반영 (1·2번을 수행했다면 `HWP_SDK_TOKEN` / `PDF_SDK_TOKEN` 값은 직접 입력하지 말 것)

5. 실행 [doc-parser-build.sh](../build-script/doc-parser-build.sh)

6. [register.config](preprocessor/scripts/register.config) 변경 사항 있을 시 변경 필요

7. 실행 [register_image.sh](preprocessor/scripts/register_image.sh) : push와 디비에 등록해준다.

8. 사이트 배포 시
```shell
1. 이미지 저장
docker save mncregistry:30500/mnc/doc-parser-preprocessor:latest | gzip > doc-parser-preprocessor.tar.gz
2. 사이트에서 이미지 복원
gunzip -c doc-parser-preprocessor.tar.gz | docker load
3. register_image.sh 파일 실행
```

## 로컬 테스트 (도커 빌드 없이 test.py 실행)

도커를 거치지 않고 `genon/preprocessor/facade/test.py` 등을 로컬에서 바로 실행하려면, **HWP SDK · PDF SDK를 레포 최상위에 직접 다운로드**해야 한다. (코드가 `<repo_root>/hwp_sdk`, `<repo_root>/pdf_sdk` 경로를 자동으로 찾음)

> 아래 명령어들은 **레포가 위치한 호스트 머신의 터미널**에서 실행한다 (도커 컨테이너 안 셸이 아님). 컨테이너 안에서 macOS 절대경로로 받으면 컨테이너 내부 가상 경로에 저장돼 호스트에 반영되지 않으니 주의. 만약 컨테이너 안에서 받고 싶다면 cwd를 마운트된 repo root(예: `/app/docparser_work_187/doc_parser`)로 옮긴 뒤 상대경로(`./hwp_sdk`, `./pdf_sdk`)로 받아야 한다.

1. HuggingFace 인증 (위 빌드 단계 1번/2번에서 발급한 두 fine-grained 토큰 사용)
   ```shell
   export HWP_SDK_TOKEN=hf_xxx_your_hwp_sdk_token_here
   export PDF_SDK_TOKEN=hf_yyy_your_pdf_sdk_token_here   # PDF SDK 도 받을 경우만
   ```
2. 레포 최상위(`doc_parser/`) 경로에서 두 SDK 다운로드 (각 레포에 대응되는 토큰 사용)
   ```shell
   # HWP SDK
   huggingface-cli download HeechanKim-Genon/hwp_sdk \
     --repo-type dataset \
     --local-dir ./hwp_sdk \
     --local-dir-use-symlinks False \
     --token "${HWP_SDK_TOKEN}"
   chmod +x ./hwp_sdk/convtext

   # PDF SDK
   huggingface-cli download HeechanKim-Genon/pdf_sdk \
     --repo-type dataset \
     --local-dir ./pdf_sdk \
     --token "${PDF_SDK_TOKEN}" \
     --local-dir-use-symlinks False
   chmod +x ./pdf_sdk/pdfConverter
   ```
3. 두 디렉토리(`hwp_sdk/`, `pdf_sdk/`)는 `.gitignore`에 포함되어 있어 커밋되지 않음
4. 이후 `genon/preprocessor/facade/test.py` 실행 시 별도 환경변수 설정 없이 동작

> 도커 환경에서는 빌드 단계에서 두 SDK가 자동으로 `/app/hwp_sdk`, `/app/pdf_sdk` 에 설치되며, `PDF_SDK_HOME` 환경변수로 SDK 경로가 제어됨. 로컬에서는 위 경로 fallback이 자동 적용됨.

### SDK 바이너리 단독 실행 (디버깅 / 동작 확인용)

> ⚠️ **두 SDK 모두 Linux용 바이너리**라서 macOS · Windows 호스트에서는 직접 실행 불가. 반드시 **Linux 환경(또는 Linux 도커 컨테이너 안)** 에서 호출해야 한다. 도커 컨테이너 안에서 cwd를 마운트된 repo root로 옮기고 아래 명령어 실행.

#### PDF SDK (HWP/DOCX/PPT/이미지 → PDF)

```shell
SDK=$(pwd)/pdf_sdk    # repo root 에서 실행한다고 가정
SAMPLES=$(pwd)/genon/preprocessor/sample_files

# fonts_gen.conf 경로 한 번 보정 (HF에 올라간 원본 conf는 옛 경로를 가리킴)
sed -i "s|<dir>[^<]*</dir>|<dir>${SDK}/fonts</dir>|" ${SDK}/fonts/fonts_gen.conf
sed -i "s|<cachedir>[^<]*</cachedir>|<cachedir>${SDK}/font_cache</cachedir>|" ${SDK}/fonts/fonts_gen.conf
mkdir -p ${SDK}/font_cache

# 환경변수 + 변환 실행
export LD_LIBRARY_PATH=${SDK}/moduledata:${LD_LIBRARY_PATH:-}
export FONTCONFIG_FILE=${SDK}/fonts/fonts_gen.conf
export FONTCONFIG_PATH=${SDK}/fonts
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

${SDK}/pdfConverter \
  -i ${SAMPLES}/hwp_sample.hwp \
  -o ${SAMPLES} \
  -t /tmp \
  -f ${SDK}/fonts \
  -e -1 -p 1 --log info
```

→ `genon/preprocessor/sample_files/hwp_sample.hwp.pdf` 생성. 호스트에서 그대로 열어 변환 품질 확인 가능.

#### HWP SDK (HWP/HWPX → JSON · info · 이미지)

```shell
SDK=$(pwd)/hwp_sdk
SAMPLES=$(pwd)/genon/preprocessor/sample_files
OUT=/tmp/hwp_out
mkdir -p ${OUT}/images

cd ${SDK} && ./convtext \
  ${SAMPLES}/hwp_sample.hwp \
  ${OUT}/output.json \
  ${OUT}/output.info \
  ${OUT}/images/
```

→ `${OUT}/output.json` 에 파싱된 텍스트/구조, `${OUT}/images/` 에 추출 이미지.

## 🧩 HWP → PDF 변환용 rhwp-pdf-api 배포 (이슈 #199)

HWP → PDF 변환 backend 중 `rhwp` 는 OCR / VLM 과 동일하게 **별도 HTTP 서비스**로 호출한다. 호출 client 코드는 [preprocessor/converters/hwp_to_pdf/rhwp.py](preprocessor/converters/hwp_to_pdf/rhwp.py) 에 이미 들어있고, 서버 측 자산(Dockerfile / k8s 매니페스트)은 [genonai/genos-rhwp](https://github.com/genonai/genos-rhwp) 레포에 있다.

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

런타임에 같은 이름의 env 변수를 새로 주입하면 chain config 가 자동으로 반영한다 ([preprocessor/converters/hwp_to_pdf/availability.py](preprocessor/converters/hwp_to_pdf/availability.py) 의 `rhwp_pdf_api_url()` 참고).

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

이슈 #199 가 명시한 "표 / 이미지 / 다단 / 머리말꼬리말" 회귀를 일찍 잡으려면 다음 유형 1~2건씩을 `genon/preprocessor/sample_files/` 에 추가하면 [preprocessor/tests/regression/test_hwp_to_pdf_regression.py](preprocessor/tests/regression/test_hwp_to_pdf_regression.py) 가 backend 별로 자동 회귀를 돌린다.

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

## paddle-ocr 빌드 및 배포

1. build-script 디렉토리 이동
2. [paddle-ocr-build.config](../build-script/paddle-ocr-build.config) 설정 파일 변경
3. [paddle-ocr-build.sh](../build-script/paddle-ocr-build.sh) 스크립트 실행 : 빌드 및 레지스트리 푸쉬
4. [doc-parser-ocr-deployment.yaml](serving/paddle/k8s-manifest/doc-parser-ocr-deployment.yaml)
```shell
kubectl apply -f doc-parser-ocr-deployment.yaml
```
5. 노드 포트로 배포시는 [doc-parser-ocr-deployment-node-port.yaml](serving/paddle/k8s-manifest/doc-parser-ocr-deployment-node-port.yaml)

## paddle-ocr B300 (Blackwell) 빌드 및 배포

B300 (compute capability `sm_100`) GPU 환경 대응을 위한 별도 빌드 경로. 기존 cu126 이미지와는 파일 세트가 완전히 분리되어 있다. 빌드 / 배포 / smoke test 절차는 [serving/paddle/README-B300.md](serving/paddle/README-B300.md) 참고.

## dots mocr vllm 서빙

1. 모델 다운로드

```shell
pip install huggingface_hub

huggingface-cli download rednote-hilab/dots.mocr \
  --local-dir ./dots-mocr
```

- huggingface: https://huggingface.co/rednote-hilab/dots.mocr

2. Genos 모델서빙 기능으로 서빙생성
- 주요 옵션은 아래 서빙명령어 참고

* 참고

- 내부 별도서버에 서비스 했던 vllm 서빙명령어

```shell
CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr \
 --host 0.0.0.0 \
 --port 26001 \
 --tensor-parallel-size 1 \
 --dtype bfloat16 \
 --gpu-memory-utilization 0.9 \
 --max-model-len 20000 \
 --max-num-seqs 32 \
 --chat-template-content-format string \
 --served-model-name dots-mocr \
 --trust-remote-code
```
