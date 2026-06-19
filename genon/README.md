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

### A. 토큰 / 변수 설정 (1~3번)

1. `HWP_SDK_TOKEN` 값 설정 (HWP SDK private 레포 다운로드용)

   **(배경 설명)**
   - HWP SDK란? → HWP를 고품질로 파싱하기 위한 전용 SDK. 사전에 사내 private 토큰 발급후 적용을 완료해야만, 전처리기 빌드시 같이 다운로드 및 설치 진행됨(다운로드 불가시 전처리기 빌드 실패).

   **(진행 순서)**
   - **[1번]** 토큰 값을 [제논 내부 드라이브 (HWP_SDK_TOKEN)](https://docs.google.com/document/d/1c2kHPus5QxFN0jhfH37EDFORd6pt2rermkINfDFlQbs/edit?usp=drive_link) 에서 확인한다.
   - **[2번]** 확인한 토큰 값으로, `doc_parser/` (레포 최상위 경로) 에서 아래 명령어 실행 (이후 재실행 불필요, Git 미추적):
     - `hf_xxx_your_hwp_sdk_token_here` 부분을 해당 토큰 값으로 교체후 다음 명령어 실행.
     ```shell
     echo "HWP_SDK_TOKEN=hf_xxx_your_hwp_sdk_token_here" >> build-script/hf_private_token.env
     ```
     - `doc-parser-build.config` 에 직접 입력하거나 push 하지 말 것 (토큰은 반드시 위 명령어를 통해 `hf_private_token.env` 파일에만 존재해야함)

2. `BUILD_VARIANT` 값 선택 - 기본(**`standard`**) / 유료 PDF SDK 포함(**`synap`**) 버전 선택

   > **주의 (배포 정책):** `synap` 빌드는 유료 PDF SDK가 포함된다. 운영계 Genos 도커 레지스트리에는 **`standard` 이미지만** 올린다. 일반 사이트 배포는 **기본적으로 `standard`** 로 진행하며, 특정 사이트에 `synap`(유료 PDF SDK) 버전을 들고가야 하는 요청이 생길 경우엔, **AI Search 팀 내부에서 자체적으로 빌드후, 이미지를 전달**한다 (PDF SDK 토큰/라이선스는 AI Search 팀이 관리, 비공개).

   **(배경 설명)**
   - PDF SDK란? → `타문서(HWP,docx 등) → PDF`를 고품질로 변환하기 위한 유료용 SDK. 이 SDK의 유무에 따라 `standard` 또는 `synap` 버전으로 구분됨.
   - **`standard`** — 기본 버전.
     - 오픈소스(LibreOffice + rhwp)만 사용하여 PDF로 변환. PDF SDK 자산이 이미지에 일절 들어가지 않음 (다운로드 단계 자체가 없음).
     - 사내 운영계 GenOS 환경/일반적인 외부 사이트 배포용. **`HWP_SDK_TOKEN` 외에 별도의 추가 토큰값 필요 없음**.
   - **`synap`** — 오픈소스(LibreOffice + rhwp) + 유료 PDF SDK 포함 버전.
     - PDF로의 문서 변환시, `pdf_sdk → rhwp(hwp/hwpx 전용) → libreoffice` 순서로 fallback 동작(=변환 실패시 후순위 로직 사용).
     - `PDF SDK`는 **전처리기 빌드시 자동 설치되며, `PDF_SDK_TOKEN`값이 사전에 필수로 적용되어야함(`HWP_SDK_TOKEN`과 별개 값)**.

   **(진행 순서)**

    - **[0번(synap의 경우만)]** **토큰을 발급후**, `doc_parser/` (레포 최상위 경로) 에서 아래 명령어를 실행한다 (이후 재실행 불필요, Git 미추적). 발급받은 값으로 `hf_yyy_...` 부분을 교체후 실행:
       ```shell
       echo "PDF_SDK_TOKEN=hf_yyy_your_pdf_sdk_token_here" >> build-script/hf_private_token.env
       ```
       - `doc-parser-build.config` 에 직접 입력하거나 push 하지 말 것 (토큰은 반드시 위 명령어를 통해 `hf_private_token.env` 파일에만 존재해야함)
   - **[1번]** build-script 디렉토리 이동
   - **[2번]**[`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 `BUILD_VARIANT=` 을 둘 중 하나로 설정:
     ```bash
     # build-script/doc-parser-build.config
     BUILD_VARIANT=standard   # 또는 synap
     ```
     - **비워둔 채 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단**된다 (의도치 않게 **`synap`** 가 배포될 위험을 막기 위한 안전장치).
     - 빌드 시 `DOCKERFILE_PATH` 가 자동으로 `genon/preprocessor/docker/Dockerfile.${BUILD_VARIANT}` 로 결정된다.
     - 두 variant 의 런타임 동작 차이 / chain 우선순위는 [`preprocessor/docker/README.md`](preprocessor/docker/README.md) 참고.

3. `HW_VARIANT` 값 선택 - GPU(**`gpu`**) / CPU(**`cpu`**) 빌드 선택

   **(배경 설명)**

   - **`gpu`** — `uv.lock` 기준 그대로. torch CUDA wheel + nvidia-* / triton 포함. GPU 가속 환경용.
   - **`cpu`** — builder 단계에서 torch / torchvision 을 CPU wheel(`https://download.pytorch.org/whl/cpu`)로 재설치하고 nvidia-* / triton 패키지를 제거한 경량 이미지. GPU 없는 환경용.
    - 최종 이미지명 태그 규칙:
       - **기본 조합(`cpu` + `standard`)** 은 가장 기본 산출물이라 접미사 없이 `:${IMAGE_VERSION}` (예: `:2.1.5`).
       - 그 외 조합은 hw 먼저·variant 나중 순서로 `:${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}` (예: `:2.1.5-gpu-synap`).
   - `BUILD_VARIANT` × `HW_VARIANT` 조합으로 최대 4종의 이미지를 만들 수 있다:
     | 조합 | 태그 예시 |
     |---|---|
     | cpu + standard (기본) | `:2.1.5` |
     | gpu + standard | `:2.1.5-gpu-standard` |
     | cpu + synap | `:2.1.5-cpu-synap` |
     | gpu + synap | `:2.1.5-gpu-synap` |
    > **정리** — 위 2번 (`BUILD_VARIANT`) × 3번 (`HW_VARIANT`) 의 조합으로 **총 4종의 Dockerfile(이미지)** 가 만들어진다. 운영 환경에 맞는 1개를 골라서 빌드하면 된다. `synap` 는 빌드를 위해 AI Search 팀에서 관리하는 `PDF_SDK_TOKEN` 값이 필요함.

   **(진행 순서)**

   - **[1번]** [`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 `HW_VARIANT=` 라인을 둘 중 하나로 설정:
     ```bash
     # build-script/doc-parser-build.config
     HW_VARIANT=gpu   # 또는 cpu
     ```
     - 비워둔 채 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단된다.

### A-2. (선택) rhwp / LibreOffice 제외 빌드 (이슈 [#286](https://github.com/genonai/doc_parser/issues/286))

일부 사이트는 정책상 rhwp · LibreOffice 를 이미지에 넣지 않기를 요구한다. 이 경우 [`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 두 플래그를 끄고 **이미지를 새로 빌드**한다 (이미 빌드된 운영 이미지에는 두 패키지가 포함돼 있으므로, 제외하려면 재빌드가 필수다).

```bash
# build-script/doc-parser-build.config
INSTALL_LIBREOFFICE=false   # 기본 true. false 면 LibreOffice + Java + H2Orestart 미설치
INSTALL_RHWP=false          # 기본 true. false 면 rhwp 바이너리(Rust 빌드 stage) 미포함
```

- 둘 다 `true`(기본)면 기존과 동일하다. `true` / `false` 외의 값은 빌드가 즉시 에러로 중단된다.
- **동작 영향** — HWP/오피스 → PDF 변환은 가용한 backend 만 자동 등록된다 (미설치 backend 는 graceful 제외). **`standard` + 둘 다 `false`** 면 변환 backend 가 0개가 되며, 전처리기별로 영향이 다르다:
  - **적재형(지능형)** — PDF 가 아닌 입력을 내부에서 PDF 로 변환한 뒤 파싱하므로, 변환기가 없으면 HWP·docx·ppt 등을 처리하지 못한다 (명확한 안내와 함께 실패). → **PDF 로 변환된 문서를 입력**해야 함.
  - **첨부형 / 변환형 / 파싱형** — HWP·HWPX 는 이미지에 항상 포함되는 **HWP SDK** 로, docx·ppt 는 원본을 직접 파싱하므로 변환 backend 없이도 동작한다 (영향 적음). 단 변환형의 PDF 표준화 산출물 등 일부 부가 기능은 제한.
  - **`synap`** 은 PDF SDK 가 비-HWP 변환의 1순위라, 둘 다 꺼도 docx/ppt 등은 PDF SDK 로 변환된다 (HWP/HWPX 는 PDF SDK 만으로 처리).
- **태그 반영** — `false` 로 끄면 이미지 태그 끝에 `-nolibre` / `-norhwp` 가 **자동으로** 붙는다 (둘 다 끄면 `-nolibre-norhwp`). 둘 다 `true`(기본)면 접미사 없이 기존 태그 그대로다.
  - 예) `cpu`+`standard` + 둘 다 off → `:2.2.0-nolibre-norhwp`, `gpu`+`synap` + rhwp off → `:2.2.0-gpu-synap-norhwp`.
  - 덕분에 패키지를 끈 특수 이미지가 운영 이미지(둘 다 on)와 **같은 태그로 push 돼 덮어쓰는 사고가 자동 차단**된다. 표준 4종 카탈로그(2번 표)와도 자연히 구분된다.
  - 등록 시 [`register.config`](preprocessor/scripts/register.config) 에도 `INSTALL_LIBREOFFICE` / `INSTALL_RHWP` 를 빌드와 동일하게 넣어야 태그가 일치한다.
  - 빌드 시 두 값은 `docker inspect` 로도 보이도록 OCI 라벨(`ai.genon.install.libreoffice` / `ai.genon.install.rhwp`)에 함께 기록된다.

### B. 이미지 빌드 (4~5번)

4. [doc-parser-build.config](../build-script/doc-parser-build.config) 에 기타 변경 사항 반영 (1·2번을 수행했다면 `HWP_SDK_TOKEN` / `PDF_SDK_TOKEN` 값은 직접 입력하지 말 것)

5. 실행 [doc-parser-build.sh](../build-script/doc-parser-build.sh)
   - `SMOKE_TEST=true`(기본값) 면 빌드 직후 컨테이너를 띄워 `SMOKE_TEST_FILE`(기본 `pdf_sample.pdf`) 한 건을 파싱하고 torch 의 CUDA 포함 여부가 `HW_VARIANT` 와 일치하는지 검증한다. 실패하면 빌드도 실패한다.

### C. 레지스트리 등록 (6~7번)

6. [register.config](preprocessor/scripts/register.config) 설정 — `IMAGE_VERSION` / `BUILD_VARIANT` / `HW_VARIANT` 세 값은 **`doc-parser-build.config` 와 동일한 값**으로 둔다. `register_image.sh` 가 빌드와 동일한 태그 규칙으로 자동 조합한다 (기본 조합 `cpu`+`standard` 는 `${IMAGE_VERSION}`, 그 외는 `${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}`).

7. 실행 [register_image.sh](preprocessor/scripts/register_image.sh) : push와 디비에 등록해준다.
   - 다른 조합을 등록하려면 `register.config` 의 세 값을 그 조합으로 바꾸고 다시 실행 (interactive prompt 가 있어 한 번에 batch 자동화 불가).

### D. 사이트 배포 (8번)

8. 사이트 배포 시 (조합별로 동일하게 진행 — 아래는 기본 조합인 `cpu`+`standard` 예시. 기본 조합일때는 태그 접미사가 없는 점에 주의)
```shell
# 1. 이미지 저장
docker save mncregistry:30500/mnc/doc-parser-preprocessor:2.1.5 | gzip > doc-parser-preprocessor-2.1.5.tar.gz
# 2. 사이트에서 이미지 복원
gunzip -c doc-parser-preprocessor-2.1.5.tar.gz | docker load
# 3. 해당 조합으로 register_image.sh 실행 (BUILD_VARIANT / HW_VARIANT 지정)
```
   - `gpu`/`synap` 등 다른 조합이면 태그가 `:2.1.5-gpu-synap` 처럼 붙으니 파일명/명령어를 그에 맞게 바꾼다.

## 로컬 테스트 (도커 빌드 없이 test.py 실행)

도커를 거치지 않고 `genon/preprocessor/facade/test.py` 등을 로컬에서 바로 실행하려면, **HWP SDK · PDF SDK를 레포 최상위에 직접 다운로드**해야 한다. (코드가 `<repo_root>/hwp_sdk`, `<repo_root>/pdf_sdk` 경로를 자동으로 찾음)

> 아래 명령어들은 **레포가 위치한 호스트 머신의 터미널**에서 실행한다 (도커 컨테이너 안 셸이 아님). 컨테이너 안에서 macOS 절대경로로 받으면 컨테이너 내부 가상 경로에 저장돼 호스트에 반영되지 않으니 주의. 만약 컨테이너 안에서 받고 싶다면 cwd를 마운트된 repo root(예: `/app/docparser_work_187/doc_parser`)로 옮긴 뒤 상대경로(`./hwp_sdk`, `./pdf_sdk`)로 받아야 한다.

1. HuggingFace 인증 (위 빌드 단계 1번/2번에서 발급한 fine-grained 토큰 사용)
   ```shell
   export HWP_SDK_TOKEN=hf_xxx_your_hwp_sdk_token_here
   export PDF_SDK_TOKEN=hf_yyy_your_pdf_sdk_token_here   #(비공개 토큰, AI Search 팀 관리)
   ```
   - `PDF_SDK_TOKEN` 은 공개 값이 아님(AI Search 팀 내부에서 private 하게 관리).
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

GPU 메모리 크기에 따라 권장 옵션이 갈린다 (이슈 [#205](https://github.com/genonai/doc_parser/issues/205) 운영 분석 + 메모리 한계 기반 계산식 — 자세한 근거는 [`dotsocr_vllm_max_num_seqs.md`](dotsocr_vllm_max_num_seqs.md) 참고).

### 공통 옵션 (모든 케이스 동일)

```shell
--host 0.0.0.0 \
--port 26001 \
--tensor-parallel-size 1 \
--dtype bfloat16 \
--max-model-len 20480 \            # 이미지 ~2.5k + 출력 16384 + 여유분 (짧으면 OOM)
--limit-mm-per-prompt image=1 \
--enable-chunked-prefill \         # 긴 prefill(이미지) + decode 혼합 효율↑
--chat-template-content-format string \
--served-model-name dots-mocr \
--trust-remote-code
```

### 케이스별 가변 옵션 (`--gpu-memory-utilization` / `--max-num-seqs`)

| GPU 메모리 | `--gpu-memory-utilization` | `--max-num-seqs` (표준 권장) | 비고 |
|---|---|---|---|
| 24G (L4 등) | 0.9 | **64** | 메모리 한계 기준 |
| 40G (MIG 슬라이스) | 0.9 | **128** | |
| 80G (H100, 안전) | 0.9 | **256** | 표준 권장. CUDA graph capture 천장이 256 |
| 80G (KV cache 극대화) | 0.95 | **128** | KV cache 더 끌어쓰되 동시성 보수 |

> 모든 요청이 `max completion token` 을 꽉 채우는 worst case가 운영상 우려되면 더 보수적 마진(24/40G:32, 80G:64)을 사용한다. 자세한 표는 [`dotsocr_vllm_max_num_seqs.md` "7.3 보수적 옵션"](dotsocr_vllm_max_num_seqs.md#73-보수적-옵션-worst-case-oom-회피) 참고.



### 예시: 80G 표준 권장 (다른 GPU 크기는 위 표에 따라 두 옵션만 바꾸면 동일 패턴)

```shell
CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr \
  --host 0.0.0.0 \
  --port 26001 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 20480 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt image=1 \
  --enable-chunked-prefill \
  --chat-template-content-format string \
  --served-model-name dots-mocr \
  --trust-remote-code
```

`CUDA_VISIBLE_DEVICES` 는 MIG 환경이면 `MIG-<instance-UUID>` 로, L4 단일 GPU 환경이면 `0` 등으로 지정한다.
