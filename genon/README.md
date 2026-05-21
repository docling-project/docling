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

1. `HWP_SDK_TOKEN` 설정 (HWP SDK private 레포 다운로드용 — **두 variant 모두 필수**)
   - 대상 레포: [`HeechanKim-Genon/hwp_sdk`](https://huggingface.co/datasets/HeechanKim-Genon/hwp_sdk) — 이 레포 전용 read 토큰
   - 토큰 값은 [제논 내부 드라이브 (HWP_SDK_TOKEN)](https://docs.google.com/document/d/1c2kHPus5QxFN0jhfH37EDFORd6pt2rermkINfDFlQbs/edit?usp=drive_link) 에서 확인
   - `doc_parser/` (레포 최상위 경로) 에서 아래 명령어 실행 (이후 재실행 불필요, Git 미추적):
     - `hf_xxx_your_hwp_sdk_token_here` 부분을 위 드라이브에 적힌 토큰 값으로 교체후 아래 명령어 실행.
     ```shell
     echo "HWP_SDK_TOKEN=hf_xxx_your_hwp_sdk_token_here" >> build-script/hf_private_token.env
     ```
   - `doc-parser-build.config` 에 직접 입력하거나 push 하지 말 것 (토큰은 반드시 위 명령어를 통해 `hf_private_token.env` 파일에만 존재해야함)

2. `BUILD_VARIANT` 선택 - 무료용(**`opensource`**) / 유료용(**`enterprise`**) 버전 선택
   - **`opensource`** — LibreOffice + rhwp(외부 HTTP API 호출) 만 포함. PDF SDK 자산이 이미지에 일절 들어가지 않음 (다운로드 단계 자체가 없음). 회사 내부 PDF SDK 라이선스가 없는 환경/외부 배포용. **1번 과정인 `HWP_SDK_TOKEN` 만 있으면 됨**.
   - **`enterprise`** — 위 + 유료 PDF SDK 포함. HWP → PDF 변환 chain 이 `pdf_sdk → rhwp → libreoffice` 순으로 동작. **`HWP_SDK_TOKEN` 에 더해 `PDF_SDK_TOKEN` 도 필수**.
     - `PDF_SDK_TOKEN` — 대상 레포 [`HeechanKim-Genon/pdf_sdk`](https://huggingface.co/datasets/HeechanKim-Genon/pdf_sdk) 전용 read 토큰
     - 토큰 값은 [제논 내부 드라이브 (PDF_SDK_TOKEN)](https://docs.google.com/document/d/1amoktYvYYk74m81u2hxlBn7ljzoe-apNB56doNRZmhQ/edit?usp=drive_link) 에서 확인. 
       - 1번 과정에서 사용된 [(HWP_SDK_TOKEN)](https://docs.google.com/document/d/1c2kHPus5QxFN0jhfH37EDFORd6pt2rermkINfDFlQbs/edit?usp=drive_link) 이랑은 다른 값임.
     - `doc_parser/` (레포 최상위 경로) 에서 아래 명령어 실행 (이후 재실행 불필요, Git 미추적):
       - `hf_xxx_your_hwp_sdk_token_here` 부분을 위 드라이브에 적힌 토큰 값으로 교체후 아래 명령어 실행.
       ```shell
       echo "PDF_SDK_TOKEN=hf_yyy_your_pdf_sdk_token_here" >> build-script/hf_private_token.env
       ```
   - **`opensource`**/**`enterprise`** 버전에 따른 안내 적용후, build-script 디렉토리 이동 
   - [`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 `BUILD_VARIANT=` 라인을 둘 중 하나로 설정 해야한다:
     ```bash
     # build-script/doc-parser-build.config
     BUILD_VARIANT=opensource   # 또는 enterprise
     ```
     - 비워둔 채 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단된다 (의도치 않게 **`enterprise`** 가 배포될 위험을 막기 위한 안전장치).
     - 빌드 시 `DOCKERFILE_PATH` 가 자동으로 `genon/preprocessor/docker/Dockerfile.${BUILD_VARIANT}` 로 결정된다.
     - 두 variant 의 런타임 동작 차이 / chain 우선순위는 [`preprocessor/docker/README.md`](preprocessor/docker/README.md) 참고.

3. `HW_VARIANT` 선택 - GPU(**`gpu`**) / CPU(**`cpu`**) 빌드 선택
   - **`gpu`** — `uv.lock` 기준 그대로. torch CUDA wheel + nvidia-* / triton 포함. GPU 가속 환경용.
   - **`cpu`** — builder 단계에서 torch / torchvision 을 CPU wheel(`https://download.pytorch.org/whl/cpu`)로 재설치하고 nvidia-* / triton 패키지를 제거한 경량 이미지. GPU 없는 환경용.
   - [`doc-parser-build.config`](../build-script/doc-parser-build.config) 의 `HW_VARIANT=` 라인을 둘 중 하나로 설정:
     ```bash
     # build-script/doc-parser-build.config
     HW_VARIANT=gpu   # 또는 cpu
     ```
     - 비워둔 채 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단된다.
     - 최종 이미지 태그는 `${IMAGE_VERSION}-${BUILD_VARIANT}-${HW_VARIANT}` 형태가 된다 (예: `:1.3.6.3-opensource-cpu`).
   - `BUILD_VARIANT` × `HW_VARIANT` 조합으로 최대 4종(`opensource-gpu` / `opensource-cpu` / `enterprise-gpu` / `enterprise-cpu`)의 이미지를 만들 수 있다.

> **정리** — 위 2번 (`BUILD_VARIANT`) × 3번 (`HW_VARIANT`) 의 조합으로 **총 4종의 Dockerfile(이미지)** 가 만들어진다. 운영 환경에 맞는 1개를 골라서 빌드하면 된다.

### B. 이미지 빌드 (4~5번)

4. [doc-parser-build.config](../build-script/doc-parser-build.config) 기타 변경 사항 반영 (1·2번을 수행했다면 `HWP_SDK_TOKEN` / `PDF_SDK_TOKEN` 값은 직접 입력하지 말 것)

5. 실행 [doc-parser-build.sh](../build-script/doc-parser-build.sh)
   - `SMOKE_TEST=true`(기본값) 면 빌드 직후 컨테이너를 띄워 `SMOKE_TEST_FILE`(기본 `pdf_sample.pdf`) 한 건을 파싱하고 torch 의 CUDA 포함 여부가 `HW_VARIANT` 와 일치하는지 검증한다. 실패하면 빌드도 실패한다.

### C. 레지스트리 등록 (6~7번)

6. [register.config](preprocessor/scripts/register.config) 변경 사항 있을 시 변경 필요

7. 실행 [register_image.sh](preprocessor/scripts/register_image.sh) : push와 디비에 등록해준다.
   - `BUILD_VARIANT` / `HW_VARIANT` 환경변수를 함께 주면 베이스 `IMAGE_TAG` 에 자동으로 `-${BUILD_VARIANT}-${HW_VARIANT}` suffix 가 붙는다 (빌드 태그와 동일 규칙). 예:
     ```shell
     # opensource + cpu 조합 등록 (실제 등록할 조합으로 교체해서 실행)
     BUILD_VARIANT=opensource HW_VARIANT=cpu bash genon/preprocessor/scripts/register_image.sh
     ```
   - 스크립트는 실행 시 interactive prompt(Registry / Image / Tag / MySQL 사용자명 / Redis FLUSHALL)를 띄운다. 각 prompt 의 default 값(`[...]` 안의 값) 이 위 env 로 박힌 태그라서 그대로 Enter 만 쳐도 진행 가능.
   - 등록할 조합이 여러 개면 위 명령을 조합 수만큼(`BUILD_VARIANT` / `HW_VARIANT` 값만 바꿔서) 한 번씩 직접 실행한다 (한 번에 batch 자동화 불가).

### D. 사이트 배포 (8번)

8. 사이트 배포 시 (조합별로 동일하게 진행 — 아래는 opensource-cpu 예시)
```shell
# 1. 이미지 저장
docker save mncregistry:30500/mnc/doc-parser-preprocessor:1.3.6.3-opensource-cpu | gzip > doc-parser-preprocessor-opensource-cpu.tar.gz
# 2. 사이트에서 이미지 복원
gunzip -c doc-parser-preprocessor-opensource-cpu.tar.gz | docker load
# 3. 해당 조합으로 register_image.sh 실행 (BUILD_VARIANT / HW_VARIANT 지정)
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
