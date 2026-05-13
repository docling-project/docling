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

1. `HF_TOKEN` 설정 (HWP SDK · PDF SDK를 private 레포에서 다운로드하기 위한 토큰)
   - 동일한 토큰으로 두 SDK 모두 다운로드됨 (`HeechanKim-Genon/hwp_sdk`, `HeechanKim-Genon/pdf_sdk`)
   - 토큰 값은 [제논 내부 드라이브](https://drive.google.com/file/d/1m8aom4_zo3ZuQ-HdHHpkRsVOJakN-Lt5/view?usp=sharing)에서 확인
   - `doc_parser/` (레포 최상위 경로) 에서 아래 명령어 한 번 실행 (이후 재실행 불필요, Git에 커밋되지 않음):
     ```shell
     echo "HF_TOKEN=hf_your_token_here" > build-script/hf_private_token.env
     ```
   - `doc-parser-build.config`에 직접 입력하거나 push하지 말 것 (토큰은 반드시 `hf_private_token.env` 파일에만)
2. build-script 디렉토리 이동
3. [doc-parser-build.config](../build-script/doc-parser-build.config) 기타 변경 사항 반영 (1번을 수행했다면 `HF_TOKEN`값은 직접 입력하지 말 것)
4. 실행 [doc-parser-build.sh](../build-script/doc-parser-build.sh)
5. [register.config](preprocessor/scripts/register.config) 변경 사항 있을 시 변경 필요
6. 실행 [register_image.sh](preprocessor/scripts/register_image.sh) : push와 디비에 등록해준다.
7. 사이트 배포 시
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

1. HuggingFace 인증 (위 빌드 단계 1번과 동일한 토큰)
   ```shell
   export HF_TOKEN=hf_your_token_here
   ```
2. 레포 최상위(`doc_parser/`) 경로에서 두 SDK 다운로드
   ```shell
   # HWP SDK
   huggingface-cli download HeechanKim-Genon/hwp_sdk \
     --repo-type dataset \
     --local-dir ./hwp_sdk \
     --local-dir-use-symlinks False
   chmod +x ./hwp_sdk/convtext

   # PDF SDK
   huggingface-cli download HeechanKim-Genon/pdf_sdk \
     --repo-type dataset \
     --local-dir ./pdf_sdk \
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
