#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트는 반드시 repo 루트(/workspace/doc_parser)에서 실행된다고 가정
# 다른 데서 실행해도 되게 하려면 아래에서 루트 계산
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 기본값
CONFIG_FILE="${ROOT_DIR}/build-script/doc-parser-build.config"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
else
  echo "[WARN] ${CONFIG_FILE} 이(가) 없어서 기본값으로 빌드합니다."
fi

# 로컬 전용 토큰 파일이 있으면 추가로 읽음 (HWP_SDK_TOKEN / PDF_SDK_TOKEN 민감 정보용, Git 미추적)
LOCAL_CONFIG_FILE="${ROOT_DIR}/build-script/hf_private_token.env"
if [[ -f "${LOCAL_CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_CONFIG_FILE}"
fi

# build.config 에서 못 가져오면 기본값 세팅
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-doc-parser-preprocessor}"
IMAGE_VERSION="${IMAGE_VERSION:-latest}"
BUILD_VARIANT="${BUILD_VARIANT:-}"
HW_VARIANT="${HW_VARIANT:-}"
INSTALL_LIBREOFFICE="${INSTALL_LIBREOFFICE:-true}"
INSTALL_RHWP="${INSTALL_RHWP:-true}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
APP_UNAME="${APP_UNAME:-genos}"
APP_GNAME="${APP_GNAME:-genos}"
APP_NLTK_PACKAGES="${APP_NLTK_PACKAGES:-all}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
SMOKE_TEST="${SMOKE_TEST:-true}"
SMOKE_TEST_FILE="${SMOKE_TEST_FILE:-pdf_sample.pdf}"
export HWP_SDK_TOKEN="${HWP_SDK_TOKEN:-}"
export PDF_SDK_TOKEN="${PDF_SDK_TOKEN:-}"

# BUILD_VARIANT 분기 (이슈 #199, #236 — standard/synap 두 빌드 산출물 분리)
# standard / synap 둘 중 하나는 반드시 명시되어야 한다.
# 비워둔 채 빌드를 돌리면 의도치 않게 유료 SDK 가 포함될 위험이 있어 명시적 에러 처리.
case "${BUILD_VARIANT}" in
  standard|synap)
    DOCKERFILE_PATH="genon/preprocessor/docker/Dockerfile.${BUILD_VARIANT}"
    ;;
  "")
    echo "[ERROR] BUILD_VARIANT 가 비어 있습니다."
    echo "        build-script/doc-parser-build.config 에서 다음 중 하나로 명시하세요:"
    echo "          BUILD_VARIANT=standard   # 기본 (LibreOffice + rhwp, PDF SDK 미포함)"
    echo "          BUILD_VARIANT=synap        # 위 + 유료 PDF SDK(Synap)"
    exit 1
    ;;
  *)
    echo "[ERROR] BUILD_VARIANT 는 standard 또는 synap 만 허용됩니다 (현재: '${BUILD_VARIANT}')."
    exit 1
    ;;
esac

# HW_VARIANT 분기 (이슈 #210 — GPU/CPU 빌드 분리)
# gpu / cpu 둘 중 하나는 반드시 명시되어야 한다.
case "${HW_VARIANT}" in
  gpu|cpu)
    ;;
  "")
    echo "[ERROR] HW_VARIANT 가 비어 있습니다."
    echo "        build-script/doc-parser-build.config 에서 다음 중 하나로 명시하세요:"
    echo "          HW_VARIANT=gpu   # GPU 빌드 (CUDA torch 포함)"
    echo "          HW_VARIANT=cpu   # CPU 빌드 (nvidia-* / triton 제거, CPU torch wheel)"
    exit 1
    ;;
  *)
    echo "[ERROR] HW_VARIANT 는 gpu 또는 cpu 만 허용됩니다 (현재: '${HW_VARIANT}')."
    exit 1
    ;;
esac

# INSTALL_LIBREOFFICE / INSTALL_RHWP 분기 (이슈 #286 — rhwp/LibreOffice 설치 on/off)
# Dockerfile 의 stage alias(rhwp_builder_${INSTALL_RHWP})·shell 조건이 정확히 true/false 를
# 기대하므로 다른 값은 즉시 에러 처리한다.
case "${INSTALL_LIBREOFFICE}" in
  true|false) ;;
  *)
    echo "[ERROR] INSTALL_LIBREOFFICE 는 true 또는 false 만 허용됩니다 (현재: '${INSTALL_LIBREOFFICE}')."
    exit 1
    ;;
esac
case "${INSTALL_RHWP}" in
  true|false) ;;
  *)
    echo "[ERROR] INSTALL_RHWP 는 true 또는 false 만 허용됩니다 (현재: '${INSTALL_RHWP}')."
    exit 1
    ;;
esac

# standard 에서 rhwp/LibreOffice 를 둘 다 끄면 변환 backend 가 0개 → 적재형(지능형)은 비-PDF 처리 불가.
# (첨부형/변환형/파싱형은 HWP SDK·원본 파싱으로 동작, synap 은 PDF SDK 가 남아 해당 없음.)
# 의도된 구성일 수 있으니 막지 않고 경고만.
if [[ "${BUILD_VARIANT}" == "standard" && "${INSTALL_LIBREOFFICE}" == "false" && "${INSTALL_RHWP}" == "false" ]]; then
  echo "[WARN] standard + INSTALL_LIBREOFFICE=false + INSTALL_RHWP=false 조합입니다."
  echo "[WARN] 이 이미지에는 HWP/오피스 → PDF 변환 backend 가 전혀 없습니다. 영향은 전처리기별로 다릅니다:"
  echo "[WARN]   - 적재형(지능형): 비-PDF 입력은 내부 PDF 변환이 필요 → 처리 불가. PDF 로 변환된 문서를 입력해야 함."
  echo "[WARN]   - 첨부형/변환형/파싱형: HWP 는 내장 HWP SDK, docx/ppt 는 원본을 직접 파싱하므로 동작(영향 적음)."
  echo "[WARN]   (의도된 구성이면 무시)"
fi

# rhwp/LibreOffice 를 끈 경우 태그 접미사 (이슈 #286)
# 기본(둘 다 on)이면 빈 문자열 → 기존 태그 그대로. 끈 패키지만 명시적으로 붙인다
# (off 이미지가 운영 이미지와 같은 태그로 push 돼 덮어쓰는 사고 방지).
#   LibreOffice off → -nolibre / rhwp off → -norhwp / 둘 다 off → -nolibre-norhwp
CONV_SUFFIX=""
if [[ "${INSTALL_LIBREOFFICE}" == "false" ]]; then CONV_SUFFIX="${CONV_SUFFIX}-nolibre"; fi
if [[ "${INSTALL_RHWP}" == "false" ]]; then CONV_SUFFIX="${CONV_SUFFIX}-norhwp"; fi

# 최종 이미지 태그 (이슈 #236, #286)
# 기본 조합(cpu + standard)은 가장 기본 산출물이므로 접미사 없이 ${IMAGE_VERSION} 만 사용.
# 그 외 조합은 hw 먼저, variant 나중 순서로 접미사: ${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}
# 마지막으로 rhwp/LibreOffice off 면 ${CONV_SUFFIX} 가 더 붙는다.
#   예) gpu+synap → :1.3.6.3-gpu-synap / cpu+standard → :1.3.6.3
#       cpu+standard + 둘 다 off → :1.3.6.3-nolibre-norhwp
if [[ "${HW_VARIANT}" == "cpu" && "${BUILD_VARIANT}" == "standard" ]]; then
  IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}${CONV_SUFFIX}"
else
  IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}${CONV_SUFFIX}"
fi

echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] BUILD_VARIANT   = ${BUILD_VARIANT}"
echo "[INFO] HW_VARIANT      = ${HW_VARIANT}"
echo "[INFO] INSTALL_LIBREOFFICE = ${INSTALL_LIBREOFFICE}"
echo "[INFO] INSTALL_RHWP        = ${INSTALL_RHWP}"
echo "[INFO] DOCKERFILE_PATH = ${DOCKERFILE_PATH}"
echo "[INFO] IMAGE_TAG       = ${IMAGE_TAG}"
echo "[INFO] UID:GID         = ${APP_UID}:${APP_GID} (${APP_UNAME}:${APP_GNAME})"
echo "[INFO] NLTK_PACKAGES  = ${APP_NLTK_PACKAGES}"
echo "[INFO] SMOKE_TEST      = ${SMOKE_TEST} (file: ${SMOKE_TEST_FILE})"

# HuggingFace 토큰 존재 여부 확인 (이슈 #199 — SDK 별 fine-grained 토큰 분리)
# HWP_SDK_TOKEN  : HeechanKim-Genon/hwp_sdk 전용 read 토큰 (두 variant 모두 필수)
# PDF_SDK_TOKEN  : HeechanKim-Genon/pdf_sdk 전용 read 토큰 (synap 일 때만 필수)
if [[ -z "${HWP_SDK_TOKEN}" ]]; then
  echo "[ERROR] HWP_SDK_TOKEN 이 설정되지 않았습니다. HeechanKim-Genon/hwp_sdk 다운로드에 필요합니다."
  echo "[ERROR] build-script/hf_private_token.env 또는 환경변수에 HWP_SDK_TOKEN 을 설정하세요."
  exit 1
fi
echo "[INFO] HWP_SDK_TOKEN 감지됨."

if [[ "${BUILD_VARIANT}" == "synap" && -z "${PDF_SDK_TOKEN}" ]]; then
  echo "[ERROR] synap 빌드는 PDF_SDK_TOKEN 도 필요합니다 (HeechanKim-Genon/pdf_sdk 다운로드용)."
  echo "[ERROR] build-script/hf_private_token.env 또는 환경변수에 PDF_SDK_TOKEN 을 설정하세요."
  exit 1
fi
if [[ -n "${PDF_SDK_TOKEN}" ]]; then
  echo "[INFO] PDF_SDK_TOKEN 감지됨."
fi

# BuildKit secret mount: 두 토큰을 각자 다른 secret id 로 노출.
# standard Dockerfile 은 HWP_SDK_TOKEN 만 사용 → PDF_SDK_TOKEN 마운트는 무해.
SECRET_ARGS=(--secret id=HWP_SDK_TOKEN,env=HWP_SDK_TOKEN)
if [[ -n "${PDF_SDK_TOKEN}" ]]; then
  SECRET_ARGS+=(--secret id=PDF_SDK_TOKEN,env=PDF_SDK_TOKEN)
fi

# BuildKit plain 로그로 보기 + 루트(.)를 컨텍스트로 빌드
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -f "${ROOT_DIR}/${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  "${SECRET_ARGS[@]}" \
  --build-arg HW_VARIANT="${HW_VARIANT}" \
  --build-arg INSTALL_LIBREOFFICE="${INSTALL_LIBREOFFICE}" \
  --build-arg INSTALL_RHWP="${INSTALL_RHWP}" \
  --build-arg TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL}" \
  --build-arg UID="${APP_UID}" \
  --build-arg GID="${APP_GID}" \
  --build-arg UNAME="${APP_UNAME}" \
  --build-arg GNAME="${APP_GNAME}" \
  --build-arg NLTK_PACKAGES="${APP_NLTK_PACKAGES}" \
  "${ROOT_DIR}"

echo "[INFO] Build done: ${IMAGE_TAG}"

# 이슈 #210 — 빌드 직후 컨테이너에서 torch variant 검증 + 샘플 1건 파싱 smoke
if [[ "${SMOKE_TEST}" == "true" ]]; then
  SAMPLE_HOST_PATH="${ROOT_DIR}/genon/preprocessor/sample_files/${SMOKE_TEST_FILE}"
  if [[ ! -f "${SAMPLE_HOST_PATH}" ]]; then
    echo "[WARN] smoke test 샘플 파일이 없습니다: ${SAMPLE_HOST_PATH} — smoke test 스킵"
  else
    echo "[INFO] Smoke test 실행 (BUILD_VARIANT=${BUILD_VARIANT}, HW_VARIANT=${HW_VARIANT}): ${SMOKE_TEST_FILE}"
    docker run --rm \
      --platform linux/amd64 \
      -e HW_VARIANT="${HW_VARIANT}" \
      -e SMOKE_SAMPLE="/app/sample_files/${SMOKE_TEST_FILE}" \
      -v "${SAMPLE_HOST_PATH}:/app/sample_files/${SMOKE_TEST_FILE}:ro" \
      --entrypoint /bin/bash \
      "${IMAGE_TAG}" \
      -c '
        set -euo pipefail
        cd /app/src
        # venv python 절대경로로 호출 (로그인 셸 /etc/profile 이 PATH 를 덮어쓰면
        # `python` 이 시스템 python 으로 가서 venv 의 torch 를 못 찾는 문제 회피)
        /app/.venv/bin/python - <<"PY"
import os, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

variant = os.environ["HW_VARIANT"]
sample = os.environ["SMOKE_SAMPLE"]

import torch
has_cuda_build = torch.version.cuda is not None
print(f"[SMOKE] torch={torch.__version__} cuda_build={torch.version.cuda}")
expected_cuda = (variant == "gpu")
assert has_cuda_build == expected_cuda, (
    f"HW_VARIANT={variant} 인데 torch.version.cuda={torch.version.cuda} — "
    "GPU/CPU 빌드 분기가 깨졌습니다."
)

# 변종 확인 후 실제 파싱 1건
from preprocessor import DocumentProcessor
dp = DocumentProcessor()
docs = dp.load_documents(sample)
assert docs, "load_documents returned empty"
chunks = dp.split_documents(docs)
assert chunks, "split_documents returned empty"
print(f"[SMOKE] ok — hw={variant} sample={sample} chunks={len(chunks)}")
PY
      '
    echo "[INFO] Smoke test passed"
  fi
fi

# 푸시 여부 플래그
# 이슈 #236 — synap 이미지는 공유 레지스트리(Genos)에 올리지 않는다. 빌드만 하고 push 스킵.
#   (Synap 유료 SDK 가 공유 레지스트리로 새어나가는 것을 막기 위함. synap 배포는 AI Search 팀 경유)
if [[ "${BUILD_VARIANT}" == "synap" ]]; then
  echo "[INFO] BUILD_VARIANT=synap — 공유 레지스트리 push 를 건너뜁니다 (이미지 빌드만 수행)."
  echo "[INFO] synap 이미지 배포가 필요하면 AI Search 팀에 문의하세요."
elif [[ "${PUSH_IMAGE:-false}" == "true" ]]; then
  echo "[INFO] Pushing ${IMAGE_TAG} ..."
  docker push "${IMAGE_TAG}"
  echo "[INFO] Push done"
fi
