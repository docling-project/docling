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
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
APP_UNAME="${APP_UNAME:-genos}"
APP_GNAME="${APP_GNAME:-genos}"
APP_NLTK_PACKAGES="${APP_NLTK_PACKAGES:-all}"
export HWP_SDK_TOKEN="${HWP_SDK_TOKEN:-}"
export PDF_SDK_TOKEN="${PDF_SDK_TOKEN:-}"

# BUILD_VARIANT 분기 (이슈 #199 — 오픈소스/엔터프라이즈 두 빌드 산출물 분리)
# opensource / enterprise 둘 중 하나는 반드시 명시되어야 한다.
# 비워둔 채 빌드를 돌리면 의도치 않게 유료 SDK 가 포함될 위험이 있어 명시적 에러 처리.
case "${BUILD_VARIANT}" in
  opensource|enterprise)
    DOCKERFILE_PATH="genon/preprocessor/docker/Dockerfile.${BUILD_VARIANT}"
    IMAGE_TAG_SUFFIX="-${BUILD_VARIANT}"
    ;;
  "")
    echo "[ERROR] BUILD_VARIANT 가 비어 있습니다."
    echo "        build-script/doc-parser-build.config 에서 다음 중 하나로 명시하세요:"
    echo "          BUILD_VARIANT=opensource   # 오픈소스 (LibreOffice + rhwp HTTP)"
    echo "          BUILD_VARIANT=enterprise   # 엔터프라이즈 (위 + 유료 PDF SDK)"
    exit 1
    ;;
  *)
    echo "[ERROR] BUILD_VARIANT 는 opensource 또는 enterprise 만 허용됩니다 (현재: '${BUILD_VARIANT}')."
    exit 1
    ;;
esac

# 최종 이미지 태그
IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}${IMAGE_TAG_SUFFIX}"

echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] BUILD_VARIANT   = ${BUILD_VARIANT}"
echo "[INFO] DOCKERFILE_PATH = ${DOCKERFILE_PATH}"
echo "[INFO] IMAGE_TAG       = ${IMAGE_TAG}"
echo "[INFO] UID:GID         = ${APP_UID}:${APP_GID} (${APP_UNAME}:${APP_GNAME})"
echo "[INFO] NLTK_PACKAGES  = ${APP_NLTK_PACKAGES}"

# HuggingFace 토큰 존재 여부 확인 (이슈 #199 — SDK 별 fine-grained 토큰 분리)
# HWP_SDK_TOKEN  : HeechanKim-Genon/hwp_sdk 전용 read 토큰 (두 variant 모두 필수)
# PDF_SDK_TOKEN  : HeechanKim-Genon/pdf_sdk 전용 read 토큰 (enterprise 일 때만 필수)
if [[ -z "${HWP_SDK_TOKEN}" ]]; then
  echo "[ERROR] HWP_SDK_TOKEN 이 설정되지 않았습니다. HeechanKim-Genon/hwp_sdk 다운로드에 필요합니다."
  echo "[ERROR] build-script/hf_private_token.env 또는 환경변수에 HWP_SDK_TOKEN 을 설정하세요."
  exit 1
fi
echo "[INFO] HWP_SDK_TOKEN 감지됨."

if [[ "${BUILD_VARIANT}" == "enterprise" && -z "${PDF_SDK_TOKEN}" ]]; then
  echo "[ERROR] enterprise 빌드는 PDF_SDK_TOKEN 도 필요합니다 (HeechanKim-Genon/pdf_sdk 다운로드용)."
  echo "[ERROR] build-script/hf_private_token.env 또는 환경변수에 PDF_SDK_TOKEN 을 설정하세요."
  exit 1
fi
if [[ -n "${PDF_SDK_TOKEN}" ]]; then
  echo "[INFO] PDF_SDK_TOKEN 감지됨."
fi

# BuildKit secret mount: 두 토큰을 각자 다른 secret id 로 노출.
# opensource Dockerfile 은 HWP_SDK_TOKEN 만 사용 → PDF_SDK_TOKEN 마운트는 무해.
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
  --build-arg UID="${APP_UID}" \
  --build-arg GID="${APP_GID}" \
  --build-arg UNAME="${APP_UNAME}" \
  --build-arg GNAME="${APP_GNAME}" \
  --build-arg NLTK_PACKAGES="${APP_NLTK_PACKAGES}" \
  "${ROOT_DIR}"

echo "[INFO] Build done: ${IMAGE_TAG}"

# 푸시 여부 플래그
if [[ "${PUSH_IMAGE:-false}" == "true" ]]; then
  echo "[INFO] Pushing ${IMAGE_TAG} ..."
  docker push "${IMAGE_TAG}"
  echo "[INFO] Push done"
fi
