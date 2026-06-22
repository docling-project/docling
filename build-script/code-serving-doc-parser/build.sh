#!/usr/bin/env bash
set -euo pipefail

# doc-parser 코드서빙 base 이미지 빌드 스크립트.
# build context = repo 루트 (doc-parser-build.sh 와 동일 방식). genon/preprocessor/resources 와
# build-script/hf_private_token.env(HWP_SDK_TOKEN) 를 재사용한다.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_FILE="${SCRIPT_DIR}/build.config"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
else
  echo "[WARN] ${CONFIG_FILE} 이(가) 없어서 기본값으로 빌드합니다."
fi

# 로컬 전용 토큰 파일(Git 미추적) — 전처리기 빌드와 동일 파일 재사용
LOCAL_TOKEN_FILE="${ROOT_DIR}/build-script/hf_private_token.env"
if [[ -f "${LOCAL_TOKEN_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_TOKEN_FILE}"
fi

DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-doc-parser-serving}"
IMAGE_VERSION="${IMAGE_VERSION:-latest}"
HW_VARIANT="${HW_VARIANT:-}"
SMOKE_TEST="${SMOKE_TEST:-true}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
NLTK_PACKAGES="${NLTK_PACKAGES:-all}"
RHWP_GIT_REF="${RHWP_GIT_REF:-genos/main}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
APP_UNAME="${APP_UNAME:-genos}"
APP_GNAME="${APP_GNAME:-genos}"
export HWP_SDK_TOKEN="${HWP_SDK_TOKEN:-}"

# HW_VARIANT 분기
case "${HW_VARIANT}" in
  cpu)
    DOCKERFILE_PATH="build-script/code-serving-doc-parser/Dockerfile"
    IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}"
    ;;
  gpu)
    DOCKERFILE_PATH="build-script/code-serving-doc-parser/Dockerfile.gpu"
    IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}-gpu"
    ;;
  "")
    echo "[ERROR] HW_VARIANT 가 비어 있습니다."
    echo "        build-script/code-serving-doc-parser/build.config 에서 다음 중 하나로 명시하세요:"
    echo "          HW_VARIANT=cpu   # CPU 빌드 (python:3.12-slim, CPU torch)"
    echo "          HW_VARIANT=gpu   # GPU 빌드 (nvidia/cuda 12.4.1 + torch cu124)"
    exit 1
    ;;
  *)
    echo "[ERROR] HW_VARIANT 는 gpu 또는 cpu 만 허용됩니다 (현재: '${HW_VARIANT}')."
    exit 1
    ;;
esac

# HWP_SDK_TOKEN 필수 (HWP SDK 다운로드용)
if [[ -z "${HWP_SDK_TOKEN}" ]]; then
  echo "[ERROR] HWP_SDK_TOKEN 이 설정되지 않았습니다 (HWP SDK 다운로드에 필요)."
  echo "[ERROR] build-script/hf_private_token.env 에 다음을 추가하세요:"
  echo "          echo \"HWP_SDK_TOKEN=hf_xxx\" >> build-script/hf_private_token.env"
  exit 1
fi

echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] HW_VARIANT      = ${HW_VARIANT}"
echo "[INFO] DOCKERFILE_PATH = ${DOCKERFILE_PATH}"
echo "[INFO] IMAGE_TAG       = ${IMAGE_TAG}"
echo "[INFO] UID:GID         = ${APP_UID}:${APP_GID} (${APP_UNAME}:${APP_GNAME})"
echo "[INFO] NLTK_PACKAGES   = ${NLTK_PACKAGES}"
echo "[INFO] HWP_SDK_TOKEN   = (감지됨)"
echo "[INFO] SMOKE_TEST      = ${SMOKE_TEST}"

# 빌드 (context = repo 루트, BuildKit secret 으로 HWP_SDK_TOKEN 주입)
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -f "${ROOT_DIR}/${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --secret id=HWP_SDK_TOKEN,env=HWP_SDK_TOKEN \
  --build-arg HW_VARIANT="${HW_VARIANT}" \
  --build-arg TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL}" \
  --build-arg NLTK_PACKAGES="${NLTK_PACKAGES}" \
  --build-arg RHWP_GIT_REF="${RHWP_GIT_REF}" \
  --build-arg UID="${APP_UID}" \
  --build-arg GID="${APP_GID}" \
  --build-arg UNAME="${APP_UNAME}" \
  --build-arg GNAME="${APP_GNAME}" \
  "${ROOT_DIR}"

echo "[INFO] Build done: ${IMAGE_TAG}"

# 빌드 직후 smoke: venv import + torch variant + 아티팩트 존재 + docling_parse 설치 위치
if [[ "${SMOKE_TEST}" == "true" ]]; then
  echo "[INFO] Smoke test 실행 (HW_VARIANT=${HW_VARIANT})"
  docker run --rm \
    --platform linux/amd64 \
    -e HW_VARIANT="${HW_VARIANT}" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    -c '
      set -euo pipefail
      # 아티팩트 존재 확인
      test -d /models || { echo "[SMOKE] /models 없음"; exit 1; }
      test -x /app/hwp_sdk/convtext || { echo "[SMOKE] hwp_sdk/convtext 없음"; exit 1; }
      test -x /usr/local/bin/rhwp || { echo "[SMOKE] rhwp 없음"; exit 1; }
      /app/.venv/bin/python - <<"PY"
import os, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
variant = os.environ["HW_VARIANT"]

import fastapi, uvicorn          # noqa: F401  (web)
import docling_core              # noqa: F401  (parsing)
import transformers              # noqa: F401  (NLP)

# docling-parse 패치본이 실제 실행 venv 에 설치됐는지 위치 검증
import docling_parse
loc = docling_parse.__file__
print(f"[SMOKE] docling_parse @ {loc}")
assert loc.startswith("/app/.venv/"), f"docling_parse 가 venv 밖에 설치됨: {loc}"

import torch
has_cuda = torch.version.cuda is not None
print(f"[SMOKE] torch={torch.__version__} cuda_build={torch.version.cuda}")
assert has_cuda == (variant == "gpu"), \
    f"HW_VARIANT={variant} 인데 torch.version.cuda={torch.version.cuda}"
print(f"[SMOKE] ok — hw={variant}")
PY
    '
  echo "[INFO] Smoke test passed"
fi

if [[ "${PUSH_IMAGE:-false}" == "true" ]]; then
  echo "[INFO] Pushing ${IMAGE_TAG} ..."
  docker push "${IMAGE_TAG}"
  echo "[INFO] Push done"
fi
