#!/usr/bin/env bash
set -euo pipefail

# doc-parser 코드서빙 base 이미지 빌드 스크립트.
# 자기완결: build context = 이 폴더 자신(build-script/code-serving-doc-parser).
# genon/code-serving / repo 루트의 파일을 전혀 참조하지 않는다.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${ROOT_DIR}/build.config"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${CONFIG_FILE}"
else
  echo "[WARN] ${CONFIG_FILE} 이(가) 없어서 기본값으로 빌드합니다."
fi

# config 에서 못 가져오면 기본값 세팅
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-doc-parser-serving}"
IMAGE_VERSION="${IMAGE_VERSION:-latest}"
HW_VARIANT="${HW_VARIANT:-}"
APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"
APP_UNAME="${APP_UNAME:-genos}"
APP_GNAME="${APP_GNAME:-genos}"
SMOKE_TEST="${SMOKE_TEST:-true}"

# HW_VARIANT 분기 (gpu / cpu 빌드 분리)
# gpu / cpu 둘 중 하나는 반드시 명시되어야 한다.
case "${HW_VARIANT}" in
  cpu)
    DOCKERFILE_PATH="Dockerfile"
    ;;
  gpu)
    DOCKERFILE_PATH="Dockerfile.gpu"
    ;;
  "")
    echo "[ERROR] HW_VARIANT 가 비어 있습니다."
    echo "        build-script/code-serving-doc-parser/build.config 에서 다음 중 하나로 명시하세요:"
    echo "          HW_VARIANT=cpu   # CPU 빌드 (python:3.12-slim, CPU torch)"
    echo "          HW_VARIANT=gpu   # GPU 빌드 (nvidia/cuda + torch cu124)"
    exit 1
    ;;
  *)
    echo "[ERROR] HW_VARIANT 는 gpu 또는 cpu 만 허용됩니다 (현재: '${HW_VARIANT}')."
    exit 1
    ;;
esac

# 최종 이미지 태그
# 기본(cpu)은 접미사 없이 ${IMAGE_VERSION}, gpu 는 ${IMAGE_VERSION}-gpu
if [[ "${HW_VARIANT}" == "cpu" ]]; then
  IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}"
else
  IMAGE_TAG="${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:${IMAGE_VERSION}-${HW_VARIANT}"
fi

echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] HW_VARIANT      = ${HW_VARIANT}"
echo "[INFO] DOCKERFILE_PATH = ${DOCKERFILE_PATH}"
echo "[INFO] IMAGE_TAG       = ${IMAGE_TAG}"
echo "[INFO] UID:GID         = ${APP_UID}:${APP_GID} (${APP_UNAME}:${APP_GNAME})"
echo "[INFO] SMOKE_TEST      = ${SMOKE_TEST}"

# BuildKit plain 로그로 보기 + 이 폴더(.)를 컨텍스트로 빌드
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -f "${ROOT_DIR}/${DOCKERFILE_PATH}" \
  -t "${IMAGE_TAG}" \
  --build-arg UID="${APP_UID}" \
  --build-arg GID="${APP_GID}" \
  --build-arg UNAME="${APP_UNAME}" \
  --build-arg GNAME="${APP_GNAME}" \
  "${ROOT_DIR}"

echo "[INFO] Build done: ${IMAGE_TAG}"

# 빌드 직후 컨테이너에서 venv 의존성 import + torch GPU/CPU variant 검증
# (base 이미지엔 repo 소스가 없어 full parse 는 불가 → 의존성/torch 분기만 확인)
if [[ "${SMOKE_TEST}" == "true" ]]; then
  echo "[INFO] Smoke test 실행 (HW_VARIANT=${HW_VARIANT})"
  docker run --rm \
    --platform linux/amd64 \
    -e HW_VARIANT="${HW_VARIANT}" \
    --entrypoint /bin/bash \
    "${IMAGE_TAG}" \
    -c '
      set -euo pipefail
      /app/.venv/bin/python - <<"PY"
import os, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

variant = os.environ["HW_VARIANT"]

# pre-bake 된 핵심 런타임 의존성 import 검증
import fastapi, uvicorn          # noqa: F401  (web)
import docling_core              # noqa: F401  (parsing)
import transformers              # noqa: F401  (NLP)

import torch
has_cuda_build = torch.version.cuda is not None
print(f"[SMOKE] torch={torch.__version__} cuda_build={torch.version.cuda}")
expected_cuda = (variant == "gpu")
assert has_cuda_build == expected_cuda, (
    f"HW_VARIANT={variant} 인데 torch.version.cuda={torch.version.cuda} — "
    "GPU/CPU 빌드 분기가 깨졌습니다."
)
print(f"[SMOKE] ok — hw={variant} deps import + torch variant 검증 통과")
PY
    '
  echo "[INFO] Smoke test passed"
fi

# 푸시 여부 플래그
if [[ "${PUSH_IMAGE:-false}" == "true" ]]; then
  echo "[INFO] Pushing ${IMAGE_TAG} ..."
  docker push "${IMAGE_TAG}"
  echo "[INFO] Push done"
fi
