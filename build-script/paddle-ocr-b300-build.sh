#!/usr/bin/env bash
# B300 (Blackwell / cu129) PaddleOCR 이미지 빌드 스크립트.
# 기존 paddle-ocr-build.sh 와 분리된 진입점이며, paddle-ocr-b300-build.config 를 읽는다.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONF_FILE="${BASE_DIR}/build-script/paddle-ocr-b300-build.config"
if [[ ! -f "$CONF_FILE" ]]; then
  echo "config file not found: $CONF_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

CONTEXT="${CONTEXT:-$BASE_DIR}"
DOCKERFILE="${DOCKERFILE:-genon/serving/paddle/docker/Dockerfile-b300}"
# 상대 경로로 들어왔으면 cwd 가 아니라 repo root (BASE_DIR) 기준으로 정규화한다.
# (README 안내대로 `cd build-script` 후 실행해도 빌드가 깨지지 않게 하기 위함)
[[ "$CONTEXT" != /* ]] && CONTEXT="${BASE_DIR}/${CONTEXT}"
[[ "$DOCKERFILE" != /* ]] && DOCKERFILE="${BASE_DIR}/${DOCKERFILE}"
IMAGE_NAME="${IMAGE_NAME:-doc-parser-ocr-b300}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

if [[ -n "$REGISTRY" ]]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "[INFO] building B300 image: $FULL_IMAGE"
echo "[INFO] context            : $CONTEXT"
echo "[INFO] dockerfile         : $DOCKERFILE"
echo "[INFO] paddle index       : ${PADDLE_EXTRA_INDEX_URL:-<from-pyproject>}"

docker build \
  --file "$DOCKERFILE" \
  --build-arg PADDLE_EXTRA_INDEX_URL="${PADDLE_EXTRA_INDEX_URL:-}" \
  --tag "$FULL_IMAGE" \
  "$CONTEXT"

echo "[INFO] build done: $FULL_IMAGE"

if [[ -n "$REGISTRY" ]]; then
  echo "[INFO] pushing to $FULL_IMAGE"
  docker push "$FULL_IMAGE"
fi
