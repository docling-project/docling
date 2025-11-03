#!/usr/bin/env bash
set -euo pipefail

# 이 스크립트는 /workspace/doc_parser 에서 실행된다고 가정
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONF_FILE="${BASE_DIR}/build-script/paddle-ocr-build.config"
if [[ ! -f "$CONF_FILE" ]]; then
  echo "config file not found: $CONF_FILE"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONF_FILE"

# 기본값
CONTEXT="${CONTEXT:-$BASE_DIR}"
DOCKERFILE="${DOCKERFILE:-genon/serving/paddle/docker/Dockerfile}"
IMAGE_NAME="${IMAGE_NAME:-doc-parser-ocr}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"

# 태그 조합
if [[ -n "$REGISTRY" ]]; then
  FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
  FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "[INFO] building image: $FULL_IMAGE"
echo "[INFO] context      : $CONTEXT"
echo "[INFO] dockerfile   : $DOCKERFILE"

# buildx 안 쓰는 기본 빌드
docker build \
  --file "$DOCKERFILE" \
  --build-arg PADDLE_EXTRA_INDEX_URL="${PADDLE_EXTRA_INDEX_URL:-}" \
  --tag "$FULL_IMAGE" \
  "$CONTEXT"

echo "[INFO] build done: $FULL_IMAGE"

# 레지스트리 있으면 푸시
if [[ -n "$REGISTRY" ]]; then
  echo "[INFO] pushing to $FULL_IMAGE"
  docker push "$FULL_IMAGE"
fi
