#!/usr/bin/env bash
# B300 PaddleOCR smoke test entrypoint.
#
# 컨테이너 내부에서 실행한다. GPU 가 실제로 붙어있는 노드에서 수행되어야 한다.
#
# 사용법:
#   bash /app/etc/smoke_test.sh              # 전체 (health → inference) 순차 수행
#   bash /app/etc/smoke_test.sh health        # health 엔드포인트 응답만 확인
#   bash /app/etc/smoke_test.sh inference     # in-process OCR 추론 + GPU 사용 여부 확인
#   bash /app/etc/smoke_test.sh compare <기존_baseline.json> <현재_result.json>
#
# 환경변수:
#   OCR_PORT           기본 8080
#   HEALTH_TIMEOUT_SEC 헬스 응답 대기 최대 초 (기본 120)
#   SMOKE_OUT_DIR      결과 JSON 저장 디렉토리 (기본 /tmp/ocr_smoke)
set -euo pipefail

PORT="${OCR_PORT:-8080}"
TIMEOUT="${HEALTH_TIMEOUT_SEC:-120}"
OUT_DIR="${SMOKE_OUT_DIR:-/tmp/ocr_smoke}"
APP_DIR="${APP_DIR:-/app}"
PY="${APP_DIR}/.venv/bin/python"

mkdir -p "${OUT_DIR}"

cmd_health() {
  echo "[smoke] waiting for http://127.0.0.1:${PORT}/health (timeout ${TIMEOUT}s)"
  local elapsed=0
  while ! curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "${elapsed}" -ge "${TIMEOUT}" ]; then
      echo "[smoke] FAIL: health endpoint did not respond within ${TIMEOUT}s"
      exit 1
    fi
  done
  echo "[smoke] PASS: health endpoint OK"
}

cmd_inference() {
  echo "[smoke] running in-process OCR inference (GPU check + result dump)"
  "${PY}" "${APP_DIR}/etc/smoke_test_inference.py" \
      --out "${OUT_DIR}/result.json"
  echo "[smoke] result written to ${OUT_DIR}/result.json"
}

cmd_compare() {
  if [ "$#" -lt 2 ]; then
    echo "usage: smoke_test.sh compare <baseline.json> <current.json>"
    exit 2
  fi
  "${PY}" "${APP_DIR}/etc/smoke_test_compare.py" --baseline "$1" --current "$2"
}

case "${1:-all}" in
  health)    cmd_health ;;
  inference) cmd_inference ;;
  compare)   shift; cmd_compare "$@" ;;
  all|"")    cmd_health; cmd_inference ;;
  *)
    echo "unknown subcommand: $1"
    echo "usage: smoke_test.sh [health|inference|compare|all]"
    exit 2
    ;;
esac
