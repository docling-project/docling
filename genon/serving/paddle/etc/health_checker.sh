#!/usr/bin/env bash
# Supervisor EventListener 프로토콜: READY/RESULT 로 통신
# 참고: https://supervisord.org/events.html

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${OCR_PORT:-8080}/health}"
TIMEOUT="${HEALTH_TIMEOUT:-3}"
RETRIES="${HEALTH_RETRIES:-3}"

# supervisor 이벤트 루프
while true; do
  echo "READY"
  read -r line || exit 1
  # 이벤트 페이로드 읽기 (버리기)
  headers=""
  while read -r h && [ "$h" != "" ]; do
    headers+="$h"$'\n'
  done

  # --- 실제 헬스 체크 ---
  ok=0
  for i in $(seq 1 "$RETRIES"); do
    if curl -fsS --max-time "$TIMEOUT" "$HEALTH_URL" >/dev/null 2>&1; then
      ok=1
      break
    fi
    sleep 1
  done

  if [ $ok -eq 1 ]; then
    # 헬시: 아무것도 안 함
    echo -ne "RESULT 2\nOK"
  else
    # 언헬시: paddlex 재시작
    /usr/bin/supervisorctl restart paddlex >/dev/null 2>&1 || true
    echo -ne "RESULT 13\nRESTARTED"
  fi
done
