#!/bin/sh
# Python code-serving entrypoint.
# - commit hash 별 marker 로 init.sh 실행 가드 — commit 바뀌면 재실행, fail 시 marker 안 박힘.
# - START_COMMAND 가 주어졌으면 사용자 시작 커맨드 실행 (cwd=/app/src/service).
# - 비어있으면 자동 감지:
#     1) /app/src/service/main.py 가 있으면 uvicorn main:app 실행
#     2) 그 외 fallback — 기존 디폴트(gunicorn src.main:app)
# PORT 는 admin-api에서 8080으로 주입됨.
set -eu

# commit hash 별 marker — 새 commit 으로 재배포 시 init.sh 자동 재실행.
# COMMIT_HASH 미설정이면 'noref' 로 fallback (개발 편의).
MARKER_DIR="/app"
MARKER="${MARKER_DIR}/.init_done.${COMMIT_HASH:-noref}"

if [ ! -f "$MARKER" ]; then
  # 이전 commit 의 marker 들 제거 (디스크 누적 방지)
  rm -f "${MARKER_DIR}/.init_done"* 2>/dev/null || true
  if sh scripts/init.sh; then
    touch "$MARKER"
  else
    echo "[entrypoint] WARNING: init.sh failed (exit=$?). Marker not set — will retry on next boot." >&2
    # init.sh 자체는 || echo WARNING 으로 감싸여 거의 항상 0 으로 끝나지만
    # set -e 환경에서 명시적 fail 시 marker 미박힘 보장.
  fi
fi

cd /app/src/service

if [ -n "${START_COMMAND:-}" ]; then
  echo "[entrypoint] Running user START_COMMAND: ${START_COMMAND}"
  exec sh -c "${START_COMMAND}"
fi

# 디폴트 — main.py 자동 감지 (사용자가 START_COMMAND 안 채워도 동작)
if [ -f "main.py" ]; then
  echo "[entrypoint] Default: uvicorn main:app on port ${PORT:-8080}"
  exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"
fi

if [ -f "src/main.py" ]; then
  echo "[entrypoint] Default: uvicorn src.main:app on port ${PORT:-8080}"
  exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8080}"
fi

# 최후 fallback — 기존 호환 (베이스 이미지 자체 main 실행. 사용자 코드 없을 때)
echo "[entrypoint] No main.py found, falling back to base gunicorn"
cd /app
exec gunicorn -c gunicorn/gunicorn_conf.py src.main:app
