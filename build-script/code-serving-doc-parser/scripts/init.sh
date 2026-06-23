#!/bin/sh
# Python code-serving 초기화 스크립트.
# 컨테이너 최초 부팅 시 한 번만 실행 (entrypoint.sh의 marker로 가드).
#
# 동작:
#   1. Gitea repo clone (REPOSITORY_URL + COMMIT_HASH).
#   2. BUILD_COMMAND 가 주어졌으면 그것을 실행.
#      비어있으면 requirements.txt 가 있을 때 pip install 디폴트 동작.
set -eu

DESTINATION="/app/src/service"

# Expect: REPOSITORY_URL like
#   http://llmops:<TOKEN>@llmops-gitea-service:3000/llmops/22
# And COMMIT_HASH is set.

if [ "${GIT_HOST_TYPE:-INTERNAL}" = "EXTERNAL" ]; then
  # EXTERNAL — REPOSITORY_URL 에 credential 임베드 → 표준 git clone (extraHeader 불필요)
  if [ -d "$DESTINATION/.git" ]; then
    echo "[init.sh] Repo already exists at $DESTINATION. Skipping clone."
  else
    mkdir -p "$(dirname "$DESTINATION")"
    git clone "${REPOSITORY_URL}" "$DESTINATION"
  fi
  git -C "$DESTINATION" fetch --all --tags --prune || true
  git -C "$DESTINATION" checkout "${COMMIT_HASH}"
else
  # INTERNAL — 사내 gitea Bearer extraHeader 흐름 (기존)
  repo_url="${REPOSITORY_URL}"

  # Extract token (between "llmops:" and "@")
  token="$(printf '%s' "$repo_url" | sed -n 's#^http://llmops:\([^@]*\)@.*#\1#p')"
  if [ -z "${token:-}" ]; then
    echo "ERROR: Could not extract token from REPOSITORY_URL" >&2
    echo "       Expected: http://llmops:<TOKEN>@host:port/owner/repo" >&2
    exit 1
  fi

  # Build HTTPS URL without credentials: https://host:port/owner/repo(.git)
  host_path="$(printf '%s' "$repo_url" | sed -n 's#^http://llmops:[^@]*@\([^/]*\)/\(.*\)$#\1/\2#p')"
  if [ -z "${host_path:-}" ]; then
    echo "ERROR: Could not parse host/path from REPOSITORY_URL" >&2
    exit 1
  fi

  clean_url="http://${host_path}"
  case "$clean_url" in
    *.git) : ;;
    *) clean_url="${clean_url}.git" ;;
  esac

  extra_header="Authorization: Bearer ${token}"

  if [ -d "$DESTINATION/.git" ]; then
    echo "Repo already exists at $DESTINATION. Skipping clone."
  else
    mkdir -p "$(dirname "$DESTINATION")"
    git -c "http.extraHeader=${extra_header}" clone "$clean_url" "$DESTINATION"
  fi

  git -C "$DESTINATION" -c "http.extraHeader=${extra_header}" fetch --all --tags --prune || true
  git -C "$DESTINATION" checkout "${COMMIT_HASH}"
fi

cd "$DESTINATION"

# 빌드 단계 — 사용자 BUILD_COMMAND 우선.
# pip 의 dependency resolver 가 사내 미러에서 backtracking 무한루프 빠지는 케이스를
# `--no-deps + --upgrade-strategy only-if-needed` 로 차단.
PIP_OPTS="--upgrade-strategy only-if-needed --no-cache-dir"

if [ -n "${BUILD_COMMAND:-}" ]; then
  echo "[init.sh] Running user BUILD_COMMAND: ${BUILD_COMMAND}"
  sh -c "${BUILD_COMMAND}" || echo "[init.sh] WARNING: BUILD_COMMAND failed"
  echo "[init.sh] BUILD_COMMAND completed."
else
  # 디폴트: requirements.txt 있으면 pip install (--find-links로 packages/도 참조).
  REQ_FILE="$DESTINATION/requirements.txt"
  if [ -f "$REQ_FILE" ]; then
    FIND_LINKS=""
    if [ -d "$DESTINATION/packages" ]; then
      FIND_LINKS="--find-links $DESTINATION/packages"
    fi
    echo "[init.sh] requirements.txt detected, installing packages..."
    # pip 은 PATH 상 /app/.venv/bin/pip (base deps 와 동일 venv) 로 해석된다.
    pip install $PIP_OPTS -r "$REQ_FILE" $FIND_LINKS 2>&1 || echo "[init.sh] WARNING: pip install failed"
    echo "[init.sh] Package installation completed."
  fi
fi
