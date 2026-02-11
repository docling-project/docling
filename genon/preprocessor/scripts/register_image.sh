#!/usr/bin/env bash
set -euo pipefail

# ── 경로/로그 ────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/register.config"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/register_image_${TS}.log"
STATE_FILE="${LOG_DIR}/register_image_${TS}.state"
exec > >(tee -a "${LOG_FILE}") 2>&1
step(){ echo "[STEP] $*"; echo "$(date +'%F %T') STEP $*" >> "${STATE_FILE}"; }
ok(){   echo "✅ $*";      echo "$(date +'%F %T') OK   $*" >> "${STATE_FILE}"; }
fail(){ echo "❌ $*";      echo "$(date +'%F %T') FAIL $*" >> "${STATE_FILE}"; }
trap 'fail "스크립트 실패 (line $LINENO). 로그: ${LOG_FILE}"' ERR

echo "=== $(date +'%F %T') 이미지 등록 시작 ==="
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "LOG_FILE=${LOG_FILE}"

# ── 설정 로드 ────────────────────────────────────────────────
step "설정 파일 로드"
[[ -f "${CONFIG_FILE}" ]] || { fail "설정 파일 없음: ${CONFIG_FILE}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
ok "설정 로드 완료"

: "${K8S_NAMESPACE:?}"
: "${MARIADB_POD:?}"

# ── 기본값 + 사용자 입력 (Enter=기본값 유지) ────────────────
echo ""
echo "※ Enter 를 누르면 config 기본값을 사용합니다."
read -rp "Registry [${REGISTRY_NAME:-}]: " _REG
read -rp "Image    [${IMAGE_NAME:-}]: " _IMG
read -rp "Tag      [${IMAGE_TAG:-}]: " _TAG

REGISTRY_NAME="${_REG:-${REGISTRY_NAME}}"
IMAGE_NAME="${_IMG:-${IMAGE_NAME}}"
IMAGE_TAG="${_TAG:-${IMAGE_TAG}}"
FULL_IMAGE_NAME="${REGISTRY_NAME}${IMAGE_NAME}:${IMAGE_TAG}"

echo "📦 대상 이미지 : ${FULL_IMAGE_NAME}"
echo "📝 설명         : ${DESCRIPTION:-N/A}"
echo "🔎 IMAGE_NAME='${IMAGE_NAME}' IMAGE_TAG='${IMAGE_TAG}' REGISTRY='${REGISTRY_NAME}'"

# DB 계정(Enter=기본값)
read -rp "MySQL 사용자명 [${DEFAULT_MYSQL_USER:-}]: " MYSQL_USER_IN
MYSQL_USER="${MYSQL_USER_IN:-${DEFAULT_MYSQL_USER:-}}"
if [[ -z "${MYSQL_USER}" ]]; then fail "MySQL 사용자명 비어있음"; exit 1; fi

if [[ -n "${DEFAULT_MYSQL_PASS:-}" ]]; then
  MYSQL_PASS="${DEFAULT_MYSQL_PASS}"
  echo "MySQL 비밀번호: (config 기본값 사용)"
else
  read -srp "MySQL 비밀번호: " MYSQL_PASS; echo
fi

# ── 로컬 이미지 확인 ────────────────────────────────────────
step "로컬 Docker 이미지 확인"
if docker image inspect "${FULL_IMAGE_NAME}" >/dev/null 2>&1; then
  ok "로컬 이미지 존재"
  HAS_LOCAL_IMAGE="yes"
else
  echo "⚠️ 로컬에 ${FULL_IMAGE_NAME} 없음."
  HAS_LOCAL_IMAGE="no"
fi

# ── docker push (포그라운드 / 재시도) ───────────────────────
step "docker push"
SKIP_PUSH="no"
if [[ "${HAS_LOCAL_IMAGE}" != "yes" ]]; then
  if [[ -n "${REGISTRY_API_URL:-}" ]]; then
    step "레지스트리 존재 확인 (${REGISTRY_API_URL})"
    if curl -fsS "${REGISTRY_API_URL}/v2/${IMAGE_NAME}/manifests/${IMAGE_TAG}" >/dev/null 2>&1; then
      ok "레지스트리에 동일 태그 존재, push 스킵"
      SKIP_PUSH="yes"
    else
      fail "로컬 이미지 없음 + 레지스트리에도 태그 없음. build/push 필요"
      exit 1
    fi
  else
    read -rp "로컬 이미지 없음. 레지스트리에 이미 푸쉬된 상태면 y 입력 (y/N): " _SKIP
    if [[ "${_SKIP:-N}" =~ ^[Yy]$ ]]; then
      ok "사용자 확인으로 push 스킵"
      SKIP_PUSH="yes"
    else
      fail "로컬 이미지 없음. build/push 필요"
      exit 1
    fi
  fi
fi

if [[ "${SKIP_PUSH}" != "yes" ]]; then
  PUSH_MAX_RETRY="${PUSH_MAX_RETRY:-3}"
  for i in $(seq 1 "${PUSH_MAX_RETRY}"); do
    echo "push ${i}/${PUSH_MAX_RETRY}: ${FULL_IMAGE_NAME}"
    if docker push "${FULL_IMAGE_NAME}"; then ok "docker push 성공"; break; fi
    [[ $i -lt ${PUSH_MAX_RETRY} ]] || { fail "docker push 실패"; exit 1; }
    echo "10초 대기 후 재시도..."; sleep 10
  done
else
  echo "⏩ docker push 스킵"
fi

# (옵션) 레지스트리 API 확인
if [[ -n "${REGISTRY_API_URL:-}" ]]; then
  step "레지스트리 API 확인 (${REGISTRY_API_URL})"
  if curl -fsS "${REGISTRY_API_URL}/v2/_catalog" >/dev/null 2>&1; then
    ok "레지스트리 API OK"
  else
    echo "⚠️ API 응답 없음(무시 가능). push는 완료됨."
  fi
fi

# ────────────────────────────────────────────────────────────
# ⬇⬇⬇ 여기부터 DB 파트 *원하신 형태 그대로* (유저/패스/설정만 치환)
# ────────────────────────────────────────────────────────────
step "DB 등록 확인"
echo "2. DB 등록 확인 중..."
mysql_query() {
  local sql="$1"
  local out=""
  # Prefer mariadb binary if present (Bitnami)
  if out=$(kubectl exec -i "${MARIADB_POD}" -n "${K8S_NAMESPACE}" -- \
      /opt/bitnami/mariadb/bin/mariadb -u "${MYSQL_USER}" -p"${MYSQL_PASS}" llmops -se --show-warnings \
      "${sql}" 2>&1); then
    printf '%s' "${out}"
    return 0
  fi
  # Fallback to mysql
  out=$(kubectl exec -i "${MARIADB_POD}" -n "${K8S_NAMESPACE}" -- \
      mysql -u "${MYSQL_USER}" -p"${MYSQL_PASS}" llmops -se --show-warnings \
      "${sql}" 2>&1)
  local rc=$?
  printf '%s' "${out}"
  return "${rc}"
}

SQL_EXISTING="SELECT id FROM system_docker_image_tb WHERE name='${IMAGE_NAME}' AND tag='${IMAGE_TAG}';"
MYSQL_OUT=""
if ! MYSQL_OUT="$(mysql_query "${SQL_EXISTING}")"; then
  fail "DB 조회 실패. 아래 로그 확인 필요."
  echo "${MYSQL_OUT}"
  exit 1
fi
echo "DB 조회 결과(raw): ${MYSQL_OUT}"
EXISTING_ID="$(printf '%s' "${MYSQL_OUT}" | tr -d '\r\n' | grep -Eo '^[0-9]+$' || true)"

if [ -z "${EXISTING_ID}" ]; then
  echo "새로운 이미지 등록 중..."
  TYPE_LIST_JSON='["IT0301"]'
  SQL_INSERT="
      INSERT INTO llmops.system_docker_image_tb
        (name, tag, description, type, status, is_active, reg_date, mod_date, reg_user_id, mod_user_id)
      VALUES
        ('${IMAGE_NAME}', '${IMAGE_TAG}', '${DESCRIPTION}', '${TYPE_LIST_JSON}', 'COMPLETED', 1, NOW(), NOW(), 1, 1);
      INSERT INTO llmops.resource_meta_tb
        (resource_id, resource_type, resource_group_id, is_active, reg_date, mod_date, reg_user_id, mod_user_id)
      VALUES
        (LAST_INSERT_ID(), 'DOCKER_IMAGE', 2, 1, NOW(), NOW(), 1, 1);
  "
  if ! MYSQL_OUT="$(mysql_query "${SQL_INSERT}")"; then
    fail "DB 등록 실패. 아래 로그 확인 필요."
    echo "${MYSQL_OUT}"
    exit 1
  fi

  MYSQL_OUT=""
  if ! MYSQL_OUT="$(mysql_query "${SQL_EXISTING}")"; then
    fail "DB 조회 실패(등록 후). 아래 로그 확인 필요."
    echo "${MYSQL_OUT}"
    exit 1
  fi
  IMAGE_ID="$(printf '%s' "${MYSQL_OUT}" | tr -d '\r\n' | grep -Eo '^[0-9]+$' || true)"
  ok "DB 등록 완료. 이미지 ID: ${IMAGE_ID}"
else
  ok "이미 등록된 이미지입니다. ID: ${EXISTING_ID}"
  IMAGE_ID="${EXISTING_ID}"
fi
# ────────────────────────────────────────────────────────────
# ⬆⬆⬆ DB 파트 끝
# ────────────────────────────────────────────────────────────

# ── Redis Flush (선택) ───────────────────────────────────────
step "Redis 캐시 초기화 여부"
read -rp "Redis 캐시 FLUSHALL 할까요? (y/N): " REDIS_FLUSH
if [[ "${REDIS_FLUSH:-N}" =~ ^[Yy]$ ]]; then
  step "Redis FLUSHALL 실행"
  REDIS_POD="$(kubectl get pods -n "${K8S_NAMESPACE}" -l app=llmops-redis -o jsonpath='{.items[0].metadata.name}')"
  if [[ -n "${REDIS_POD}" ]]; then
    kubectl exec -n "${K8S_NAMESPACE}" "${REDIS_POD}" -- redis-cli FLUSHALL
    ok "Redis FLUSHALL 완료"
  else
    fail "Redis Pod를 찾지 못함(건너뜀)"
  fi
else
  echo "⏩ Redis 초기화 건너뜀"
fi

ok "모든 단계 완료"
echo ""
echo "=== 완료 ==="
echo "이미지   : ${FULL_IMAGE_NAME}"
echo "로그파일 : ${LOG_FILE}"
echo "상태파일 : ${STATE_FILE}"
