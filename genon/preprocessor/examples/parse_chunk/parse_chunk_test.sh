#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESSOR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "${PREPROCESSOR_DIR}/.venv/bin/python" ]; then
    PYTHON="${PREPROCESSOR_DIR}/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:/usr/local/lib:/usr/lib"

# ── 풀 E2E: PDF/문서 → 파싱(docling) → 청킹  (파싱에 layout/OCR 모델서빙 필요) ──
# python parse_chunk_test.py "../genon/preprocessor/sample_files/docx_sample.pdf" result_parse_chunk/
# python parse_chunk_test.py "./20260617_table_doc/10.여비규정_20240129_인사경영국_20240129.pdf" result_parse_chunk/

# ── docling JSON 입력 → 청킹만  (모델서버 불필요) ──
# python parse_chunk_test.py "result_parse_chunk/docx_sample.docling.json" result_parse_chunk/

# ── 디렉터리 일괄 ──
# python parse_chunk_test.py "../genon/preprocessor/sample_files" result_parse_chunk/

cd "${SCRIPT_DIR}"
# "${PYTHON}" parse_chunk_test.py "../../sample_files/pdf_sample.pdf" result_parse_chunk/


# ── #329: LLM 캐시 / error_policy / deadline 테스트 (parse→chunk 분리 경로) ──────
# 캐시는 parse 단계 LLM 호출(OCR VLM/TOC/이미지·표 desc/메타데이터)에서 동작한다.
# 로컬은 NFS 가 없으므로 interim_root 를 로컬 쓰기가능 경로로 준다.
# 각 LLM 호출마다 로그에 HIT(캐시 재사용) / MISS(실제 호출) / STORE(저장)가 찍힌다.
FILE="../../sample_files/pdf_sample.pdf"
OUT="result_parse_chunk"
INTERIM="${OUT}/interim"          # <INTERIM>/<workflow_id>/<run_id>/llm_cache/
WF="wf-parse-001"                 # 재실행 간 동일해야 캐시 재사용
RUN="run-1"

# 1) 1회차(MISS=LLM 실제 호출 후 저장):
"${PYTHON}" parse_chunk_test.py --llm_cache --interim_root "${INTERIM}" --workflow_id "${WF}" --run_id "${RUN}" "${FILE}" "${OUT}/"
# 2) 2회차(HIT=캐시 재사용): 로그에서 페이지별 "HIT" 및 요약 "[llm_cache] hit=.. miss=.." 확인
# "${PYTHON}" parse_chunk_test.py --llm_cache --interim_root "${INTERIM}" --workflow_id "${WF}" --run_id "${RUN}" "${FILE}" "${OUT}/"

# 캐시 파일 확인: ls -R "${INTERIM}/${WF}/${RUN}/llm_cache/"

# error_policy=strict (enrichment 실패 시 예외 전파):
# "${PYTHON}" parse_chunk_test.py --llm_cache --interim_root "${INTERIM}" --workflow_id "${WF}" --run_id "${RUN}" --error_policy strict "${FILE}" "${OUT}/"

# 요청 deadline(초) — 초과 시 timeout(행잉 방지):
# "${PYTHON}" parse_chunk_test.py --request_deadline 60 "${FILE}" "${OUT}/"

# 캐시 미지정(기본): 기존과 완전히 동일(캐시 코드 미진입):
# "${PYTHON}" parse_chunk_test.py "${FILE}" "${OUT}/"
