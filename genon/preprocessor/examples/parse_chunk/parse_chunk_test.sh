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
"${PYTHON}" parse_chunk_test.py "../../sample_files/pdf_sample.pdf" result_parse_chunk/
