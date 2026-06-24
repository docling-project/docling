#!/usr/bin/env bash
# main.py(/health, /parser, /chunker) 게이트웨이 curl 예제 모음.
#
# 게이트웨이 URL 패턴: {BASE}/api/gateway/code_serving/{SERVING_ID}/{route}
# 단일 서빙(139)이 /parser 와 /chunker 를 모두 노출한다고 가정한다.
#
# 주의: /parser 의 file_path 는 *서빙 컨테이너 내부의 로컬 경로*다(MinIO 키 아님).
#       서버가 접근 가능한 경로를 넣어야 한다.

set -euo pipefail

BASE="https://genos.genon.ai"
SERVING_ID="139"
AUTH="b8c0b48f7b4d410699ed1aa8f2c0da8a"
GW="${BASE}/api/gateway/code_serving/${SERVING_ID}"

# 파싱할 문서(서버 기준 경로)
FILE_PATH="/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf"
RESULT_DIR="./result_serving_gateway_test"

# ── 1) health ─────────────────────────────────────────────────────────────
# (참고: 기존 health 체크 curl 과 동일)
curl --location "${GW}/health" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}"
echo

# ── 2) /parser : 문서 파싱 → data.document(docling JSON) ────────────────────
# 응답 envelope: {"code":0,"errMsg":"success","data":{"document":{...},"usage":{"pages":N}}}
# data.document 만 추출해 /tmp/doc.json 으로 저장(jq 필요).
curl --location "${GW}/parser" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}" \
  --data "$(jq -nc --arg fp "${FILE_PATH}" '{file_path:$fp, params:{}}')" \
  | jq '.data.document' > "${RESULT_DIR}/doc.json"
echo "saved docling document → ${RESULT_DIR}/doc.json"

# ── 3) /chunker : 위 docling JSON 을 params.document 로 실어 청킹 ────────────
# 응답 envelope: {"code":0,"errMsg":"success","data":[{...청크...}, ...]}
# 주의: doc.json(수 MB)을 --data "$(...)" 로 인자 전달하면 ARG_MAX(1MB) 초과로
#       "Argument list too long" 발생. jq 출력을 파이프로 넘겨 --data-binary @- 로 stdin 에서 읽는다.
jq -nc --slurpfile doc "${RESULT_DIR}/doc.json" \
  '{file_path:"report.pdf", params:{document:$doc[0], chunk_size:0}}' \
| curl --location "${GW}/chunker" \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer ${AUTH}" \
    --data-binary @- \
  | jq '.data | length as $n | "chunks: \($n)"'

# ── 참고: 동등한 Python 스크립트 실행 ───────────────────────────────────────
#   python serving_gateway_test.py --mode health
#   python serving_gateway_test.py --mode e2e --file-path "${FILE_PATH}" --out /tmp/chunks.json
#   python serving_gateway_test.py --mode parser --file-path "${FILE_PATH}" --out-doc /tmp/doc.json
#   python serving_gateway_test.py --mode chunker --doc-json /tmp/doc.json
