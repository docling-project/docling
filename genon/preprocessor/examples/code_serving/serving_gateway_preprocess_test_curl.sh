#!/usr/bin/env bash
# main.py(/preprocess*) 게이트웨이 curl 예제 모음 (첨부/적재/변환).
#
# 게이트웨이 URL 패턴: {BASE}/api/gateway/code_serving/{SERVING_ID}/{route}
# 단일 서빙(139)이 /preprocess_* 엔드포인트를 노출한다고 가정한다.
# 주의: 게이트웨이는 route 를 단일 세그먼트로만 포워딩하므로 평탄 경로(preprocess_*)를 쓴다
#       (중첩 경로 /preprocess/xxx 는 게이트웨이로 호출 불가).
#
# 응답 envelope: {"code":0,"errMsg":"success","data":[ {...청크...}, ... ]}
#   세 엔드포인트 모두 data 는 청크 리스트(list[GenOSVectorMeta])다.
#
# 주의: file_path 는 *서빙 컨테이너 내부의 로컬 경로*다(MinIO 키 아님).
#       서버가 접근 가능한 경로를 넣어야 한다.

set -euo pipefail

BASE="https://genos.genon.ai"
SERVING_ID="139"
AUTH="b8c0b48f7b4d410699ed1aa8f2c0da8a"
GW="${BASE}/api/gateway/code_serving/${SERVING_ID}"

# 전처리할 문서(서버 기준 경로)
FILE_PATH="/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf"

# ── 1) health ─────────────────────────────────────────────────────────────
curl --location "${GW}/health" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}"
echo

# ── 2) /preprocess_attachment : 첨부용 전처리 → data(청크 리스트) ────────────
curl --location "${GW}/preprocess_attachment" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}" \
  --data "$(jq -nc --arg fp "${FILE_PATH}" '{file_path:$fp, params:{}}')" \
  | jq '.data | length as $n | "attachment chunks: \($n)"'

# ── 3) /preprocess_intelligent : 적재용(지능형) 전처리 ──────────────────────
curl --location "${GW}/preprocess_intelligent" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}" \
  --data "$(jq -nc --arg fp "${FILE_PATH}" '{file_path:$fp, params:{}}')" \
  | jq '.data | length as $n | "intelligent chunks: \($n)"'

# ── 4) /preprocess_convert : 변환용 전처리 ──────────────────────────────────
curl --location "${GW}/preprocess_convert" \
  --header 'Content-Type: application/json' \
  --header "Authorization: Bearer ${AUTH}" \
  --data "$(jq -nc --arg fp "${FILE_PATH}" '{file_path:$fp, params:{}}')" \
  | jq '.data | length as $n | "convert chunks: \($n)"'

# ── 참고: 동등한 Python 스크립트 실행 ───────────────────────────────────────
#   python serving_gateway_preprocess_test.py --mode health
#   python serving_gateway_preprocess_test.py --mode attachment  --file-path "${FILE_PATH}"
#   python serving_gateway_preprocess_test.py --mode intelligent --file-path "${FILE_PATH}"
#   python serving_gateway_preprocess_test.py --mode convert     --file-path "${FILE_PATH}"
#   python serving_gateway_preprocess_test.py --mode all         --file-path "${FILE_PATH}" --out /tmp/preprocess/
