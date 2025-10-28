#!/usr/bin/env bash
set -euo pipefail
echo "[OCR] Serving on :${OCR_PORT}"
exec paddlex serving ocr \
  --det_model_dir "${DET_MODEL_DIR}" \
  --rec_model_dir "${REC_MODEL_DIR}" \
  --rec_char_dict_path "${REC_CHAR_DICT_PATH}" \
  --port "${OCR_PORT}"
