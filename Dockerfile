FROM python:3.11-slim

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

# This will install torch with *only* cpu support
# Remove the --extra-index-url part if you want to install all the gpu requirements
# For more details in the different torch distribution visit https://pytorch.org/.
# feat-ocr-easyocr pulls in easyocr (optional); CPU-only torch via the extra index.
RUN pip install --no-cache-dir "docling[feat-ocr-easyocr]" --extra-index-url https://download.pytorch.org/whl/cpu
# FastAPI + uvicorn serve the web UI; python-multipart parses multipart/form-data uploads.
RUN pip install --no-cache-dir fastapi uvicorn python-multipart
RUN pip install --no-cache-dir "docling[easyocr]" --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

# Pre-download the EasyOCR recognition models for the languages we need:
#   ru     -> Cyrillic model (cyrillic_g2)
#   ch_sim -> Simplified Chinese model (zh_sim_g2)
# `docling-tools models download` resolves these codes to the right weights
# and stores them under DOCLING_ARTIFACTS_PATH so the container is self-contained.
RUN docling-tools models download easyocr --easyocr-lang ru --easyocr-lang ru --easyocr-lang en
RUN docling-tools models download-hf-repo docling-project/docling-layout-heron
RUN docling-tools models download tableformer

# Use the model weights baked into the image instead of re-downloading at runtime.
ENV DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models

# Batch conversion helper: reads PDFs from /data, runs OCR (ru + ch_sim), and
# writes one Markdown file per input to /out.
COPY convert.py /root/convert.py

# Web UI: upload a PDF in the browser, convert with OCR, download the Markdown.
COPY web_app.py /root/web_app.py

# On container environments, always set a thread budget to avoid undesired thread congestion.
ENV OMP_NUM_THREADS=4

# Default entrypoint is the web UI (uvicorn on :8080).
# Override for batch mode:  docker compose run --rm docling python /root/convert.py
ENTRYPOINT ["python", "/root/web_app.py"]