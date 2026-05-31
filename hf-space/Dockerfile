# ── Docling API — Hugging Face Spaces (Docker SDK) ──────────────────────────
  # Free, permanent deployment on https://huggingface.co/spaces/<user>/docling-api
  #
  # HF Spaces Docker requirements:
  #   • The app MUST listen on port 7860
  #   • Run as a non-root user (UID 1000 is the HF standard)
  # ─────────────────────────────────────────────────────────────────────────────

  FROM python:3.11-slim-bookworm

  # System dependencies required by docling (poppler, libgl, etc.)
  RUN apt-get update && apt-get install -y --no-install-recommends \
          libgl1 libglib2.0-0 libgomp1 curl wget git \
      && rm -rf /var/lib/apt/lists/*

  # Install PyTorch CPU-only first (reduces image size vs full GPU wheels)
  RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

  # Install docling + API server dependencies
  RUN pip install --no-cache-dir \
          docling \
          fastapi==0.115.* \
          uvicorn[standard]==0.32.* \
          python-multipart==0.0.*

  # Cache directory used by HF Hub and torch
  ENV HF_HOME=/tmp/hf_cache
  ENV TORCH_HOME=/tmp/torch_cache
  # Limit OpenMP threads to avoid congestion on shared CPUs
  ENV OMP_NUM_THREADS=4

  # Pre-download docling ML models into the image so the first request is instant
  RUN docling-tools models download

  # ── App ──────────────────────────────────────────────────────────────────────
  RUN useradd -m -u 1000 appuser
  WORKDIR /app
  COPY hf-space/app.py .
  RUN chown -R appuser:appuser /app /tmp

  USER appuser

  EXPOSE 7860

  CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
  