# syntax=docker/dockerfile:1.7

############################
# 0) 공통 ARG/ENV
############################
ARG PY_VER=3.12
ARG DEBIAN_DIST=bookworm
ARG H2ORESTART_URL="https://github.com/ebandal/H2Orestart/releases/latest/download/H2Orestart.oxt"
ARG APP_VENV=/app/.venv

############################
# 1) Base (APT + LibreOffice)
############################
FROM python:${PY_VER}-slim-${DEBIAN_DIST} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    NLTK_DATA=/root/nltk_data
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      net-tools \
      curl \
      wget \
      fontconfig \
      procps \
      supervisor \
      vim \
      libmagic1 \
      libmagic-dev \
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libcairo2 \
      libgdk-pixbuf2.0-0 \
      libffi-dev \
      libreoffice \
      ffmpeg \
      unzip \
      openjdk-17-jre-headless  \
      ca-certificates-java  \
      libreoffice-java-common \
      libreoffice-writer \
      libreoffice-impress \
      libreoffice-l10n-ko \
      fonts-dejavu \
      fonts-noto-cjk \
      fonts-noto-cjk-extra \
      fonts-nanum \
      imagemagick \
      poppler-utils \
    && rm -rf /var/lib/apt/lists/*
RUN sed -i 's/# \(ko_KR.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && fc-cache -f -v || true

############################
# 2) LibreOffice 확장 (공통)
############################
FROM base AS loext
ARG H2ORESTART_URL
COPY ./genon/preprocessor/resources /app
RUN --mount=type=cache,target=/opt/h2o_cache \
    set -eux; \
    mkdir -p /opt/h2orestart /usr/lib/libreoffice/share/extensions/h2orestart; \
    if [ ! -f /opt/h2o_cache/H2Orestart.oxt ]; then \
      wget -O /opt/h2o_cache/H2Orestart.oxt "${H2ORESTART_URL}"; \
    fi; \
    cp /opt/h2o_cache/H2Orestart.oxt /opt/h2orestart/H2Orestart.oxt; \
    unzip -q -o /opt/h2orestart/H2Orestart.oxt -d /usr/lib/libreoffice/share/extensions/h2orestart; \
    soffice --headless --nologo --nodefault --nofirststartwizard --terminate_after_init
RUN set -eux; \
    mkdir -p /usr/share/fonts && \
    tar zxf /app/HCRBatang.ttf.tar.gz -C /usr/share/fonts/ && \
    fc-cache -f -v && rm /app/HCRBatang.ttf.tar.gz

############################
# 3) Builder (앱 venv만)
############################
FROM loext AS builder
ARG APP_VENV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_VIRTUALENVS_CREATE=1 \
    UV_VIRTUALENVS_IN_PROJECT=0 \
    UV_NO_INTERACTION=1 \
    UV_PYTHON=/usr/local/bin/python \
    UV_PROJECT_ENVIRONMENT=${APP_VENV} \
    APP_VENV=${APP_VENV} \
    PATH="${APP_VENV}/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface \
    NLTK_DATA=/root/nltk_data

# 앱 관련 pyproject만
COPY docling /app/docling
COPY pyproject.toml /app/pyproject.toml
COPY genon/preprocessor/pyproject.toml /app/genon/preprocessor/pyproject.toml
COPY genon/preprocessor/uv.lock /app/genon/preprocessor/uv.lock

WORKDIR /app/genon/preprocessor

# ✅ 공통 위치에 venv 생성
RUN uv venv ${APP_VENV}

# ✅ 그 venv에 설치
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen || \
    uv sync

############################
# 4) 모델 아티팩트
############################
FROM builder AS models_artifacts
ARG APP_VENV
ARG DOCLING_MODELS_VER=2.41
ARG HF_MODELS_VER=2.41
ARG EASYOCR_VER=2.41

# 👉 여기서도 APP_VENV 그대로 씀
ENV PATH="${APP_VENV}/bin:${PATH}" \
    HF_HOME=/root/.cache/huggingface

RUN --mount=type=cache,target=/root/.cache/models \
    echo "Docling models ver: ${DOCLING_MODELS_VER}"; \
    if [ ! -d /models ] || [ -z "$(ls -A /models)" ]; then \
        docling-tools models download -o /models; \
    else \
        echo "[INFO] /models already present, skip"; \
    fi

RUN --mount=type=cache,target=${HF_HOME} \
    echo "HF models ver: ${HF_MODELS_VER}"; \
    mkdir -p /models/doc_parser_models; \
    huggingface-cli download mncai/doc_parser_models --local-dir /models/doc_parser_models && \
    huggingface-cli download mncai/doc_parser_models --include "sentence-transformers-all-MiniLM-L6-v2/*" --local-dir /models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2

RUN --mount=type=cache,target=/root/.cache/nltk \
    python -c "import nltk; nltk.download('all')"

RUN --mount=type=cache,target=/root/.cache/easyocr_korean \
    echo "EasyOCR ver: ${EASYOCR_VER}"; \
    mkdir -p /models/EasyOcr && \
    if [ ! -f /models/EasyOcr/korean_g2.zip ]; then \
      curl -L -o /root/.cache/easyocr_korean/korean_g2.zip "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip"; \
      unzip -q -o /root/.cache/easyocr_korean/korean_g2.zip -d /models/EasyOcr; \
    fi

############################
# 5) Runtime
############################
FROM loext AS runtime
ARG APP_VENV
ENV UV_PROJECT_ENVIRONMENT=${APP_VENV} \
    PATH="${APP_VENV}/bin:${PATH}" \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    HF_HOME=/root/.cache/huggingface \
    NLTK_DATA=/root/nltk_data \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    LC_CTYPE=ko_KR.UTF-8 \
    DOCLING_ARTIFACTS_PATH=/models

WORKDIR /app
COPY ./genon/preprocessor/configs /app/configs
COPY ./genon/preprocessor/env /app/env
COPY ./genon/preprocessor/src /app/src

# ✅ 공통 venv 위치만 복사
COPY --from=builder ${APP_VENV} ${APP_VENV}

# ✅ 모델/자료
COPY --from=models_artifacts /root/nltk_data /root/nltk_data
COPY --from=models_artifacts /models /models

RUN ln -sf /app/configs/supervisor.conf /etc/supervisor/conf.d/supervisor.conf
CMD ["supervisord", "-n"]
