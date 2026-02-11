# syntax=docker/dockerfile:1.7

############################
# 0) 공통 ARG/ENV
############################
ARG PY_VER=3.12.12
ARG DEBIAN_DIST=trixie
ARG H2ORESTART_URL="https://github.com/ebandal/H2Orestart/releases/latest/download/H2Orestart.oxt"
ARG APP_VENV=/app/.venv
ARG NLTK_PACKAGES=all
ARG UID=1000
ARG GID=1000
ARG UNAME=genos
ARG GNAME=genos

############################
# 1) Base (APT + LibreOffice)
############################
FROM python:${PY_VER}-slim-${DEBIAN_DIST} AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/.config \
    XDG_DATA_HOME=/app/.local/share \
    XDG_RUNTIME_DIR=/tmp/xdg \
    TMPDIR=/app/tmp \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    echo "deb http://deb.debian.org/debian sid main" > /etc/apt/sources.list.d/sid.list && \
    printf "Package: python3.13 python3.13-minimal libpython3.13 libpython3.13-minimal libpython3.13-stdlib\nPin: release a=sid\nPin-Priority: 990\n\nPackage: *\nPin: release a=sid\nPin-Priority: 100\n" > /etc/apt/preferences.d/pin-python3.13 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
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
      libgdk-pixbuf-xlib-2.0-0 \
      libffi-dev \
      libgl1 \
      libreoffice \
      ffmpeg \
      unzip \
      openjdk-21-jre-headless  \
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
      tesseract-ocr \
      tesseract-ocr-eng \
      tesseract-ocr-kor \
      sudo \
    && apt-get install -y --no-install-recommends --only-upgrade \
      imagemagick \
      imagemagick-7.q16 \
      libmagickcore-7.q16-10 \
      libmagickwand-7.q16-10 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/.cache/huggingface /app/.cache/fontconfig /app/.config /app/.local/share /app/.fontconfig /app/nltk_data /app/tmp /tmp/xdg
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
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/.config \
    XDG_DATA_HOME=/app/.local/share \
    XDG_RUNTIME_DIR=/tmp/xdg \
    TMPDIR=/app/tmp \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data

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
ARG NLTK_PACKAGES
ARG DOCLING_MODELS_VER=2.41
ARG HF_MODELS_VER=2.41
ARG EASYOCR_VER=2.41

# 👉 여기서도 APP_VENV 그대로 씀
ENV PATH="${APP_VENV}/bin:${PATH}" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/.config \
    XDG_DATA_HOME=/app/.local/share \
    XDG_RUNTIME_DIR=/tmp/xdg \
    TMPDIR=/app/tmp \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data

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
    python - <<'PY'
import os
import nltk

target = os.environ.get("NLTK_DATA", "/app/nltk_data")
os.makedirs(target, exist_ok=True)

packages = os.environ.get("NLTK_PACKAGES", "all")
if packages == "all":
    nltk.download("all", download_dir=target)
else:
    for pkg in [p.strip() for p in packages.split(",") if p.strip()]:
        nltk.download(pkg, download_dir=target)
PY

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
ARG UID
ARG GID
ARG UNAME
ARG GNAME
ENV UV_PROJECT_ENVIRONMENT=${APP_VENV} \
    PATH="${APP_VENV}/bin:${PATH}" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/.config \
    XDG_DATA_HOME=/app/.local/share \
    XDG_RUNTIME_DIR=/tmp/xdg \
    TMPDIR=/app/tmp \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data \
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
COPY --from=models_artifacts /app/nltk_data /app/nltk_data
COPY --from=models_artifacts /models /models

RUN ln -sf /app/configs/supervisor.conf /etc/supervisor/conf.d/supervisor.conf
RUN getent group "${GID}" || groupadd -g "${GID}" "${GNAME}" && \
    getent passwd "${UID}" || useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${UNAME}" && \
    UNAME_ACTUAL="$(getent passwd "${UID}" | cut -d: -f1)" && \
    usermod -aG sudo "${UNAME_ACTUAL}" && \
    echo "${UNAME_ACTUAL} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /app/.cache/huggingface /app/.config/libreoffice /app/.cache/libreoffice /app/.local/share /app/tmp /tmp/xdg && \
    chown -R "${UID}:${GID}" /app /run /var/log /etc/supervisor /models /app/nltk_data /app/.cache /app/.config /app/.local /tmp/xdg
USER ${UID}:${GID}
CMD ["supervisord", "-n"]
