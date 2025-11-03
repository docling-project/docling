# syntax=docker/dockerfile:1.7

########################################
# 0) 공통 ARG
########################################
ARG PY_VER=3.10
ARG DEBIAN_DIST=bookworm

# Paddle GPU wheel 있는 곳 (변수화)
ARG PADDLE_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# 이미지 안에서 우리가 쓸 경로
ARG APP_DIR=/app
ARG OCR_DIR=${APP_DIR}

########################################
# 1) Base (OS + uv + supervisor)
########################################
FROM python:${PY_VER}-slim-${DEBIAN_DIST} AS base

ARG APP_DIR
ENV APP_DIR=${APP_DIR}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tini \
        supervisor \
        curl \
        ca-certificates \
        git \
        build-essential \
        unzip \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv 바이너리
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 작업 디렉터리
RUN mkdir -p ${APP_DIR}
WORKDIR ${APP_DIR}

########################################
# 2) deps (pyproject → venv 만들기)
########################################
FROM base AS deps

# 너가 위에서 썼던 거 그대로
ARG APP_DIR
ENV APP_DIR=${APP_DIR:-/app}

# paddle 레포는 환경변수로만 넘겨
ENV PADDLE_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

WORKDIR ${APP_DIR}

# pyproject / uv.lock
COPY genon/serving/paddle/pyproject.toml ${APP_DIR}/pyproject.toml
COPY genon/serving/paddle/README.md ${APP_DIR}/README.md
COPY genon/serving/paddle/uv.lock        ${APP_DIR}/uv.lock

# venv 만들기
RUN uv venv ${APP_DIR}/.venv
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ❗❗ 여기가 핵심
# 1) --frozen 없애고
# 2) paddle 레포 + pypi 둘 다 보라고 하고
# 3) 인덱스 순서 고정 안 됐을 때도 뒤에 있는 것도 보라고 함
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_EXTRA_INDEX_URL="${PADDLE_EXTRA_INDEX_URL}" \
    uv sync --index-strategy unsafe-best-match --python ${VIRTUAL_ENV}/bin/python

########################################
# 3) models (OCR 모델만 받는 스테이지)
########################################
FROM deps AS models

ARG APP_DIR
ENV APP_DIR=${APP_DIR}
ENV MODEL_DIR=/models
# 모델을 한 곳에만
ENV MODEL_DIR=/models

# 네가 올려둔 zip이 있을 때
COPY genon/serving/paddle/resources/paddleocr_model.zip /tmp/paddleocr_model.zip

RUN mkdir -p ${MODEL_DIR}/paddleocr_model && \
    unzip -q -o /tmp/paddleocr_model.zip -d ${MODEL_DIR} && \
    rm -f /tmp/paddleocr_model.zip

# 필요하면 추가 모델도 여기서 받으면 됨

########################################
# 4) runtime (최종)
########################################
FROM base AS runtime

ARG APP_DIR=/app
ARG OCR_DIR=${APP_DIR}
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR ${APP_DIR}

# 2) 앱 소스 들여오기
#    config / docker / etc / resources ... 전부 살려서 가져옴
COPY genon/serving/paddle ${OCR_DIR}

# 3) 모델 들여오기
COPY --from=models /models /models

COPY --from=deps ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# supervisor 설정이 ocr_paddle_server/config/supervisord.conf 라고 했으니 링크만 잡아줌
RUN ln -sf ${OCR_DIR}/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8080

# healthchecker 있으면 그대로
# RUN chmod +x ${OCR_DIR}/etc/health_checker.sh
CMD ["supervisord", "-n"]
