# syntax=docker/dockerfile:1.7

########################################
# 0) 공통 ARG
########################################
ARG PY_VER=3.12
ARG DEBIAN_DIST=bookworm

# Paddle GPU wheel 있는 곳 (변수화)
ARG PADDLE_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# 이미지 안에서 우리가 쓸 경로
ARG APP_DIR=/app
ARG OCR_DIR=${APP_DIR}/ocr_paddle_server

########################################
# 1) Base (OS + uv + supervisor)
########################################
FROM python:${PY_VER}-slim-${DEBIAN_DIST} AS base

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

# venv 하나만 전역으로
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# ocr 프로젝트 pyproject만 먼저 복사
# (여기서 경로 주의: 너 구조가 /workspace/doc_parser/genon/serving/paddle/ocr_paddle_server/pyproject.toml 이라서)
COPY genon/serving/paddle/pyproject.toml ${APP_DIR}/pyproject.toml
# lock 있으면 같이
COPY genon/serving/paddle/uv.lock ${APP_DIR}/uv.lock

# venv 만들고 uv sync
# pyproject.toml 안에
# [tool.uv]
# extra-index-url = ["https://www.paddlepaddle.org.cn/packages/stable/cu126/"]
# 가 있으면 이 RUN 한방으로 paddlepaddle-gpu까지 들어온다.
RUN uv venv ${VIRTUAL_ENV}

# 중국 레포를 pyproject에 안 넣고 도커에서만 넣고 싶을 때를 대비해서 ENV로도 한 번 더 줌
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_EXTRA_INDEX_URL="${PADDLE_EXTRA_INDEX_URL}" \
    uv sync --frozen || \
    UV_EXTRA_INDEX_URL="${PADDLE_EXTRA_INDEX_URL}" \
    uv sync

########################################
# 3) models (OCR 모델만 받는 스테이지)
########################################
FROM deps AS models
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
ARG OCR_DIR=${APP_DIR}/ocr_paddle_server
ENV VIRTUAL_ENV=${APP_DIR}/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR ${APP_DIR}

# 1) venv 들여오기
COPY --from=deps ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# 2) 앱 소스 들여오기
#    config / docker / etc / resources ... 전부 살려서 가져옴
COPY genon/serving/paddle ${OCR_DIR}

# 3) 모델 들여오기
COPY --from=models /models /models

# supervisor 설정이 ocr_paddle_server/config/supervisord.conf 라고 했으니 링크만 잡아줌
RUN ln -sf ${OCR_DIR}/config/supervisord.conf /etc/supervisor/supervisord.conf && \
    mkdir -p /var/log/supervisor /var/log/paddlex

EXPOSE 8080

# healthchecker 있으면 그대로
# RUN chmod +x ${OCR_DIR}/etc/health_checker.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
