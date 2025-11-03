# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 필수 툴
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates tini supervisor git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv 바이너리
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 👇 네 프로젝트 pyproject만 먼저
COPY serving/ocr_paddle_server/pyproject.toml /app/pyproject.toml
# 필요하면 README도
# COPY serving/ocr_paddle_server/README.md /app/README.md

# 👇 여기서 uv가 paddle까지 전부 설치
# (pyproject의 [tool.uv]에 extra-index-url이 있기 때문에 가능)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv sync --frozen || uv sync

# 앱 코드
COPY serving/ocr_paddle_server /app

# 로그 디렉터리
RUN mkdir -p /var/log/supervisor /var/log/paddlex

EXPOSE 8080

CMD ["/usr/bin/tini", "--", "supervisord", "-n", "-c", "/app/config/supervisord.conf"]
