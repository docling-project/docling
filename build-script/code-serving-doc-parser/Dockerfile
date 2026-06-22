FROM python:3.12.12-slim-trixie AS builder

RUN pip install uv

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# build context = 이 폴더(build-script/code-serving-doc-parser). 모든 COPY 는 로컬 상대경로.
# 의존성은 pyproject.toml(genon/preprocessor uv 정보 기반)에 모두 선언되어 있어
# uv pip install . 로 빌드 시점에 pre-bake 된다 (별도 requirements.txt 없음).
COPY ./pyproject.toml ./uv.lock /app/

RUN uv venv --seed && \
    uv pip install .

FROM python:3.12.12-slim-trixie AS runtime

ARG UID=1000
ARG GID=1000
ARG UNAME=genos
ARG GNAME=genos

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# facade(docling/easyocr/unstructured 등) 실행에 필요한 시스템 패키지.
# 정본: genon/preprocessor/docker/Dockerfile.standard apt-get 블록.
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        net-tools curl procps supervisor vim git \
        build-essential \
        libreoffice antiword poppler-utils \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
        libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0 \
        fonts-noto-cjk \
        libmagic1 libgl1 libspatialindex-c6 imagemagick && \
    rm -rf /var/lib/apt/lists/*

# 사용자 생성 및 권한 설정
# 1. 그룹 및 유저 생성
# 2. 필요한 디렉토리 생성 (Supervisor 로그/실행, 앱 디렉토리)
# 3. 소유권 변경
RUN groupadd -g $GID $GNAME && \
    useradd --create-home --no-log-init -u $UID -g $GID $UNAME && \
    mkdir -p /var/log/supervisor /var/run/supervisor /app && \
    chown -R $UNAME:$GNAME /var/log/supervisor /var/run/supervisor /app

WORKDIR /app

# 빌드 결과물 복사 (권한 설정 포함)
COPY --from=builder --chown=$UID:$GID ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Supervisor 설정 복사
# 1. 메인 설정 (Non-root 대응)
COPY --chown=$UID:$GID ./supervisor/non-root-base.conf /etc/supervisor/supervisord.conf
# 2. 프로그램 설정 (기존 설정)
COPY --chown=$UID:$GID ./supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 소스 코드 복사 (code-serving harness — 사용자 코드는 런타임에 /app/src/service 로 clone)
COPY --chown=$UID:$GID ./ /app

# 불필요한 파일 삭제
RUN rm -rf /app/src/service

# 사용자 코드 wrapper 스크립트 실행 권한 보장 (다른 언어 이미지와 동일 패턴).
RUN chmod +x /app/scripts/*.sh

# 사용자 전환 (K8s runAsNonRoot 검증을 위해 UID 숫자로 지정)
USER $UID

# 명시적인 설정 파일 경로로 실행
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
