# syntax=docker/dockerfile:1.7
#
# code-serving-doc-parser (CPU) — doc-parser 코드서빙 base 이미지.
#
# 라이브러리 설치 방식은 genon/code-serving 과 동일: `uv venv --seed` + `uv pip install .` → /app/.venv.
# 다운로드 아티팩트(모델/HWP SDK/rhwp/H2Orestart/폰트/NLTK/EasyOCR/docling-parse 패치)는
# genon/preprocessor/docker/Dockerfile.standard 와 full parity 로 복제한다.
#
# build context = repo 루트 (build-script/code-serving-doc-parser/build.sh 가 ROOT_DIR=repo 루트 사용).
# 따라서 모든 COPY 는 repo 루트 기준 경로. HWP_SDK_TOKEN 은 --secret 로 주입.

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
ARG HW_VARIANT=cpu
ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG RHWP_GIT_URL=https://github.com/genonai/genos-rhwp.git
ARG RHWP_GIT_REF=genos/main
# models_artifacts 단계에서 모델 다운로드용으로만 임시 설치하는 docling 패키지 버전
# (vendored docling 과 동일 계열. 런타임 venv 에는 포함되지 않음)
ARG DOCLING_PKG_VER=2.41.0

############################
# 1) Base (APT + LibreOffice) — Dockerfile.standard 와 동일 세트
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
      net-tools curl wget fontconfig procps supervisor vim \
      libmagic1 libmagic-dev \
      libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-xlib-2.0-0 \
      libffi-dev libgl1 \
      libreoffice ffmpeg unzip \
      openjdk-21-jre-headless ca-certificates-java \
      libreoffice-java-common libreoffice-writer libreoffice-impress libreoffice-l10n-ko \
      fonts-dejavu fonts-noto-cjk fonts-noto-cjk-extra fonts-nanum \
      imagemagick poppler-utils \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-kor \
      sudo \
      libspatialindex-c6 \
    && apt-get install -y --no-install-recommends --only-upgrade \
      imagemagick imagemagick-7.q16 libmagickcore-7.q16-10 libmagickwand-7.q16-10 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/.cache/huggingface /app/.cache/fontconfig /app/.config /app/.local/share /app/.fontconfig /app/nltk_data /app/tmp /tmp/xdg
RUN sed -i 's/# \(ko_KR.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && fc-cache -f -v || true

############################
# 2) LibreOffice 확장 + HCRBatang 폰트 (Dockerfile.standard 와 동일)
#    resources 에서 H2Orestart 패치 헤더(stream_fix_xobject_reload.h)/HCRBatang 폰트가 들어온다.
############################
FROM base AS loext
ARG H2ORESTART_URL
RUN --mount=type=cache,target=/opt/h2o_cache \
    set -eux; \
    mkdir -p /opt/h2orestart /usr/lib/libreoffice/share/extensions/h2orestart; \
    if [ ! -f /opt/h2o_cache/H2Orestart.oxt ]; then \
      wget -O /opt/h2o_cache/H2Orestart.oxt "${H2ORESTART_URL}"; \
    fi; \
    cp /opt/h2o_cache/H2Orestart.oxt /opt/h2orestart/H2Orestart.oxt; \
    unzip -q -o /opt/h2orestart/H2Orestart.oxt -d /usr/lib/libreoffice/share/extensions/h2orestart; \
    soffice --headless --nologo --nodefault --nofirststartwizard --terminate_after_init
# HCRBatang 폰트 (resources 의 특정 파일만 /tmp 로 가져와 설치 — /app 은 오염시키지 않는다)
COPY ./genon/preprocessor/resources/HCRBatang.ttf.tar.gz /tmp/HCRBatang.ttf.tar.gz
RUN set -eux; \
    mkdir -p /usr/share/fonts && \
    tar zxf /tmp/HCRBatang.ttf.tar.gz -C /usr/share/fonts/ && \
    fc-cache -f -v && rm /tmp/HCRBatang.ttf.tar.gz

############################
# 3) Builder — venv 설치는 genon/code-serving 방식 (uv venv --seed + uv pip install .)
############################
FROM loext AS builder
ARG APP_VENV
ARG HW_VARIANT
ARG TORCH_CPU_INDEX_URL
ENV VIRTUAL_ENV=${APP_VENV} \
    PATH="${APP_VENV}/bin:${PATH}" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data

RUN pip install uv

WORKDIR /app
# 의존성 정의는 build-script/code-serving-doc-parser 의 pyproject/uv.lock (preprocessor uv 정보 기반).
COPY build-script/code-serving-doc-parser/pyproject.toml build-script/code-serving-doc-parser/uv.lock /app/

# genon/code-serving 와 동일: uv venv --seed 로 /app/.venv 생성 후 uv pip install . 로 venv 에 설치.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed && \
    uv pip install .

# HW_VARIANT=cpu — torch/torchvision 을 CPU wheel 로 재설치 + nvidia-*/triton 제거(경량화).
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${HW_VARIANT}" = "cpu" ]; then \
      echo "[INFO] HW_VARIANT=cpu — reinstalling torch/torchvision as CPU wheels" && \
      uv pip install --python ${APP_VENV}/bin/python \
        --reinstall-package torch --reinstall-package torchvision \
        --index-url "${TORCH_CPU_INDEX_URL}" \
        torch torchvision && \
      uv pip uninstall --python ${APP_VENV}/bin/python \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
        nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-cufile-cu12 \
        nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
        nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 triton \
      || true; \
    else \
      echo "[INFO] HW_VARIANT=${HW_VARIANT} — keeping torch as-is"; \
    fi

############################
# 3-1) docling-parse 4.1.0 소스 패치 빌드 (이슈 #245)
#  - 패치를 /app/.venv 의 venv 인터프리터로 설치 → wheel 설치본을 덮어쓴다 (런타임 venv 와 동일 위치).
############################
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      git g++ make zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
# 패치 헤더만 /tmp 로 (─ /app 오염 방지)
COPY ./genon/preprocessor/resources/stream_fix_xobject_reload.h /tmp/stream_fix_xobject_reload.h
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    uv pip install --python ${APP_VENV}/bin/python \
      pybind11 cmake scikit-build-core poetry -q; \
    git clone --depth 1 --branch v4.1.0 \
      https://github.com/docling-project/docling-parse /tmp/docling-parse-4.1.0; \
    cp /tmp/stream_fix_xobject_reload.h \
      /tmp/docling-parse-4.1.0/src/v2/pdf_decoders/stream.h; \
    sed -i 's|-DCMAKE_INSTALL_PREFIX=${EXTERNALS_PREFIX_PATH}|-DCMAKE_INSTALL_PREFIX=${EXTERNALS_PREFIX_PATH}\n        -DCXXOPTS_BUILD_EXAMPLES=OFF\n        -DCXXOPTS_BUILD_TESTS=OFF|' \
      /tmp/docling-parse-4.1.0/cmake/extlib_cxxopts.cmake; \
    sed -i 's|set(QPDF_EXTRA_CXX_FLAGS "-fPIC ")|set(QPDF_EXTRA_CXX_FLAGS "-fPIC -include cstdint -Wno-error")|' \
      /tmp/docling-parse-4.1.0/cmake/extlib_qpdf_v11.cmake; \
    sed -i 's|set(QPDF_EXTRA_C_FLAGS "-fPIC ")|set(QPDF_EXTRA_C_FLAGS "-fPIC -Wno-error")|' \
      /tmp/docling-parse-4.1.0/cmake/extlib_qpdf_v11.cmake; \
    sed -i 's|-DCMAKE_C_FLAGS=${ENV_ARCHFLAGS}|-DCMAKE_C_FLAGS=-Wno-error|' \
      /tmp/docling-parse-4.1.0/cmake/extlib_jpeg.cmake; \
    uv pip install --python ${APP_VENV}/bin/python --no-build-isolation /tmp/docling-parse-4.1.0/; \
    uv pip install --python ${APP_VENV}/bin/python "docling-core[chunking]==2.62.0"; \
    rm -rf /tmp/docling-parse-4.1.0

############################
# 4-1) HWP SDK (HF private, 무료 자산) — Dockerfile.standard 와 동일
############################
FROM builder AS sdk_artifacts
RUN --mount=type=cache,target=${HF_HOME} \
    --mount=type=secret,id=HWP_SDK_TOKEN \
    echo "[INFO] Downloading HWP SDK from Private HF Repo..."; \
    HWP_SDK_TOKEN_VAL=$(cat /run/secrets/HWP_SDK_TOKEN) && \
    mkdir -p /app/hwp_sdk && \
    huggingface-cli download HeechanKim-Genon/hwp_sdk \
      --repo-type dataset --local-dir /app/hwp_sdk \
      --local-dir-use-symlinks False --token "${HWP_SDK_TOKEN_VAL}" && \
    chmod +x /app/hwp_sdk/convtext

############################
# 4-2) 추가 폰트 (HF public)
############################
FROM builder AS font_artifacts
RUN --mount=type=cache,target=${HF_HOME} \
    echo "[INFO] Downloading extra fonts from HF (public)..." && \
    mkdir -p /app/extra_fonts && \
    huggingface-cli download HeechanKim-Genon/additional_fonts \
      --repo-type dataset --local-dir /app/extra_fonts --local-dir-use-symlinks False

############################
# 4-2b) rhwp 바이너리 (rust 빌드 → 결과만 COPY)
############################
FROM rust:latest AS rhwp_builder
ARG RHWP_GIT_URL
ARG RHWP_GIT_REF
WORKDIR /src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    git clone --depth 1 --branch "${RHWP_GIT_REF}" "${RHWP_GIT_URL}" /src
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/target \
    cargo build --release --bin rhwp \
 && cp /src/target/release/rhwp /usr/local/bin/rhwp

############################
# 4-3) 모델/NLTK/EasyOCR
#  ⚠️ docling-tools 는 docling 패키지 CLI 이므로 이 스테이지에서만 docling 임시 설치 후 모델 다운로드.
#     (이 stage 의 venv 는 runtime 에 COPY 되지 않음 — /models, /app/nltk_data 만 가져감)
############################
FROM builder AS models_artifacts
ARG APP_VENV
ARG NLTK_PACKAGES
ARG DOCLING_PKG_VER
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${APP_VENV}/bin/python "docling==${DOCLING_PKG_VER}"
RUN --mount=type=cache,target=/root/.cache/models \
    if [ ! -d /models ] || [ -z "$(ls -A /models)" ]; then \
        docling-tools models download -o /models; \
    fi
RUN --mount=type=cache,target=${HF_HOME} \
    mkdir -p /models/doc_parser_models && \
    huggingface-cli download mncai/doc_parser_models --local-dir /models/doc_parser_models && \
    huggingface-cli download mncai/doc_parser_models --include "sentence-transformers-all-MiniLM-L6-v2/*" \
      --local-dir /models/doc_parser_models/sentence-transformers-all-MiniLM-L6-v2
RUN --mount=type=cache,target=/root/.cache/nltk \
    python - <<'PY'
import os, nltk
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
    mkdir -p /models/EasyOcr && \
    if [ ! -f /root/.cache/easyocr_korean/korean_g2.zip ]; then \
      curl -L -o /root/.cache/easyocr_korean/korean_g2.zip \
        "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip"; \
    fi; \
    unzip -q -o /root/.cache/easyocr_korean/korean_g2.zip -d /models/EasyOcr

############################
# 5) Runtime — genon/code-serving harness + 아티팩트 + ENV parity
############################
FROM loext AS runtime
ARG APP_VENV
ARG UID
ARG GID
ARG UNAME
ARG GNAME
ENV VIRTUAL_ENV=${APP_VENV} \
    PATH="${APP_VENV}/bin:${PATH}" \
    HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    XDG_CONFIG_HOME=/app/.config \
    XDG_DATA_HOME=/app/.local/share \
    XDG_RUNTIME_DIR=/tmp/xdg \
    TMPDIR=/app/tmp \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    LC_CTYPE=ko_KR.UTF-8 \
    DOCLING_ARTIFACTS_PATH=/models \
    DOC_PARSER_BUILD_VARIANT=standard \
    RHWP_BIN=/usr/local/bin/rhwp

WORKDIR /app

# 사용자 생성 + 디렉토리/권한
RUN groupadd -g $GID $GNAME && \
    useradd --create-home --no-log-init -u $UID -g $GID $UNAME && \
    echo "$UNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /var/log/supervisor /var/run/supervisor /app/.cache/huggingface \
      /app/.config/libreoffice /app/.cache/libreoffice /app/.local/share /app/tmp /tmp/xdg

# venv (패치된 docling_parse 포함) + 아티팩트
COPY --from=builder ${APP_VENV} ${APP_VENV}
COPY --from=sdk_artifacts /app/hwp_sdk /app/hwp_sdk
COPY --from=rhwp_builder /usr/local/bin/rhwp /usr/local/bin/rhwp
COPY --from=font_artifacts /app/extra_fonts /usr/share/fonts/additional
RUN fc-cache -f -v
COPY --from=models_artifacts /models /models
COPY --from=models_artifacts /app/nltk_data /app/nltk_data

# hwp_sdk import 경로 (venv site-packages 심링크). python3.12 고정(slim 3.12.x).
RUN ln -s /app/hwp_sdk ${APP_VENV}/lib/python3.12/site-packages/hwp_sdk

# 코드서빙 harness (genon/code-serving 와 동일 구성)
COPY build-script/code-serving-doc-parser/supervisor/non-root-base.conf /etc/supervisor/supervisord.conf
COPY build-script/code-serving-doc-parser/supervisor/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY build-script/code-serving-doc-parser/scripts /app/scripts
COPY build-script/code-serving-doc-parser/src /app/src
COPY build-script/code-serving-doc-parser/gunicorn /app/gunicorn

# 샘플 service 제거(런타임에 /app/src/service 로 clone) + 스크립트 실행권한
RUN rm -rf /app/src/service && chmod +x /app/scripts/*.sh

RUN chown -R $UID:$GID /app /run /var/log /etc/supervisor /models /app/nltk_data

USER $UID
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
