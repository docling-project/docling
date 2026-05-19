# preprocessor Dockerfiles

이슈 [#199](https://github.com/genonai/doc_parser/issues/199) 에 따라 두 variant 로 분리됨.

| 파일 | 용도 | 포함 |
|---|---|---|
| `Dockerfile.opensource` | 오픈소스 배포용 | LibreOffice + rhwp (PDF SDK **미포함**, 다운로드 단계도 없음) |
| `Dockerfile.enterprise` | 유료 (PDF SDK 보유) 환경용 | 위 + PDF SDK (`HF_TOKEN` 필요) |
| `Dockerfile` | 레거시 단일 빌드 | 기존 운영용 — 점진 마이그레이션 동안 유지 |

## HWP → PDF 변환 chain (런타임 동작)

`genon.preprocessor.converters.hwp_to_pdf.build_chain()` 이 컨테이너에 설치된 backend 만 자동 등록한다.

- 엔터프라이즈: `pdf_sdk → rhwp → libreoffice` (PDF SDK 우선, 실패 시 자동 fallback)
- 오픈소스: `rhwp → libreoffice` (PDF SDK 자산 자체가 없으므로 가용성 false)

env override (운영 시 일시 변경 가능):
- `HWP_TO_PDF_PRIMARY=<backend>` — 단일 backend 를 1순위로
- `HWP_TO_PDF_ORDER=<a>,<b>` — chain 순서 직접 지정
- `HWP_TO_PDF_DISABLE_FALLBACK=1` — primary 만 시도, 실패 시 None
- `HWP_TO_PDF_TIMEOUT_SEC=600` — backend 당 subprocess timeout
- `RHWP_BIN=/usr/local/bin/rhwp` — rhwp 바이너리 경로 override

## 빌드 방법

```bash
# 오픈소스 빌드
BUILD_VARIANT=opensource bash build-script/doc-parser-build.sh

# 엔터프라이즈 빌드
BUILD_VARIANT=enterprise bash build-script/doc-parser-build.sh

# 레거시 (기존 그대로)
BUILD_VARIANT= bash build-script/doc-parser-build.sh
```

이미지 태그는 자동으로 `:${IMAGE_VERSION}-${BUILD_VARIANT}` 형태가 된다 (variant 미지정 시 suffix 없음).

`HF_TOKEN` 은 두 variant 모두 HWP SDK 다운로드용으로 여전히 필요하다 (HWP SDK 는 무료 자산이지만 현재 HF private dataset 에 호스팅됨).

## rhwp 바이너리

`Dockerfile.opensource` / `Dockerfile.enterprise` 모두 `rhwp_builder` multi-stage 에서 [`genonai/genos-rhwp`](https://github.com/genonai/genos-rhwp) 를 `cargo build --release --bin rhwp` 로 빌드해 `/usr/local/bin/rhwp` 에 설치한다.

- `--build-arg RHWP_GIT_REF=<tag-or-sha>` 로 빌드 ref 고정 가능 (기본 `main`).
- Cargo cache mount 로 incremental 가능 — 두 번째 이후 빌드는 빠름.
