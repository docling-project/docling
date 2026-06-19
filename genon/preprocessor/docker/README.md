# preprocessor Dockerfiles

PDF SDK의 사용 여부에 따라 `standard` 또는 `synap`로 분리됨. 

| 파일 | 용도 | 포함 |
|---|---|---|
| `Dockerfile.standard` | 기본 배포용 | LibreOffice + rhwp (PDF SDK 미포함) |
| `Dockerfile.synap` | 유료 PDF SDK(Synap) 보유 환경용 | 기본형(LibreOffice + rhwp) + PDF SDK (`PDF_SDK_TOKEN` 추가 필요, AI Search 팀 발급) |

> 기존의 단일 `Dockerfile` 은 PDF SDK 다운로드 단계가 그대로 포함돼 있어, 의도치 않게 유료 변형으로 빌드될 위험이 있었기 때문에 본 PR에서 삭제함. 신규 빌드는 반드시 위 두 variant 중 하나로 진행한다 — `doc-parser-build.sh`에서 `BUILD_VARIANT` 를 비워두고 실행하면 즉시 에러로 중단된다.

## GPU / CPU 분기 (이슈 [#210](https://github.com/genonai/doc_parser/issues/210))

위 두 Dockerfile 은 각각 `HW_VARIANT` 빌드 인자(`gpu` | `cpu`)로 다시 갈린다. 같은 Dockerfile 이 builder 단계에서 분기 처리한다.

| `HW_VARIANT` | torch | 용도 |
|---|---|---|
| `gpu` | `uv.lock` 기준 CUDA wheel + nvidia-* / triton 포함 | GPU 가속 환경 |
| `cpu` | CPU wheel 재설치 + nvidia-* / triton 제거 (경량) | GPU 없는 환경 |

`BUILD_VARIANT` × `HW_VARIANT` 조합으로 최대 4종 이미지가 만들어진다. 최종 태그는 기본 조합(`cpu`+`standard`)이면 접미사 없이 `:${IMAGE_VERSION}` (예: `:1.3.6.3`), 그 외 조합은 `:${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}` (예: `:1.3.6.3-gpu-synap`) — 이슈 #236. `HW_VARIANT` 도 비워두면 `doc-parser-build.sh` 가 에러로 중단된다.

## rhwp / LibreOffice 설치 on/off (이슈 [#286](https://github.com/genonai/doc_parser/issues/286))

두 Dockerfile 모두 `INSTALL_LIBREOFFICE` · `INSTALL_RHWP` 빌드 인자(`true` | `false`, 기본 `true`)로 해당 패키지를 빼고 빌드할 수 있다. rhwp/LibreOffice 를 쓰지 않는 사이트용.

| 빌드 인자 | `false` 일 때 |
|---|---|
| `INSTALL_LIBREOFFICE` | LibreOffice + Java apt 패키지, H2Orestart 확장(`loext` 단계) 미설치 |
| `INSTALL_RHWP` | rhwp 바이너리 미포함. `rhwp_builder_${INSTALL_RHWP}` stage alias 로 분기해 **Rust 빌드 stage 자체를 건너뜀** (`false` → 빈 `/rhwp_out/` 만 복사) |

- 미설치 backend 는 런타임 chain 에서 자동 제외된다(아래 가용성 판정 참조). `standard` 에서 둘 다 `false` 면 변환 backend 가 0개가 되며, 영향은 전처리기별로 다르다:
  - 적재형(지능형) — 비-PDF 입력을 내부 PDF 변환 후 파싱하므로 처리 불가 → "PDF 직접 입력/재빌드" 안내와 함께 실패.
  - 첨부형/변환형/파싱형 — HWP 는 내장 HWP SDK, docx/ppt 는 원본 직접 파싱이라 변환 backend 없이도 동작(영향 적음).
- `synap` 은 PDF SDK 가 남아 docx/ppt 등은 계속 변환된다.
- ⚠️ 두 플래그는 이미지 태그에 반영되지 않으니, 끈 이미지는 `IMAGE_VERSION` 에 식별자를 붙여 구분한다. 설정/빌드 절차는 [`../../README.md` "A-2. (선택) rhwp / LibreOffice 제외 빌드"](../../README.md#a-2-선택-rhwp--libreoffice-제외-빌드-이슈-286) 참고.

## HWP → PDF 변환 chain (런타임 동작)

`genon.preprocessor.converters.hwp_to_pdf.build_chain()` 이 가용한 backend 만 자동 등록한다. rhwp 는 이미지 안에 바이너리로 직접 포함되어 별도 외부 서비스 없이 동작한다.

- synap: `pdf_sdk → rhwp → libreoffice` (PDF SDK 우선, 실패 시 자동 fallback)
- standard: `rhwp → libreoffice` (PDF SDK 미포함)

각 backend 가용성 판정:
- `pdf_sdk` — `PDF_SDK_HOME/pdfConverter` 가 실행 가능 (synap 이미지에서만 true)
- `rhwp` — `/usr/local/bin/rhwp` 가 실행 가능 (양쪽 이미지 다 true)
- `libreoffice` — `shutil.which("soffice")` 결과 존재

env override (운영 시 일시 변경 가능):
- `RHWP_BIN=/usr/local/bin/rhwp` — rhwp 바이너리 경로 override (다른 위치 빌드 시)
- `HWP_TO_PDF_PRIMARY=<backend>` — 단일 backend 를 1순위로
- `HWP_TO_PDF_ORDER=<a>,<b>` — chain 순서 직접 지정
- `HWP_TO_PDF_DISABLE_FALLBACK=1` — primary 만 시도, 실패 시 None
- `HWP_TO_PDF_TIMEOUT_SEC=600` — backend 당 subprocess timeout

## rhwp 바이너리 (이미지 내 포함)

[`genonai/genos-rhwp`](https://github.com/genonai/genos-rhwp) 를 multi-stage builder 에서 `cargo build --release --bin rhwp` 로 빌드해 `/usr/local/bin/rhwp` 에 설치한다. Python 측은 subprocess 로 `rhwp export-pdf <input.hwp> -o <output.pdf>` 를 호출한다 — 외부 서비스/네트워크 의존 없음.

- `--build-arg RHWP_GIT_REF=<tag-or-sha>` 로 빌드 ref 고정 가능 (기본 `main`)
- Cargo cache mount 로 incremental build, 두 번째 이후 빌드는 빠름

## 빌드 방법

```bash
# 기본(standard) / CPU  → 태그 접미사 없음
BUILD_VARIANT=standard HW_VARIANT=cpu bash build-script/doc-parser-build.sh
# 기본(standard) / GPU
BUILD_VARIANT=standard HW_VARIANT=gpu bash build-script/doc-parser-build.sh
# synap / CPU
BUILD_VARIANT=synap HW_VARIANT=cpu bash build-script/doc-parser-build.sh
# synap / GPU
BUILD_VARIANT=synap HW_VARIANT=gpu bash build-script/doc-parser-build.sh
```

이미지 태그는 기본 조합(`cpu`+`standard`)이면 `:${IMAGE_VERSION}`, 그 외는 `:${IMAGE_VERSION}-${HW_VARIANT}-${BUILD_VARIANT}` 형태로 자동 결정된다 (예: `:1.3.6.3-gpu-synap`). `BUILD_VARIANT` / `HW_VARIANT` 는 `build-script/doc-parser-build.config` 에 적어두고 실행해도 된다.

토큰은 SDK 별로 fine-grained 분리되어 있다 (이슈 #199). `HWP_SDK_TOKEN` 은 두 variant 모두 필수 (HWP SDK 가 무료 자산이지만 현재 HF private dataset 에 호스팅됨), `PDF_SDK_TOKEN` 은 `synap` 일 때만 필수이며 **비공개 값(AI Search 팀 발급)** 이다. `synap` 빌드/배포가 필요하면 AI Search 팀에 문의한다. 토큰 안내는 [`../../README.md` 의 "전처리기 빌드 및 등록" 1번 / 2번 항목](../../README.md#전처리기-빌드-및-등록) 참고.
