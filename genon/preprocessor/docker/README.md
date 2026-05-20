# preprocessor Dockerfiles

이슈 [#199](https://github.com/genonai/doc_parser/issues/199) 에 따라 두 variant 로 분리됨.

| 파일 | 용도 | 포함 |
|---|---|---|
| `Dockerfile.opensource` | 오픈소스 배포용 | LibreOffice (PDF SDK / rhwp 바이너리 **미포함**) |
| `Dockerfile.enterprise` | 유료 (PDF SDK 보유) 환경용 | 위 + PDF SDK (`PDF_SDK_TOKEN` 추가 필요) |

> 기존 단일 `Dockerfile` 은 PDF SDK 다운로드 단계가 그대로 포함돼 있어 의도치 않게 유료 변형으로 빌드될 위험이 있었기 때문에 본 PR에서 삭제했다. 신규 빌드는 반드시 위 두 variant 중 하나로 진행한다 — `BUILD_VARIANT` 를 비워두고 `doc-parser-build.sh` 를 실행하면 즉시 에러로 중단된다.

## HWP → PDF 변환 chain (런타임 동작)

`genon.preprocessor.converters.hwp_to_pdf.build_chain()` 이 가용한 backend 만 자동 등록한다.

- 엔터프라이즈 + `RHWP_PDF_API_URL` 설정 시: `pdf_sdk → rhwp → libreoffice`
- 엔터프라이즈 + URL 미설정: `pdf_sdk → libreoffice`
- 오픈소스 + `RHWP_PDF_API_URL` 설정 시: `rhwp → libreoffice`
- 오픈소스 + URL 미설정: `libreoffice`

각 backend 가용성 판정:
- `pdf_sdk` — `PDF_SDK_HOME/pdfConverter` 가 실행 가능 (엔터프라이즈 이미지에서만 true)
- `rhwp` — `RHWP_PDF_API_URL` 환경변수가 비어있지 않음 (외부 서비스 endpoint)
- `libreoffice` — `shutil.which("soffice")` 결과 존재

env override (운영 시 일시 변경 가능):
- `RHWP_PDF_API_URL=http://rhwp-pdf-api:7878` — genos-rhwp 의 serve-pdf endpoint base URL
- `HWP_TO_PDF_PRIMARY=<backend>` — 단일 backend 를 1순위로
- `HWP_TO_PDF_ORDER=<a>,<b>` — chain 순서 직접 지정
- `HWP_TO_PDF_DISABLE_FALLBACK=1` — primary 만 시도, 실패 시 None
- `HWP_TO_PDF_TIMEOUT_SEC=600` — backend 당 timeout (subprocess 또는 HTTP)

## 빌드 방법

```bash
# 오픈소스 빌드
BUILD_VARIANT=opensource bash build-script/doc-parser-build.sh

# 엔터프라이즈 빌드
BUILD_VARIANT=enterprise bash build-script/doc-parser-build.sh
```

이미지 태그는 자동으로 `:${IMAGE_VERSION}-${BUILD_VARIANT}` 형태가 된다 (예: `:1.3.6.3-enterprise`).

토큰은 SDK 별로 fine-grained 분리되어 있다 (이슈 #199). `HWP_SDK_TOKEN` 은 두 variant 모두 필수 (HWP SDK 가 무료 자산이지만 현재 HF private dataset 에 호스팅됨), `PDF_SDK_TOKEN` 은 enterprise 일 때만 필수. 두 토큰 값은 [`../../README.md` 의 "전처리기 빌드 및 등록" 1번 / 2번 항목](../../README.md#전처리기-빌드-및-등록) 에 안내된 내부 드라이브 링크에서 확인한다.

## rhwp PDF API (외부 서비스)

[`genonai/genos-rhwp`](https://github.com/genonai/genos-rhwp) 의 `Dockerfile.pdf-api` + `k8s/rhwp-pdf-api.yaml` 로 별도 Deployment + ClusterIP Service `rhwp-pdf-api:7878` 가 클러스터에 배포된다 (OCR / VLM endpoint 와 동일한 운영 패턴).

> 회사 클러스터에 아직 떠 있지 않다면 [`genon/README.md` 의 §HWP → PDF 변환용 rhwp-pdf-api 배포](../../README.md#-hwp--pdf-변환용-rhwp-pdf-api-배포-이슈-199) 를 step-by-step 으로 따라간다. 빌드 시 만나는 두 가지 누락(`Cargo.lock`, `saved/`) 우회 방법 + curl 검증까지 포함.

doc_parser 컨테이너는 `RHWP_PDF_API_URL` 환경변수로 base URL 만 주입받아 다음 endpoint 로 HTTP 호출한다:

- `POST {RHWP_PDF_API_URL}/api/convert/hwp-to-pdf`
- Request: `Content-Type: application/octet-stream`, body = HWP 바이트
- Response: `Content-Type: application/pdf`, body = PDF 바이트

같은 namespace 면 `http://rhwp-pdf-api:7878`, 다른 namespace 면 FQDN `http://rhwp-pdf-api.<ns>.svc.cluster.local:7878`.

> ⚠ 현재 두 Dockerfile 의 `RHWP_PDF_API_URL` 기본값은 **임시 placeholder** (`http://rhwp-pdf-api:7878`) 임. 실제 운영 endpoint 가 확정되면 deploy manifest 의 env 로 override 하거나 Dockerfile 의 placeholder 를 교체할 것. 이슈 #199 TODO 항목.

## 로컬 dev 에서 rhwp 사용

`docker-compose.yml` 에 `genonai/genos-rhwp` 의 `Dockerfile.pdf-api` 로 빌드한 컨테이너를 sidecar 로 띄우고 `RHWP_PDF_API_URL=http://localhost:7878` 로 설정하면 동일하게 동작한다. 별도 띄우지 않으면 chain 이 자동으로 LibreOffice 로 fallback.
