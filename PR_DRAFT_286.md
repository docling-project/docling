# PR 초안 (이슈 #286) — 아직 PR 생성 안 함, 검토용 초안

- **Base** ← **Head**: `develop` ← `task/286-전처리기-빌드시-rhwp-및-libreoffice-설치-onoff-기능-추가`
- **Closes** #286
- **커밋**: `3b2146e` (8 files, +213 / -34)

---

## 제목

```
feat(#286): 전처리기 빌드 시 rhwp/libreoffice 설치 on/off 기능 추가
```

---

## 배경

- 한국은행 등 일부 사이트는 정책상 `rhwp` · `LibreOffice` 를 전처리기 이미지에 넣지 않기를 요구함.
- 기존엔 두 패키지가 항상 설치됨 → 빌드 단계에서 설치 여부를 플래그로 끌 수 있어야 함.

## 변경 요약

빌드 플래그 2개(`INSTALL_LIBREOFFICE` / `INSTALL_RHWP`, 기본 `true`)를 추가하고, 미설치 시에도 런타임이 깨지지 않도록 폴백 안내를 보강했음.

| 영역 | 파일 | 변경 |
|---|---|---|
| 빌드 설정 | `build-script/doc-parser-build.config` | 플래그 2개 + 주석(동작 영향/태그 주의) |
| 빌드 스크립트 | `build-script/doc-parser-build.sh` | 기본값·`true/false` 검증·INFO 출력·`--build-arg` 전달·standard+둘다off 경고 |
| 도커 | `docker/Dockerfile.standard`, `Dockerfile.synap` | LibreOffice apt/H2Orestart 조건 분기, rhwp stage-alias 분기 |
| 폴백 | `facade/intelligent_processor.py` | 변환 backend 전무 시 "PDF 직접 입력/재빌드" 안내 예외 |
| 문서 | `genon/README.md`, `docker/README.md`, `gitbook_doc/installation.md` | 제외 빌드 절차/안내 |

## 구현 상세

### 1. 빌드 플래그

- `INSTALL_LIBREOFFICE=false` → `base` 단계 apt에서 LibreOffice + Java 스택 제외, `loext` 단계 H2Orestart 확장 스킵.
- `INSTALL_RHWP=false` → rhwp 미포함.
- `build.sh` 에서 `true`/`false` 외 값은 즉시 에러 처리(Dockerfile stage alias·shell 조건이 정확한 값을 기대하므로).

### 2. rhwp — stage alias 로 Rust 빌드 자체 skip

```dockerfile
FROM rust:latest AS rhwp_builder_true      # cargo build → /rhwp_out/rhwp
FROM base        AS rhwp_builder_false      # mkdir /rhwp_out (빈 디렉토리)
FROM rhwp_builder_${INSTALL_RHWP} AS rhwp_builder
...
COPY --from=rhwp_builder /rhwp_out/ /usr/local/bin/
```

- `false` 면 BuildKit 이 `rhwp_builder_false` 만 빌드 → **무거운 Rust 빌드 stage 가 그래프에서 prune** 됨.
- runtime 은 `/rhwp_out/` 의 *내용* 을 복사하므로, off 일 때는 `/usr/local/bin/` 에 아무것도 안 들어감 → `rhwp_available()` 이 `False` → HWP→PDF chain 에서 자동 제외.

### 3. 미설치 시 graceful (코드 점검 결과)

- `converters/hwp_to_pdf` 의 `build_chain()` 은 원래부터 가용 backend 만 등록하고, 없으면 `ConverterChain.try_each()` 가 `None` 반환 → **예외 없음**(점검 완료).
- 단, `intelligent_processor` 는 변환 실패 시 일반 메시지(`PDF 변환 실패`)만 던졌음. backend 가 0개인 경우를 `_has_any_pdf_converter()` 로 구분해, 아래처럼 **행동 가능한 안내**로 교체:
  > 이 전처리기 이미지에는 PDF 변환기(rhwp/LibreOffice/PDF SDK)가 설치되어 있지 않아 '<파일>' 를 PDF 로 변환할 수 없습니다. PDF 로 변환한 파일을 입력하거나, 변환기를 포함해 전처리기 이미지를 다시 빌드하세요 (genon/README.md 참고).

### 4. 그 외 on/off 후보 점검

- apt 목록 점검 결과, 분리 가능한 무거운 묶음은 LibreOffice+Java 스택과 rhwp 둘뿐이라 그 둘만 플래그화.
- tesseract/imagemagick/ffmpeg/poppler/fonts 는 OCR·렌더링 공용 의존이라 분리하지 않음.

## 동작 매트릭스

| variant | LibreOffice | rhwp | 결과 |
|---|---|---|---|
| standard | on | on | 기존과 동일 (`rhwp → libreoffice`) |
| standard | off | off | **변환 backend 0개 → PDF 입력만 처리.** HWP/오피스는 안내와 함께 실패 (= 한국은행 케이스) |
| synap | off | off | PDF SDK 가 1순위 → docx/ppt 등 변환됨, HWP/HWPX 는 PDF SDK 로 처리 |

## 주의 / 검토 포인트

- ⚠️ **두 플래그는 이미지 태그에 반영되지 않음.** 패키지를 끈 특수 이미지는 `IMAGE_VERSION` 에 식별자(예: `1.3.6.3-nooffice`)를 수동으로 붙여 운영 이미지와 구분해야 함 (config/문서에 명시). → *태그 규칙에 자동 반영할지 여부 리뷰 필요.*
- 폴백 안내는 **`intelligent_processor` 에만** 적용함(이슈 범위). `convert_processor` / `attachment_processor` 에도 같은 안내가 필요하면 후속.
- `genon/README.md` 에 "A-2. (선택) rhwp/LibreOffice 제외 빌드" 절을 추가함 (installation.md/docker README 가 이 절을 링크). 불필요하면 제거 가능.

## 테스트

- `python -m py_compile intelligent_processor.py` 통과
- `bash -n doc-parser-build.sh` 통과
- ⚠️ **실제 도커 빌드는 미실행** (rust 빌드·모델 다운로드 비용). 리뷰 시 4종 조합 + `INSTALL_*` off 조합 빌드 확인 권장.
