# code-serving-doc-parser — doc-parser 코드서빙 base 이미지 빌드

doc-parser 의 코드서빙(code-serving) base 이미지를 빌드하기 위한 **자기완결(self-contained)** 폴더.
이 폴더 내용만으로 빌드되며 `genon/code-serving` 이나 repo 루트 파일을 참조하지 않는다
(build context = 이 폴더 자신).

> **이 이미지는 전처리기(preprocessor) 이미지와 별개다.**
> - 전처리기(`doc-parser-preprocessor`) = Genos 에서 실행되는 전처리 API 이미지. 빌드/등록은
>   [`genon/README.md` "전처리기 빌드 및 등록"](../../genon/README.md#전처리기-빌드-및-등록) /
>   [`genon/preprocessor/docker/README.md`](../../genon/preprocessor/docker/README.md) 참고.
> - 이 이미지(`doc-parser-serving`) = **코드서빙 base 이미지**. 런타임에 git 소스를 clone 해
>   `main.py` 를 실행하는 harness 에, facade 실행에 필요한 무거운 deps 를 미리 깔아둔 것.
> - 전처리기와 달리 **`HWP_SDK_TOKEN` / `PDF_SDK_TOKEN` / `BUILD_VARIANT`(standard/synap) 가
>   필요 없다** (유료 PDF SDK 미포함). 분기 축은 `HW_VARIANT`(cpu/gpu) 하나뿐이다.

## 코드서빙 동작 (런타임)

1. 시스템이 이 base 이미지를 컨테이너로 띄운다 (CMD = `supervisord`).
2. `scripts/init.sh` 가 현재 git 소스를 `/app/src/service` 로 clone 하고, clone 된 repo 루트의
   `requirements.txt` 로 환경을 구성한다 — 이 base 이미지에 동일 deps 가 **pre-bake** 되어 있어
   대부분 이미 충족 → 빠르게 통과한다.
3. `scripts/entrypoint.sh` 가 `/app/src/service/main.py` 를 감지해 `uvicorn main:app`(port 8080) 실행.
4. `supervisor` 가 위 프로세스 + 로그 발행(`src/log_watcher.py` → RabbitMQ)을 관리한다.

facade 는 docling/easyocr/transformers 등 매우 무거운 deps 를 쓰므로, 콜드스타트 단축을 위해
이 base 이미지에 **빌드 시점에 pre-bake** 한다. 의존성은 `pyproject.toml` 에 선언되어 있고
`Dockerfile` / `Dockerfile.gpu` 의 `uv pip install .` 로 설치된다 (별도 requirements.txt 사본 없음).
facade 실행에 필요한 **시스템 패키지**(libreoffice/poppler/tesseract/libmagic/imagemagick 등)는
전처리기 `genon/preprocessor/docker/Dockerfile.standard` 의 apt-get 세트를 따른다.

## HW_VARIANT (cpu / gpu)

`HW_VARIANT` 한 값으로 두 산출물이 갈린다. **비워둔 채 `build.sh` 를 실행하면 즉시 에러로 중단**된다.

| `HW_VARIANT` | Dockerfile | base 이미지 | torch | 최종 태그 |
|---|---|---|---|---|
| `cpu` (기본) | `Dockerfile` | `python:3.12-slim-trixie` (multi-stage) | pyproject deps 의 CPU torch | `:${IMAGE_VERSION}` |
| `gpu` | `Dockerfile.gpu` | `nvidia/cuda:12.4.1-cudnn-runtime` | `cu124` wheel 로 재고정 | `:${IMAGE_VERSION}-gpu` |

- 태그 규칙은 전처리기 컨벤션을 따른다 — 기본 산출물(`cpu`)은 접미사 없이 `:${IMAGE_VERSION}`,
  그 외(`gpu`)는 `:${IMAGE_VERSION}-gpu`.
- 최종 이미지명: `${DOCKER_REGISTRY}/mnc/${IMAGE_NAME}:<태그>`
  (기본값 `mncregistry:30500/mnc/doc-parser-serving:<버전>`).
- GPU 빌드는 `uv pip install .` 후 마지막에 torch 를 `cu124` wheel 로 재설치해 CPU torch 를 override 한다.

## 빌드 / 배포

### A. 설정

[`build.config`](build.config) 에서 다음을 설정한다 (전처리기와 달리 토큰 불필요):

```bash
# build-script/code-serving-doc-parser/build.config
HW_VARIANT=cpu          # 또는 gpu (비우면 에러)
IMAGE_VERSION=0.1.0     # 필요 시 교체
PUSH_IMAGE=true         # 먼저 false 로 로컬 검증 권장
SMOKE_TEST=true
```

### B. 빌드

```bash
bash build-script/code-serving-doc-parser/build.sh
```

- 빌드 직후 smoke test 로 핵심 deps import(`fastapi`/`docling_core`/`transformers`) +
  torch 의 CUDA 포함 여부가 `HW_VARIANT` 와 일치하는지 검증한다. 실패하면 빌드도 실패한다
  (base 이미지엔 repo 소스가 없어 full parse 는 하지 않음 — 의존성/torch 분기만 확인).
- 처음에는 `PUSH_IMAGE=false` 로 로컬 빌드를 검증한 뒤, 정상 확인되면 `true` 로 push 한다.

### C. push

`PUSH_IMAGE=true` 면 `build.sh` 가 빌드 + smoke 통과 후 자동으로 `docker push` 한다
(전처리기의 `register_image.sh`/DB 등록 흐름과는 별개 — 이 이미지는 레지스트리 push 만 수행).

### D. 사이트(오프라인) 배포

레지스트리 직접 접근이 어려운 사이트는 전처리기와 동일한 save/load 패턴을 쓴다:

```bash
# 1. 이미지 저장 (예: cpu)
docker save mncregistry:30500/mnc/doc-parser-serving:0.1.0 | gzip > doc-parser-serving-0.1.0.tar.gz
# 2. 사이트에서 복원
gunzip -c doc-parser-serving-0.1.0.tar.gz | docker load
```

- gpu 이미지면 태그가 `:0.1.0-gpu` 이니 파일명/명령어를 그에 맞게 바꾼다.

## ⚠️ 의존성 동기화 (pyproject.toml / uv)

`pyproject.toml` 의 의존성은 **`genon/preprocessor` 의 uv 정보를 기반**으로 구성한 것이다:

- preprocessor 직접 deps (`genon/preprocessor/pyproject.toml`) — 단, vendored `docling` 패키지
  자체는 제외(런타임에 clone 된 repo 가 `/app/src/service` 에서 제공).
- vendored docling(repo 루트 `pyproject.toml`) 의 직접 deps — docling import 에 필요한
  easyocr/rtree/tree-sitter*/scipy/pandas 등.
- harness 필수 deps — pydantic-settings, python-dotenv.

preprocessor 또는 repo 루트 docling 의 deps 가 바뀌면 `pyproject.toml` 을 갱신하고 lock 을 재생성한다:

```bash
cd build-script/code-serving-doc-parser
# pyproject.toml 의존성 수정 후
uv lock        # uv.lock 재생성/검증
```

## 구성

- `Dockerfile` / `Dockerfile.gpu` — CPU / GPU base 이미지 (deps 는 `uv pip install .` 로 pre-bake).
- `pyproject.toml` / `uv.lock` — uv 환경(의존성). preprocessor uv 정보 기반.
- `build.sh` / `build.config` — 빌드 스크립트/설정 (context = 이 폴더).
- `scripts/` `supervisor/` `src/` `gunicorn/` — 코드서빙 harness (genon/code-serving 형태).
