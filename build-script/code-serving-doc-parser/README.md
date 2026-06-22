# code-serving-doc-parser — doc-parser 코드서빙 base 이미지 빌드

doc-parser 의 코드서빙(code-serving) base 이미지를 빌드한다. 런타임에 시스템이 이 이미지를 띄우고
git 소스를 `/app/src/service` 로 clone 해 `main.py`(facade) 를 실행한다. facade 의 무거운 deps 와
다운로드 아티팩트를 **빌드 시점에 pre-bake** 해 콜드스타트 단축 + 에어갭(오프라인) 동작을 보장한다.

> **전처리기(preprocessor) 이미지와 별개**다. 전처리기 빌드/등록은
> [`genon/README.md` "전처리기 빌드 및 등록"](../../genon/README.md#전처리기-빌드-및-등록) /
> [`genon/preprocessor/docker/README.md`](../../genon/preprocessor/docker/README.md) 참고.

## 빌드 방식 (요약)

- **라이브러리 설치는 `genon/code-serving` 과 동일**: `uv venv --seed` + `uv pip install .` → venv `/app/.venv`.
  (preprocessor 의 `uv sync` 방식이 아님.)
- **의존성 목록은 `genon/preprocessor` 의 uv 정보 기반**(`pyproject.toml`): preprocessor 직접 deps
  (− vendored `docling`) ∪ repo 루트 docling deps ∪ harness 필수(pydantic-settings, python-dotenv).
- **다운로드 아티팩트는 `Dockerfile.standard` 와 full parity** 로 복제(아래).
- **build context = repo 루트** (자기완결 아님): `genon/preprocessor/resources`(HCRBatang 폰트, docling-parse
  패치 헤더) 와 `build-script/hf_private_token.env`(HWP_SDK_TOKEN) 를 재사용.
- 구조: `Dockerfile`(CPU, python:3.12-slim) / `Dockerfile.gpu`(GPU, nvidia/cuda 12.4.1).

## pre-bake 되는 다운로드 아티팩트 (Dockerfile.standard parity)

| 아티팩트 | 위치/환경변수 | 출처 |
|---|---|---|
| Docling 모델 | `/models` (`DOCLING_ARTIFACTS_PATH`) | `docling-tools models download` (models 스테이지서 docling 임시설치) |
| HF 모델 (MiniLM 등) | `/models/doc_parser_models/...` | `mncai/doc_parser_models` |
| HWP SDK (convtext) | `/app/hwp_sdk` (+ venv 심링크) | HF `HeechanKim-Genon/hwp_sdk` (**HWP_SDK_TOKEN 필요**) |
| rhwp 바이너리 | `/usr/local/bin/rhwp` (`RHWP_BIN`) | `genonai/genos-rhwp` rust 빌드 |
| H2Orestart.oxt | LibreOffice 확장 | GitHub 릴리스 |
| 폰트 | HCRBatang/additional + noto-cjk/nanum/dejavu | resources + HF + apt |
| NLTK 데이터 | `/app/nltk_data` (`NLTK_DATA`) | `nltk.download` |
| EasyOCR korean_g2 | `/models/EasyOcr` | JaidedAI 릴리스 |
| docling-parse 4.1.0 패치 | `/app/.venv/.../site-packages/docling_parse` | 소스 패치 빌드(#245), venv 인터프리터로 설치해 wheel 덮어씀 |

## 빌드

### 1) 토큰 설정 (1회, Git 미추적)

HWP SDK 다운로드용 HF 토큰을 전처리기와 **동일한 파일**에 둔다:

```bash
echo "HWP_SDK_TOKEN=hf_xxx" >> build-script/hf_private_token.env
```

(토큰 발급은 [genon/README "전처리기 빌드 및 등록" 1번](../../genon/README.md#전처리기-빌드-및-등록) 참고.)

### 2) 설정 & 실행

```bash
# build.config 에서 HW_VARIANT(cpu|gpu)/버전/푸시여부 설정 후
bash build-script/code-serving-doc-parser/build.sh
```

- 태그: cpu → `mncregistry:30500/mnc/doc-parser-serving:<버전>`, gpu → `...:<버전>-gpu`.
- 빌드 직후 smoke: deps import + torch GPU/CPU variant + 아티팩트 존재 + **`docling_parse` 가 `/app/.venv`
  하위에 설치됐는지**(패치본 위치) 검증.
- 먼저 `PUSH_IMAGE=false` 로 로컬 검증 후 `true` 로 push.

## 의존성/아티팩트 동기화 주의

- `pyproject.toml` 의 deps 는 `genon/preprocessor/pyproject.toml` + repo 루트 docling deps 기반이다.
  그쪽이 바뀌면 여기 `pyproject.toml` 갱신 후 `uv lock` 재생성.
- 다운로드 단계는 `genon/preprocessor/docker/Dockerfile.standard` 와 ~중복이다. standard 의 아티팩트
  버전(모델/H2Orestart/rhwp ref 등)이 바뀌면 이 Dockerfile 들도 **수동 동기화** 필요.
- GPU(`Dockerfile.gpu`) 는 ubuntu 22.04(jammy) base 라 apt 패키지명/버전(특히 `openjdk-21`,
  `libgdk-pixbuf`)이 다를 수 있다. 첫 빌드 실패 시 해당 패키지명만 조정.

## 구성

- `Dockerfile` / `Dockerfile.gpu` — CPU / GPU base 이미지.
- `pyproject.toml` / `uv.lock` — uv 환경(의존성). preprocessor uv 정보 기반.
- `build.sh` / `build.config` — 빌드 스크립트/설정 (context = repo 루트, HWP_SDK_TOKEN secret).
- `scripts/` `supervisor/` `src/` `gunicorn/` — 코드서빙 harness (genon/code-serving 형태).
