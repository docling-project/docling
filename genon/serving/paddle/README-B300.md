# PaddleOCR B300 (Blackwell) 빌드 가이드

NVIDIA B300 (Blackwell, compute capability `sm_100`) GPU 환경용 PaddleOCR 서빙 이미지.
기존 cu126 이미지(`doc-parser-ocr`) 와 **완전히 분리된 sibling 파일 세트** 로 운용한다.

> 사내 B300 GPU 부재로 인해 코드 단계에서 실측 검증 불가. 실제 B300 노드에서
> [smoke test](#5-smoke-test) 통과 전까지는 "지원 보장" 으로 간주하지 말 것.

## 1. 사용 버전 (cu129 호환 PaddleOCR 스택)

| 패키지 | 버전 | cu129 호환 근거 |
| --- | --- | --- |
| `paddlepaddle-gpu` | **3.2.0** (cu129 wheel) | PaddlePaddle 공식 install 매트릭스에 `Blackwell sm_100 → CUDA 12.9 (Recommend)` + 빌드 태그 `cuda12.9-cudnn9.9-mkl-gcc12.2-avx` 명시 ([Tables_en](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html)). 인덱스에 `paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl` 실재. |
| `paddlex` | **3.3.6** (extras: `serving`, `ocr`) | PaddleX 3.3.x 는 OCR 파이프라인(PP-OCRv5) 을 내장 — **별도의 `paddleocr` 패키지 설치 불필요** (`uv-b300.lock` 에 `paddleocr` 항목 없음). cu126 이미지와 동일 버전 유지로 모델 거동 차이 최소화. |
| OCR 모델 | **PP-OCRv5_server_det + korean_PP-OCRv5_mobile_rec** | `resources/paddleocr_model.zip` 그대로 사용 (cu126 이미지와 동일). |
| `nvidia-*-cu12` | 12.9.x (cuDNN 9.9.0.52, NCCL 2.27.3 등) | paddlepaddle-gpu 3.2.0 cu129 의 transitive 의존. cuDNN 9.9 는 위 빌드 태그와 일치. NCCL 2.27.3 release notes 가 *"supports ... CUDA 12.9"* 명시 ([NCCL 2.27.3](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/release-notes/rel_2-27-3.html)). |
| Python · Base image | 3.10 · `python:3.10-slim-bookworm` | cu126 이미지와 동일. |

### 호스트 노드 요구
- **NVIDIA Linux driver ≥ 575.51.03** (CUDA 12.9 GA 최소 — [release notes Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html))
- B300 SKU 인식용 driver 는 NVIDIA 가 SKU 단위 매핑을 공개하지 않으므로 **운영팀이 datasheet 로 확인**

## 2. 파일 구성

cu126 경로에 영향을 주지 않기 위해 모두 sibling 파일.

| 용도 | cu126 | B300 (cu129) |
| --- | --- | --- |
| 의존성 | `pyproject.toml` / `uv.lock` | `pyproject-b300.toml` / `uv-b300.lock` |
| Dockerfile | `docker/Dockerfile` | `docker/Dockerfile-b300` |
| Build script | `build-script/paddle-ocr-build.{config,sh}` | `build-script/paddle-ocr-b300-build.{config,sh}` |
| K8s manifest | `k8s-manifest/doc-parser-ocr-*.yaml` | `k8s-manifest/doc-parser-ocr-b300-*.yaml` (+ `-smoke-job.yaml`) |
| Smoke test | — | `etc/smoke_test.{sh,_inference.py,_compare.py}` |

`config/`, `resources/`, `etc/health_checker.sh` 는 양쪽 공유.

## 3. 빌드

```shell
cd build-script
./paddle-ocr-b300-build.sh
```

기본 태그: `mncregistry:30500/doc-parser-ocr-b300:0.0.0`. 빌드 단계 검증은 `python -c "import paddle"` 한 줄뿐 (GPU 동작 검증은 smoke test 에서 별도 수행).

## 4. 배포

```shell
kubectl apply -f genon/serving/paddle/k8s-manifest/doc-parser-ocr-b300-deployment.yaml
# NodePort (30881) 사용 시
kubectl apply -f genon/serving/paddle/k8s-manifest/doc-parser-ocr-b300-deployment-node-port.yaml
```

이름과 NodePort 가 cu126 과 다르므로 동시 배포 가능.

## 5. Smoke test

이슈 #211 검토 기준 3 단계를 컨테이너 안에서 수행 — **B300 GPU 노드에서만 의미**.

```shell
POD=$(kubectl -n llmops get pod -l app=doc-parser-ocr-b300 -o name | head -1)

# 단계 1+2: health(/health 200) + GPU inference (paddle.device.get_device() == gpu:*)
kubectl -n llmops exec -it "$POD" -- bash /app/etc/smoke_test.sh

# 단계 3: 기존 cu126 baseline 과 결과 비교
#   (cu126 pod 에 smoke_test*.{sh,py} 3 파일 복사 후 동일 스크립트로 baseline 생성)
kubectl cp ./baseline_cu126.json llmops/"${POD#pod/}":/tmp/ocr_smoke/baseline_cu126.json
kubectl -n llmops exec -it "$POD" -- \
  bash /app/etc/smoke_test.sh compare \
    /tmp/ocr_smoke/baseline_cu126.json /tmp/ocr_smoke/result.json
```

전용 Job 으로도 가능: `k8s-manifest/doc-parser-ocr-b300-smoke-job.yaml`.

비교 통과 기준: 인식 텍스트 정규화 일치 + score 차이 ≤ `--score-tol` (기본 0.05).

PaddleX `--serve` 의 OCR endpoint path 는 버전·파이프라인에 따라 다르므로 본 가이드에서는 단정하지 않음. 실 배포 시 `paddlex --serve` 시작 로그 또는 `/docs` (FastAPI auto-docs) 로 확인.

## 6. 영향도

기존 cu126 경로 파일은 한 줄도 수정하지 않음 → 기존 환경 영향 0. 문제 시 B300 Deployment 만 삭제하면 cu126 환경 그대로 유지.

## 7. 공식 docs 가 침묵하는 항목 (실 환경 검증 필요)

| 항목 | 1차 출처 |
| --- | --- |
| B300 SKU ↔ compute capability 매핑 (sm_100 vs sm_103) | [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) 는 *"compute capability 10.0"* 만 언급, SKU 매핑 없음. [CUDA 12.9 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) 가 sm_103 추가만 명시. |
| paddlepaddle 3.2.0 release notes 자체에 Blackwell/cu129 언급 | [v3.2.0 release](https://github.com/PaddlePaddle/Paddle/releases/tag/v3.2.0) — 없음 (install 매트릭스에만 존재) |
| PaddleX 3.3.6 ↔ paddlepaddle 3.2.0 명시적 호환 매트릭스 | 없음 ([PyPI 메타](https://pypi.org/pypi/paddlex/3.3.6/json) 에 paddle 버전 핀 부재) |
| PaddleOCR v3.2.0 release notes 의 paddle 호환 명시 | *"Full support for PaddlePaddle framework versions 3.1.0 and 3.1.1"* — 3.2.0 명시 없음 ([링크](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.2.0)) |
| cu129 wheel 의 sm_100 cubin 사전 포함 여부 | 공식 문서 미언급. 컨테이너 안 `cuobjdump --list-elf <libpaddle.so>` 로 확인 가능 |

### 잠재 위험 (런타임 검증)

- **protobuf 6/7**: paddle 메타 상한 부재로 형식상 통과. import 시 즉시 드러나므로 smoke test 로 검증.
- **starlette 0.49 vs 1.0 / fastapi 0.128 vs 0.136**: uv 가 environment marker 별로 둘 다 lock. 빌드 후 `pip list | grep -E "fastapi|starlette"` 로 실제 설치 버전 확인.
- **score drift**: 부동소수 inference 차이가 허용 오차(0.05) 초과 시 운영팀 합의 후 tol 조정.
