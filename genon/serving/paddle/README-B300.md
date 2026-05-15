# PaddleOCR B300 (Blackwell) 빌드 가이드

이 문서는 NVIDIA B300 (Blackwell, compute capability `sm_100`) GPU 환경에서 동작하는
PaddleOCR 서빙 이미지의 빌드 · 배포 · 검증 절차를 다룬다.
기존 cu126 기반 이미지(`doc-parser-ocr`) 와는 **완전히 분리된 파일 세트**로 운용된다.

> ⚠️ 현재 사내에 B300 GPU 가 없어, 본 가이드의 절차 자체는 외부 공식 문서와
> 패키지 인덱스 메타데이터를 기준으로 작성되었다. 실제 B300 노드에서 [smoke test](#smoke-test)
> 를 수행해 결과를 확인하기 전까지는 "검증된 동작" 으로 간주하지 말 것.

## 1. 파일 구성

cu126 경로에 영향을 주지 않기 위해 모두 sibling 파일로 추가했다.

| 용도 | 기존 (cu126) | B300 (cu129) |
| --- | --- | --- |
| Python 의존성 | `pyproject.toml` | `pyproject-b300.toml` |
| Lock | `uv.lock` | `uv-b300.lock` |
| Dockerfile | `docker/Dockerfile` | `docker/Dockerfile-b300` |
| 빌드 config | `build-script/paddle-ocr-build.config` | `build-script/paddle-ocr-b300-build.config` |
| 빌드 스크립트 | `build-script/paddle-ocr-build.sh` | `build-script/paddle-ocr-b300-build.sh` |
| K8s Deployment | `k8s-manifest/doc-parser-ocr-deployment.yaml` | `k8s-manifest/doc-parser-ocr-b300-deployment.yaml` |
| K8s Deployment (NodePort) | `k8s-manifest/doc-parser-ocr-deployment-node-port.yaml` | `k8s-manifest/doc-parser-ocr-b300-deployment-node-port.yaml` |
| Smoke test Job | — | `k8s-manifest/doc-parser-ocr-b300-smoke-job.yaml` |
| Smoke test 스크립트 | — | `etc/smoke_test.sh`, `etc/smoke_test_inference.py`, `etc/smoke_test_compare.py` |

`config/ocr.yaml`, `config/supervisord.conf`, `resources/`, `etc/health_checker.sh` 는
B300 / cu126 양쪽에서 그대로 공유한다 (변경할 이유가 없음).

## 2. 패키지 버전 근거

| 항목 | 값 | 근거 (1차 출처) |
| --- | --- | --- |
| PaddlePaddle wheel 인덱스 | `https://www.paddlepaddle.org.cn/packages/stable/cu129/` | PaddlePaddle 공식 install 매트릭스가 `Blackwell sm_100 → CUDA 12.9 (Recommend), CUDA 13.0` + 빌드 태그 `cuda12.9-cudnn9.9-mkl-gcc12.2-avx` 로 등재. 출처: [Tables_en](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html). |
| `paddlepaddle-gpu` | `3.2.0` | cu129 인덱스에 `paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl` 실재. cu126 이미지와 같은 버전을 사용해 모델 동작 차이를 최소화. |
| `paddlex` | `3.3.6` | cu126 이미지와 동일. OCR 파이프라인/모델 호환을 위해 같은 버전 유지. PyPI 메타에 paddlepaddle 버전 핀 없음 ([paddlex 3.3.6 JSON](https://pypi.org/pypi/paddlex/3.3.6/json)). |
| Python | `3.10` | 기존 이미지와 동일. cu129 wheel 의 cp310 빌드 존재. |
| `nvidia-*-cu12` | `12.9.x` (cublas, cudnn 9.9.0.52, cufft, ...) | paddlepaddle-gpu 3.2.0 cu129 의 의존으로 자동 설치 (`uv-b300.lock` 참조). cuDNN 9.9 는 위 빌드 태그와 일치. |
| `nvidia-nccl-cu12` | `2.27.3` | NCCL 2.27 공식 release notes에 *"This NCCL release supports CUDA 12.2, CUDA 12.4, and CUDA 12.9"* 명시 ([링크](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/release-notes/rel_2-27-3.html)). |
| Base image | `python:3.10-slim-bookworm` | 기존 이미지와 동일. CUDA runtime 라이브러리는 wheel 의존으로 컨테이너 안에 자동 설치되므로 NVIDIA CUDA base image 를 별도로 사용하지 않음. |

### 호스트 노드 요구 (운영팀)

| 항목 | 최소값 | 출처 |
| --- | --- | --- |
| NVIDIA Linux driver | **>= 575.51.03** | [CUDA 12.9 GA release notes Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). CUDA 12.9 런타임 최소 driver. |
| B300 HW 인식용 driver | 위 값 이상이되, **B300 datasheet 별도 확인 필요** | NVIDIA 공식 docs 가 B300 SKU 단위 driver 최소 버전을 공개적으로 명시하지 않음 (운영팀이 datasheet 로 확인). |
| nvidia-container-toolkit | 최신 안정 버전 권장 | NVIDIA 공식 문서가 CUDA 12.9 컨테이너 실행 최소 toolkit 버전을 별도로 명시하지 않음. |

## 3. 빌드

```shell
cd build-script
./paddle-ocr-b300-build.sh
```

- 이미지 태그: `mncregistry:30500/doc-parser-ocr-b300:0.0.0` (기본값, config 에서 변경 가능)
- 빌드 단계에서 수행되는 검증은 `python -c "import paddle"` 한 줄뿐이다.
  GPU 동작 검증은 [smoke test](#smoke-test) 단계에서 별도로 수행한다.

빌드 머신에 GPU 가 없어도 빌드는 통과해야 한다. 만약 빌드 단계에서 GPU 관련 에러가 난다면
스펙이 잘못된 것이므로 Dockerfile-b300 의 import 헬스체크를 의도적으로 유지하라.

## 4. 배포

```shell
# 일반 ClusterIP
kubectl apply -f genon/serving/paddle/k8s-manifest/doc-parser-ocr-b300-deployment.yaml

# NodePort (30881)
kubectl apply -f genon/serving/paddle/k8s-manifest/doc-parser-ocr-b300-deployment-node-port.yaml
```

기존 cu126 서비스(`doc-parser-ocr-service`) 와 이름이 다르므로 동시 배포 가능하다.
NodePort 도 cu126(30880) 와 충돌하지 않도록 30881 을 사용한다.

## 5. Smoke test

이슈 #211 검토 기준을 충족하기 위한 3 단계 검증을 컨테이너 안에서 실행한다.
**B300 GPU 가 실제로 붙은 노드** 에서만 의미가 있다.

### 5-1. 단계 ① 컨테이너 기동 확인 (`/health` 200 응답)

```shell
# K8s 환경
kubectl -n llmops port-forward svc/doc-parser-ocr-b300-service 18080:8080
curl -fsS http://127.0.0.1:18080/health
```

또는 docker 단독 실행 시:

```shell
docker run --rm --gpus all -p 8080:8080 \
  mncregistry:30500/doc-parser-ocr-b300:0.0.0
# 다른 터미널에서:
curl -fsS http://127.0.0.1:8080/health
```

### 5-2. 단계 ② OCR 호출 시 GPU inference 정상 수행 확인

컨테이너 안에서 `smoke_test.sh inference` 를 실행한다. 이 스크립트는

1. `paddle.device.get_device()` 가 `gpu:*` 인지 검사 → 아니면 즉시 실패
2. `/app/config/ocr.yaml` 파이프라인을 로드해 합성 이미지 1장으로 OCR 수행
3. 결과를 JSON 으로 dump (`/tmp/ocr_smoke/result.json`)

배포된 pod 에 접속해서 실행:

```shell
POD=$(kubectl -n llmops get pod -l app=doc-parser-ocr-b300 -o name | head -1)
kubectl -n llmops exec -it "${POD}" -- bash /app/etc/smoke_test.sh inference
kubectl -n llmops exec -it "${POD}" -- cat /tmp/ocr_smoke/result.json
```

또는 한 번에 health + inference 모두 수행:

```shell
kubectl -n llmops exec -it "${POD}" -- bash /app/etc/smoke_test.sh
```

전용 Job 으로 띄우려면:

```shell
kubectl apply -f genon/serving/paddle/k8s-manifest/doc-parser-ocr-b300-smoke-job.yaml
kubectl -n llmops logs -f job/doc-parser-ocr-b300-smoke
```

추가로 NVIDIA 사이드에서도 점유율을 한 번 더 확인하면 좋다:

```shell
kubectl -n llmops exec -it "${POD}" -- nvidia-smi
```

### 5-3. 단계 ③ 기존 OCR 결과와 기능적 동일성 확인

먼저 **기존 cu126 컨테이너** 에서 같은 스크립트로 baseline 을 만든다:

```shell
# cu126 pod 안에서
bash /app/etc/smoke_test.sh inference
cp /tmp/ocr_smoke/result.json /tmp/ocr_smoke/baseline_cu126.json
# 호스트로 꺼내기
kubectl cp llmops/<cu126-pod>:/tmp/ocr_smoke/baseline_cu126.json ./baseline_cu126.json
```

> 참고: cu126 이미지에는 `smoke_test.sh` 가 포함되어 있지 않다. 동일 검증을 하려면
> 본 브랜치의 `etc/smoke_test*.{sh,py}` 3 파일을 cu126 pod 안으로 복사한 뒤 실행한다.
>
> ```shell
> kubectl cp ./genon/serving/paddle/etc/smoke_test.sh           llmops/<cu126-pod>:/app/etc/
> kubectl cp ./genon/serving/paddle/etc/smoke_test_inference.py llmops/<cu126-pod>:/app/etc/
> kubectl cp ./genon/serving/paddle/etc/smoke_test_compare.py   llmops/<cu126-pod>:/app/etc/
> ```

다음으로 B300 pod 안에 baseline 을 주입하고 compare 실행:

```shell
kubectl cp ./baseline_cu126.json llmops/<b300-pod>:/tmp/ocr_smoke/baseline_cu126.json
kubectl -n llmops exec -it <b300-pod> -- \
  bash /app/etc/smoke_test.sh compare \
    /tmp/ocr_smoke/baseline_cu126.json \
    /tmp/ocr_smoke/result.json
```

비교 통과 기준:
- 인식 텍스트 집합이 정규화(공백 제거 + 소문자) 기준으로 일치
- 각 텍스트 score 차이가 `--score-tol` (기본 0.05) 이내

부동소수 inference 특성상 score 가 비트 단위로 같지는 않으므로 허용 오차를 둔다.

### 5-4. 단계 ④ 외부 OCR API 호출 검증 (선택)

PaddleX `--serve` 모드의 OCR endpoint path 는 PaddleX 버전/파이프라인에 따라
달라질 수 있어 본 가이드에서는 단정적으로 명시하지 않는다.
실 배포 시에는 `paddlex --serve` 시작 로그 또는 `/docs` (FastAPI auto-docs) 를 통해
실제 path 를 확인한 뒤 다음과 같이 호출해 확인한다:

```shell
curl -fsS -X POST "http://127.0.0.1:18080/<확인된_OCR_endpoint>" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$(base64 -w 0 sample.png)" '{file:$img, fileType:1}')"
```

호출 동안 `nvidia-smi -l 1` 로 GPU 메모리/사용률이 움직이는지 확인하면 "API 경로
호출 → GPU inference 수행" 의 end-to-end 가 검증된다.

## 6. 롤백 / 영향도

- 본 작업은 모두 신규 파일 추가이며 기존 cu126 경로 (Dockerfile, pyproject, build script,
  manifest) 는 한 줄도 수정하지 않았다 → 기존 환경 영향 0.
- B300 이미지에 문제가 있을 경우 K8s 에서 b300 Deployment 만 삭제하면 cu126 환경은 그대로 유지된다.

## 7. 공식 docs 호환성 검증 결과

cu126 → cu129 전환에 따른 패키지/프레임워크 충돌 여부를 1차 출처 (벤더 공식 docs / PyPI 메타) 기준으로
검증한 결과를 정리한다. 모든 인용은 url 동봉.

### 7-1. 공식 docs 로 확정된 정합 항목

- **PaddlePaddle 공식 install 매트릭스**: *Blackwell / sm_100 / CUDA 12.9 (Recommend) / 빌드 태그 `cuda12.9-cudnn9.9-mkl-gcc12.2-avx`* — 본 이미지의 (paddle 3.2.0 cu129 wheel + cuDNN 9.9.0.52) 조합이 매트릭스와 일치 ([Tables_en](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables_en.html)).
- **NCCL 2.27.3**: *"This NCCL release supports CUDA 12.2, CUDA 12.4, and CUDA 12.9"* + Blackwell GB200 최적화 명시 ([release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2303/release-notes/rel_2-27-3.html)).
- **cuDNN 9.7+** 부터 compute capability 10.0 (sm_100) 지원 ([cuDNN release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.16.0/release-notes.html)).
- **CUDA 12.9 GA**: 최소 NVIDIA Linux driver `>= 575.51.03` ([CUDA 12.9 release notes Table 3](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
- **numpy 2.x**: paddlepaddle 3.2.0 / paddlex 3.3.6 모두 상한 없음 (각각 `numpy>=1.21`, `numpy>=1.24`). 충돌 가능성 없음.

### 7-2. 공식 docs 가 침묵 (보장도 아니고 위험도 아님 — 실 환경 검증 필수)

- **B300 ↔ compute capability 매핑**: NVIDIA [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) 는 *"With versions 12.8 of the CUDA Toolkit, `nvcc` can generate cubin native to the Blackwell architecture (compute capability 10.0)"* 만 언급하고 B100/B200/B300 같은 SKU 단위 매핑은 한 번도 등장시키지 않는다. CUDA 12.9 release notes 는 별도로 *"Improved performance on Blackwell (`sm_100` and `sm_103`)..."* 만 언급 — `sm_103 = B300` 같은 단정은 공식 문서로 불가. → 실제 B300 노드에서 동작이 확인되기 전까지 "지원 보장" 표현 금지.
- **PaddlePaddle 3.2.0 release notes** 자체에는 Blackwell/cu129/sm_100 직접 언급 없음 ([v3.2.0 release](https://github.com/PaddlePaddle/Paddle/releases/tag/v3.2.0)). install 매트릭스(별도 페이지) 에는 있음.
- **PaddleX 3.3.6 ↔ paddlepaddle 3.2.0**: 공식 호환 매트릭스 부재. PaddleX 3.3.6 PyPI 메타에 paddlepaddle 버전 핀 자체가 없어 *형식상* 모든 paddle 버전 허용.
- **PaddleOCR + paddlepaddle 3.2.0**: PaddleOCR v3.2.0 release notes 는 *"Full support for PaddlePaddle framework versions 3.1.0 and 3.1.1"* 만 명시 ([링크](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.2.0)). 3.2.0 명시 언급 없음. (paddleocr 패키지는 본 이미지에 직접 설치되지 않고 paddlex 가 모델만 사용하지만, 모델/파이프라인 호환성은 동일한 release notes 가 1차 근거.)
- **cu129 wheel 의 sm_100 cubin 포함 여부**: 공식 문서 미언급. 포함 안 됐을 경우 첫 OCR 호출에서 PTX JIT 컴파일이 일어나 latency 증가 가능. 실 노드에서 `cuobjdump --list-elf $(python -c "import paddle, pathlib; print(pathlib.Path(paddle.__file__).parent / 'libs/libpaddle.so')")` 또는 첫 호출 시간으로 확인.
- **PaddleX `--serve` 의 정확한 OCR endpoint path / payload 스키마**: 버전·파이프라인에 따라 달라질 수 있어 본 가이드에서는 단정하지 않음. 실 배포 시 `/docs` 또는 paddlex --serve 시작 로그로 확인.

### 7-3. 잠재 위험 (런타임 검증 필요)

- **protobuf 6.x/7.x**: paddlepaddle 3.2.0 PyPI 메타는 `protobuf>=3.20.2` (상한 없음). uv-b300.lock 은 `protobuf 6.33.6` 또는 `7.34.1` 을 environment marker 별로 핀. paddlepaddle 가 protobuf 6/7 의 새 API와 런타임 호환되는지 공식 문서가 침묵. import 단계에서 즉시 드러나므로 smoke test inference 통과 시 OK 로 간주.
- **starlette 0.49 → 1.0 메이저 점프 / fastapi 0.128 vs 0.136 중복 lock**: paddlex serving extras 가 `starlette>=0.36, fastapi>=0.110` 의 하한만 가져 lock 이 두 marker 분기를 모두 잡았다. 실 빌드 후 컨테이너 안에서 `pip list | grep -E "fastapi|starlette"` 로 어느 버전 하나가 살아남는지 확인 권장.
- **score drift**: cu126 → cu129 사이 부동소수 inference 결과 차이는 `smoke_test_compare.py` 의 기본 허용 오차(0.05) 내에 들어와야 한다. 그 외면 cu126 baseline 으로 회귀하거나 허용 오차를 운영팀 합의 후 조정.

### 7-4. 운영팀 사전 확인 체크리스트

1. 호스트 NVIDIA driver `>= 575.51.03` (CUDA 12.9 GA 최소 요구).
2. B300 datasheet 또는 R570 분기 driver release notes 로 B300 SKU 가 인식되는 driver 최소 버전 확인.
3. 컨테이너 기동 후 `nvidia-smi` 가 B300 을 정상 출력하는지 확인.
4. [smoke test](#smoke-test) 3 단계 모두 통과.
