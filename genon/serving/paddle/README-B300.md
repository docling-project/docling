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

| 항목 | 값 | 근거 |
| --- | --- | --- |
| PaddlePaddle wheel 인덱스 | `https://www.paddlepaddle.org.cn/packages/stable/cu129/` | B300 (Blackwell) 가 요구하는 `sm_100` 커널은 CUDA 12.8 이상에서 지원. PaddlePaddle 공식 stable wheel 채널 중 가장 가까운 채널이 `cu129`. |
| `paddlepaddle-gpu` | `3.2.0` | cu129 인덱스에 `paddlepaddle_gpu-3.2.0-cp310-cp310-linux_x86_64.whl` 존재 확인. cu126 이미지와 같은 버전을 사용해 결과 차이를 최소화. |
| `paddlex` | `3.3.6` | cu126 이미지와 동일. OCR 파이프라인/모델 호환을 위해 같은 버전 유지. |
| Python | `3.10` | 기존 이미지와 동일. cu129 wheel 의 cp310 빌드가 존재. |
| Base image | `python:3.10-slim-bookworm` | 기존 이미지와 동일. PaddlePaddle GPU wheel 이 자체 CUDA runtime 라이브러리를 포함하므로 NVIDIA CUDA base image 를 별도로 사용하지 않는다. 단 호스트 NVIDIA 드라이버는 CUDA 12.9 호환 버전이어야 한다. |

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

## 7. 알려진 미검증 사항

다음은 사내 B300 GPU 부재로 코드/문서 단계에서 검증할 수 없는 항목이다.
실 B300 노드에서 [smoke test](#smoke-test) 를 통해 확인해야 한다.

- PaddlePaddle 3.2.0 cu129 wheel 이 `sm_100` 커널을 사전 빌드 형태로 포함하는지
  (포함 안 될 경우 첫 호출에서 JIT 컴파일이 발생해 매우 느릴 수 있음).
- 기존 cu126 결과 대비 score drift 가 기본 허용 오차(0.05) 내에 들어오는지.
- PaddleX OCR serving endpoint 의 정확한 path / payload 스키마.
