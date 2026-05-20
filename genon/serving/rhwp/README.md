# rhwp-pdf-api 서빙 가이드 (이슈 #199)

doc_parser 의 HWP → PDF 변환 backend 중 `rhwp` 는 OCR / VLM 과 동일하게 **별도 HTTP 서비스**로 운영된다.

- 호출 client (doc_parser 측): [`../../preprocessor/converters/hwp_to_pdf/rhwp.py`](../../preprocessor/converters/hwp_to_pdf/rhwp.py)
- 서버 측 자산 (Dockerfile / k8s 매니페스트): [genonai/genos-rhwp](https://github.com/genonai/genos-rhwp) 레포

회사 클러스터에 아직 떠 있지 않다면 다음 절차로 직접 배포한다.

## 1. genos-rhwp 클론 + 빌드 사전 준비

```bash
git clone --depth 1 https://github.com/genonai/genos-rhwp.git
cd genos-rhwp
```

원본 `Dockerfile.pdf-api` 그대로는 다음 두 가지가 빠져 빌드되지 않는다 — 임시 사본 `Dockerfile.pdf-api.local` 을 만들어 우회한다.

```bash
cat > Dockerfile.pdf-api.local <<'DOCKERFILE'
FROM rust:latest AS builder
WORKDIR /app
COPY Cargo.toml ./
COPY src ./src
COPY ttfs ./ttfs
COPY saved ./saved
RUN cargo build --release --bin rhwp

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates fontconfig fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/rhwp /usr/local/bin/rhwp
COPY --from=builder /app/ttfs /app/ttfs
EXPOSE 7878
CMD ["rhwp", "serve-pdf", "--host", "0.0.0.0", "--port", "7878"]
DOCKERFILE
```

차이는 두 가지:
- `COPY Cargo.toml Cargo.lock ./` → `COPY Cargo.toml ./` (레포에 `Cargo.lock` 이 커밋되어 있지 않음. cargo 가 fresh resolve)
- `COPY saved ./saved` 추가 (`src/document_core/commands/document.rs` 의 `include_bytes!("../../../saved/blank2010.hwp")` 가 빌드시 필요)

## 2. 이미지 빌드 + 회사 registry push

```bash
# 빌드 (Rust cargo 의존성 컴파일로 5~15분 소요, BuildKit cargo cache mount 권장)
docker build --platform linux/amd64 \
  -f Dockerfile.pdf-api.local \
  -t mncregistry:30500/mnc/rhwp-pdf-api:0.1.0 .

# 회사 registry 로 push
docker push mncregistry:30500/mnc/rhwp-pdf-api:0.1.0
```

태그/registry 는 회사 컨벤션에 맞춰 조정한다.

## 3. k8s 매니페스트 적용

genos-rhwp 레포의 `k8s/rhwp-pdf-api.yaml` 을 사용한다. image 만 위 push 한 태그로 교체:

```bash
sed -i 's|image: rhwp-pdf-api:latest|image: mncregistry:30500/mnc/rhwp-pdf-api:0.1.0|' k8s/rhwp-pdf-api.yaml
kubectl apply -f k8s/rhwp-pdf-api.yaml
kubectl rollout status deploy/rhwp-pdf-api
kubectl get svc rhwp-pdf-api
```

기본 노출 — ClusterIP Service `rhwp-pdf-api:7878`.

## 4. doc_parser 측 endpoint 주입

placeholder 가 박혀 있는 좌표 (양쪽 Dockerfile 동일 값):
- [`../../preprocessor/docker/Dockerfile.opensource:265`](../../preprocessor/docker/Dockerfile.opensource#L265) — `ENV RHWP_PDF_API_URL=http://rhwp-pdf-api:7878`
- [`../../preprocessor/docker/Dockerfile.enterprise:287`](../../preprocessor/docker/Dockerfile.enterprise#L287) — 같은 줄

**같은 namespace** 면 추가 설정 없음 — 위 placeholder 가 그대로 동작한다. 컨테이너 안에서 한 번 확인:

```shell
kubectl exec -it deploy/doc-parser-preprocessor -- printenv RHWP_PDF_API_URL
# 출력: http://rhwp-pdf-api:7878
```

**다른 namespace** 면 doc_parser preprocessor 의 deploy 매니페스트에서 FQDN 으로 override. 본 레포에는 preprocessor 의 k8s deploy yaml 이 포함되지 않으므로 (운영팀 GenOS UI / `register_image.sh` 디비 등록 흐름으로 관리), 운영팀에 다음 env 추가를 요청하거나 GenOS 환경변수 설정 화면에서 직접 추가:

```yaml
spec:
  template:
    spec:
      containers:
        - name: doc-parser-preprocessor
          env:
            - name: RHWP_PDF_API_URL
              value: http://rhwp-pdf-api.<rhwp-namespace>.svc.cluster.local:7878
```

런타임에 같은 이름의 env 변수를 새로 주입하면 chain config 가 자동으로 반영한다 ([`../../preprocessor/converters/hwp_to_pdf/availability.py`](../../preprocessor/converters/hwp_to_pdf/availability.py) 의 `rhwp_pdf_api_url()` 참고). 주입 후 `printenv` 로 한 번 더 확인.

## 5. 동작 검증

서버 단독 검증 (doc_parser pod 안에서):

```bash
kubectl exec -it deploy/doc-parser-preprocessor -- bash
curl -sS -X POST \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/app/sample_files/hwp_sample.hwp \
  -o /tmp/out.pdf \
  http://rhwp-pdf-api:7878/api/convert/hwp-to-pdf
file /tmp/out.pdf  # "PDF document, version 1.7, N pages" 가 나와야 정상
```

doc_parser 코드 경로 검증 (chain 이 rhwp 우선 시도하는지):

```bash
kubectl logs deploy/doc-parser-preprocessor -f | grep '\[hwp_to_pdf'
# HWP 첨부 처리 시 다음과 같은 로그가 보여야 함:
#   [hwp_to_pdf] chain start file=... order=['pdf_sdk', 'rhwp', 'libreoffice']  (enterprise)
#   [hwp_to_pdf] try backend=rhwp file=...
#   [hwp_to_pdf:rhwp] POST http://rhwp-pdf-api:7878/api/convert/hwp-to-pdf (N bytes, ...)
#   [hwp_to_pdf:rhwp] success -> ...pdf
```

## 6. 변환 품질 검증용 HWP 추천

이슈 #199 가 명시한 "표 / 이미지 / 다단 / 머리말꼬리말" 회귀를 일찍 잡으려면 다음 유형 1~2건씩을 `genon/preprocessor/sample_files/` 에 추가하면 [`../../preprocessor/tests/regression/test_hwp_to_pdf_regression.py`](../../preprocessor/tests/regression/test_hwp_to_pdf_regression.py) 가 backend 별로 자동 회귀를 돌린다.

- 표 위주 — 단순 표 / 병합 셀 / 중첩 표 각각 1건
- 이미지 위주 — PNG 만, WMF/EMF 만, 그리고 둘 혼합 각각 1건 (HWP 의 WMF/EMF 는 rhwp 가 SVG 로 풀어내는 흐름이라 회귀 빈도가 잦음)
- 다단 (2단·3단) — 본문 + 표가 단을 가로지르는 케이스 1건
- 머리말 / 꼬리말 — 페이지 번호 포함 머리말꼬리말 1건
- 각주 / 미주 — 미주 있는 학술/법규 문서 1건
- 수식 (LaTeX) — `<math>` 영역 포함 (이슈 #195 회귀 — `hwp_sample.hwp` 가 이미 일부 케이스 커버)
- HWPX — `.hwpx` 도 1건 이상 (rhwp 는 .hwp 만 받을 가능성이 있어 chain 이 LibreOffice 로 자동 fallback 되는 흐름을 함께 검증)

파일명 규칙: `<유형>_<설명>.hwp` (예: `table_merged_cells.hwp`, `image_wmf.hwpx`).

## 7. 트러블슈팅

- `is_available()=False` 로 rhwp 가 chain 에서 빠짐 → `RHWP_PDF_API_URL` env 가 비어 있음. deploy yaml 의 env 확인.
- HTTP 200 이지만 응답이 PDF 가 아님 → 서버 측 stderr 로그 확인 (`kubectl logs deploy/rhwp-pdf-api`). 입력 HWP 가 손상되었거나 rhwp 가 미지원 요소를 만난 경우 흔하다. 우리 client 가 자동으로 `None` 반환 후 LibreOffice 로 fallback.
- 타임아웃 → 큰 HWP 의 경우 기본 600s 초과 가능. `HWP_TO_PDF_TIMEOUT_SEC` env 로 조정.
