# vLLM max-num-seqs 권장값 정리 (dots.mocr)

> 결론: **`--max-num-seqs 256`** 권장. 근거는 서버 시작 로그의
> `max_cudagraph_capture_size: 256` (CUDA graph 효율 천장).
> 측정/분석 기준 서버: `rednote-hilab/dots.mocr`, vLLM v0.19.0 (V1 engine), H100 1개,
> `--gpu-memory-utilization 0.9 --max-model-len 20000`.

## 1. 결론

| 항목 | 값 |
|---|---|
| 권장 `--max-num-seqs` | **256** |
| 권장 클라이언트 `page_batch_size` | **256** (서버 값과 반드시 일치) |
| `--max-model-len` | 20000 유지 (큰 페이지 vision 토큰 안전 마진) |

256을 넘기면 메모리는 버텨도 CUDA graph가 적용되지 않아 step당 느려지므로,
**성능을 깎지 않는 최대값이 256**이다.

## 2. 256의 직접 근거 — CUDA graph capture (시작 로그)

```
'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88,
 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216,
 224, 232, 240, 248, 256],
'max_cudagraph_capture_size': 256
```
캡처 수행 확인:
```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|...| 35/35
Capturing CUDA graphs (decode, FULL): 100%|...| 19/19
Graph capturing finished in 2 secs, took 0.23 GiB
```

- CUDA graph = 특정 batch 크기에 대해 GPU 커널 실행을 미리 "녹화"해 launch 오버헤드를 없애는 최적화.
- vLLM이 **1~256만** 캡처 → decode 동시 batch가 256 이하면 graph replay(빠름),
  257 이상이면 eager 실행(커널을 매번 launch → 느림).
- `max-num-seqs` = 동시 실행 sequence 수 상한 → **256으로 두면 항상 graph 적용 범위 내**.

## 3. 천장 3종 비교 — 256은 "메모리"가 아니라 "효율(graph)" 한계

| 천장 종류 | 로그 근거 | 값 |
|---|---|---|
| **효율 (CUDA graph) ← 256의 근거** | `max_cudagraph_capture_size: 256` | **256** |
| 메모리 (worst-case, 요청이 max-model-len 20000 꽉 채움) | `Maximum concurrency for 20,000 tokens per request: 115.86x` | 115 |
| 메모리 (현실 부하) | `GPU KV cache usage: 3.6% @ Running 27` → 요청당 ~3,000 토큰 | ~770 |

관련 로그:
```
Available KV cache memory: 61.88 GiB
GPU KV cache size: 2,317,168 tokens
Maximum concurrency for 20,000 tokens per request: 115.86x
...
Running: 27 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.6%
```
→ KV cache는 현실 부하서 256개를 ~33%(약 768k/2.3M 토큰)로 여유 있게 수용.
   **256 제한은 메모리 때문이 아니라 CUDA graph 때문**이다.

## 4. 256 초과 문서 처리

256페이지를 넘는 문서도 문제 없음. 예) 300p → `⌈300/256⌉ = 2`라운드:
- 1라운드 256p 동시, 2라운드 44p 동시.
각 라운드가 graph 적용 범위 내에서 GPU를 채우므로,
한 번에 다 밀어넣는 것(eager)보다 256씩 나누는 편이 더 빠르다.

## 5. 권장 실행 커맨드

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve rednote-hilab/dots.mocr \
  --host 0.0.0.0 --port 26001 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 20000 \
  --max-num-seqs 256 \
  --chat-template-content-format string \
  --served-model-name dots-mocr \
  --trust-remote-code
```
**필수:** 클라이언트 `page_batch_size`도 256으로 맞출 것. 32로 두면 max-num-seqs=256은 무의미
(서버는 여전히 ~32개만 수신).

## 6. 검증 / 모니터링 포인트

- 부하 중 `GPU KV cache usage`가 한 자릿수~수십 %면 안전.
- `num_requests_waiting > 0` 또는 `num_preemptions_total` 증가가 보이면 = 대형 페이지가
  256개 동시 몰린 worst-case → max-num-seqs를 192 등으로 하향 검토.
- batch sweep를 256까지 확장 측정해 128 대비 추가 이득(=decode compute 포화 지점) 확인 권장.

## 참고: max-num-seqs 미지정 시 기본값
- V0 엔진: 256
- V1 엔진(현재, v0.19.0): ~1024. 단 실효 동시성은 위 천장(115~256)에 묶이므로
  명시적으로 256 지정 권장.

---

## 7. GPU 크기별 일반화 (메모리 한계 계산식)

위 1~6절은 **H100 80G 1개** + `--gpu-memory-utilization 0.9` + `--max-model-len 20000`
기준의 측정·분석이고, 256은 그 환경의 **CUDA graph capture 천장**이다.
GPU 크기가 다른 환경(L4 24G, MIG 40G 슬라이스 등)에서는 메모리 한계가 먼저 천장이 되므로,
동시성을 메모리 기준으로 일반화한 계산식을 함께 둔다 (이슈 [#205](https://github.com/genonai/doc_parser/issues/205) 후속 분석).

### 7.1 계산식

```
max-num-seqs ≤ (GPU 할당 - 고정사용량) × (1G당 token 개수) ÷ (요청당 토큰 평균)
```

- **GPU 할당** = 전체 GPU 메모리 × `--gpu-memory-utilization` (0.9 기준)
- **고정사용량** = 모델 + 기타 고정 메모리 ≈ **10 GiB**
- **1G당 token 개수** ≈ **37,000** (1~3절 로그상 61.8 GiB 에서 2,317,169 token → 약 37,494/GiB)
- **요청당 토큰 평균** ≈ **5,000** (페이지당 평균)

### 7.2 GPU 크기별 권장값

| GPU 메모리 | GPU 할당 (× 0.9) | 계산식 결과 | 권장 `--max-num-seqs` |
|---|---:|---:|---:|
| 24G (L4 등) | 21.6 GiB | (21.6 − 10) × 37,000 ÷ 5,000 ≈ **88** | **64** |
| 40G (MIG 슬라이스) | 36.0 GiB | (36.0 − 10) × 37,000 ÷ 5,000 ≈ **194** | **128** |
| 80G (H100, 표준) | 71.6 GiB | (71.6 − 10) × 37,000 ÷ 5,000 ≈ **458** | **256** |

계산값보다 권장값이 작은 이유는 **CUDA graph capture 천장(=256)** + **현실 부하 안정성 마진**을 함께 고려한 것 — 1~3절 참고.

### 7.3 보수적 옵션 (worst case OOM 회피)

모든 요청이 `max completion token`을 꽉 채우는 worst case가 운영상 우려되면,
표준 권장보다 보수적 안전 마진값을 사용한다.

| GPU 메모리 | 표준 권장 | 보수적 안전 마진 | 비고 |
|---|---|---|---|
| 24G / 40G | 64 / 128 | **32** | 동시성 축소로 OOM 위험 회피 |
| 80G | 256 | **64** | worst case에서는 64 batch 가 128 batch 보다 더 빠를 수 있음 |
| 80G (`--gpu-memory-utilization 0.95`, KV cache 극대화) | — | **128** | 메모리를 KV cache로 더 끌어쓰는 대안 |

> 보수안은 동시성이 줄어 throughput 은 다소 낮아지지만, max completion token 꽉 채우는
> 요청이 다수 들어와도 OOM 위험을 더 확실히 피한다.

### 7.4 매뉴얼/안내 위치

설치·운영 vllm 명령어 예시는 [`README.md` "dots mocr vllm 서빙"](README.md#dots-mocr-vllm-서빙) 섹션에 위 권장값을 반영해 두었다.
