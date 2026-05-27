# Layout Model 학습 가이드

## 개요
`genon/train/layout/fine_tuning.py`는 COCO 포맷 데이터셋으로 문서 레이아웃 객체 탐지 모델을 파인튜닝하는 스크립트입니다.

- 기본 모델: `PekingU/rtdetr_r50vd`
- 입력 데이터: `_annotations.coco.json` + `images/`
- 학습 프레임워크: Hugging Face `Trainer`
- 기본 동작: 단일 GPU(`CUDA_VISIBLE_DEVICES=0`)

## 디렉터리 구조
기본값 기준으로 아래 구조를 사용합니다.

```text
genon/train/layout/
├─ fine_tuning.py
├─ _annotations.coco.json
└─ images/
```

## 사전 요구사항

- Python 3.10+
- CUDA 환경(권장)
- 패키지 설치:

```bash
python3 -m pip install -U torch torchvision transformers datasets pillow
```

## 빠른 시작

프로젝트 루트에서 실행합니다.

### 1) 스모크 테스트(권장)
학습 파이프라인이 정상 동작하는지만 빠르게 확인합니다.

```bash
python3 genon/train/layout/fine_tuning.py \
  --max-steps 1 \
  --batch-size 2 \
  --logging-steps 1 \
  --output-dir genon/train/layout/model_output_smoke
```

### 2) 전체 학습

```bash
python3 genon/train/layout/fine_tuning.py \
  --num-train-epochs 10 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --output-dir genon/train/layout/model_output
```

### 3) 데이터/모델 경로를 명시해서 실행

```bash
python3 genon/train/layout/fine_tuning.py \
  --annotations /path/to/_annotations.coco.json \
  --images-dir /path/to/images \
  --model-name-or-path PekingU/rtdetr_r50vd \
  --output-dir /path/to/model_output
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--annotations` | `genon/train/layout/_annotations.coco.json` | COCO annotation JSON 경로 |
| `--images-dir` | `genon/train/layout/images` | 학습 이미지 폴더 |
| `--model-name-or-path` | `PekingU/rtdetr_r50vd` | Hugging Face 모델 ID 또는 로컬 모델 경로 |
| `--output-dir` | `genon/train/layout/model_output` | 학습 결과 저장 경로 |
| `--learning-rate` | `5e-5` | 학습률 |
| `--batch-size` | `2` | 디바이스당 배치 크기 |
| `--num-train-epochs` | `10` | 학습 epoch 수 |
| `--max-steps` | `-1` | 양수면 해당 step 수만 학습(스모크 테스트용) |
| `--logging-steps` | `10` | 로그 출력 간격(step) |
| `--cuda-visible-devices` | `0` | 사용할 GPU 인덱스. `all`이면 기존 환경변수 유지 |

## 출력 결과

- 체크포인트/모델 파일: `--output-dir`
- 트레이닝 로그: 콘솔 출력 및 Trainer 로그 파일

## 트러블슈팅

- `ModuleNotFoundError: No module named 'datasets'` -> `python3 -m pip install -U datasets`
- `FileNotFoundError: ... _annotations.coco.json` -> `--annotations`, `--images-dir` 경로 재확인
- 멀티 GPU에서 RT-DETR shape mismatch 발생 -> 기본값(단일 GPU)으로 실행하거나 `--cuda-visible-devices 0` 지정

## 참고

- 학습 진행 문서: https://www.notion.so/genonai/layout-detection-finetuning-1dcfea8aef3c80a5b624ec297a0fe2f7?source=copy_link
