# Doc Parser 결과 뷰어 스크립트

문서 파싱 파이프라인의 각 단계별 결과를 브라우저에서 시각화하는 로컬 웹 서버 스크립트 모음입니다.

---

## 파일 구조

```
viewer/
├── hwp_sdk_viewer.py   # HWP SDK 원시 파싱 결과 시각화
├── docling_viewer.py   # Docling 변환 결과 시각화
└── vectors_viewer.py   # 최종 벡터 청킹 결과 시각화
```

---

## 뷰어별 설명

### 1. `hwp_sdk_viewer.py` — HWP SDK 원시 결과 뷰어

**대상 파일**: `hwp_sdk_result/output.json`

HWP/HWPX 파서 SDK(`convtext`)가 출력한 raw JSON을 시각화합니다.
텍스트, 테이블, 이미지(BMP 등)를 페이지별로 렌더링합니다.

**실행**
```bash
python hwp_sdk_viewer.py
# 경로 입력: results/hwpx_sample/hwp_sdk_result/output.json
```

**포트**: 8000 (수정 가능)

---

### 2. `docling_viewer.py` — Docling 변환 결과 뷰어

**대상 파일**: `docling_result/docling.json`

HWP SDK 결과를 Docling 파이프라인이 처리한 중간 결과물(`DoclingDocument`)을 시각화합니다.
texts / tables / pictures를 페이지별로 분류하여 렌더링합니다.

**실행**
```bash
python docling_viewer.py
# 경로 입력: results/hwpx_sample/docling_result/docling.json
```

**포트**: 8000 (수정 가능)

---

### 3. `vectors_viewer.py` — 벡터 청킹 결과 뷰어

**대상 파일**: `vectors_result/vectors.json`

청킹 및 메타데이터 조립이 완료된 최종 `GenOSVectorMeta` 객체 리스트를 시각화합니다.
각 청크의 텍스트, 페이지 범위, 글자/단어/줄 수, 미디어 파일 참조를 카드 형태로 렌더링합니다.
마크다운 테이블은 HTML 테이블로 자동 변환됩니다.

**실행**
```bash
python vectors_viewer.py
# 경로 입력: results/hwpx_sample/vectors_result/vectors.json
```

**포트**: 8000 (수정 가능)

---

## 파이프라인 흐름과 뷰어 대응

```
HWP/HWPX 파일
    │
    ▼
[HWP SDK]         →  output.json       →  hwp_sdk_viewer.py
    │
    ▼
[Docling 변환]    →  docling.json      →  docling_viewer.py
    │
    ▼
[청킹 + 벡터화]   →  vectors.json      →  vectors_viewer.py
```

---

## 공통 조작법

| 동작 | 방법 |
|------|------|
| 페이지 이동 | 버튼 클릭 또는 키보드 `←` `→` |
| 서식 보기 | `서식` 버튼 |
| 원본 JSON 보기 | `JSON` 버튼 |
| 서버 종료 | 터미널에서 `Ctrl+C` |
