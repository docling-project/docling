# Regression 테스트

문서 처리 결과(vectors)를 baseline과 비교하여 의도하지 않은 변경을 감지합니다.

## 테스트 실행

### 환경 활성화
```bash
source .venv/bin/activate
```

### Regression 테스트 실행
```bash
source .venv/bin/activate && pytest -m regression
```

### Baseline 업데이트
```bash
source .venv/bin/activate && pytest -m update_baseline
```

### 모든 테스트 실행 (regression, smoke, unit)
```bash
source .venv/bin/activate && pytest
# update_baseline은 자동으로 제외됨 (pytest.ini 설정)
```

## 테스트 구조

### 지원하는 파일 형식
- `test_pdf_regression.py`: PDF 파일 regression 테스트
- `test_docx_regression.py`: DOCX 파일 regression 테스트
- `test_md_regression.py`: Markdown 파일 regression 테스트
- `test_hwp_regression.py`: HWP 파일 regression 테스트
- `test_hwpx_regression.py`: HWPX 파일 regression 테스트
- `test_pptx_regression.py`: PPTX 파일 regression 테스트
- `test_html_regression.py`: HTML 파일 regression 테스트
- `baselines/`: 각 테스트 파일의 baseline 데이터 (JSON 형식)

### 파일 자동 검색
각 테스트는 `sample_files/` 디렉토리에서 해당 확장자의 모든 파일을 자동으로 검색하여 테스트합니다.
- 새 파일 추가 시 코드 수정 없이 자동으로 테스트 대상에 포함됨
- Baseline 파일명: `{확장자}_{파일명}.json` (예: `docx_FinalPaperTemplate.json`)

## 체크 항목

> ⚠️ **형식별 활성화 상태가 다릅니다.**
> - **활성화**: HWP, HWPX, HTML — assert가 모두 활성화되어 있습니다.
> - **비활성화**: PDF, DOCX, MD, PPTX — assert가 주석 처리된 상태입니다.

### HWP / HWPX / HTML (활성화)

1. **Vector 개수** (`num_vectors`)
   - 문서 처리 결과(vectors)의 개수 일관성 확인
   - 현재값 == baseline값

2. **전체 텍스트 글자 수** (`total_characters`)
   - 전체 텍스트 길이 변화 감지
   - 허용 범위: baseline 대비 ±5% 이내

3. **텍스트 유사도** (각 vector별)
   - 각 vector의 텍스트 내용 유사도 확인
   - 최소 유사도: 85% 이상 (difflib.SequenceMatcher 사용)

### PDF / DOCX / MD / PPTX (비활성화)

아래 항목들은 코드에 구현되어 있으나 현재 assert가 주석 처리된 상태입니다:

- Vector 개수 일치 확인
- Label 분포(`label_distribution`) 일치 확인
- 전체 글자 수 ±5% 이내 확인
- 각 vector 텍스트 유사도 ≥85% 확인

## 새로운 파일 형식 추가

### 기존 형식에 파일 추가
1. `sample_files/`에 테스트할 파일 추가
2. `pytest -m update_baseline`로 baseline 생성
3. Baseline 검토 후 git commit

### 새로운 확장자 추가 (예: CSV)
1. `test_csv_regression.py` 생성 (다른 파일 참고하여 작성)
2. `sample_files/`에 CSV 파일 추가
3. `pytest -m update_baseline`로 baseline 생성
4. Baseline 검토 후 git commit

## HTML 테스트 상세

### 테스트 파일
- `test_html_regression.py`: `sample_files/*.html` 파일을 자동 검색하여 테스트

### 샘플 파일 구성 (`sample_files/`)

| 파일명 | 커버하는 HTML 요소 |
|--------|------------------|
| `html_sample.html` | h1~h6 헤딩 계층, 단락, bold/italic/code/sub/sup 포맷팅, 하이퍼링크, 코드블록(`<pre>`), 인용문(`<blockquote>`), `<address>`, `<footer>` |
| `html_tables.html` | 단순 테이블, `colspan`, `rowspan`, colspan+rowspan 혼합, `<caption>`, `<thead>`/`<tfoot>`, 셀 내 포맷팅 |
| `html_lists.html` | `<ul>`, `<ol>`, `start` 속성, 3단계 중첩 리스트, ul+ol 혼합, 단락이 포함된 리스트 항목, `<dl>` 정의 목록 |

### 활성화된 assert

```python
# 1. 벡터 수 정확 일치
assert current_result["num_vectors"] == baseline["num_vectors"]

# 2. 전체 문자 수 ±5% 이내
char_ratio = char_diff / max(baseline["total_characters"], 1)
assert char_ratio < 0.05

# 3. 벡터별 텍스트 유사도 ≥ 85%
similarity = difflib.SequenceMatcher(None, current_text, baseline_text).ratio()
assert similarity > 0.85
```

### HTML 샘플 파일 추가 방법
1. `sample_files/`에 `.html` 파일 추가 (코드 수정 불필요, 자동 감지)
2. `pytest -m update_baseline -k test_update_html_baselines`로 baseline 생성
3. Baseline JSON 검토 후 git commit

### HTML baseline 파일명 규칙
`baselines/html_{파일명}.json` (예: `html_html_sample.json`)

## 주의사항

### Baseline 관리
- ⚠️ **Baseline은 자동 생성되지 않음** - 명시적으로 `pytest -m update_baseline` 실행 필요
- ✅ **pytest 실행 시 update_baseline은 자동 제외됨** (`pytest.ini` 설정)
- 📝 Baseline 파일들은 git에 commit하여 버전 관리
- 🔍 Baseline 변경 시 git diff로 변경사항을 반드시 검토할 것

### 테스트 실패 시
테스트 실패 시 다음 정보가 출력됩니다:
- 어느 항목에서 차이가 발생했는지
- 현재값과 baseline값의 구체적인 차이
- 파일명 (예: `[FinalPaperTemplate.docx]`)

### 개발 워크플로우
1. 코드 수정 후 `source .venv/bin/activate && pytest -m regression` 실행
2. 의도한 변경이면: `source .venv/bin/activate && pytest -m update_baseline`로 baseline 업데이트
3. 의도하지 않은 변경이면: 코드 수정 후 다시 테스트
