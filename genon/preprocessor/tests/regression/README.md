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
- `test_hwpx_regression.py`: HWPX 파일 regression 테스트
- `test_pptx_regression.py`: PPTX 파일 regression 테스트
- `baselines/`: 각 테스트 파일의 baseline 데이터 (JSON 형식)

### 파일 자동 검색
각 테스트는 `sample_files/` 디렉토리에서 해당 확장자의 모든 파일을 자동으로 검색하여 테스트합니다.
- 새 파일 추가 시 코드 수정 없이 자동으로 테스트 대상에 포함됨
- Baseline 파일명: `{확장자}_{파일명}.json` (예: `docx_FinalPaperTemplate.json`)

## 체크 항목

각 테스트는 다음 항목들을 검증합니다:

1. **Vector 개수** (`num_vectors`)
   - 문서 처리 결과(vectors)의 개수 일관성 확인
   - 현재값 == baseline값

2. **Label 분포** (`label_distribution`)
   - 문서 구조 분석 결과의 일관성
   - `chunk_bboxes` 내부의 각 bbox `type` 필드에서 추출
   - Label 종류 예시:
     - DOCX: `list_item`, `paragraph`, `table`
     - PDF: `picture`, `section_header`, `text`, `page_footer`, `list_item`, `checkbox_unselected`, `table`
     - MD: `title`, `text`, `section_header`, `list_item`, `code`
   - 현재값 == baseline값 (각 label별 개수 완전 일치)

3. **전체 텍스트 글자 수** (`total_characters`)
   - 전체 텍스트 길이 변화 감지
   - 허용 범위: baseline 대비 ±5% 이내

4. **텍스트 유사도** (각 vector별)
   - 각 vector의 텍스트 내용 유사도 확인
   - 최소 유사도: 85% 이상 (difflib.SequenceMatcher 사용)
   - ⚠️ HWPX는 비결정적 처리로 인해 텍스트 유사도 검사 비활성화됨

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
