# 샘플 파일 폴더

이 폴더는 테스트용 샘플 파일들을 저장하는 곳입니다.

## 사용법

1. 이 폴더에 `sample.pdf` 파일을 추가하세요
2. `genos_di/test.py`를 실행하면 해당 파일을 처리합니다

## 지원되는 파일 형식

- PDF (.pdf)
- HWP (.hwp) 
- DOCX (.docx)
- TXT (.txt)
- 기타 DocumentProcessor가 지원하는 형식

## 파일명 규칙

- 기본적으로 `sample.pdf`를 찾습니다
- 다른 파일명을 사용하려면 `test.py`의 `file_path` 변수를 수정하세요

## HWP → PDF 회귀 검증 자산 (이슈 #199)

본 폴더에 `*.hwp` / `*.hwpx` 파일을 추가하면 `tests/regression/test_hwp_to_pdf_regression.py` 가 가용한 모든 backend (pdf_sdk / rhwp / libreoffice) 로 자동 변환·검증한다. 별도 등록 절차 없음.

검증 범위 확대를 위해 다음 유형의 HWP 샘플을 추가 권장:

- 표 (병합 셀 / 중첩 표 포함)
- 이미지 / 그림 (PNG / WMF / EMF 혼재)
- 다단 (2단·3단)
- 머리말·꼬리말
- 각주·미주
- 폼·필드

각 자산은 `<유형>_<설명>.hwp` 패턴 권장 (예: `table_merged_cells.hwp`, `image_wmf.hwpx`).
