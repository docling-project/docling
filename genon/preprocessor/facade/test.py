import os
from fastapi import Request
import logging
import asyncio
import json
import time

# 테스트할 전처리기 임포트
# from attachment_processor import DocumentProcessor # 첨부용
# from basic_processor import DocumentProcessor # 기본형
# from intelligent_processor import DocumentProcessor # 지능형
from intelligent_processor_ocr import DocumentProcessor # 지능형 + OCR
# from intelligent_processor_law import DocumentProcessor # 지능형 + 법률문서특화

# 파일 경로
file_path = "../sample_files/pdf_sample.pdf"

# 파일 존재 여부 확인
if not os.path.exists(file_path):
    print(f"Sample file not found: {file_path}")
    print("Please add a file to the sample_files folder.")
    exit(1)

# DocumentProcessor 인스턴스 생성
doc_processor = DocumentProcessor()

# FastAPI 요청 예제
mock_request = Request(scope={"type": "http"})

# 비동기 메서드 실행
async def process_document():
    # print(file_path)
    kwargs = {}
    kwargs['org_filename'] = os.path.basename(file_path)
    vectors = await doc_processor(mock_request, file_path, **kwargs)
    # WMF 변환 여부는 include_wmf 파라미터 전달: 현재 한글만 지원
    # vectors = await doc_processor(mock_request, file_path, save_images=True, include_wmf=False)
    return vectors

begin = time.time()
# 메인 루프 실행
result = asyncio.run(process_document())

result_list_as_dict = [item.model_dump() for item in result]

# 최종적으로 이 리스트를 JSON으로 저장
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result_list_as_dict, f, ensure_ascii=False, indent=4)

end = time.time()
print(f"Processing time: {end - begin:.2f} seconds")
