import os
from fastapi import Request
import logging
import asyncio
import json
import time

import sys
sys.path.insert(0, "../../../") # 현재 doc_parser의 docling 폴더 참조

# 테스트할 전처리기 임포트
from attachment_processor import DocumentProcessor # 첨부용
# from convert_processor import DocumentProcessor # 변환형
#from intelligent_processor import DocumentProcessor # 지능형

# 파일 경로
#file_path = "../sample_files/pdf_sample.pdf"
#file_path = "../sample_files/docx_sample.docx"
#file_path = "../sample_files/docx_sample/롯데손해보험 데이터경영팀 MLOps 운영자 매뉴얼.docx"
file_path = "../sample_files/hwpx_sample.hwpx"
#file_path = "../sample_files/hwpx_sample/01_[핵심이슈]가계별 금리익스포저를 감안한 금리상승의 소비 영향 점검_24.2.19_공개용.hwp"
#file_path = "../sample_files/hwpx_sample/16_(통화정책국)의결문(안) 및 참고자료(1804)_송부용.hwpx"
#file_path = "../sample_files/hwpx_sample/14_′24년도 DDoS 공격 방어훈련 시행 계획(안)(정보보안실-2409).hwpx"

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
    kwargs['max_tokens'] = 512
    # 🚀 True로 설정하면 jayu_sdk_result / docling_result / vectors_result 저장
    # ※ save_result / save_path 는 attachment_processor 전용 기능입니다.
    #    결과 파일은 save_path 하위에 파일명 기준 디렉토리로 저장됩니다.
    kwargs['save_result'] = True
    kwargs['save_path'] = './results'
    
    vectors = await doc_processor(mock_request, file_path, **kwargs)
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
