# # 1) 헬스체크
# python serving_gateway_preprocess_test.py --mode health

# # 2) 첨부용 전처리
python serving_gateway_preprocess_test.py --mode attachment \
    --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf \
    --out ./result/attachment.json

# 3) 적재용(지능형) 전처리 (+ 추가 파라미터 예시)
# python serving_gateway_preprocess_test.py --mode intelligent \
#     --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf --ocr-mode auto --chunk-size 4096 --out ./result/intelligent.json

# # 4) 변환용 전처리
# python serving_gateway_preprocess_test.py --mode convert \
#     --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf --out ./result/convert.json

# # 5) 세 엔드포인트 순차 호출
# python serving_gateway_preprocess_test.py --mode all \
#     --file-path /app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf

# # 6) 임의 파라미터 전달(반복 가능, 값은 JSON 파싱 시도)
# python serving_gateway_preprocess_test.py --mode intelligent \
#     --file-path /data/report.pdf --param save_images=true --param chunk_overlap=120

# python serving_gateway_preprocess_test.py
