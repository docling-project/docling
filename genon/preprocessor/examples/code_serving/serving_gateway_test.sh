
# python serving_gateway_test.py --mode e2e --file-path "/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf" --out result_serving_gateway_test/
# python serving_gateway_test.py --mode parser_upload --upload-file "../../../../shkim_labs/20260609_convert_test/10.여비규정_20240129_인사경영국_20240129.pdf" --out-doc result_serving_gateway_test/doc.json
python serving_gateway_test.py --mode parser --file-path "/app/src/service/genon/preprocessor/sample_files/pdf_sample.pdf" --out-doc result_serving_gateway_test/doc.json
python serving_gateway_test.py --mode chunker --doc-json result_serving_gateway_test/doc.json --out result_serving_gateway_test/chunks.json

