
from pathlib import Path
import json
from law_preprocess import DocumentProcessor  # 클래스 임포트
from pydantic import BaseModel
from schemas import ParserResult
# DocumentProcessor 클래스 인스턴스 생성
processor = DocumentProcessor()

# 문서 처리
input_file = Path("/root/workspaces/264627_data_268423.json")
output_dir = Path("/root/workspaces/mynewtest/version3")

# 법률 문서 처리
documents = processor.process_law_document(input_file, output_dir)

print(f"처리된 문서가 {output_dir}에 저장되었습니다.")


# def load_json_as_model(json_file: str, model: type[BaseModel]) -> BaseModel:
#     with open(json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return model.model_validate(data)  # Pydantic v2 (v1은 model.parse_obj(data))


# if __name__ == "__main__":
#     law_instance:ParserResult = load_json_as_model("../metadata_2100000213205_20250407_0651.json", ParserResult)

#     print(law_instance.law.metadata.admrule_id) # 파일명 길이 보고 바꾸기

#     print(f"처리된 문서가 {output_dir}에 저장되었습니다.")
