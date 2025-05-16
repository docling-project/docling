import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, FormatToExtensions
import json

INPUT_DIR = Path("tests/input")
OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(exist_ok=True)

def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(i) for i in obj]
    elif hasattr(obj, "__module__") and obj.__module__.startswith("pathlib"):
        return str(obj)
    else:
        return obj

def test_backend_msexcel_xlsm():

    supported_ext = 'xlsm'
    
    input_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix[1:].lower() == supported_ext]

    converter = DocumentConverter()
    
    xlsm_files_processed = 0
    for file in input_files:
        try:

            result = converter.convert(str(file))
     
            assert result is not None, f"Conversion failed for {file.name}"
            out_path = OUTPUT_DIR / (file.stem + ".json")
            result_dict = convert_paths(result.model_dump())
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            assert out_path.exists(), f"Output file {out_path} was not created"
            xlsm_files_processed += 1
                
        except Exception as e:
            assert False, f"Failed to convert {file.name}: {e}"

    if input_files:
        assert xlsm_files_processed > 0, "No xlsm files were processed despite being present"
    else:
        assert True, "No xlsm files were found to process"
