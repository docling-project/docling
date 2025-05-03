import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat, FormatToExtensions
import json

INPUT_DIR = Path("tests/input")
OUTPUT_DIR = Path("tests/output")
OUTPUT_DIR.mkdir(exist_ok=True)

supported_exts = set()
for fmt in InputFormat:
    exts = FormatToExtensions.get(fmt, [])
    if exts:
        supported_exts.update(exts)
    else:
        supported_exts.add(fmt.value)
supported_exts.add('xlsm')

print(f"Supported extensions: {sorted(supported_exts)}")

input_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix[1:].lower() in supported_exts]
print(f"Found {len(input_files)} files to process: {[f.name for f in input_files]}")

converter = DocumentConverter()

def convert_paths(obj):
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(i) for i in obj]
    elif hasattr(obj, "__module__") and obj.__module__.startswith("pathlib"):
        return str(obj)
    else:
        return obj

for file in input_files:
    try:
        print(f"Processing {file}...")
        result = converter.convert(str(file))
        out_path = OUTPUT_DIR / (file.stem + ".json")
        result_dict = convert_paths(result.model_dump())
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        print(f"Converted {file.name} -> {out_path.name}")
    except Exception as e:
        print(f"Failed to convert {file.name}: {e}")