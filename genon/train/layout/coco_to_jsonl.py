import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_to_jsonl(coco_path, docling_path, image_root, output_jsonl):
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    with open(docling_path, 'r', encoding='utf-8') as f:
        docling = json.load(f)

    id2filename = {img["id"]: img["file_name"] for img in coco["images"]}
    filename2anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        fname = id2filename[img_id]
        filename2anns.setdefault(fname, []).append(ann)

    label_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    label2id = {name: idx for idx, name in enumerate(sorted(set(label_map.values())))}

    examples = []
    doc_id = 0
    for item in tqdm(docling):
        if not ("text" in item and "bbox" in item and "source_image" in item):
            continue

        text = item["text"].strip()
        words = text.split()
        if not words:
            continue

        bbox = item["bbox"]
        x0, y0, x1, y1 = bbox
        bboxes = [bbox for _ in words]

        # 추출된 label
        raw_label = item.get("label", "O")
        ner_tag = label2id.get(raw_label, 0)
        ner_tags = [ner_tag] * len(words)

        image_file = os.path.basename(item["source_image"])
        image_path = str(Path(image_root) / image_file)

        examples.append({
            "id": str(doc_id),
            "image_path": image_path,
            "words": words,
            "bboxes": bboxes,
            "ner_tags": ner_tags
        })
        doc_id += 1

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 변환 완료: 총 {len(examples)}개 샘플 → {output_jsonl}")

# 사용 예시
convert_to_jsonl(
    coco_path="path/to/_annotations.coco.json",
    docling_path="path/to/docling_result.json",
    image_root="path/to/image/folder",
    output_jsonl="train.jsonl"
)
