import json
import os

# 경로 설정
coco_path = "/workspaces/mytest/fine_tuning/_annotations.coco.json"
image_dir = "/workspaces/mytest/fine_tuning/images"

# COCO JSON 로드
with open(coco_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# 이미지 파일명 리스트 (정렬 보장)
image_files = sorted(os.listdir(image_dir))

# 이미지 수 = 어노테이션의 "images" 수인지 확인
if len(image_files) != len(coco["images"]):
    raise ValueError(f"🚨 이미지 수({len(image_files)})와 COCO 'images' 수({len(coco['images'])})가 다릅니다!")

# 각 이미지 entry에 실제 파일명 부여
for img_entry, fname in zip(coco["images"], image_files):
    img_entry["file_name"] = fname

# 덮어쓰기 저장
with open(coco_path, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2, ensure_ascii=False)

print("✅ COCO 파일 내 file_name을 실제 이미지 파일명으로 모두 업데이트했습니다.")
