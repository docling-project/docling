import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 입력 경로
result_json = Path("/workspace/doc_parser/doc_preprocessors/result.json")
pdf_folder = Path(
    "/workspace/dots_ocr_test/images/연수규정(20250113)_일부개정_72dpi"
)
# image_path = Path(
#     "/workspace/dots_ocr_test/images/연수규정(20250113)_일부개정/0_dpi72.png"
# )


output_path = "./results"

os.makedirs(output_path, exist_ok=True)


# 색상 매핑
COLOR = {
    "text": (0, 255, 0),
    "section_header": (255, 165, 0),
    "page_header": (0, 191, 255),
    "page_footer": (30, 144, 255),
    "list_item": (138, 43, 226),
    "picture": (255, 0, 0),
    "table": (255, 215, 0),
}
DEFAULT_COLOR = (0, 255, 255)

font = ImageFont.load_default(size=30)  # 폴백

# result.json 읽기
data = json.loads(result_json.read_text(encoding="utf-8"))

img_paths = sorted(os.listdir(pdf_folder), key=lambda x: int(x.split("_")[0]))
for img_name in img_paths:
    img_path = os.path.join(pdf_folder, img_name)
    im = Image.open(img_path).convert("RGBA")
    im.save(os.path.join(output_path, img_name))

bboxes = []
for d in data:
    bbox = d.get("chunk_bboxes", "[]")
    bbox = json.loads(bbox)
    bboxes.extend(bbox)


for b_idx, item in enumerate(bboxes):
    bbox = item.get("bbox", {})

    img_path = os.path.join(output_path, img_paths[item["page"] - 1])
    im = Image.open(img_path).convert("RGBA")
    W, H = im.size
    draw = ImageDraw.Draw(im, "RGBA")

    l = float(bbox.get("l", 0.0))
    t = float(bbox.get("t", 0.0))
    r = float(bbox.get("r", 0.0))
    b = float(bbox.get("b", 0.0))
    origin = item.get("coord_origin", "BOTTOMLEFT")
    typ = item.get("type", "text")

    # 정규화 → 픽셀
    x1 = int(l * W)
    x2 = int(r * W)

    if origin.upper() == "BOTTOMLEFT":
        y1 = int((1.0 - t) * H)  # top
        y2 = int((1.0 - b) * H)  # bottom
    else:  # 이미 TOPLEFT라면 그대로
        y1 = int(t * H)
        y2 = int(b * H)

    color = COLOR.get(typ, DEFAULT_COLOR)
    # 외곽선 + 반투명 채움
    draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
    # draw.rectangle([x1, y1, x2, y2], fill=color + (40,))

    # 라벨
    label = f"{typ}"
    tw, th = draw.textlength(label), 12
    pad = 2
    draw.rectangle([x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color + (200,))
    draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0, 255))

    # 몇번째인지~
    text = str(b_idx + 1)  # 1부터 시작
    # text_position = (x_min, y_min - 15)  # 숫자를 BBox 위쪽에 배치
    draw.text([x1, y1 - 15], text, fill="blue", font=font)

    im.save(os.path.join(output_path, img_paths[item["page"] - 1]))
