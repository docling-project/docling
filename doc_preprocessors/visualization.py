import json
from pathlib import Path
from PIL import Image, ImageDraw

# 입력 경로
result_json = Path("/workspaces/md_attach_preprocess/doc_parser/result.json")
image_path = Path("/workspaces/koreabank/자른_1-1. BOK_우리나라 물가수준의 특징 및 시사점_최종/image_000004_139c0e4ffddf013830b635da02e7b6f891f9748599a85095f98e939ac58a5e28.png")

# 출력 경로
out_path = image_path.with_name(image_path.stem + "_annotated.png")

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

# 이미지 열기
im = Image.open(image_path).convert("RGBA")
W, H = im.size
draw = ImageDraw.Draw(im, "RGBA")

# result.json 읽기
data = json.loads(result_json.read_text(encoding="utf-8"))
# 첫 항목의 chunk_bboxes 사용 (필요시 인덱스/조건 변경)
bboxes = data[0].get("chunk_bboxes", "[]")
if isinstance(bboxes, str):
    bboxes = json.loads(bboxes)

# 박스 그리기
for item in bboxes:
    bbox = item.get("bbox", {})
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
    draw.rectangle([x1, y1 - th - 2*pad, x1 + tw + 2*pad, y1], fill=color + (200,))
    draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0, 255))

# 저장
im.save(out_path)
print(f"Saved: {out_path}")