
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

# 예: JSON 데이터를 로드한다고 가정
with open("result.json", "r", encoding="utf-8") as f:
    data = json.load(f)
import fitz  # PyMuPDF
from PIL import Image


def visualize_bboxes(data, pdf_path, dpi=72):
    """
    data: JSON 리스트 (각 원소는 문서 아이템),
          'i_page' (내부 페이지 인덱스),
          'chunk_bboxes' (bbox 목록),
          'page': 실제 PDF 페이지 번호(1-based)
          'bbox': {l, t, r, b} (origin=bottom-left)
    pdf_path: PDF 파일 경로
    dpi: PDF 페이지를 이미지로 렌더링할 때의 해상도 (PyMuPDF)
    """

    # PyMuPDF를 이용해 PDF 열기
    doc = fitz.open(pdf_path)

    # (i_page, page_num) 별로 bounding box 목록을 모읍니다
    page_dict = defaultdict(list)
    for item in data:
        i_page = item["i_page"]  # 문서 아이템 내부 페이지 인덱스
        # chunk_bboxes가 문자열인 경우 JSON으로 파싱
        chunk_bboxes = item.get("chunk_bboxes", [])
        if isinstance(chunk_bboxes, str):
            chunk_bboxes = json.loads(chunk_bboxes)

        for cb in chunk_bboxes:
            page_num = cb["page"]  # 실제 PDF 페이지 번호(1-based)
            bbox_info = cb["bbox"]  # {l, t, r, b}
            typ = cb["type"]
            page_dict[(i_page, page_num)].append((bbox_info, typ))

    # (i_page, page_num) 조합마다 페이지 렌더링 후, bbox 시각화
    for (i_page, page_num), bboxes in page_dict.items():
        # 페이지 범위 체크
        if page_num < 1 or page_num > len(doc):
            print(f"Warning: page_num={page_num} is out of range.")
            continue

        # PyMuPDF의 페이지 index는 0-based이므로 page_num-1
        page = doc[page_num - 1]

        # dpi를 조정하기 위해 zoom factor를 계산 (기본 72 dpi가 1.0 배율)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # 페이지 렌더링하여 Pixmap 획득
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Pixmap → PIL Image 변환
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_width, img_height = img.size

        # Matplotlib으로 시각화
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=(0, img_width, 0, img_height))

        # bbox를 추가로 그려주기 위해 x, y의 최대/최소값 추적
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for (bbox_info, typ) in bboxes:
            l = bbox_info["l"] * img_width
            t = bbox_info["t"] * img_height
            r = bbox_info["r"] * img_width
            b = bbox_info["b"] * img_height

            width = r - l
            height = t - b

            # 축 범위 업데이트
            min_x = min(min_x, l, r)
            max_x = max(max_x, l, r)
            min_y = min(min_y, b, t)
            max_y = max(max_y, b, t)

            # 타입별 색상 매핑
            color_map = {
                "text": "red",
                "picture": "blue",
                "list_item": "green"
            }
            rect_color = color_map.get(typ, "black")

            # Rectangle로 bbox 시각화
            rect = Rectangle(
                (l, b), width, height,
                fill=False,
                edgecolor=rect_color,
                linewidth=2
            )
            ax.add_patch(rect)

            # bbox 라벨 표시
            ax.text(
                l, t, typ,
                fontsize=8, color=rect_color,
                verticalalignment="bottom"
            )

        # 보기 좋게 약간의 여백 설정
        pad = 10
        # min_x, max_x, min_y, max_y가 inf인 경우(=bbox가 없을 때)를 대비
        if min_x == float('inf'):
            min_x, max_x, min_y, max_y = 0, img_width, 0, img_height

        ax.set_xlim([max(0, min_x - pad), min(img_width, max_x + pad)])
        ax.set_ylim([max(0, min_y - pad), min(img_height, max_y + pad)])

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"i_page={i_page}, page_num={page_num}")
        plt.show()

if __name__ == "__main__":
    # 시각화 함수 호출
    path = './../sample_files/test.pdf'
    visualize_bboxes(data, path)
