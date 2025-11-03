import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

import fitz  # PyMuPDF
from PIL import Image

def visualize_bboxes_item_superset(data, pdf_path, dpi=72):
    """
    data: JSON 리스트. 각 원소(item)가 하나의 문서 조각이라 가정.
       예) [
         {
           "i_page": ... (사용 여부는 자유),
           "chunk_bboxes": [
               {
                 "page": 1-based PDF 페이지 번호,
                 "bbox": {"l":..., "t":..., "r":..., "b":...},  # 좌표
                 "type": ... (optional)
               },
               ...
           ]
         },
         ...
       ]
    pdf_path: PDF 파일 경로
    dpi: 렌더링 해상도 (기본 72 dpi에서 배율로 환산)
    """

    # 1) 열려는 PDF
    doc = fitz.open(pdf_path)

    # 2) (page_num, item_index)마다 superset bbox를 계산하기 위한 dict
    #    superset_dict[page_num][item_index] = [min_left, min_bottom, max_right, max_top]
    superset_dict = defaultdict(lambda: defaultdict(lambda: [float('inf'), float('inf'), float('-inf'), float('-inf')]))

    for item_index, item in enumerate(data):
        # chunk_bboxes를 순회하며 좌표를 합산
        # chunk_bboxes가 문자열인 경우 JSON으로 파싱
        chunk_bboxes = item.get("chunk_bboxes", [])
        if isinstance(chunk_bboxes, str):
            chunk_bboxes = json.loads(chunk_bboxes)

        for cb in chunk_bboxes:
            page_num = cb["page"]  # 1-based
            bbox_info = cb["bbox"]
            l, b, r, t = bbox_info["l"], bbox_info["b"], bbox_info["r"], bbox_info["t"]

            # superset bbox 갱신
            sb = superset_dict[page_num][item_index]
            sb[0] = min(sb[0], l)
            sb[1] = min(sb[1], b)
            sb[2] = max(sb[2], r)
            sb[3] = max(sb[3], t)

    # 3) item마다 색을 달리 할당할 수 있도록 color_map 준비
    color_list = [
        "red", "blue", "green", "purple", "orange", "magenta", "cyan", "brown", "pink"
    ]
    # item_index가 몇 개인지 모르므로, 필요하다면 동적으로 할당
    # 여기서는 data 길이만큼 반복
    item_color_map = {}
    for i in range(len(data)):
        item_color_map[i] = color_list[i % len(color_list)]

    # 4) 이제 page_num별로 이미지를 렌더링하고, 해당 페이지에 있는 모든 item의 superset bbox를 그립니다.
    #    superset_dict.items()는 (page_num, dict(item_index -> [l, b, r, t])) 형태
    for page_num, item_dict in sorted(superset_dict.items()):
        if page_num < 1 or page_num > len(doc):
            print(f"Warning: page_num={page_num} is out of range.")
            continue

        page = doc[page_num - 1]  # fitz는 0-based 인덱스
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        # 페이지 렌더링
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_width, img_height = img.size

        # Matplotlib 설정
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=(0, img_width, 0, img_height))

        # 표시 범위를 잡기 위해 min/max
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        # item_dict는 {item_index -> [l, b, r, t]} 형태
        for item_index, bbox_vals in item_dict.items():
            l, b, r, t = bbox_vals
            l = l * img_width
            t = t * img_height
            r = r * img_width
            b = b * img_height
            if l == float('inf'):
                # 유효 bbox가 없을 수도 있으므로 체크
                continue

            width = r - l
            height = t - b

            min_x = min(min_x, l)
            max_x = max(max_x, r)
            min_y = min(min_y, b)
            max_y = max(max_y, t)

            # 색상 결정 (item_index별로 고유)
            rect_color = item_color_map[item_index]

            # Rectangle 생성
            rect = Rectangle(
                (l, b), width, height,
                fill=False,
                edgecolor=rect_color,
                linewidth=2
            )
            ax.add_patch(rect)

            # 라벨 표시 (item 인덱스, 혹은 원한다면 다른 정보)
            ax.text(
                l, t,
                f"Item {item_index}",
                fontsize=8, color=rect_color,
                verticalalignment="bottom"
            )

        # bbox 없을 경우 대비
        if min_x == float('inf'):
            min_x, max_x = 0, img_width
            min_y, max_y = 0, img_height

        # 보기 좋게 여백
        pad = 10
        ax.set_xlim([max(0, min_x - pad), min(img_width, max_x + pad)])
        ax.set_ylim([max(0, min_y - pad), min(img_height, max_y + pad)])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Superset BBox - page={page_num}")
        plt.show()

if __name__ == "__main__":
    # JSON 예시 로드
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # PDF 파일 경로
    pdf_path = "./../sample_files/가스산업 인사이트 6호.pdf"

    # item 단위로 Superset bbox 시각화
    visualize_bboxes_item_superset(data, pdf_path)
