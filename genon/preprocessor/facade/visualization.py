import argparse
import difflib
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

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

DEFAULT_RESULT_JSON = Path("/workspace/doc_parser/doc_preprocessors/result.json")
DEFAULT_PDF_PATH = Path(
    "/workspace/dots_ocr_test/images/연수규정(20250113)_일부개정.pdf"
)
DEFAULT_OUTPUT_PATH = Path("./results")
TEXT_SCALE = 1
BASE_LABEL_FONT_SIZE = 12
BASE_INDEX_FONT_SIZE = 30


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            # Pillow 기본 폰트도 size를 명시해서 실제 크기가 커지도록 한다.
            return ImageFont.load_default(size=size)


def _load_bboxes(data: List[dict]) -> List[dict]:
    bboxes: List[dict] = []
    if isinstance(data, dict):
        items = data.get("data") or data.get("result") or []
    else:
        items = data

    for item in items:
        if not isinstance(item, dict):
            continue
        raw_bbox = item.get("chunk_bboxes", "[]")
        if isinstance(raw_bbox, str):
            try:
                parsed = json.loads(raw_bbox)
            except json.JSONDecodeError:
                parsed = []
        else:
            parsed = raw_bbox
        if isinstance(parsed, list):
            bboxes.extend(parsed)
    return bboxes


def _norm_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lower()


def _pick_pdf_path(result_json: Path, pdf_path: Optional[Path], pdf_folder: Path) -> Path:
    if pdf_path is not None:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return pdf_path

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    if pdf_folder.is_file():
        if pdf_folder.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file: {pdf_folder}")
        return pdf_folder

    candidates = sorted([p for p in pdf_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])
    if not candidates:
        raise FileNotFoundError(f"No PDF files found under: {pdf_folder}")
    if len(candidates) == 1:
        return candidates[0]

    target = _norm_text(result_json.stem)
    scored = sorted(
        candidates,
        key=lambda p: difflib.SequenceMatcher(None, target, _norm_text(p.stem)).ratio(),
        reverse=True,
    )
    return scored[0]


def _render_pdf_pages_to_images(
    pdf_path: Path,
    output_path: Path,
    render_scale: float = 2.0,
) -> Dict[int, Path]:
    page_map: Dict[int, Path] = {}
    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(pdf_doc)
        for page_index in range(total_pages):
            page = pdf_doc[page_index]
            bitmap = page.render(scale=render_scale)
            pil_image = bitmap.to_pil().convert("RGBA")
            out_img_path = output_path / f"page_{page_index + 1:05}.png"
            pil_image.save(out_img_path)
            page_map[page_index] = out_img_path
    finally:
        pdf_doc.close()

    return page_map


def visualize(
    result_json: Path,
    pdf_path: Path,
    output_path: Path,
    render_scale: float = 2.0,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    data = json.loads(result_json.read_text(encoding="utf-8"))
    page_image_map = _render_pdf_pages_to_images(
        pdf_path=pdf_path,
        output_path=output_path,
        render_scale=render_scale,
    )
    if not page_image_map:
        raise ValueError(
            f"Could not render page images from PDF: {pdf_path}"
        )

    bboxes = _load_bboxes(data)
    label_font = _get_font(BASE_LABEL_FONT_SIZE * TEXT_SCALE)
    index_font = _get_font(BASE_INDEX_FONT_SIZE * TEXT_SCALE)

    for b_idx, item in enumerate(bboxes):
        bbox = item.get("bbox", {})
        page_no = int(item["page"])
        img_src = page_image_map.get(page_no - 1) or page_image_map.get(page_no)
        if img_src is None:
            raise IndexError(
                f"No image mapped for page={page_no}. Available page keys={sorted(page_image_map.keys())[:20]}"
            )
        img_name = img_src.name
        img_path = output_path / img_name

        im = Image.open(img_path).convert("RGBA")
        w, h = im.size
        draw = ImageDraw.Draw(im, "RGBA")

        l = float(bbox.get("l", 0.0))
        t = float(bbox.get("t", 0.0))
        r = float(bbox.get("r", 0.0))
        b = float(bbox.get("b", 0.0))
        origin = str(item.get("coord_origin", "BOTTOMLEFT")).upper()
        typ = str(item.get("type", "text"))

        # 정규화 → 픽셀
        x1 = int(l * w)
        x2 = int(r * w)

        if origin == "BOTTOMLEFT":
            y1 = int((1.0 - t) * h)  # top
            y2 = int((1.0 - b) * h)  # bottom
        else:  # 이미 TOPLEFT라면 그대로
            y1 = int(t * h)
            y2 = int(b * h)

        color = COLOR.get(typ, DEFAULT_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

        # 라벨
        label = f"{typ}"
        label_bbox = draw.textbbox((x1, y1), label, font=label_font)
        tw = int(label_bbox[2] - label_bbox[0])
        th = int(label_bbox[3] - label_bbox[1])
        pad = 2 * TEXT_SCALE
        draw.rectangle(
            [x1, y1 - th - 2 * pad, x1 + tw + 2 * pad, y1], fill=color + (200,)
        )
        draw.text((x1 + pad, y1 - th - pad), label, fill=(0, 0, 0, 255), font=label_font)

        # 몇번째인지~
        text = str(b_idx + 1)  # 1부터 시작
        index_bbox = draw.textbbox((0, 0), text, font=index_font)
        index_h = int(index_bbox[3] - index_bbox[1])
        index_pad = 2 * TEXT_SCALE
        index_x = x1 + index_pad
        index_y = y1 - index_h - index_pad
        if index_y < 0:
            index_y = y1 + index_pad
        draw.text((index_x, index_y), text, fill="blue", font=index_font)

        im.save(img_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw chunk bounding boxes by rendering page images from a PDF."
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_JSON,
        help="Path to result json file.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=None,
        help="Path to source PDF file.",
    )
    parser.add_argument(
        "--pdf-folder",
        type=Path,
        default=Path("."),
        help="Fallback: folder containing PDF files when --pdf-path is omitted.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Folder path to save visualized images.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="PDF rendering scale. 1.0 is 72dpi-equivalent.",
    )
    args = parser.parse_args()

    resolved_result_json = args.result_json.expanduser().resolve()
    resolved_pdf_path = (
        args.pdf_path.expanduser().resolve() if args.pdf_path is not None else None
    )
    resolved_pdf_folder = args.pdf_folder.expanduser().resolve()
    resolved_output_path = args.output_path.expanduser().resolve()

    selected_pdf_path = _pick_pdf_path(
        result_json=resolved_result_json,
        pdf_path=resolved_pdf_path,
        pdf_folder=resolved_pdf_folder,
    )

    visualize(
        result_json=resolved_result_json,
        pdf_path=selected_pdf_path,
        output_path=resolved_output_path,
        render_scale=float(args.render_scale),
    )


if __name__ == "__main__":
    main()
