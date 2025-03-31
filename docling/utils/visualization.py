from docling_core.types.doc import DocItemLabel
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from docling.datamodel.base_models import Cluster


def draw_clusters(
    image: Image.Image, clusters: list[Cluster], scale_x: float, scale_y: float
) -> None:
    """
    Draw clusters on an image
    """
    draw = ImageDraw.Draw(image, "RGBA")
    # Create a smaller font for the labels
    font: ImageFont.ImageFont | FreeTypeFont
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        # Fallback to default font if arial is not available
        font = ImageFont.load_default()
    for c_tl in clusters:
        all_clusters = [c_tl, *c_tl.children]
        for c in all_clusters:
            # Draw cells first (underneath)
            cell_color = (0, 0, 0, 40)  # Transparent black for cells
            for tc in c.cells:
                cx0, cy0, cx1, cy1 = tc.rect.to_bounding_box().as_tuple()
                cx0 *= scale_x
                cx1 *= scale_x
                cy0 *= scale_x
                cy1 *= scale_y

                draw.rectangle(
                    [(cx0, cy0), (cx1, cy1)],
                    outline=None,
                    fill=cell_color,
                )
            # Draw cluster rectangle
            x0, y0, x1, y1 = c.bbox.as_tuple()
            x0 *= scale_x
            x1 *= scale_x
            y0 *= scale_x
            y1 *= scale_y

            if y1 <= y0:
                y1, y0 = y0, y1
            if x1 <= x0:
                x1, x0 = x0, x1

            cluster_fill_color = (*list(DocItemLabel.get_color(c.label)), 70)
            cluster_outline_color = (
                *list(DocItemLabel.get_color(c.label)),
                255,
            )
            draw.rectangle(
                [(x0, y0), (x1, y1)],
                outline=cluster_outline_color,
                fill=cluster_fill_color,
            )
            # Add label name and confidence
            label_text = f"{c.label.name} ({c.confidence:.2f})"
            # Create semi-transparent background for text
            text_bbox = draw.textbbox((x0, y0), label_text, font=font)
            text_bg_padding = 2
            draw.rectangle(
                [
                    (
                        text_bbox[0] - text_bg_padding,
                        text_bbox[1] - text_bg_padding,
                    ),
                    (
                        text_bbox[2] + text_bg_padding,
                        text_bbox[3] + text_bg_padding,
                    ),
                ],
                fill=(255, 255, 255, 180),  # Semi-transparent white
            )
            # Draw text
            draw.text(
                (x0, y0),
                label_text,
                fill=(0, 0, 0, 255),  # Solid black
                font=font,
            )
