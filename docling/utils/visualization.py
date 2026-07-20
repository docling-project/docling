from pathlib import Path, PurePath

from docling_core.types.doc import DocItemLabel
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from docling.datamodel.base_models import Cluster, Page
from docling.datamodel.settings import settings


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
                cy0 *= scale_y
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
            y0 *= scale_y
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


def draw_clusters_and_cells_side_by_side(
    input_file: PurePath,
    page: Page,
    clusters: list[Cluster],
    mode_prefix: str,
    show: bool = False,
) -> None:
    """
    Draws a page image side by side with clusters filtered into two categories:
    - Left: Clusters excluding FORM, KEY_VALUE_REGION, and PICTURE.
    - Right: Clusters including FORM, KEY_VALUE_REGION, and PICTURE.
    Includes label names and confidence scores for each cluster.
    """
    assert page.image is not None
    assert page.size is not None
    scale_x = page.image.width / page.size.width
    scale_y = page.image.height / page.size.height

    # Filter clusters for left and right images
    exclude_labels = {
        DocItemLabel.FORM,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.PICTURE,
    }
    left_clusters = [c for c in clusters if c.label not in exclude_labels]
    right_clusters = [c for c in clusters if c.label in exclude_labels]
    # Create a deep copy of the original image for both sides
    left_image = page.image.copy()
    right_image = page.image.copy()

    # Draw clusters on both images
    draw_clusters(left_image, left_clusters, scale_x, scale_y)
    draw_clusters(right_image, right_clusters, scale_x, scale_y)
    # Combine the images side by side
    combined_width = left_image.width * 2
    combined_height = left_image.height
    combined_image = Image.new("RGB", (combined_width, combined_height))
    combined_image.paste(left_image, (0, 0))
    combined_image.paste(right_image, (left_image.width, 0))
    if show:
        combined_image.show()
    else:
        out_path: Path = (
            Path(settings.debug.debug_output_path) / f"debug_{input_file.stem}"
        )
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
        combined_image.save(str(out_file), format="png")
