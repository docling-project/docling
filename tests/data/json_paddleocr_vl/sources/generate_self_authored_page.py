"""Generate the self-authored source image for PaddleOCR-VL adapter tests."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = Path(__file__).with_name("self_authored_page.png")


def main() -> None:
    image = Image.new("RGB", (960, 1280), "white")
    draw = ImageDraw.Draw(image)

    title_font = ImageFont.load_default(size=42)
    heading_font = ImageFont.load_default(size=30)
    body_font = ImageFont.load_default(size=24)
    formula_font = ImageFont.load_default(size=30)

    draw.text(
        (70, 70),
        "PaddleOCR-VL Adapter Fixture",
        fill="black",
        font=title_font,
    )
    draw.line((70, 130, 890, 130), fill="#2f5597", width=4)

    draw.text((70, 180), "Quarterly Metrics", fill="#1f4e79", font=heading_font)
    draw.multiline_text(
        (70, 240),
        "Resume: this self-authored page verifies structured result import.\n"
        "All names and values below are synthetic test data.",
        fill="black",
        font=body_font,
        spacing=12,
    )

    draw.rounded_rectangle(
        (210, 365, 750, 445),
        radius=12,
        outline="#2f5597",
        width=3,
        fill="#eef3f8",
    )
    draw.text(
        (315, 388),
        "Profit = Revenue - Cost",
        fill="black",
        font=formula_font,
    )

    table_left = 70
    table_top = 520
    column_widths = [240, 190, 190, 190]
    row_height = 62
    rows = [
        ["Region", "Revenue", "Cost", "Profit"],
        ["North", "120", "80", "40"],
        ["South", "90", "60", "30"],
    ]

    y = table_top
    for row_index, row in enumerate(rows):
        x = table_left
        for value, width in zip(row, column_widths):
            fill = "#d9e7f5" if row_index == 0 else "white"
            draw.rectangle(
                (x, y, x + width, y + row_height),
                outline="black",
                width=2,
                fill=fill,
            )
            draw.text(
                (x + 14, y + 17),
                value,
                fill="black",
                font=body_font,
            )
            x += width
        y += row_height

    draw.text(
        (70, 735),
        "Table 1. Synthetic values for adapter testing.",
        fill="#444444",
        font=body_font,
    )

    draw.text((70, 850), "Notes", fill="#1f4e79", font=heading_font)
    draw.multiline_text(
        (70, 910),
        "1. Source content is original and contains no private information.\n"
        "2. The provider output must remain unedited after serialization.",
        fill="black",
        font=body_font,
        spacing=14,
    )

    image.save(OUTPUT_PATH, format="PNG", optimize=False)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
