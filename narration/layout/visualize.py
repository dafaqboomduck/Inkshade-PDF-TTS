"""
Visualisation helpers for layout classification.

Draws colour-coded overlays showing the final label assigned to each text
block.  Used both by the main pipeline (optional ``debug_layout_dir`` output)
and by the standalone ``debug.debug_classifier`` script.
"""

from PIL import Image, ImageDraw, ImageFont

from narration.layout.models import ClassifiedBlock, LayoutLabel

LABEL_COLORS = {
    LayoutLabel.TITLE: (220, 40, 40),
    LayoutLabel.SECTION_HEADER: (230, 130, 20),
    LayoutLabel.TEXT: (50, 160, 50),
    LayoutLabel.LIST_ITEM: (50, 180, 130),
    LayoutLabel.CAPTION: (100, 100, 220),
    LayoutLabel.FOOTNOTE: (160, 100, 200),
    LayoutLabel.FORMULA: (200, 180, 50),
    LayoutLabel.TABLE: (50, 130, 200),
    LayoutLabel.PICTURE: (180, 180, 180),
    LayoutLabel.PAGE_HEADER: (180, 120, 100),
    LayoutLabel.PAGE_FOOTER: (140, 140, 140),
    LayoutLabel.UNKNOWN: (255, 255, 255),
}

# Labels that the narration will skip
SKIP_LABELS = {
    LayoutLabel.PAGE_HEADER,
    LayoutLabel.PAGE_FOOTER,
    LayoutLabel.PICTURE,
    LayoutLabel.TABLE,
    LayoutLabel.FORMULA,
}


def draw_classified_blocks(
    image: Image.Image,
    classified: list[ClassifiedBlock],
    scale: float,
    line_width: int = 2,
) -> Image.Image:
    """
    Draw colour-coded boxes around each text block using the
    *classified* label.  Skipped blocks get a dashed outline + ×.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    for cb in classified:
        color = LABEL_COLORS.get(cb.label, (255, 255, 255))
        # Convert block bbox (PDF pts) → pixel coords
        x0 = cb.bbox[0] * scale
        y0 = cb.bbox[1] * scale
        x1 = cb.bbox[2] * scale
        y1 = cb.bbox[3] * scale

        is_skip = cb.label in SKIP_LABELS

        # Semi-transparent fill
        alpha = 35 if not is_skip else 20
        draw.rectangle([x0, y0, x1, y1], fill=(*color, alpha))

        # Border
        for i in range(line_width):
            draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)

        # Label tag
        ref = " ®" if cb.refined else ""
        skip = " SKIP" if is_skip else ""
        tag = f"{cb.label.name} {cb.confidence:.0%}{ref}{skip}"
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        draw.rectangle([x0, y0 - th - 4, x0 + tw + 6, y0], fill=(*color, 220))
        draw.text((x0 + 3, y0 - th - 2), tag, fill=(255, 255, 255), font=font)

    return img.convert("RGB")
