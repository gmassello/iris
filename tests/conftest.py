from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

RECEIPT_LINES = [
    "SUPERMERCADO LA ESQUINA",
    "CUIT 30-71234567-8",
    "Fecha 14/03/2026",
    "",
    "Cafe molido 1250,00",
    "Leche entera 890,50",
    "Pan lactal 620,00",
    "",
    "Subtotal 2760,50",
    "IVA 21% 579,70",
    "TOTAL 3340,20",
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_receipt(*, rotate: float = 0.0) -> np.ndarray:
    """Recibo sintetico. Sin esto, cualquier test del pipeline depende de tener fotos reales
    en el repo, y el CI no podria correrlo."""
    width, height = 700, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(34)

    y = 60
    for line in RECEIPT_LINES:
        if line:
            draw.text((50, y), line, fill="black", font=font)
        y += 60

    if rotate:
        image = image.rotate(rotate, resample=Image.Resampling.BICUBIC, fillcolor="white")

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@pytest.fixture
def receipt_image() -> np.ndarray:
    return render_receipt()


@pytest.fixture
def skewed_receipt_image() -> np.ndarray:
    return render_receipt(rotate=7.0)
