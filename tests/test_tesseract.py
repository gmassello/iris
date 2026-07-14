from __future__ import annotations

import shutil
from decimal import Decimal

import cv2
import numpy as np
import pytest

from iris.engines.tesseract import TesseractEngine
from iris.preprocess import estimate_skew, to_gray

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract no instalado")


def test_deskew_endereza_una_foto_torcida(skewed_receipt_image: np.ndarray) -> None:
    from iris.preprocess import preprocess

    gray = to_gray(skewed_receipt_image)
    assert abs(estimate_skew(gray)) > 3.0

    processed, _ = preprocess(skewed_receipt_image)
    assert abs(estimate_skew(processed)) < 1.0


def test_las_cajas_vuelven_al_espacio_de_la_imagen_original(
    skewed_receipt_image: np.ndarray,
) -> None:
    """El preprocesamiento endereza la foto, pero el cliente dibuja sobre la que subio. Si las
    cajas se devolvieran en coordenadas de la imagen rotada, quedarian corridas justo en los
    recibos torcidos, que son los que el deskew existe para atender."""
    image = skewed_receipt_image
    height, width = image.shape[:2]

    result = TesseractEngine(lang="spa").extract(image)

    assert (result.image_width, result.image_height) == (width, height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for word in result.words:
        box = word.bbox
        crop = gray[box.y : box.y + box.height, box.x : box.x + box.width]
        # Si la caja cae sobre el texto, adentro hay tinta oscura. Si esta corrida, cae en el
        # fondo blanco del recibo.
        assert crop.size and crop.mean() < 250, f"la caja de {word.text!r} cayo en el fondo"


def test_un_recibo_torcido_igual_se_parsea(skewed_receipt_image: np.ndarray) -> None:
    """El parser agrupa renglones por altura de caja: sobre una foto torcida, las palabras de
    una misma linea no comparten altura. Por eso se parsea con las cajas enderezadas, no con
    las que se le devuelven al cliente."""
    receipt = TesseractEngine(lang="spa").extract(skewed_receipt_image).receipt

    assert receipt is not None
    assert receipt.total == Decimal("3340.20")


def test_tesseract_extrae_un_recibo_end_to_end(receipt_image: np.ndarray) -> None:
    """El test que importa: de imagen a Receipt, sin mocks. Si el preprocesamiento, el OCR
    o el parser se rompen, esto falla."""
    result = TesseractEngine(lang="spa").extract(receipt_image)

    assert result.words
    assert result.receipt is not None
    receipt = result.receipt

    assert receipt.merchant is not None
    assert "ESQUINA" in receipt.merchant.upper()
    assert receipt.total == Decimal("3340.20")
    assert receipt.date == "14/03/2026"
    assert receipt.tax_id == "30-71234567-8"
