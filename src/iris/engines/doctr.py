from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from iris.engines.base import EngineError, EngineUnavailable, result_from_words
from iris.schema import BoundingBox, OCRResult, Word


class DocTREngine:
    """docTR: deteccion y reconocimiento son dos redes separadas, a diferencia de Tesseract.
    Corre en CPU ARM dentro del contenedor (PyTorch tiene ruedas arm64).

    No preprocesa: sus redes fueron entrenadas sobre fotos, no sobre imagenes binarizadas.
    Sus cajas ya estan en coordenadas de la imagen de entrada.
    """

    name = "doctr"

    def __init__(self) -> None:
        try:
            from doctr.models import ocr_predictor
        except ImportError as exc:
            raise EngineUnavailable(
                "docTR no esta instalado. Instalalo con: uv sync --extra doctr"
            ) from exc

        self._predictor = ocr_predictor(pretrained=True)

    def extract(self, image: np.ndarray) -> OCRResult:
        started = time.perf_counter()

        # docTR espera RGB; OpenCV entrega BGR. Sin esto el reconocimiento degrada en silencio.
        code = cv2.COLOR_GRAY2RGB if image.ndim == 2 else cv2.COLOR_BGR2RGB
        rgb = cv2.cvtColor(image, code)

        try:
            document = self._predictor([rgb])
        except (RuntimeError, ValueError) as exc:
            raise EngineError(f"docTR fallo: {exc}") from exc

        height, width = rgb.shape[:2]
        words = _words_from_document(document.export(), width, height)
        return result_from_words(self.name, words, image, started)


def _words_from_document(exported: dict[str, Any], width: int, height: int) -> list[Word]:
    """docTR devuelve las cajas en coordenadas relativas (0-1). El resto de iris trabaja en
    pixeles, asi que se desnormalizan aca y no en el consumidor."""
    words: list[Word] = []
    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    (x0, y0), (x1, y1) = word["geometry"]
                    words.append(
                        Word(
                            text=word["value"],
                            confidence=float(word["confidence"]),
                            bbox=BoundingBox(
                                x=int(x0 * width),
                                y=int(y0 * height),
                                width=int((x1 - x0) * width),
                                height=int((y1 - y0) * height),
                            ),
                        )
                    )
    return words
