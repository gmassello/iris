from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytesseract
from pytesseract import Output

from iris.engines.base import EngineError, result_from_words
from iris.preprocess import invert_points, preprocess
from iris.schema import BoundingBox, OCRResult, Word

# PSM 6 = "un bloque uniforme de texto". Un recibo es una columna unica; los modos que buscan
# columnas multiples (3, el default) parten los items en pedazos.
DEFAULT_CONFIG = "--oem 3 --psm 6"


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "spa+eng", config: str = DEFAULT_CONFIG) -> None:
        self.lang = lang
        self.config = config

    def extract(self, image: np.ndarray) -> OCRResult:
        started = time.perf_counter()
        processed, matrix = preprocess(image)

        try:
            data = pytesseract.image_to_data(
                processed,
                lang=self.lang,
                config=self.config,
                output_type=Output.DICT,
            )
        except pytesseract.TesseractError as exc:
            raise EngineError(f"tesseract failed: {exc}") from exc

        layout_words, words = _words_from_tsv(data, matrix)
        return result_from_words(self.name, words, image, started, layout_words=layout_words)


def _words_from_tsv(
    data: dict[str, list[Any]], matrix: np.ndarray
) -> tuple[list[Word], list[Word]]:
    """`image_to_data` devuelve columnas TSV paralelas: una lista por atributo, alineadas
    por indice. Los valores llegan como str o int segun la columna, de ahi el `Any`.

    Devuelve las palabras dos veces: en el espacio enderezado (para el parser, que agrupa
    renglones por altura) y en el de la imagen de entrada (para el cliente, que dibuja sobre
    la foto original).
    """
    keep = [
        index
        for index, raw_text in enumerate(data["text"])
        # Tesseract emite -1 para las cajas de layout (bloque, parrafo, linea), que no son
        # palabras. Filtrarlas por confianza negativa es el modo canonico de quedarse con tokens.
        if str(raw_text).strip() and float(data["conf"][index]) >= 0
    ]
    if not keep:
        return [], []

    corners = np.array(
        [
            [
                [data["left"][i], data["top"][i]],
                [data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]],
            ]
            for i in keep
        ],
        dtype=np.float64,
    )
    original = invert_points(matrix, corners.reshape(-1, 2)).reshape(-1, 2, 2)

    layout_words: list[Word] = []
    words: list[Word] = []
    for position, index in enumerate(keep):
        text = str(data["text"][index]).strip()
        confidence = float(data["conf"][index]) / 100.0

        layout_words.append(
            Word(
                text=text,
                confidence=confidence,
                bbox=BoundingBox(
                    x=int(data["left"][index]),
                    y=int(data["top"][index]),
                    width=int(data["width"][index]),
                    height=int(data["height"][index]),
                ),
            )
        )

        (x0, y0), (x1, y1) = original[position]
        words.append(
            Word(
                text=text,
                confidence=confidence,
                bbox=BoundingBox(
                    x=int(min(x0, x1)),
                    y=int(min(y0, y1)),
                    width=int(abs(x1 - x0)),
                    height=int(abs(y1 - y0)),
                ),
            )
        )
    return layout_words, words
