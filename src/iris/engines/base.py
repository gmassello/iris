from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from iris.parse import parse_words
from iris.schema import OCRResult

if TYPE_CHECKING:
    from iris.schema import Word


class OCREngine(Protocol):
    """Interfaz unica sobre tres clases de backend muy distintas:

    - `in-container`: corre en CPU dentro de Docker (Tesseract, docTR).
    - `sidecar`: corre en el host con Metal/MLX y se le habla por HTTP (PaddleOCR-VL).
    - `api`: modelo multimodal remoto (Claude).

    Docker sobre Apple Silicon no expone Metal al contenedor, asi que un VLM local no puede
    vivir dentro del Compose. Esta interfaz es lo que permite compararlos igual.

    Contrato: las `bbox` y el `image_width`/`image_height` del `OCRResult` estan siempre en
    coordenadas de la imagen **de entrada**. Si un motor preprocesa (rota, reescala), es su
    responsabilidad devolver las cajas al espacio original: el cliente dibuja sobre la imagen
    que subio, no sobre una intermedia que nunca vio.
    """

    name: str

    def extract(self, image: np.ndarray) -> OCRResult: ...


class EngineError(RuntimeError):
    """El motor no pudo procesar la imagen. Se propaga: no se devuelve un recibo vacio
    haciendo pasar un fallo por un resultado."""


class EngineUnavailable(EngineError):
    """El motor existe pero no se puede usar aca: falta una dependencia opcional, una API key
    o el sidecar no esta levantado. Es un problema de configuracion del servidor, no del
    pedido del cliente, y por eso se distingue de un fallo de inferencia."""


def result_from_words(
    name: str,
    words: list[Word],
    image: np.ndarray,
    started: float,
    layout_words: list[Word] | None = None,
) -> OCRResult:
    """Arma el OCRResult de un motor que devuelve cajas. Centralizarlo evita que cada motor
    reinvente el mismo bloque y garantiza que todos pasen por el mismo parser.

    `words` va en coordenadas de la imagen de entrada: es lo que el cliente dibuja encima de la
    foto que subio. `layout_words`, si el motor enderezo la imagen, son las mismas palabras en
    el espacio ya enderezado, y es lo que se parsea: el parser agrupa renglones comparando la
    altura de las cajas, y en una foto torcida las palabras de una misma linea no comparten
    altura, asi que agruparlas ahi partiria los renglones al medio. Dos consumidores, dos
    sistemas de coordenadas, cada uno con el que necesita.
    """
    height, width = image.shape[:2]
    return OCRResult(
        engine=name,
        text=" ".join(word.text for word in words),
        words=words,
        image_width=width,
        image_height=height,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        receipt=parse_words(layout_words if layout_words is not None else words),
    )


def encode_image(image: np.ndarray, max_edge: int | None = None) -> str:
    """Imagen a JPEG en base64, para los motores que la mandan por la red.

    JPEG y no PNG: un recibo es una foto, y el modelo la va a reescalar y tokenizar igual.
    PNG cuesta ~13x mas CPU y produce ~9x mas bytes para una entrada que el modelo trata
    identico.
    """
    if max_edge is not None:
        longest = max(image.shape[:2])
        if longest > max_edge:
            scale = max_edge / longest
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise EngineError("could not encode the image as JPEG")
    return base64.b64encode(buffer.tobytes()).decode()
