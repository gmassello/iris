from __future__ import annotations

import os
import time

import httpx
import numpy as np

from iris.engines.base import EngineError, encode_image
from iris.parse import parse_lines
from iris.schema import OCRResult

# Docker sobre Apple Silicon no expone Metal al contenedor, asi que el VLM no puede vivir
# dentro del Compose: corre en el host con `mlx_vlm.server` y le hablamos por HTTP.
# Desde el contenedor, el host es `host.docker.internal`.
DEFAULT_URL = os.environ.get("IRIS_MLX_URL", "http://localhost:8080/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("IRIS_MLX_MODEL", "PaddlePaddle/PaddleOCR-VL")

# PaddleOCR-VL es un modelo de *transcripcion*, no un LLM instructible: no sabe seguir un JSON
# Schema. Pedirle "devolve JSON con esta forma" hace que responda vacio. Lo que sabe hacer, y
# muy bien, es leer la imagen a texto. La estructura la pone despues nuestro parser, el mismo
# que usan Tesseract y docTR.
#
# El prompt tiene que ser este y no una instruccion en prosa: con "Extract all text from this
# image" el modelo se va a modo VQA y contesta cosas como "la imagen no contiene ningun grafico
# del cual extraer datos", sin leer nada. Con "OCR:" transcribe.
PROMPT = "OCR:"

# El modelo se traba en bucles de repeticion (una misma linea emitida cien veces hasta agotar
# max_tokens). Una penalizacion minima lo corta sin degradar la lectura.
REPETITION_PENALTY = 1.05

# El modelo reescala internamente igual: mandarle la foto a resolucion completa solo infla el
# cuerpo HTTP y el tiempo de encoding.
MAX_EDGE = 1600


class MLXVLMEngine:
    """VLM de OCR corriendo en Metal via mlx_vlm.server (API compatible con OpenAI).

    Transcribe la imagen a texto con saltos de linea; no emite bounding boxes en este modo, asi
    que `words` va vacio y el overlay del frontend no tiene cajas que dibujar para este motor.
    """

    name = "mlx-vlm"

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> None:
        self.url = url
        self.model = model
        self._client = httpx.Client(timeout=180.0)

    def extract(self, image: np.ndarray) -> OCRResult:
        started = time.perf_counter()
        height, width = image.shape[:2]

        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.0,
            "repetition_penalty": REPETITION_PENALTY,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image, MAX_EDGE)}"
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        }

        try:
            response = self._client.post(self.url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EngineError(
                f"no se pudo hablar con mlx_vlm.server en {self.url}: {exc}. "
                f"Levantalo en el host con: mlx_vlm.server --model {self.model} --port 8080"
            ) from exc

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            # Sin esto, una respuesta con otra forma escapa como KeyError y corta el benchmark
            # entero a mitad de una corrida.
            raise EngineError(f"respuesta inesperada de mlx_vlm.server: {exc}") from exc

        if not content:
            raise EngineError("el modelo devolvio una respuesta vacia")

        lines = [line.strip() for line in content.splitlines() if line.strip()]

        return OCRResult(
            engine=self.name,
            text=content,
            words=[],
            image_width=width,
            image_height=height,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            receipt=parse_lines(lines),
        )
