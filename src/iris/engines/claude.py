from __future__ import annotations

import os
import time

import numpy as np

from iris.engines.base import EngineError, EngineUnavailable, encode_image
from iris.schema import OCRResult, Receipt

MODEL = os.environ.get("IRIS_CLAUDE_MODEL", "claude-opus-4-8")

# Opus 4.8 lee imagenes de hasta 2576px de lado largo. Mandar mas grande no mejora la lectura
# y multiplica los tokens de imagen (y el costo).
MAX_EDGE = 2576

PROMPT = (
    "Extraé este recibo a JSON. Copiá los importes exactamente como figuran impresos, "
    "sin recalcular ni corregir. Si un campo no está en el recibo, devolvé null: "
    "no lo inventes ni lo infieras."
)


class ClaudeEngine:
    """Modelo multimodal via API. El JSON estructurado no sale de un parser con regex sino de
    structured outputs, usando el mismo schema Pydantic que valida todo lo demas: una sola
    fuente de verdad para la forma del recibo.

    Es el unico motor que manda la imagen fuera de la maquina. Esta en el benchmark como techo
    de referencia contra el cual medir a los locales.
    """

    name = "claude"

    def __init__(self, model: str = MODEL) -> None:
        try:
            import anthropic  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise EngineUnavailable(
                "the anthropic SDK is not installed. Install it with: uv sync --extra claude"
            ) from exc

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise EngineUnavailable("ANTHROPIC_API_KEY is not set")

        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic()

    def extract(self, image: np.ndarray) -> OCRResult:
        started = time.perf_counter()
        height, width = image.shape[:2]

        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(image, MAX_EDGE),
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
                output_format=Receipt,
            )
        except self._anthropic.APIError as exc:
            # Un 429 o un timeout tienen que salir como EngineError, o cortan el benchmark
            # entero a mitad de una corrida de 100 imagenes que ya se pago.
            raise EngineError(f"the Claude API failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise EngineError("Claude refused the image on content policy grounds")

        return OCRResult(
            engine=self.name,
            text=response.parsed_output.model_dump_json() if response.parsed_output else "",
            words=[],  # la API no devuelve bounding boxes
            image_width=width,
            image_height=height,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            receipt=response.parsed_output,
        )
