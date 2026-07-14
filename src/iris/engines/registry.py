from __future__ import annotations

from functools import cache
from importlib import import_module

from iris.engines.base import EngineError, OCREngine

# Los motores se importan perezosamente: docTR arrastra PyTorch, el de Claude necesita API key
# y el sidecar MLX necesita un server corriendo en el host. Importarlos todos al arrancar haria
# que el contenedor no levante si falta cualquiera de las tres cosas.
_ENGINES = {
    "tesseract": "iris.engines.tesseract:TesseractEngine",
    "doctr": "iris.engines.doctr:DocTREngine",
    "mlx-vlm": "iris.engines.mlx_vlm:MLXVLMEngine",
    "claude": "iris.engines.claude:ClaudeEngine",
}


def available_engines() -> list[str]:
    return sorted(_ENGINES)


@cache
def get_engine(name: str) -> OCREngine:
    """Instancia el motor una sola vez: cargan pesos o abren clientes HTTP, y reconstruirlos
    por request costaria segundos."""
    if name not in _ENGINES:
        raise EngineError(
            f"motor desconocido: {name!r}. Disponibles: {', '.join(available_engines())}"
        )

    module, class_name = _ENGINES[name].split(":")
    engine: OCREngine = getattr(import_module(module), class_name)()
    return engine
