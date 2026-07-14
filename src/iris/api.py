from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from iris.engines.base import EngineError, EngineUnavailable
from iris.engines.registry import available_engines, get_engine
from iris.schema import OCRResult

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="iris",
    description="OCR de recibos con motores intercambiables",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, list[str]]:
    return {"engines": available_engines()}


@app.post("/extract", response_model=OCRResult)
async def extract(
    file: UploadFile = File(...),
    engine: str = Query(default="tesseract", description="Motor de OCR a usar"),
) -> OCRResult:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="archivo vacio")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="la imagen supera los 20 MB")

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="no se pudo decodificar la imagen")

    if engine not in available_engines():
        raise HTTPException(status_code=400, detail=f"motor desconocido: {engine!r}")

    # Construir el motor tambien va al threadpool: la primera llamada a docTR importa PyTorch y
    # arma dos redes, y eso son segundos de trabajo CPU-bound. En el event loop congelaria el
    # server entero, healthcheck incluido.
    #
    # La inferencia va por el mismo camino: Tesseract y PyTorch no ceden el GIL.
    try:
        ocr = await run_in_threadpool(get_engine, engine)
        return await run_in_threadpool(ocr.extract, image)
    except EngineUnavailable as exc:
        # El motor existe pero al servidor le falta una dependencia o una credencial. Es un
        # problema de configuracion nuestro, no del pedido del cliente.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
