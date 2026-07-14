from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from iris.api import app

client = TestClient(app)


def test_health_lista_los_motores() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert "tesseract" in response.json()["engines"]


def test_motor_desconocido_da_400() -> None:
    response = client.post(
        "/extract",
        params={"engine": "no-existe"},
        files={"file": ("r.png", b"\x89PNG\r\n\x1a\n" + b"0" * 50, "image/png")},
    )

    assert response.status_code == 400


def test_imagen_corrupta_da_400() -> None:
    response = client.post(
        "/extract",
        files={"file": ("r.png", b"esto no es una imagen", "image/png")},
    )

    assert response.status_code == 400


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract no instalado")
def test_extract_devuelve_el_recibo_estructurado(receipt_image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", receipt_image)
    assert ok

    response = client.post(
        "/extract",
        params={"engine": "tesseract"},
        files={"file": ("recibo.png", buffer.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["engine"] == "tesseract"
    assert body["words"]
    assert body["receipt"]["total"] == 3340.20
    # El schema declara `number`; si esto vuelve string, el structured output de Claude
    # tampoco va a cuadrar.
    assert isinstance(body["receipt"]["total"], float)
