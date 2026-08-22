# linux/arm64 explicito: en un Mac ARM, una imagen amd64 corre bajo emulacion y el OCR
# se vuelve inusablemente lento.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# tesseract-ocr-spa y -eng son los modelos de idioma; sin ellos el binario instala pero
# no puede leer nada. libgl1/libglib son dependencias de OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Las dependencias en una capa aparte del codigo: cambiar una linea de iris no reinstala PyTorch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-install-project --extra doctr

COPY src/ ./src/
RUN uv sync --locked --extra doctr

ENV PATH="/app/.venv/bin:$PATH"

# Los pesos de docTR (~100 MB) se bajan la primera vez que se instancia el modelo. Sin esta
# capa, esa descarga ocurre en el primer request a /extract?engine=doctr, dentro del contenedor
# ya en produccion y con el cliente esperando.
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(pretrained=True)"

EXPOSE 8000
CMD ["uvicorn", "iris.api:app", "--host", "0.0.0.0", "--port", "8000"]
