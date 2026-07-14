# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos

```bash
make install        # uv sync (dev + doctr + bench + claude) + npm install en web/
make api            # uvicorn iris.api:app --reload --port 8000
make web            # vite dev en :5173
make check          # ruff check + ruff format --check + ty check src/ bench/ + pytest -q
make bench          # bench sobre tesseract,doctr (100 recibos) -> bench/results.{md,json,png}
make bench-all      # idem + mlx-vlm (requiere `make mlx-server` corriendo aparte, en el host)
make mlx-server     # sirve PaddleOCR-VL en :8080 sobre Metal (HOST, nunca en Docker)
```

Un solo test: `uv run pytest tests/test_parse.py::test_nombre -q`.
Front: `cd web && npx tsc --noEmit` (el CI lo corre; `make check` no).
CI corre las tres cosas: check de Python, typecheck+build del front, y un smoke test de `docker compose up`.

## Arquitectura

Pipeline único: imagen → `OCREngine.extract()` → `OCRResult` (texto + `Word`s con bbox) → `parse_words()` → `Receipt`.

**Todos los motores comparten el mismo parser.** Incluso el VLM: PaddleOCR-VL no es instructible, no devuelve JSON — devuelve texto, y ese texto pasa por `parse.py` igual que Tesseract. No agregar un parser por motor: eso rompería la premisa del benchmark (aislar "calidad de lectura", no comparar parsers).

Tres clases de backend detrás de la misma interfaz (`engines/base.py`), y el motivo importa:
- **in-container** (tesseract, doctr): CPU ARM dentro del Compose.
- **sidecar** (mlx-vlm): corre en el **host** sobre Metal, se le habla por HTTP a `host.docker.internal:8080`. Docker en Apple Silicon no expone Metal — meter el VLM en el Compose lo dejaría en CPU pura. No "arreglar" esto moviéndolo al contenedor.
- **api** (claude): remoto; único motor que manda la imagen fuera de la máquina.

### Invariantes que no se rompen

- **Coordenadas**: las `bbox` y el `image_width/height` del `OCRResult` van siempre en el espacio de la imagen **de entrada**. Si un motor preprocesa (deskew, upscale), debe devolver las cajas al espacio original. `result_from_words()` acepta un `layout_words` aparte —las mismas palabras en el espacio enderezado— porque el parser agrupa renglones por altura de caja y una foto torcida partiría las líneas.
- **`Money` = `Decimal`** (`schema.py`), serializado a `number` en JSON vía `PlainSerializer`. Sin eso Pydantic emite string y contradice el JSON Schema que se le declara a Claude como structured output.
- **Todo campo de `Receipt` es nullable y sin default**: structured outputs exige que todos los campos estén en `required`.
- **Registry perezoso** (`engines/registry.py`): los motores se importan por string al instanciarse. docTR arrastra PyTorch, Claude necesita API key, mlx-vlm necesita el sidecar; importarlos al arrancar haría que el contenedor no levante si falta cualquiera.
- **Errores**: `EngineUnavailable` (falta dep/key/sidecar → 503) vs `EngineError` (falló la inferencia → 502). No devolver un `Receipt` vacío para disimular un fallo.
- **Inferencia en threadpool** (`api.py`): Tesseract y PyTorch no ceden el GIL; en el event loop congelan el server. La construcción del motor también va al threadpool.

## Convenciones

- Comentarios y docstrings en español; explican el **porqué** (una restricción, un trade-off), no el qué.
- `ruff` con `ANN` activo: anotar tipos en todo. `from __future__ import annotations` en cada módulo.
- Al tocar un motor o la métrica, `make bench` regenera `bench/results.md`; el README cita esos números — si cambian, actualizarlo.
