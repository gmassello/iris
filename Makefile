.PHONY: help install dev api web check bench mlx-server docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Instala todo (incluye docTR y el SDK de Claude)
	uv sync --group dev --extra doctr --extra bench --extra claude
	cd web && npm install

api: ## Levanta la API en :8000
	uv run uvicorn iris.api:app --reload --port 8000

web: ## Levanta el frontend en :5173
	cd web && npm run dev

check: ## Lint, formato, tipos y tests
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check src/ bench/
	uv run pytest -q

bench: ## Corre el benchmark y regenera bench/results.*
	uv run python -m bench.run --engines tesseract,doctr --limit 100

bench-all: ## Benchmark incluyendo el VLM local (necesita `make mlx-server` corriendo)
	uv run python -m bench.run --engines tesseract,doctr,mlx-vlm --limit 100

mlx-server: ## Sirve PaddleOCR-VL en Metal, en el HOST (no en Docker: no expone GPU en Mac)
	uv tool run --from mlx-vlm mlx_vlm.server --model PaddlePaddle/PaddleOCR-VL --port 8080

docker: ## Levanta todo con Compose
	docker compose up --build

clean:
	rm -rf .venv web/node_modules web/dist .pytest_cache .ruff_cache
