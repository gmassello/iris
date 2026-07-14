# iris

Turns a photo of a receipt into structured JSON (merchant, date, line items, subtotal, tax, total).

Behind it are **four interchangeable OCR engines** —Tesseract, docTR, a local VLM running on Metal,
and Claude— exposed through the same API, plus a **reproducible benchmark** that compares them on a
public dataset with ground truth.

```bash
make install
make api          # API on :8000
make web          # frontend on :5173
```

---

## OCR in five minutes

If you've never touched OCR, these four things will save you your entire first day.

**1. An OCR engine does not hand you JSON. It doesn't even hand you lines of text.**
It hands you **individual words**, each with a box (where it sits in the image) and a confidence:

```json
{ "text": "TOTAL", "bbox": {"x": 52, "y": 704, "width": 96, "height": 20}, "confidence": 0.96 }
```

That's it. "That `3340,20` is the total, not the price of a coffee" is not in the box — you have to
infer it.

**2. Which is why you need a parser on top.** iris's parser (`src/iris/parse.py`) does two things:

- **Rebuilds the lines from the boxes.** Two words belong to the same line if their vertical centers
  differ by less than 0.6 × the box height (`group_lines()`). Without this, `TOTAL` and `3340,20` are
  two words with no relationship to each other.
- **Only then looks for meaning**: the line containing `total` (or `importe`, `a pagar`), the one
  containing `iva` (or `tax`, `ppn`), the date, the tax ID. And from each line it takes **the last
  number**, because in `Coffee x2 350,00` the first one is the quantity.

**3. The photo needs prep work — but only for the classical engines.** Tesseract was trained on
scanned documents, not on photos of a crumpled receipt with the shadow of your hand across it.
`src/iris/preprocess.py` compensates: grayscale → denoise → **deskew** (straightens up to 20°; more
than that is almost always a badly estimated angle, and rotating makes it worse) → **upscale** to
1000px tall (Tesseract needs ~30px per character) → **adaptive thresholding**, per-region rather than
global, because lighting across a photo is uneven.

The neural engines (docTR, the VLM, Claude) get **no preprocessing**: they were trained on photos,
and binarizing strips away information they actually use.

A consequence: if an engine rotated and upscaled the image internally, the boxes it finds are in
*that* image's coordinates, not the one you uploaded. Mapping them back to the original space is the
engine's job (`invert_points()`), and it's what makes it possible to draw them on the real photo.

**4. There are two families of engines, and they return different things.**

- **Classical OCR** (Tesseract, docTR): reads pixels → words + boxes. Fast, local, inspectable.
- **VLM** (PaddleOCR-VL, Claude): a multimodal model that *transcribes* the image to text. It emits
  **no boxes** — the `words` field comes back empty and the frontend overlay has nothing to draw.

And watch out for one trap: **PaddleOCR-VL is not instructable.** Ask it for "JSON matching this
schema" and it answers empty; give it a prose prompt and it slips into question-answering mode and
tells you the image contains no charts, without ever reading the receipt. The prompt that works is
literally `OCR:`.

**5. Quality is measured with two numbers that measure different things.**

- **CER / WER** (Character / Word Error Rate): how well it *reads*. Edit distance against the real
  text. Lower is better; 0 is perfect, and it can exceed 1 if the engine invents more characters than
  were there.
- **Field-level F1**: how well it *understands*. Did it get the total right? The line items? Higher
  is better.

An engine can read beautifully and structure badly, or the other way around. That's why the benchmark
reports both.

---

## Getting started

You'll need `uv`, Python 3.12, Node 24, and the `tesseract` binary with its language packs:

```bash
brew install tesseract tesseract-lang       # macOS
make install                                # Python deps + npm
make api                                    # http://localhost:8000
```

Or install nothing locally — the image already ships Tesseract and pre-warmed docTR weights:

```bash
docker compose up --build                   # API on :8000, frontend on :5173
```

One extraction:

```bash
curl -F file=@receipt.png 'localhost:8000/extract?engine=tesseract'
```

```json
{
  "engine": "tesseract",
  "image_width": 700,
  "image_height": 900,
  "elapsed_ms": 363,
  "words": [
    { "text": "SUPERMERCADO", "bbox": { "x": 52, "y": 68, "width": 236, "height": 20 }, "confidence": 0.96 }
  ],
  "receipt": {
    "merchant": "SUPERMERCADO LA ESQUINA",
    "tax_id": "30-71234567-8",
    "date": "14/03/2026",
    "items": [
      { "description": "Cafe molido", "quantity": null, "unit_price": null, "total": 1250.0 },
      { "description": "Leche entera", "quantity": null, "unit_price": null, "total": 890.5 },
      { "description": "Pan lactal", "quantity": null, "unit_price": null, "total": 620.0 }
    ],
    "subtotal": 2760.5,
    "tax": 579.7,
    "total": 3340.2,
    "currency": null
  }
}
```

(Real output from the synthetic receipt in `tests/conftest.py`, hence the Spanish. `words` is
truncated: the actual response carries all 23.)

The frontend (`make web`, a single page) is the comfortable way to see what's happening: pick an
engine, upload a photo, and it shows you the image **with the boxes drawn on top** —hover one and it
tells you what was read and with what confidence— next to the structured receipt and the raw JSON.

---

## The API

Two endpoints (`src/iris/api.py`).

| | |
|---|---|
| `GET /health` | `{"engines": ["claude", "doctr", "mlx-vlm", "tesseract"]}` |
| `POST /extract?engine=<name>` | multipart, field `file`. `engine` defaults to `tesseract`. Returns an `OCRResult`. |

`/health` **always lists all four engines**, usable here or not: if docTR isn't installed or the API
key is missing, that's discovered when the engine is instantiated, not at boot. An unavailable engine
answers 503 — it doesn't vanish from the list.

Errors:

| Code | When |
|---|---|
| `400` | empty file · image OpenCV can't decode · unknown engine |
| `413` | image exceeds 20 MB |
| `502` | `EngineError`: the engine ran and failed (API timeout, VLM returned empty) |
| `503` | `EngineUnavailable`: a dependency, the API key, or the sidecar is missing |

It never returns an empty `Receipt` to paper over a failure: if something broke, the status says so.

---

## The JSON you get back

`src/iris/schema.py` is the single source of truth: API validation, the frontend's types, and the
JSON Schema declared to Claude all come from it.

**`OCRResult`** — what `/extract` responds with:

| Field | Type | Note |
|---|---|---|
| `engine` | `str` | which engine produced it |
| `text` | `str` | raw text, unstructured |
| `words` | `Word[]` | word + `bbox` + `confidence` (0–1). **Empty for `mlx-vlm` and `claude`** |
| `image_width` / `image_height` | `int` | dimensions of **the image you uploaded** |
| `elapsed_ms` | `float` | inference only |
| `receipt` | `Receipt \| null` | the structured result |

**`Receipt`**: `merchant`, `tax_id`, `date`, `items[]` (`description`, `quantity`, `unit_price`,
`total`), `subtotal`, `tax`, `total`, `currency`.

Two invariants worth knowing before you consume it:

- **Every field is nullable and none has a default.** A field the receipt doesn't carry comes back
  `null`, never absent. (Claude's structured outputs require every field to be in `required`; the way
  to say "may be missing" is a nullable type.)
- **Money is `Decimal` in Python and `number` in JSON.** `Decimal` because amounts get summed and
  compared, and with `float` `0.1 + 0.2 != 0.3`. `number` in JSON because a `Decimal` serialized as a
  string would contradict the schema we declare to the model.

Boxes are **always** in the input image's coordinates, even if the engine rotated or scaled
internally. The frontend draws them 1:1 with an `<svg viewBox="0 0 image_width image_height">` and
zero scaling math.

---

## The four engines

| Engine | Runs on | Requires | Boxes? | Median latency | When |
|---|---|---|---|---|---|
| `tesseract` | container, CPU | — | yes | 284 ms | baseline; the fastest, the worst reader |
| `doctr` | container, CPU | `uv sync --extra doctr` | yes | 697 ms | **the best trade**: 4.6× better CER than Tesseract for 2.5× the latency |
| `mlx-vlm` | **host**, Metal | `make mlx-server` | no | 5439 ms | a local OCR VLM, nothing leaves the machine |
| `claude` | remote API | `ANTHROPIC_API_KEY` | no | — | the only one that understands the receipt instead of reading it |

**`mlx-vlm` runs on the host, not in Compose.** Docker on Apple Silicon doesn't expose Metal to the
container: in there it would be pure CPU. It runs separately and the container talks to it over HTTP:

```bash
make mlx-server   # serves PaddleOCR-VL on :8080 over Metal
```

From the container it's reached at `IRIS_MLX_URL=http://host.docker.internal:8080/v1/chat/completions`.

**`claude` is the only engine that sends the image off the machine**, and the only one that skips the
parser: it produces the `Receipt` directly via structured outputs, using the same Pydantic model that
validates everything else.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv sync --extra claude
```

You can benchmark it (`--engines claude`, burns API credit), but the table below covers only the
three local engines.

---

## How a request flows

```
image  →  [preprocess]  →  engine.extract()  →  OCRResult (words + bboxes)
            classical          ↓                        ↓
            engines only    raw text            parse_words()  →  Receipt
```

**All four engines share the same parser.** The VLM included: since it returns newline-separated text
instead of boxes, it enters through `parse_lines()` rather than `parse_words()`, but the extraction is
the same. (The exception is Claude, which produces the JSON itself.)

A single shared parser is what keeps the benchmark honest: it isolates the "reading quality" variable
instead of comparing three different parsers.

What the parser actually knows how to do:

| Field | Heuristic |
|---|---|
| `merchant` | the first of the top 5 lines with ≥3 letters (skips barcodes and decorative dashes) |
| `tax_id` | tax-ID regex (`30-71234567-8`, the Argentine CUIT format) |
| `date` | `14/03/2026`, `2026-03-14` and variants |
| `items` | lines carrying an amount, skipping total/subtotal/tax/date/change lines. `quantity` and `unit_price` are never filled |
| `subtotal` / `tax` / `total` | the last line matching a keyword (`subtotal` · `iva`, `tax`, `impuesto`, `ppn` · `total`, `importe`, `a pagar`). `total` excludes `subtotal` lines — that's the classic bug |
| a line's amount | the **last** number; the first one would be the quantity |
| decimal separator | the last `.` or `,` is decimal only if 1–2 digits follow it: `1.234` is one thousand two hundred thirty-four, `12.34` is twelve and change |

---

## Benchmark

```bash
make bench        # tesseract + docTR, 100 receipts
make bench-all    # + mlx-vlm (needs `make mlx-server` running separately)

uv run python -m bench.run --engines doctr --limit 20 --split test
```

Writes `bench/results.md`, `results.json` and `results.png`. The dataset is **CORD v2**
(`naver-clova-ix/cord-v2`, CC-BY-4.0): 1000 Indonesian receipts with hierarchical annotation — line
items, subtotal, tax, total. Downloaded automatically on first run.

100 receipts, Apple Silicon. Full table in [`bench/results.md`](bench/results.md).

| Engine | CER ↓ | WER ↓ | Worst CER ↓ | Macro F1 ↑ | F1 total | F1 items | Latency |
|---|---|---|---|---|---|---|---|
| tesseract | 0.440 | 0.897 | **16.3** | 0.257 | 0.389 | 0.118 | **284 ms** |
| doctr | **0.096** | **0.250** | **1.2** | **0.574** | **0.807** | **0.397** | 697 ms |
| mlx-vlm | 0.202 | 0.732 | 20.6 | 0.411 | 0.496 | 0.385 | 5439 ms |

**How to read this table** (this is where you learn something, not from the ranking):

- **CER and WER are medians, not means.** On a handful of receipts with textured backgrounds,
  Tesseract reads the noise as if it were text and emits thousands of characters against a ground
  truth of two hundred. A single receipt with CER 16 moves the mean across all 100 from 0.42 to 0.89:
  the mean would describe three disasters, not the engine's behavior.
- **The column that says the most is "Worst CER".** Tesseract without a clean background **has no
  floor**: its worst case is 16.3. docTR stays bounded at 1.2 even in its own. That's robustness, not
  average accuracy, and it shows up in no mean — but it's what decides the engine in production.
- **`mlx-vlm`'s CER isn't comparable to the others'.** CER and WER compare *sequences*, and the VLM
  transcribes in a different reading order: it groups by column (descriptions first, then amounts)
  while the ground truth interleaves them. The content is right, the sequence doesn't line up, and the
  metric punishes it. For this engine look at **F1**, which is order-independent. The number is
  published anyway, with the caveat next to it, rather than hidden.
- **`merchant` and `date` are not scored: CORD doesn't annotate them.** iris extracts them anyway, but
  there's nothing here to measure them against, and scoring them F1=0 would sink every engine's macro
  average over a dataset limitation.
- **The `tax` F1 is somewhat inflated**, and that's worth saying: the parser recognizes `ppn` (the
  Indonesian VAT) and `Rp`/`IDR` alongside the Argentine terms. Meaning it knows the vocabulary of the
  set it's being measured on.

---

## Adding an engine

Three steps, no magic registration.

1. A module in `src/iris/engines/` with a `name` attribute and an `extract(image) -> OCRResult` method
   (the `Protocol` lives in `engines/base.py`).
2. If your engine returns boxes, build the result with `result_from_words()`: it times the run, fills
   in the dimensions and calls the parser for you. If it returns plain text, use `parse_lines()`.
3. One line in `_ENGINES` in `engines/registry.py`:

```python
_ENGINES = {
    ...
    "my-engine": "iris.engines.my_engine:MyEngine",
}
```

Engines are registered **by string** and imported only when instantiated. That's deliberate: docTR
drags in PyTorch, Claude needs an API key, and the VLM needs a sidecar running; importing them at boot
would keep the container from starting if any of the three were missing.

If your engine can't be used (missing dep, missing key), raise `EngineUnavailable` → 503. If it ran
and failed, `EngineError` → 502.

---

## Reference

**Environment variables**

| | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required by the `claude` engine |
| `IRIS_CLAUDE_MODEL` | `claude-opus-4-8` | |
| `IRIS_MLX_URL` | `http://localhost:8080/v1/chat/completions` | Compose points it at `host.docker.internal` |
| `IRIS_MLX_MODEL` | `PaddlePaddle/PaddleOCR-VL` | |
| `VITE_API_URL` | `http://localhost:8000` | where the frontend's `/api` proxies to |

**Commands**

```bash
make install                # Python deps (dev + doctr + bench + claude) + npm
make api / make web         # :8000 / :5173
make check                  # ruff + ty + pytest
make bench / make bench-all
make mlx-server             # the VLM on the host, over Metal
make docker                 # compose up --build

uv run pytest tests/test_parse.py -q     # a single file
cd web && npx tsc --noEmit               # frontend typecheck (not part of `make check`)
```

CI runs three things: `make check`, the frontend typecheck + build, and a smoke test that brings up
Compose and extracts a synthetic receipt end to end.

**Stack**: `uv` · `ruff` · `ty` · FastAPI · Pydantic v2 · OpenCV · React + Vite + Tailwind · Docker
Compose.
