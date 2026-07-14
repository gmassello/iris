# iris

Extrae JSON estructurado de fotos de recibos, con **motores de OCR intercambiables** y un
**benchmark reproducible** que los compara sobre un dataset con ground truth.

```bash
make install
make api          # API en :8000
make web          # frontend en :5173
make bench        # regenera bench/results.md
```

---

## La restricción que define la arquitectura

**Docker sobre Apple Silicon no expone Metal al contenedor.** No hay passthrough de GPU para
Metal: solo NVIDIA vía `nvidia-container-toolkit`. La documentación oficial de PaddleOCR-VL para
Apple Silicon lo dice sin vueltas: *"Docker not supported"*.

Consecuencia directa: **todo lo que corra dentro del Compose es CPU ARM puro**. Un VLM local
metido en un contenedor no usaría la GPU de la máquina en la que está corriendo.

En vez de pelearse con eso, iris lo asume en el diseño. `OCREngine` abstrae **tres clases de
backend que viven en lugares distintos**:

| Clase | Dónde corre | Motores |
|---|---|---|
| `in-container` | CPU ARM, dentro del Compose | Tesseract, docTR |
| `sidecar` | host, sobre Metal/MLX, se le habla por HTTP | PaddleOCR-VL |
| `api` | remoto | Claude |

El frontend y el benchmark no saben ni les importa en cuál de las tres vive el motor que están
usando. Esa indirección es el punto del proyecto.

```
┌── Docker Compose ──────────────┐      ┌── Host (Metal) ──────┐
│  web (React)  →  api (FastAPI) │ ───► │  mlx_vlm.server      │
│                  ├─ tesseract  │ HTTP │  PaddleOCR-VL        │
│                  └─ docTR      │      └──────────────────────┘
└────────────────────────────────┘ ───► API de Claude
```

---

## Resultados

100 recibos de CORD v2, en un Mac. Tabla completa en **[`bench/results.md`](bench/results.md)**,
regenerada por `make bench`.

| Motor | CER | CER peor | F1 macro | Latencia |
|---|---|---|---|---|
| tesseract | 0.440 | **16.3** | 0.257 | 284 ms |
| docTR | **0.096** | 1.2 | **0.574** | 697 ms |
| mlx-vlm (PaddleOCR-VL) | 0.202 | 20.6 | 0.411 | 5439 ms |

Lo que salió midiendo de verdad, no citando papers:

- **docTR le saca 4,6× a Tesseract en calidad de texto** y más que duplica el F1 de extracción,
  a costa de 2,5× la latencia. Es el mejor canje del benchmark.
- **PaddleOCR-VL en Apple Silicon corre a ~5,4 s por recibo.** No encontré ese número publicado en
  ningún lado: los benchmarks de VLMs de OCR que circulan son todos sobre GPU NVIDIA. Acá está,
  medido — y explica por qué el VLM no es la opción obvia aunque sea el modelo más moderno.
- **La columna que más dice es "CER peor".** Ante un recibo con fondo texturado, Tesseract lee el
  ruido como si fuera texto: emite 4861 caracteres contra un ground truth de 287. El problema no es
  que lea mal, es que **no tiene piso**. docTR se mantiene acotado (1.2) hasta en su peor caso. Esa
  diferencia —robustez, no precisión promedio— no se ve en ninguna media, y es la que decidiría el
  motor en producción.
- El VLM tiene **dos modos de fallo que los benchmarks publicados no mencionan** (abajo), y uno
  todavía se le escapa: su peor caso de 20.6 es un bucle de repetición que la penalización no llegó
  a cortar.

### Dos cosas que aparecieron rompiendo el VLM

**1. PaddleOCR-VL no es un modelo instructible.** Es de *transcripción*. Si le pedís "devolveme
un JSON con este schema", responde vacío. Peor: con un prompt en prosa como *"Extract all text
from this image"* se va a modo VQA y contesta cosas como *"la imagen no contiene ningún gráfico
del cual extraer datos"* — sin leer el recibo. El prompt que funciona es literalmente `OCR:`.

Eso reordenó el diseño: el VLM **no** produce el JSON. Produce texto, y ese texto pasa por el
**mismo parser** que usan Tesseract y docTR. Lo cual, de paso, hace el benchmark más limpio:
aísla la variable "calidad de lectura" en vez de comparar tres parsers distintos.

**2. Se traba en bucles de repetición.** Emite la misma línea (`11,000`) cien veces hasta agotar
`max_tokens`. Un `repetition_penalty` de 1.05 lo corta sin degradar la lectura: 1032 caracteres de
basura pasan a 72 caracteres correctos.

### Por qué el CER del VLM parece malo y no lo es

El CER de `mlx-vlm` da por encima de 1, lo cual es absurdo a primera vista. La causa es la
métrica, no el modelo: CER y WER comparan **secuencias**, y el VLM transcribe en otro orden de
lectura que el ground truth — agrupa por columnas (primero las descripciones, después los
importes) mientras CORD los intercala. El contenido es correcto; la secuencia no coincide.

Para ese motor, el **F1 por campo** —que no depende del orden— es la medida representativa. El
número se publica igual, con la advertencia al lado, en vez de esconderlo.

---

## Cómo está armado

```
src/iris/
  schema.py       Pydantic v2. Una sola fuente de verdad: de acá sale la validación
                  y también el JSON Schema que se le pasa a Claude como structured output
  engines/        base.py (Protocol) + un módulo por motor + registry con carga perezosa
  preprocess.py   OpenCV: gris, deskew, upscale, denoise, binarización adaptativa
  parse.py        OCRResult -> Receipt, reconstruyendo líneas por posición de las cajas
  api.py          FastAPI. La inferencia va a un threadpool: es CPU-bound y bloqueante
bench/            Dataset, métricas y el runner que escribe results.{md,json,png}
web/              React + Vite + Tailwind. Overlay de cajas en SVG plano
```

**Decisiones que un revisor puede querer discutir:**

- **Sin librería de canvas** para las bounding boxes. Konva, Fabric y Annotorious existen para
  *editar* cajas; las nuestras son de solo lectura. Un `<svg>` con `viewBox` igual a las
  dimensiones de la imagen dibuja las coordenadas del modelo 1:1, sin cuentas de escalado, en
  ~40 líneas y cero dependencias.
- **Sin cola de jobs.** Un usuario, un job de 0,3 a 6 segundos: un threadpool alcanza. Celery o
  arq serían infraestructura para un problema que todavía no existe. Si el benchmark mostrara
  latencias que lo justifiquen, entra `arq` + SSE (FastAPI trae SSE nativo).
- **Field-level F1 escrito a mano** en vez de `seqeval`. `seqeval` espera secuencias de tags BIO
  por token; nuestra salida es un JSON de campos. Adaptar el dato a la librería sería trabajar al
  revés.
- **`Decimal` para el dinero**, serializado como `number` en JSON. Sin el serializer, Pydantic
  emite Decimal como *string* y contradice el schema que le declaramos a Claude — que entonces
  escribiría strings donde pedimos números.

---

## Motores

| Motor | Licencia | Dónde corre | Requiere |
|---|---|---|---|
| `tesseract` | Apache 2.0 | contenedor (CPU) | — |
| `doctr` | Apache 2.0 | contenedor (CPU) | `uv sync --extra doctr` |
| `mlx-vlm` | Apache 2.0 | **host** (Metal) | `make mlx-server` |
| `claude` | — | API remota | `ANTHROPIC_API_KEY` |

**Descartados a propósito:** PaddleOCR clásico (ARM64 no soportado por upstream, con segfaults
reportados), EasyOCR (imágenes Docker solo amd64, y no aporta nada sobre docTR), Surya y Marker
(pesos bajo OpenRAIL-M: no son libres para uso comercial, mala licencia para un repo público),
olmOCR 2 (7B, inviable en CPU ARM).

### Usar el VLM local

No va en el Compose, por lo dicho arriba. Corre en el host:

```bash
make mlx-server   # sirve PaddleOCR-VL en :8080 sobre Metal
```

El contenedor lo alcanza en `host.docker.internal:8080`.

### Usar Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv sync --extra claude
```

Es el único motor que manda la imagen fuera de la máquina. Está en el benchmark como techo de
referencia contra el cual medir a los locales.

---

## Dataset

**CORD v2** (CC-BY-4.0): 1000 recibos indonesios con anotación jerárquica —ítems, subtotales,
impuestos, total—. Se baja solo la primera vez que corrés `make bench`.

CORD **no anota** el nombre del comercio ni la fecha. iris los extrae igual, pero el benchmark no
los puntúa: darles F1=0 hundiría el macro-F1 de todos los motores por una limitación del dataset,
no por una falla del motor.

**Una advertencia honesta sobre el F1 de `tax`:** el parser reconoce `ppn` (el IVA indonesio) y
`Rp`/`IDR` además de los términos argentinos, porque CORD son recibos indonesios. O sea que el
parser conoce el vocabulario del set con el que se lo mide, y ese F1 está algo inflado respecto de
lo que daría sobre recibos que nunca vio. Se deja así —en vez de montar un sistema de perfiles de
locale que el proyecto todavía no necesita— pero se dice, porque un benchmark que se calla esto no
vale nada.

---

## Stack

`uv` · `ruff` · `ty` · FastAPI · Pydantic v2 · React + Vite + Tailwind · Docker Compose

Sobre **ty** (el type checker de Astral, todavía en beta): la duda razonable era su soporte de
Pydantic v2, que Astral reconoce como incompleto. Se probó el día uno contra los modelos de
`schema.py` y pasa limpio, así que se quedó. Si diera problemas, el reemplazo es Pyright.
