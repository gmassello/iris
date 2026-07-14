# Benchmark de motores OCR

Dataset: CORD v2 (test split, 100 recibos). Maquina: Apple Silicon.

| Motor | CER | WER | CER peor | F1 macro | F1 subtotal | F1 tax | F1 total | F1 items | Latencia mediana | Fallos |
|---|---|---|---|---|---|---|---|---|---|---|
| **tesseract** | 0.440 | 0.897 | 16.3 | 0.257 | 0.360 | 0.163 | 0.389 | 0.118 | 284 ms | 0 |
| **doctr** | 0.096 | 0.250 | 1.2 | 0.574 | 0.752 | 0.340 | 0.807 | 0.397 | 697 ms | 0 |
| **mlx-vlm** | 0.202 | 0.732 | 20.6 | 0.411 | 0.518 | 0.245 | 0.496 | 0.385 | 5439 ms | 0 |

CER/WER: calidad del texto crudo, mas bajo es mejor. F1: extraccion estructurada, mas alto es mejor.

**CER y WER son medianas, no medias.** En un puñado de recibos con fondo texturado, Tesseract
lee el ruido como si fuera texto y emite miles de caracteres contra un ground truth de doscientos:
un CER de 16 en un solo recibo mueve la media de los 100 de 0.42 a 0.89. La media describiria esos
tres desastres, no el comportamiento del motor. La cola no se esconde: va en la columna **CER peor**,
que es justamente donde se ve que Tesseract sin un fondo limpio no tiene piso.

**Leer el CER de `mlx-vlm` con cuidado.** CER y WER comparan secuencias, y el VLM transcribe en
un orden de lectura distinto al del ground truth: agrupa por columnas (primero las descripciones,
despues los importes) mientras CORD los intercala. El contenido es correcto pero la secuencia no
coincide, y eso lo penaliza. Para este motor, el F1 por campo —que no depende del orden— es la
medida representativa. Es una limitacion de la metrica, no del modelo, y por eso el numero se
publica igual en vez de esconderlo.

No se puntuan `merchant`, `date`: **CORD no los anota**. Incluirlos daria F1=0 para todos los motores y hundiria el macro-F1 por una limitacion del dataset, no por una falla del motor. iris si los extrae; simplemente no hay contra que medirlos aca.
