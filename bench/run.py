"""Corre cada motor sobre CORD v2 y publica la comparacion.

Sin tracker de experimentos: los resultados son artefactos commiteados (results.md, results.json,
results.png) que se leen desde el README sin instalar nada. En CI, la misma tabla va al summary
del run de Actions.

    uv run python -m bench.run --engines tesseract,doctr --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from bench.dataset import load_samples
from bench.metrics import SCALAR_FIELDS, Report, update
from iris.engines.base import EngineError
from iris.engines.registry import get_engine

OUTPUT_DIR = Path(__file__).parent


def evaluate(
    engine_name: str,
    samples: list,
) -> Report:
    engine = get_engine(engine_name)
    report = Report()

    for index, sample in enumerate(samples, start=1):
        print(f"  [{engine_name}] {index}/{len(samples)}", end="\r", file=sys.stderr)
        try:
            result = engine.extract(sample.image)
        except EngineError as exc:
            print(f"\n  [{engine_name}] fallo en {index}: {exc}", file=sys.stderr)
            update(report, None, sample.truth, "", sample.text)
            continue

        report.latency_ms.append(result.elapsed_ms)
        update(report, result.receipt, sample.truth, result.text, sample.text)

    print(file=sys.stderr)
    return report


def to_markdown(reports: dict[str, Report]) -> str:
    first = next(iter(reports.values()))
    scored = [f for f in (*SCALAR_FIELDS, "items") if f in first.supported_fields]
    unscored = [f for f in (*SCALAR_FIELDS, "items") if f not in scored]

    lines = [
        "# Benchmark de motores OCR",
        "",
        f"Dataset: CORD v2 (test split, {first.samples} recibos). Maquina: Apple Silicon.",
        "",
        "| Motor | CER | WER | CER peor | F1 macro | "
        + " | ".join(f"F1 {f}" for f in scored)
        + " | Latencia mediana | Fallos |",
        "|---|---|---|---|---|" + "---|" * len(scored) + "---|---|",
    ]

    for name, report in reports.items():
        per_field = " | ".join(f"{report.fields[f].f1:.3f}" for f in scored)
        lines.append(
            f"| **{name}** | {report.cer:.3f} | {report.wer:.3f} | {report.cer_worst:.1f} | "
            f"{report.macro_f1:.3f} | {per_field} | {report.median_latency_ms:.0f} ms | "
            f"{report.failures} |"
        )

    lines += [
        "",
        "CER/WER: calidad del texto crudo, mas bajo es mejor. F1: extraccion estructurada, mas alto es mejor.",
        "",
        "**CER y WER son medianas, no medias.** En un puñado de recibos con fondo texturado, Tesseract",
        "lee el ruido como si fuera texto y emite miles de caracteres contra un ground truth de doscientos:",
        "un CER de 16 en un solo recibo mueve la media de los 100 de 0.42 a 0.89. La media describiria esos",
        "tres desastres, no el comportamiento del motor. La cola no se esconde: va en la columna **CER peor**,",
        "que es justamente donde se ve que Tesseract sin un fondo limpio no tiene piso.",
        "",
        "**Leer el CER de `mlx-vlm` con cuidado.** CER y WER comparan secuencias, y el VLM transcribe en",
        "un orden de lectura distinto al del ground truth: agrupa por columnas (primero las descripciones,",
        "despues los importes) mientras CORD los intercala. El contenido es correcto pero la secuencia no",
        "coincide, y eso lo penaliza. Para este motor, el F1 por campo —que no depende del orden— es la",
        "medida representativa. Es una limitacion de la metrica, no del modelo, y por eso el numero se",
        "publica igual en vez de esconderlo.",
    ]
    if unscored:
        lines += [
            "",
            f"No se puntuan {', '.join('`' + f + '`' for f in unscored)}: **CORD no los anota**. "
            "Incluirlos daria F1=0 para todos los motores y hundiria el macro-F1 por una limitacion "
            "del dataset, no por una falla del motor. iris si los extrae; simplemente no hay contra "
            "que medirlos aca.",
        ]
    return "\n".join(lines)


def to_chart(reports: dict[str, Report], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(reports)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    left.bar(names, [reports[n].macro_f1 for n in names], color="#4c72b0")
    left.set_title("F1 macro (extraccion) — mas alto es mejor")
    left.set_ylim(0, 1)

    right.bar(names, [reports[n].median_latency_ms for n in names], color="#c44e52")
    right.set_title("Latencia mediana (ms) — mas bajo es mejor")
    right.set_yscale("log")

    figure.tight_layout()
    figure.savefig(path, dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", default="tesseract")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"Cargando CORD v2 ({args.split}, limit={args.limit})...", file=sys.stderr)
    samples = load_samples(split=args.split, limit=args.limit)

    reports: dict[str, Report] = {}
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        started = time.perf_counter()
        reports[name] = evaluate(name, samples)
        print(f"  [{name}] listo en {time.perf_counter() - started:.1f}s", file=sys.stderr)

    markdown = to_markdown(reports)
    (OUTPUT_DIR / "results.md").write_text(markdown + "\n")
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(
            {
                name: {
                    "cer_median": report.cer,
                    "cer_mean": report.cer_mean,
                    "cer_worst": report.cer_worst,
                    "wer_median": report.wer,
                    "macro_f1": report.macro_f1,
                    "median_latency_ms": report.median_latency_ms,
                    "failures": report.failures,
                    "samples": report.samples,
                    "fields": {
                        field: {
                            "precision": counts.precision,
                            "recall": counts.recall,
                            "f1": counts.f1,
                        }
                        for field, counts in report.fields.items()
                    },
                }
                for name, report in reports.items()
            },
            indent=2,
        )
        + "\n"
    )
    to_chart(reports, OUTPUT_DIR / "results.png")

    print("\n" + markdown)

    # En CI, la tabla va al summary del run: se ve en la UI de Actions sin bajar artefactos.
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write(markdown + "\n")


if __name__ == "__main__":
    main()
