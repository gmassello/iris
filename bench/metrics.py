"""Metricas del benchmark.

CER/WER miden la calidad del texto crudo (que tan bien lee el motor). Field-level F1 mide la
extraccion estructurada (que tan bien entiende el recibo). Un motor puede ser bueno en una y
malo en la otra, y por eso se reportan las dos.

El F1 va escrito a mano y no con seqeval: seqeval espera secuencias de tags BIO por token, y
nuestra salida es un JSON de campos. Adaptar el dato a la libreria seria trabajar al reves.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

import jiwer

from iris.schema import Receipt

SCALAR_FIELDS = ("merchant", "date", "subtotal", "tax", "total")


@dataclass
class Counts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        predicted = self.true_positive + self.false_positive
        return self.true_positive / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if not precision or not recall:
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclass
class Report:
    fields: dict[str, Counts] = field(default_factory=dict)
    cer_scores: list[float] = field(default_factory=list)
    wer_scores: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    samples: int = 0
    failures: int = 0

    @property
    def cer(self) -> float:
        """Mediana, no media. En unos pocos recibos con fondo ruidoso, Tesseract emite miles de
        caracteres de basura contra un ground truth de doscientos y saca un CER de 16: la media
        de 100 recibos queda secuestrada por tres casos y deja de describir el comportamiento
        tipico. La cola larga no se pierde, se reporta aparte como `cer_worst`."""
        return statistics.median(self.cer_scores) if self.cer_scores else 1.0

    @property
    def wer(self) -> float:
        return statistics.median(self.wer_scores) if self.wer_scores else 1.0

    @property
    def cer_mean(self) -> float:
        return statistics.fmean(self.cer_scores) if self.cer_scores else 1.0

    @property
    def cer_worst(self) -> float:
        return max(self.cer_scores) if self.cer_scores else 1.0

    @property
    def supported_fields(self) -> dict[str, Counts]:
        """Solo los campos que el ground truth realmente anota. CORD no trae `merchant` ni
        `date`: promediarlos daria 0 siempre y hundiria el macro-F1 de todos los motores por
        una limitacion del dataset, no por una falla del motor."""
        return {
            name: counts
            for name, counts in self.fields.items()
            if counts.true_positive + counts.false_negative > 0
        }

    @property
    def macro_f1(self) -> float:
        supported = self.supported_fields
        if not supported:
            return 0.0
        return sum(counts.f1 for counts in supported.values()) / len(supported)

    @property
    def median_latency_ms(self) -> float:
        # statistics.median y no sorted()[n//2]: con n par, el indice del medio devuelve el
        # elemento superior, no el promedio de los dos centrales.
        return statistics.median(self.latency_ms) if self.latency_ms else 0.0


def normalize(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        # 60000 y 60000.00 son el mismo importe. Compararlos como texto diria que no.
        return str(value.normalize())
    text = " ".join(str(value).split()).strip().lower()
    return text or None


def score_scalar(counts: Counts, predicted: object, truth: object) -> None:
    prediction, reference = normalize(predicted), normalize(truth)

    if reference is None:
        # No hay nada que extraer. Si el motor igual invento un valor, es un falso positivo:
        # alucinar un total donde el recibo no lo tiene es un error, no un acierto neutro.
        if prediction is not None:
            counts.false_positive += 1
        return

    if prediction is None:
        counts.false_negative += 1
    elif prediction == reference:
        counts.true_positive += 1
    else:
        counts.false_positive += 1
        counts.false_negative += 1


def score_items(counts: Counts, predicted: Receipt, truth: Receipt) -> None:
    """Los items se comparan como multiconjunto de descripciones: el orden de lectura puede
    variar entre motores sin que ninguno este equivocado."""
    expected = Counter(normalize(item.description) for item in truth.items)
    got = Counter(normalize(item.description) for item in predicted.items)

    matched = sum((expected & got).values())
    counts.true_positive += matched
    counts.false_positive += sum(got.values()) - matched
    counts.false_negative += sum(expected.values()) - matched


def update(
    report: Report, predicted: Receipt | None, truth: Receipt, text: str, reference: str
) -> None:
    report.samples += 1

    for name in (*SCALAR_FIELDS, "items"):
        report.fields.setdefault(name, Counts())

    if predicted is None:
        report.failures += 1
        # Un motor que se cae no puntua neutro: no extrajo nada de lo que habia que extraer.
        for name in SCALAR_FIELDS:
            if normalize(getattr(truth, name)) is not None:
                report.fields[name].false_negative += 1
        report.fields["items"].false_negative += len(truth.items)
        report.cer_scores.append(1.0)
        report.wer_scores.append(1.0)
        return

    for name in SCALAR_FIELDS:
        score_scalar(report.fields[name], getattr(predicted, name), getattr(truth, name))
    score_items(report.fields["items"], predicted, truth)

    if reference.strip():
        report.cer_scores.append(jiwer.cer(reference, text or " "))
        report.wer_scores.append(jiwer.wer(reference, text or " "))
