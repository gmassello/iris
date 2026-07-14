"""Carga CORD v2 y lo traduce al schema de iris.

CORD (CC-BY-4.0) trae recibos indonesios con estructura jerarquica real: items con nombre,
cantidad y precio, subtotales y total. Es lo que permite medir extraccion estructurada y no
solo calidad de texto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import cv2
import numpy as np

from iris.parse import to_decimal
from iris.schema import LineItem, Receipt

DATASET = "naver-clova-ix/cord-v2"


@dataclass(frozen=True)
class Sample:
    image: np.ndarray
    truth: Receipt
    text: str


def _as_list(value: Any) -> list[dict[str, Any]]:
    """CORD emite un dict cuando hay un solo item y una lista cuando hay varios. Tratar
    siempre el dict como lista de uno evita perder el unico item de la mitad de los recibos."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _money(source: dict[str, Any], key: str) -> Decimal | None:
    value = source.get(key)
    return to_decimal(str(value)) if value else None


def _receipt_from_gt(gt_parse: dict[str, Any]) -> Receipt:
    sub_total = gt_parse.get("sub_total") or {}
    total = gt_parse.get("total") or {}

    items = [
        LineItem(
            description=str(entry.get("nm", "")).strip(),
            quantity=_money(entry, "cnt"),
            unit_price=_money(entry, "unitprice"),
            total=_money(entry, "price"),
        )
        for entry in _as_list(gt_parse.get("menu"))
        if entry.get("nm")
    ]

    return Receipt(
        merchant=None,  # CORD no anota el nombre del comercio
        tax_id=None,
        date=None,
        items=items,
        subtotal=_money(sub_total, "subtotal_price"),
        tax=_money(sub_total, "tax_price"),
        total=_money(total, "total_price"),
        currency=None,
    )


def _text_from_valid_line(ground_truth: dict[str, Any]) -> str:
    """Reconstruye el texto de referencia para CER/WER a partir de las lineas anotadas."""
    words: list[str] = []
    for line in ground_truth.get("valid_line", []):
        for word in line.get("words", []):
            text = str(word.get("text", "")).strip()
            if text:
                words.append(text)
    return " ".join(words)


def load_samples(split: str = "test", limit: int | None = None) -> list[Sample]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    samples: list[Sample] = []
    for row in dataset:
        ground_truth = json.loads(row["ground_truth"])
        rgb = np.array(row["image"].convert("RGB"))
        samples.append(
            Sample(
                image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                truth=_receipt_from_gt(ground_truth["gt_parse"]),
                text=_text_from_valid_line(ground_truth),
            )
        )
    return samples
