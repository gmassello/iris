from __future__ import annotations

from decimal import Decimal

import pytest

from iris.parse import group_lines, last_amount, parse_words, to_decimal
from iris.schema import BoundingBox, Word


def word(text: str, x: int, y: int, *, width: int = 80, height: int = 20) -> Word:
    return Word(
        text=text,
        confidence=0.9,
        bbox=BoundingBox(x=x, y=y, width=width, height=height),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1250,00", Decimal("1250.00")),
        ("1,250.50", Decimal("1250.50")),
        ("1.250", Decimal("1250")),  # separador de miles, no decimal
        ("12.34", Decimal("12.34")),  # decimal, no miles
        ("890", Decimal("890")),
        ("basura", None),
    ],
)
def test_to_decimal_distingue_miles_de_decimales(raw: str, expected: Decimal | None) -> None:
    assert to_decimal(raw) == expected


def test_last_amount_toma_el_precio_no_la_cantidad() -> None:
    assert last_amount("Cafe x 2 350,00") == Decimal("350.00")


def test_group_lines_reagrupa_por_posicion_vertical() -> None:
    words = [
        word("TOTAL", x=10, y=100),
        word("3340,20", x=300, y=103),  # misma linea, leve desalineacion
        word("Gracias", x=10, y=200),
    ]
    lines = group_lines(words)

    assert len(lines) == 2
    assert [w.text for w in lines[0]] == ["TOTAL", "3340,20"]
    assert [w.text for w in lines[1]] == ["Gracias"]


def test_parse_receipt_extrae_los_campos_clave() -> None:
    rows = [
        ("SUPERMERCADO LA ESQUINA", 40),
        ("CUIT 30-71234567-8", 80),
        ("Fecha 14/03/2026", 120),
        ("Cafe molido 1250,00", 200),
        ("Leche entera 890,50", 240),
        ("Subtotal 2760,50", 320),
        ("IVA 21% 579,70", 360),
        ("TOTAL 3340,20", 400),
    ]
    words = [
        word(token, x=10 + index * 90, y=y)
        for text, y in rows
        for index, token in enumerate(text.split())
    ]
    receipt = parse_words(words)

    assert receipt.merchant == "SUPERMERCADO LA ESQUINA"
    assert receipt.tax_id == "30-71234567-8"
    assert receipt.date == "14/03/2026"
    assert receipt.total == Decimal("3340.20")
    assert receipt.subtotal == Decimal("2760.50")
    assert receipt.tax == Decimal("579.70")
    assert [item.description for item in receipt.items] == ["Cafe molido", "Leche entera"]


def test_total_no_se_confunde_con_subtotal() -> None:
    """El bug clasico: 'subtotal' contiene 'total', asi que un match ingenuo agarra el
    subtotal y reporta el monto equivocado."""
    words = [
        *[word(token, x=10 + i * 90, y=100) for i, token in enumerate(["Subtotal", "2760,50"])],
        *[word(token, x=10 + i * 90, y=160) for i, token in enumerate(["TOTAL", "3340,20"])],
    ]
    receipt = parse_words(words)

    assert receipt.total == Decimal("3340.20")
    assert receipt.subtotal == Decimal("2760.50")
