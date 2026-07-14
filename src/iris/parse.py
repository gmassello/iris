from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from iris.schema import LineItem, Receipt

if TYPE_CHECKING:
    from iris.schema import Word

# Un numero de dinero: 1.234,56 / 1,234.56 / 1234 / 12.50
# Una sola alternativa a proposito: con `A|B`, el motor de regex se queda con la primera que
# matchea aunque sea mas corta, y "3340,20" terminaba partido en "334" + "0,20".
AMOUNT_RE = re.compile(r"-?\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?")
DATE_RE = re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b")
CUIT_RE = re.compile(r"\b(\d{2}[-\s]?\d{8}[-\s]?\d)\b")
CURRENCY_RE = re.compile(r"(ARS|USD|EUR|IDR|Rp|\$|€)")

TOTAL_KEYWORDS = ("total", "importe", "a pagar", "amount due")
SUBTOTAL_KEYWORDS = ("subtotal", "sub total", "sub-total")
TAX_KEYWORDS = ("iva", "tax", "impuesto", "ppn")

# Dos palabras pertenecen a la misma linea si sus centros verticales estan mas cerca que
# esta fraccion de la altura de la palabra. Es lo que reconstruye la estructura del recibo
# a partir de cajas sueltas.
LINE_TOLERANCE = 0.6


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Reagrupa palabras en lineas visuales por su posicion vertical. Esto es lo que separa
    un parser layout-aware de uno que hace regex sobre texto plano: el OCR devuelve cajas,
    no renglones, y el orden de lectura hay que reconstruirlo."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.bbox.center_y, w.bbox.x))
    lines: list[list[Word]] = [[ordered[0]]]

    for word in ordered[1:]:
        current = lines[-1]
        reference = current[0]
        tolerance = max(reference.bbox.height, word.bbox.height) * LINE_TOLERANCE
        if abs(word.bbox.center_y - reference.bbox.center_y) <= tolerance:
            current.append(word)
        else:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda w: w.bbox.x)
    return lines


def line_text(line: list[Word]) -> str:
    return " ".join(word.text for word in line)


def to_decimal(raw: str) -> Decimal | None:
    """Normaliza un numero de recibo. El caso dificil es distinguir el separador decimal del
    de miles: '1.234' son mil doscientos treinta y cuatro, pero '12.34' son doce con treinta y
    cuatro. La heuristica: el ultimo separador es decimal solo si le siguen 1 o 2 digitos."""
    text = raw.strip()
    if not text:
        return None

    last_dot = text.rfind(".")
    last_comma = text.rfind(",")
    pivot = max(last_dot, last_comma)

    if pivot == -1:
        normalized = text
    else:
        decimals = len(text) - pivot - 1
        if decimals in (1, 2):
            integer_part = re.sub(r"[.,]", "", text[:pivot])
            normalized = f"{integer_part}.{text[pivot + 1 :]}"
        else:
            normalized = re.sub(r"[.,]", "", text)

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def last_amount(text: str) -> Decimal | None:
    """El monto de una linea de recibo es el ultimo numero: 'Cafe x2 350,00' -> 350,00.
    Tomar el primero devolveria la cantidad."""
    matches = AMOUNT_RE.findall(text)
    for raw in reversed(matches):
        amount = to_decimal(raw)
        if amount is not None:
            return amount
    return None


def _amount_for(
    lines: list[str], keywords: tuple[str, ...], *, exclude: tuple[str, ...] = ()
) -> Decimal | None:
    for text in reversed(lines):
        lowered = text.lower()
        if any(word in lowered for word in exclude):
            continue
        if any(word in lowered for word in keywords):
            amount = last_amount(text)
            if amount is not None:
                return amount
    return None


def _merchant(lines: list[str]) -> str | None:
    """El comercio es lo primero impreso arriba de todo. Se saltean lineas sin letras
    (codigos de barra, guiones decorativos)."""
    for text in lines[:5]:
        letters = sum(character.isalpha() for character in text)
        if letters >= 3:
            return text.strip()
    return None


def _line_items(lines: list[str]) -> list[LineItem]:
    items: list[LineItem] = []
    skip = (
        TOTAL_KEYWORDS
        + SUBTOTAL_KEYWORDS
        + TAX_KEYWORDS
        + ("cuit", "cambio", "efectivo", "vuelto", "fecha")
    )

    for text in lines:
        lowered = text.lower()
        if any(word in lowered for word in skip):
            continue
        # Una fecha tiene numeros y no es un item: sin esto, "14/03/2026" entra como
        # una compra de 2026 pesos.
        if DATE_RE.search(text):
            continue

        amount = last_amount(text)
        if amount is None:
            continue

        description = AMOUNT_RE.sub("", text).strip(" .-x*")
        if len(description) < 2:
            continue

        items.append(
            LineItem(description=description, quantity=None, unit_price=None, total=amount)
        )
    return items


def parse_words(words: list[Word]) -> Receipt:
    """Desde cajas: reconstruye las lineas por posicion y despues extrae."""
    return parse_lines([line_text(line) for line in group_lines(words)])


def parse_lines(lines: list[str]) -> Receipt:
    """Desde lineas de texto ya formadas.

    Los VLM de OCR (PaddleOCR-VL y companiia) transcriben la imagen a texto con saltos de linea,
    no devuelven cajas: la estructura visual ya viene resuelta por el modelo. Compartir esta
    etapa con los motores clasicos es lo que hace que el benchmark compare la calidad de lectura
    y no dos parsers distintos.
    """
    joined = "\n".join(lines)

    date_match = DATE_RE.search(joined)
    cuit_match = CUIT_RE.search(joined)
    currency_match = CURRENCY_RE.search(joined)

    return Receipt(
        merchant=_merchant(lines),
        tax_id=cuit_match.group(1) if cuit_match else None,
        date=date_match.group(1) if date_match else None,
        items=_line_items(lines),
        subtotal=_amount_for(lines, SUBTOTAL_KEYWORDS),
        tax=_amount_for(lines, TAX_KEYWORDS),
        total=_amount_for(lines, TOTAL_KEYWORDS, exclude=SUBTOTAL_KEYWORDS),
        currency=currency_match.group(1) if currency_match else None,
    )
