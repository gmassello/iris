from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, WithJsonSchema

# Decimal adentro (los importes se comparan y suman: float daria 0.1+0.2 != 0.3), pero number
# en JSON. Sin el serializer, Pydantic emite Decimal como string y contradice el propio schema
# que le declaramos a Claude, que entonces escribe strings donde pedimos numeros.
Money = Annotated[
    Decimal,
    WithJsonSchema({"type": "number"}),
    PlainSerializer(float, return_type=float, when_used="json"),
]


class Strict(BaseModel):
    # extra='forbid' emite additionalProperties:false, que el modo estricto de structured
    # outputs exige.
    model_config = ConfigDict(extra="forbid")


class BoundingBox(Strict):
    x: int
    y: int
    width: int
    height: int

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


class Word(Strict):
    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)


class OCRResult(Strict):
    """Salida de un motor. Las cajas y las dimensiones estan siempre en coordenadas de la
    imagen de entrada, aunque el motor haya preprocesado por dentro."""

    engine: str
    text: str
    words: list[Word]
    image_width: int
    image_height: int
    elapsed_ms: float
    receipt: Receipt | None


class LineItem(Strict):
    description: str
    quantity: Money | None
    unit_price: Money | None
    total: Money | None


class Receipt(Strict):
    """Recibo estructurado. Todo campo es nullable y ninguno tiene default: el JSON Schema
    de structured outputs exige que todos los campos esten en `required`, y la forma de
    modelar 'puede faltar' es un tipo nullable, no un campo ausente."""

    merchant: str | None
    tax_id: str | None
    date: str | None
    items: list[LineItem]
    subtotal: Money | None
    tax: Money | None
    total: Money | None
    currency: str | None
