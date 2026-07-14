from __future__ import annotations

from decimal import Decimal

from bench.metrics import Counts, Report, score_items, score_scalar, update
from iris.schema import LineItem, Receipt


def receipt(**kwargs: object) -> Receipt:
    base: dict[str, object] = {
        "merchant": None,
        "tax_id": None,
        "date": None,
        "items": [],
        "subtotal": None,
        "tax": None,
        "total": None,
        "currency": None,
    }
    base.update(kwargs)
    return Receipt.model_validate(base)


def test_importes_equivalentes_cuentan_como_acierto() -> None:
    """60000 y 60000.00 son el mismo importe: compararlos como texto diria que no."""
    counts = Counts()
    score_scalar(counts, Decimal("60000.00"), Decimal("60000"))

    assert counts.true_positive == 1


def test_alucinar_un_campo_ausente_es_falso_positivo() -> None:
    """Si el recibo no tiene impuesto y el motor igual reporta uno, es un error. Un motor que
    inventa datos no puede puntuar igual que uno que se abstiene."""
    counts = Counts()
    score_scalar(counts, Decimal("100"), None)

    assert counts.false_positive == 1
    assert counts.true_positive == 0


def test_items_se_comparan_sin_importar_el_orden() -> None:
    counts = Counts()
    predicted = receipt(
        items=[
            LineItem(description="Pan", quantity=None, unit_price=None, total=None),
            LineItem(description="Cafe", quantity=None, unit_price=None, total=None),
        ]
    )
    truth = receipt(
        items=[
            LineItem(description="Cafe", quantity=None, unit_price=None, total=None),
            LineItem(description="Pan", quantity=None, unit_price=None, total=None),
        ]
    )

    score_items(counts, predicted, truth)

    assert counts.true_positive == 2
    assert counts.false_positive == 0


def test_un_motor_que_falla_no_puntua_neutro() -> None:
    """Si el motor se cae, los campos que habia que extraer cuentan como no extraidos.
    Saltearlos le regalaria un F1 alto al motor mas fragil."""
    report = Report()
    truth = receipt(
        total=Decimal("100"),
        items=[LineItem(description="Cafe", quantity=None, unit_price=None, total=None)],
    )

    update(report, None, truth, "", "referencia")

    assert report.failures == 1
    assert report.fields["total"].false_negative == 1
    assert report.fields["items"].false_negative == 1
    assert report.macro_f1 == 0.0


def test_macro_f1_ignora_campos_sin_anotar() -> None:
    """CORD no anota `merchant`. Promediarlo daria 0 siempre y hundiria a todos los motores
    por una limitacion del dataset."""
    report = Report()
    truth = receipt(total=Decimal("100"))
    predicted = receipt(total=Decimal("100"))

    update(report, predicted, truth, "texto", "texto")

    assert "merchant" not in report.supported_fields
    assert report.macro_f1 == 1.0
