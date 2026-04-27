from tools import calculate_total, verify_invoice


def test_calculate_total():
    items = [
        {"name": "Item1", "price": 10.0},
        {"name": "Item2", "price": 20.0}
    ]
    tax = 3.0

    result = calculate_total(items, tax)
    assert result == 33.0


def test_verify_invoice_correct():
    invoice = {
        "invoice_id": "TEST1",
        "items": [{"name": "Item", "price": 10.0}],
        "tax": 1.0,
        "total": 11.0
    }

    result = verify_invoice(invoice)
    assert result["status"] == "correct"


def test_verify_invoice_incorrect():
    invoice = {
        "invoice_id": "TEST2",
        "items": [{"name": "Item", "price": 10.0}],
        "tax": 1.0,
        "total": 15.0
    }

    result = verify_invoice(invoice)
    assert result["status"] == "incorrect"