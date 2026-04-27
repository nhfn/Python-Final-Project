def calculate_total(items, tax):
    """
    Calculate total cost of an invoice.

    Args:
        items (list): List of dicts with 'price'
        tax (float): Tax amount

    Returns:
        float: total rounded to 2 decimals
    """
    subtotal = sum(item["price"] for item in items)
    return round(subtotal + tax, 2)


def verify_invoice(invoice):
    """
    Verify if an invoice total is correct.

    Args:
        invoice (dict): Full invoice object

    Returns:
        dict: result with status + message
    """
    calculated_total = calculate_total(invoice["items"], invoice["tax"])
    given_total = invoice["total"]

    if abs(calculated_total - given_total) < 0.01:
        return {
            "status": "correct",
            "invoice_id": invoice["invoice_id"],
            "expected_total": calculated_total,
            "given_total": given_total,
            "message": f"Invoice {invoice['invoice_id']} is correct."
        }
    else:
        return {
            "status": "incorrect",
            "invoice_id": invoice["invoice_id"],
            "expected_total": calculated_total,
            "given_total": given_total,
            "message": (
                f"Invoice {invoice['invoice_id']} is incorrect. "
                f"Expected ${calculated_total}, found ${given_total}."
            )
        }


def find_highest_invoice(invoices):
    """
    Find invoice with highest total.

    Args:
        invoices (list): List of invoice dicts

    Returns:
        dict: invoice with max total
    """
    if not invoices:
        return None

    return max(invoices, key=lambda x: x["total"])


def find_invoices_by_vendor(invoices, vendor_name):
    """
    Filter invoices by vendor.

    Args:
        invoices (list)
        vendor_name (str)

    Returns:
        list
    """
    return [
        invoice for invoice in invoices
        if invoice["vendor"].lower() == vendor_name.lower()
    ]

    