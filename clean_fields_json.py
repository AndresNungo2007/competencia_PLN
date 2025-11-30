from typing import Any, Dict, Optional
import json

### Funciones para transformar datos antes de guardar a CSV

# Normalizar y limpiar campos nulos en la predicción JSON
def fix_info(value):
    if value is None:
        return {}
    return value

def serialize_info(value):
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False)

def normalize_example_json_pred(js: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Si no hay salida, devolver estructura con nulls
    if js is None or not isinstance(js, dict):
        return {
            "buyer": {"name": None, "email": None, "contact": None, "addresses": None},
            "purchases": None,
            "shipping": None,
        }

    out: Dict[str, Any] = {}

    # BUYER
    buyer = js.get("buyer") if isinstance(js.get("buyer"), dict) else (js.get("buyer") if js.get("buyer") is not None else None)
    if not buyer or not isinstance(buyer, dict):
        out["buyer"] = {"name": None, "email": None, "contact": None, "addresses": None}
    else:
        out_buyer = {
            "name": buyer.get("name") if buyer.get("name") is not None else None,
            "email": buyer.get("email") if buyer.get("email") is not None else None,
            "contact": None,
            "addresses": None,
        }
        contact = buyer.get("contact")
        if contact and isinstance(contact, dict):
            out_buyer["contact"] = {
                "phone": contact.get("phone") if contact.get("phone") is not None else None,
                "alt_email": contact.get("alt_email") if contact.get("alt_email") is not None else None,
                "preferred_contact": contact.get("preferred_contact") if contact.get("preferred_contact") is not None else None,
            }
        else:
            out_buyer["contact"] = None

        addrs = buyer.get("addresses")
        if addrs and isinstance(addrs, list) and len(addrs) > 0:
            out_addrs = []
            for a in addrs:
                if not isinstance(a, dict):
                    continue
                out_addrs.append(
                    {
                        "street": a.get("street") if a.get("street") is not None else None,
                        "city": a.get("city") if a.get("city") is not None else None,
                        "state": a.get("state") if a.get("state") is not None else None,
                        "postal_code": a.get("postal_code") if a.get("postal_code") is not None else None,
                        "country": a.get("country") if a.get("country") is not None else None,
                    }
                )
            out_buyer["addresses"] = out_addrs if out_addrs else None
        else:
            out_buyer["addresses"] = None

        out["buyer"] = out_buyer

    # PURCHASES
    purchases = js.get("purchases")
    if purchases and isinstance(purchases, list) and len(purchases) > 0:
        out_p = []
        for p in purchases:
            if not isinstance(p, dict):
                continue
            qty = p.get("quantity")
            qty_parsed = None
            try:
                if qty is not None:
                    qty_parsed = int(qty)
            except Exception:
                qty_parsed = None
            out_p.append(
                {
                    "product_name": p.get("product_name") if p.get("product_name") is not None else None,
                    "quantity": qty_parsed,
                    "currency": p.get("currency") if p.get("currency") is not None else None,
                    "discount_code": p.get("discount_code") if p.get("discount_code") is not None else None,
                }
            )
        out["purchases"] = out_p if out_p else None
    else:
        out["purchases"] = None

    # SHIPPING
    shipping = js.get("shipping")
    if shipping and isinstance(shipping, dict):
        out["shipping"] = {
            "method": shipping.get("method") if shipping.get("method") is not None else None,
            "preferred_by": shipping.get("preferred_by") if shipping.get("preferred_by") is not None else None,
        }
    else:
        out["shipping"] = None

    return out
