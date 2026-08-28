import pytest
import polars as pl
from app.services.smart_schema import discover_symbolic_equations, analyze_smart_schema

def test_smart_schema_discovers_exact_symbolic_equations():
    """
    Verify discover_symbolic_equations identifies arithmetic relations between numeric features:
    - Additive sum: total = subtotal + tax
    - Multiplicative product: revenue = unit_price * quantity
    """
    subtotal = [100.0, 200.0, 150.0, 50.0, 300.0, 80.0]
    tax = [10.0, 20.0, 15.0, 5.0, 30.0, 8.0]
    total = [s + t for s, t in zip(subtotal, tax)]
    
    qty = [2.0, 5.0, 3.0, 1.0, 4.0, 2.0]
    unit_price = [50.0, 40.0, 50.0, 50.0, 75.0, 40.0]
    revenue = [q * p for q, p in zip(qty, unit_price)]
    
    df = pl.DataFrame({
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "quantity": qty,
        "unit_price": unit_price,
        "revenue": revenue
    })
    
    eqs = discover_symbolic_equations(df)
    assert len(eqs) >= 2
    
    formulas = [eq["formula"] for eq in eqs]
    assert any("total" in f and "subtotal" in f and "tax" in f for f in formulas)
    assert any("revenue" in f and "quantity" in f and "unit_price" in f for f in formulas)
    
    # Test integration in analyze_smart_schema
    res = analyze_smart_schema(df)
    assert len(res.symbolic_equations) >= 2
    assert "symbolic_equations" in res.to_dict()
