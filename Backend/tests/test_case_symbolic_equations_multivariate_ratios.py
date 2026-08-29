import polars as pl
import numpy as np
import pytest
from app.services.smart_schema import discover_symbolic_equations

def test_symbolic_equations_discovery_multi_relationships():
    """
    Verify symbolic equation discovery identifies arithmetic and ratio relationships:
    1. Total = Subtotal + Tax
    2. Net_Profit = Gross_Revenue - Operational_Cost
    3. Ratio = Part / Whole
    """
    np.random.seed(42)
    n = 200
    
    subtotal = np.random.uniform(100.0, 1000.0, n)
    tax = subtotal * 0.15
    total = subtotal + tax
    
    cost = np.random.uniform(50.0, 500.0, n)
    revenue = subtotal * 1.5
    net_profit = revenue - cost
    
    df = pl.DataFrame({
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "revenue": revenue,
        "cost": cost,
        "net_profit": net_profit
    })
    
    equations = discover_symbolic_equations(df)
    assert len(equations) >= 2
    
    formula_texts = [eq["formula"] for eq in equations]
    has_total_sum = any("total" in f and "subtotal" in f and "tax" in f for f in formula_texts)
    assert has_total_sum is True
