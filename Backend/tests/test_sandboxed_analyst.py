import pytest
import polars as pl
import numpy as np
from app.services.sandboxed_analyst import SandboxedAnalystAgent, ExecutionResult


def test_sandboxed_analyst_basic_polars_computation():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({
        "sales": [100.0, 200.0, 300.0, 400.0],
        "cost": [70.0, 120.0, 190.0, 260.0],
    })

    code = """
profit = df["sales"] - df["cost"]
mean_profit = profit.mean()
print(f"MEAN_PROFIT={mean_profit:.2f}")
"""
    res = agent.execute_polars(code, df)
    assert res.success is True
    assert "MEAN_PROFIT=90.00" in res.output
    assert res.error is None
    assert res.execution_time_ms >= 0.0


def test_sandboxed_analyst_matplotlib_capture():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 15, 30, 25],
    })

    code = """
plt.figure(figsize=(6, 4))
plt.plot(df['x'].to_list(), df['y'].to_list(), marker='o', color='crimson')
plt.title("Sample Test Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
print("Plot created successfully")
"""
    res = agent.execute_polars(code, df)
    assert res.success is True
    assert "Plot created successfully" in res.output
    assert res.chart_base64 is not None
    assert len(res.chart_base64) > 100
    assert res.chart_base64.startswith("iVBORw0KGgo")  # Standard PNG magic header in base64


def test_sandboxed_analyst_security_blocks_forbidden_imports():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({"a": [1, 2, 3]})

    forbidden_scripts = [
        "import os\nos.system('dir')",
        "import subprocess\nsubprocess.run(['cmd'])",
        "import sys\nprint(sys.version)",
        "import socket\ns = socket.socket()",
        "import shutil\nshutil.rmtree('.')",
        "import requests",
        "from os import path",
    ]

    for script in forbidden_scripts:
        res = agent.execute_polars(script, df)
        assert res.success is False
        assert "Forbidden import or AST pattern" in (res.error or "")


def test_sandboxed_analyst_security_blocks_dangerous_calls_and_dunders():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({"a": [1, 2, 3]})

    malicious_scripts = [
        "open('test.txt', 'w')",
        "eval('1 + 1')",
        "exec('x = 5')",
        "__import__('os')",
        "getattr(df, '__class__')",
        "print(().__class__.__bases__[0].__subclasses__())",
    ]

    for script in malicious_scripts:
        res = agent.execute_polars(script, df)
        assert res.success is False
        assert any(term in (res.error or "").lower() for term in ["forbidden", "dangerous", "not allowed", "prohibited"])


def test_sandboxed_analyst_polars_with_columns():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({
        "revenue": [1000, 2000, 3000],
        "expenses": [600, 1100, 1800],
    })

    code = """
augmented = df.with_columns(
    ((pl.col("revenue") - pl.col("expenses")) / pl.col("revenue")).alias("margin")
)
print("MAX_MARGIN=", augmented["margin"].max())
"""
    res = agent.execute_polars(code, df)
    assert res.success is True
    assert "MAX_MARGIN= 0.45" in res.output or "0.45" in res.output


def test_sandboxed_analyst_blocks_in_scope_file_io():
    agent = SandboxedAnalystAgent(timeout_seconds=5)
    df = pl.DataFrame({"a": [1, 2, 3]})

    io_scripts = [
        "pl.read_csv('/etc/passwd')",
        "pl.read_parquet('secret.parquet')",
        "df.write_csv('out.csv')",
        "df.write_parquet('out.parquet')",
        "np.save('test.npy', np.array([1, 2]))",
        "np.load('test.npy')",
        "plt.savefig('chart.png')",
    ]

    for script in io_scripts:
        res = agent.execute_polars(script, df)
        assert res.success is False
        assert "prohibited in the sandbox" in (res.error or "")


def test_sandboxed_analyst_enforces_timeout():
    agent = SandboxedAnalystAgent(timeout_seconds=1)
    df = pl.DataFrame({"a": [1, 2, 3]})

    infinite_loop = """
x = 0
while True:
    x = (x + 1) % 1000000
"""
    res = agent.execute_polars(infinite_loop, df)
    assert res.success is False
    assert "timed out" in (res.error or "").lower()

