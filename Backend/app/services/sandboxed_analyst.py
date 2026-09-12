"""
Backend/app/services/sandboxed_analyst.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AST-Sandboxed Python Analyst Agent inspired by PandasAI.
Enforces strict AST parsing, restricted execution namespaces,
and headless Matplotlib chart generation against Polars DataFrames.

Security: Blocks sandbox escapes via in-scope library methods (pl.read_*,
np.load, plt.savefig), enforces output size caps, and applies thread-based
execution timeouts to prevent DoS.
"""
from __future__ import annotations

import ast
import base64
import contextlib
from dataclasses import dataclass
import io
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_STDOUT_BYTES = 1_048_576  # 1 MB cap on captured stdout
MAX_RESULT_ROWS = 100         # Max rows returned from DataFrame results


class SandboxedSecurityViolation(Exception):
    """Raised when generated or user code violates AST sandbox constraints."""
    pass


class SandboxTimeoutError(Exception):
    """Raised when sandboxed code exceeds the execution time limit."""
    pass


@dataclass
class ExecutionResult:
    """Standard container for sandboxed execution outcome."""
    success: bool
    output: str
    chart_base64: Optional[str] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "chart_base64": self.chart_base64,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
            "result": self.result,
        }


class SandboxedAnalystAgent:
    """
    Validates, scopes, and executes Python data analysis scripts safely
    against an in-memory Polars DataFrame, returning verified results or charts.
    """

    FORBIDDEN_MODULES: Set[str] = {
        "os", "sys", "subprocess", "socket", "requests", "urllib",
        "shutil", "pathlib", "http", "ftplib", "builtins", "posix",
        "pickle", "importlib", "multiprocessing", "threading", "pty",
        # §1: Block low-level FFI and raw I/O escape routes
        "ctypes", "_ctypes", "io", "signal", "gc", "inspect",
        "code", "codeop", "compileall", "py_compile",
    }

    FORBIDDEN_BUILTIN_CALLS: Set[str] = {
        "eval", "exec", "open", "compile", "__import__",
        "input", "breakpoint", "memoryview", "getattr", "setattr",
        "delattr", "globals", "locals", "vars", "super", "format",
        "classmethod", "staticmethod", "property", "object", "type",
        "help", "dir", "id"
    }

    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "__class__", "__subclasses__", "__bases__", "__base__",
        "__globals__", "__code__", "__reduce__", "__reduce_ex__",
        "__mro__", "__dict__", "__builtins__", "__init__", "__new__"
    }

    # §1: Methods on in-scope objects (pl, np, plt) that allow file I/O or escape
    FORBIDDEN_METHOD_CALLS: Set[str] = {
        # Polars file I/O — read/write/scan
        "read_csv", "read_parquet", "read_json", "read_ndjson", "read_ipc",
        "read_avro", "read_excel", "read_database", "read_delta",
        "scan_csv", "scan_parquet", "scan_ndjson", "scan_ipc", "scan_delta",
        "from_csv", "from_pandas", "from_arrow",
        "write_csv", "write_parquet", "write_json", "write_ndjson",
        "write_ipc", "write_avro", "write_excel", "write_database",
        "write_delta", "sink_csv", "sink_parquet", "sink_ndjson", "sink_ipc",
        # Numpy file I/O
        "load", "save", "savez", "savez_compressed", "fromfile", "tofile",
        "savetxt", "loadtxt", "genfromtxt",
        # Matplotlib file I/O (plt.savefig is handled by us post-execution)
        "savefig",
        # Generic escape vectors
        "system", "popen", "spawn", "call", "run",
    }

    SAFE_BUILTINS: Dict[str, Any] = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(
        self,
        df: Optional[pl.DataFrame] = None,
        llm_service: Any = None,
        timeout_seconds: int = 10,
    ):
        self.df = df
        self.llm = llm_service
        self.timeout_seconds = timeout_seconds

    @classmethod
    def validate_code_ast(cls, code_str: str) -> None:
        """
        Parses code into an AST and verifies every node against security policies.
        Raises SandboxedSecurityViolation on any prohibited operation.
        """
        if not code_str or not code_str.strip():
            raise SandboxedSecurityViolation("Cannot execute empty code.")

        try:
            tree = ast.parse(code_str, mode="exec")
        except SyntaxError as se:
            raise SandboxedSecurityViolation(f"Syntax error in code: {se}")

        for node in ast.walk(tree):
            # Check for forbidden module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    if root_pkg in cls.FORBIDDEN_MODULES:
                        raise SandboxedSecurityViolation(f"Import of module '{root_pkg}' is strictly forbidden.")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split(".")[0]
                    if root_pkg in cls.FORBIDDEN_MODULES:
                        raise SandboxedSecurityViolation(f"Import from module '{root_pkg}' is strictly forbidden.")

            # Check for forbidden builtin calls AND forbidden method calls on in-scope objects
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_BUILTIN_CALLS:
                        raise SandboxedSecurityViolation(f"Calling function '{node.func.id}' is prohibited.")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith("__"):
                        raise SandboxedSecurityViolation(f"Calling private method '{node.func.attr}' is prohibited.")
                    # §1: Block dangerous methods on in-scope objects (pl, np, plt, df)
                    if node.func.attr in cls.FORBIDDEN_METHOD_CALLS:
                        raise SandboxedSecurityViolation(
                            f"Calling method '{node.func.attr}' is prohibited in the sandbox."
                        )
                    if node.func.attr in ("format", "__format__"):
                        if isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
                            if "__" in node.func.value.value:
                                raise SandboxedSecurityViolation("Introspective format strings are prohibited.")

            # Check for dunder attribute exploration / jailbreaks
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in cls.FORBIDDEN_ATTRIBUTES:
                    raise SandboxedSecurityViolation(f"Accessing private attribute '{node.attr}' is prohibited.")

            # Check for dunder subscript access (e.g. obj["__builtins__"])
            elif isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    if node.slice.value.startswith("__"):
                        raise SandboxedSecurityViolation(f"Accessing private key '{node.slice.value}' via subscript is prohibited.")

            # Check for format-string introspection payloads
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "{" in node.value and "__" in node.value:
                    raise SandboxedSecurityViolation("Format string introspection with private attributes is prohibited.")

    def execute_polars(
        self,
        code_str: str,
        df: Optional[pl.DataFrame] = None,
    ) -> ExecutionResult:
        """
        Executes Python code safely against Polars DataFrame.
        Captures print outputs, return results, and generated Matplotlib charts.
        Enforces execution timeout and output size limits.
        """
        active_df = df if df is not None else self.df
        if active_df is None:
            return ExecutionResult(
                success=False,
                output="",
                error="No DataFrame provided for analysis.",
            )

        start_time = time.perf_counter()
        try:
            self.validate_code_ast(code_str)
        except SandboxedSecurityViolation as sv:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Forbidden import or AST pattern: {str(sv)}",
                execution_time_ms=0.0,
            )
        except Exception as se:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code validation error: {str(se)}",
                execution_time_ms=0.0,
            )

        plt.close("all")
        stdout_buf = io.StringIO()
        chart_base64: Optional[str] = None

        exec_scope: Dict[str, Any] = {
            "df": active_df,
            "pl": pl,
            "np": np,
            "plt": plt,
            "result": None,
        }

        # §1: Thread-based execution timeout
        exec_error: List[Optional[Exception]] = [None]
        exec_done = threading.Event()

        def _run_code() -> None:
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(code_str, {"__builtins__": self.SAFE_BUILTINS}, exec_scope)
            except Exception as e:
                exec_error[0] = e
            finally:
                exec_done.set()

        thread = threading.Thread(target=_run_code, daemon=True)
        thread.start()
        finished = exec_done.wait(timeout=self.timeout_seconds)

        if not finished:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            plt.close("all")
            return ExecutionResult(
                success=False,
                output="",
                chart_base64=None,
                execution_time_ms=round(elapsed_ms, 2),
                error=f"Execution timed out after {self.timeout_seconds}s.",
            )

        try:
            if exec_error[0] is not None:
                raise exec_error[0]

            # Check if Matplotlib created any plots
            fig = plt.gcf()
            if fig and fig.axes:
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=130)
                img_buf.seek(0)
                chart_base64 = base64.b64encode(img_buf.read()).decode("utf-8")

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # §8: Cap stdout output size to prevent memory exhaustion
            raw_output = stdout_buf.getvalue()
            if len(raw_output) > MAX_STDOUT_BYTES:
                output_str = raw_output[:MAX_STDOUT_BYTES].strip() + "\n... [output truncated at 1MB]"
            else:
                output_str = raw_output.strip()

            raw_result = exec_scope.get("result")
            formatted_result = self._serialize_result(raw_result)

            if not output_str and raw_result is not None:
                output_str = str(raw_result)
                # Cap the stringified result too
                if len(output_str) > MAX_STDOUT_BYTES:
                    output_str = output_str[:MAX_STDOUT_BYTES] + "\n... [output truncated at 1MB]"

            return ExecutionResult(
                success=True,
                output=output_str,
                chart_base64=chart_base64,
                execution_time_ms=round(elapsed_ms, 2),
                result=formatted_result,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                output=stdout_buf.getvalue()[:MAX_STDOUT_BYTES].strip(),
                chart_base64=None,
                execution_time_ms=round(elapsed_ms, 2),
                error=str(exc),
            )
        finally:
            plt.close("all")

    def execute_sandboxed_code(self, code_str: str) -> Dict[str, Any]:
        """
        Backward-compatible dictionary-returning execution interface.
        """
        res = self.execute_polars(code_str)
        if not res.success:
            raise RuntimeError(res.error or "Sandbox execution failed")
        return {
            "result": res.result,
            "chart_base64": res.chart_base64,
            "code_executed": code_str,
            "execution_time_ms": res.execution_time_ms,
        }

    @staticmethod
    def _serialize_result(val: Any) -> Any:
        """Converts Polars DataFrames or Series to JSON-serializable structures."""
        if isinstance(val, pl.DataFrame):
            return {
                "type": "dataframe",
                "columns": val.columns,
                "rows": val.head(MAX_RESULT_ROWS).to_dicts(),
                "total_rows": val.height,
            }
        elif isinstance(val, pl.Series):
            return {
                "type": "series",
                "name": val.name,
                "values": val.head(MAX_RESULT_ROWS).to_list(),
                "length": val.len(),
            }
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, (dict, list, str, int, float, bool)) or val is None:
            return val
        else:
            return str(val)

    def generate_and_execute_analysis(self, user_prompt: str) -> Dict[str, Any]:
        """
        Synthesizes Polars Python code from user intent, validates it via AST,
        and executes it against the dataset.
        """
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("User prompt cannot be empty.")

        if self.df is None:
            raise ValueError("DataFrame must be initialized for synthesis.")

        schema_repr = {col: str(self.df[col].dtype) for col in self.df.columns}

        # Build code synthesis prompt
        system_instructions = f"""You are an elite Lead Data Scientist writing safe Polars Python analytical code.
Dataset Schema:
{schema_repr}

User Question/Request:
{clean_prompt}

STRICT REQUIREMENTS:
1. Write ONLY valid Python code inside ```python ... ```
2. Use the pre-existing Polars DataFrame `df`.
3. If computing numbers or tables, assign the output to `result`. Example: `result = df.select(...)`
4. If asked to plot or visualize, use `plt` (matplotlib.pyplot), configure titles and labels cleanly, and NEVER call `plt.show()`.
5. NEVER import os, sys, subprocess, socket, or access files.
"""
        generated_code = ""
        if self.llm is not None:
            try:
                raw_resp = self.llm.generate_text(system_instructions)
                # Strip markdown fences
                if "```python" in raw_resp:
                    generated_code = raw_resp.split("```python")[1].split("```")[0].strip()
                elif "```" in raw_resp:
                    generated_code = raw_resp.split("```")[1].split("```")[0].strip()
                else:
                    generated_code = raw_resp.strip()
            except Exception as llm_err:
                logger.warning("LLM code generation failed (%s), falling back to rule-based synthesis.", llm_err)

        if not generated_code:
            # Deterministic fallback synthesis for common keywords
            q_lower = clean_prompt.lower()
            if "plot" in q_lower or "histogram" in q_lower or "chart" in q_lower:
                num_cols = [c for c in self.df.columns if self.df[c].dtype.is_numeric()]
                target_col = num_cols[0] if num_cols else self.df.columns[0]
                generated_code = (
                    f"vals = df['{target_col}'].drop_nulls().to_numpy()\n"
                    f"plt.figure(figsize=(8, 4))\n"
                    f"plt.hist(vals, bins=15, color='#722F37', edgecolor='black', alpha=0.8)\n"
                    f"plt.title('Distribution of {target_col}')\n"
                    f"plt.xlabel('{target_col}')\n"
                    f"plt.ylabel('Frequency')\n"
                    f"plt.grid(True, linestyle='--', alpha=0.5)\n"
                    f"result = f'Generated distribution chart for {target_col}'"
                )
            else:
                generated_code = (
                    f"summary_df = df.describe()\n"
                    f"result = summary_df"
                )

        exec_res = self.execute_sandboxed_code(generated_code)
        exec_res["question"] = clean_prompt
        return exec_res
