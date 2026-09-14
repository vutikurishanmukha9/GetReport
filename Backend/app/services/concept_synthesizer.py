"""
Backend/app/services/concept_synthesizer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Virtual Concept Synthesizer inspired by Microsoft Data Formulator.
Transforms user formulas or natural language metric declarations into
validated, high-performance Polars expressions, applying them to datasets
and recording complete provenance in GetReport's Transformation DAG.
"""
from __future__ import annotations

import ast
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from pydantic import BaseModel

from app.services.transformation_dag import (
    TransformationDAG,
    TransformationNode,
    _compute_data_hash,
)

logger = logging.getLogger(__name__)


class ConceptDeclaration(BaseModel):
    concept_name: str
    formula_or_intent: str
    description: Optional[str] = ""


class ConceptSynthesizerSecurityViolation(Exception):
    """Raised when an expression attempts forbidden AST operations or jailbreaks."""
    pass


class VirtualConceptSynthesizer:
    """
    Synthesizes, validates, and materializes custom analytical concepts
    as Polars expressions while maintaining audit-grade DAG lineage.
    """

    ALLOWED_POLARS_ROOTS: Set[str] = {
        "col", "lit", "when", "coalesce", "concat_str", "min", "max",
        "sum", "mean", "median", "all", "any", "count", "duration",
        "date", "datetime", "int_range", "date_range"
    }

    FORBIDDEN_EXPR_METHODS: Set[str] = {
        "map_elements", "pipe", "apply", "to_pandas", "to_numpy", "to_arrow",
        "write_csv", "write_parquet", "write_json", "write_ndjson", "write_ipc",
        "sink_csv", "sink_parquet", "sink_ipc", "sink_ndjson", "read_csv",
        "read_parquet", "read_json", "read_ndjson", "read_ipc", "scan_csv",
        "scan_parquet", "scan_ndjson", "scan_ipc"
    }

    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "__class__", "__subclasses__", "__bases__", "__base__",
        "__globals__", "__code__", "__reduce__", "__reduce_ex__",
        "__mro__", "__dict__", "__builtins__", "__init__", "__new__"
    }

    FORBIDDEN_BUILTINS: Set[str] = {
        "eval", "exec", "open", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "super", "format", "dir", "id"
    }

    SAFE_EVAL_BUILTINS: Dict[str, Any] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    @classmethod
    def validate_expression_ast(cls, expr_str: str) -> None:
        """
        Validates that an expression string is a benign Polars calculation.
        Parsed strictly in 'eval' mode (disallowing statements, loops, or imports).
        """
        clean_expr = expr_str.strip()
        if not clean_expr:
            raise ConceptSynthesizerSecurityViolation("Expression cannot be empty.")

        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as se:
            raise ConceptSynthesizerSecurityViolation(f"Invalid expression syntax: {se}")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ConceptSynthesizerSecurityViolation("Imports are forbidden in concept expressions.")
            elif isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ConceptSynthesizerSecurityViolation("Defining functions or lambdas is prohibited.")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in cls.FORBIDDEN_BUILTINS:
                        raise ConceptSynthesizerSecurityViolation(f"Calling '{node.func.id}' is forbidden.")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith("__"):
                        raise ConceptSynthesizerSecurityViolation(f"Calling private method '{node.func.attr}' is prohibited.")
                    if node.func.attr in cls.FORBIDDEN_EXPR_METHODS:
                        raise ConceptSynthesizerSecurityViolation(f"Calling '{node.func.attr}' is prohibited.")
                    # If call is directly on pl (e.g. pl.read_csv or pl.col), enforce whitelist
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
                        if node.func.attr not in cls.ALLOWED_POLARS_ROOTS:
                            raise ConceptSynthesizerSecurityViolation(
                                f"Polars function 'pl.{node.func.attr}' is not permitted in concept expressions."
                            )
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in cls.FORBIDDEN_ATTRIBUTES:
                    raise ConceptSynthesizerSecurityViolation(f"Accessing private attribute '{node.attr}' is prohibited.")
                if node.attr in cls.FORBIDDEN_EXPR_METHODS:
                    raise ConceptSynthesizerSecurityViolation(f"Accessing method '{node.attr}' is prohibited.")
            elif isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    if node.slice.value.startswith("__"):
                        raise ConceptSynthesizerSecurityViolation(f"Accessing private key '{node.slice.value}' via subscript is prohibited.")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "{" in node.value and "__" in node.value:
                    raise ConceptSynthesizerSecurityViolation("Format string introspection with private attributes is prohibited.")

    def synthesize_polars_expression(
        self,
        schema: Dict[str, str],
        declaration: ConceptDeclaration,
    ) -> str:
        """
        Translates raw formulas or intent into a valid Polars expression.
        Examples:
          - "revenue - cost" -> "pl.col('revenue') - pl.col('cost')"
          - "(price * qty) * (1 - discount)" -> "(pl.col('price') * pl.col('qty')) * (1 - pl.col('discount'))"
          - "pl.col('sales') * 1.1" -> "pl.col('sales') * 1.1"
        """
        raw = declaration.formula_or_intent.strip()
        if not raw:
            raise ValueError("Formula or intent description cannot be empty.")

        # 1. If user provided a native Polars expression directly
        if "pl.col(" in raw or raw.startswith("pl."):
            self.validate_expression_ast(raw)
            return raw

        # 2. Check if expression is arithmetic formula referencing columns
        # Case-insensitive mapping for existing columns
        col_map = {c.lower(): c for c in schema.keys()}
        
        # Tokenize identifiers in formula
        # Replace column names with pl.col('exact_name')
        def replace_col(match: re.Match) -> str:
            ident = match.group(0)
            ident_lower = ident.lower()
            if ident_lower in col_map:
                exact_col = col_map[ident_lower]
                return f"pl.col('{exact_col}')"
            return ident

        # Regex matching word identifiers that aren't Python keywords
        candidate_expr = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", replace_col, raw)

        # Check if the transformed candidate parses as a valid Polars AST
        try:
            self.validate_expression_ast(candidate_expr)
            # Test compile in evaluation scope to verify validity
            test_scope = {"pl": pl}
            eval(candidate_expr, {"__builtins__": self.SAFE_EVAL_BUILTINS}, test_scope)
            return candidate_expr
        except Exception:
            pass

        # 3. Use LLM synthesis if available
        if self.llm is not None:
            try:
                prompt = f"""You are an expert Polars/Python compiler.
Dataset Schema:
{schema}

Concept Name: {declaration.concept_name}
User Intent / Formula: {declaration.formula_or_intent}

STRICT REQUIREMENTS:
1. Return ONLY a single Polars expression suitable for `df.with_columns(...)`.
2. Example: (pl.col("revenue") - pl.col("cost")) / pl.col("revenue")
3. Do NOT include variable assignments or `df.with_columns(...)`.
4. Output format: ```python [expression] ```
"""
                resp = self.llm.generate_text(prompt)
                expr_clean = resp.replace("```python", "").replace("```", "").strip()
                self.validate_expression_ast(expr_clean)
                return expr_clean
            except Exception as llm_err:
                logger.warning("LLM concept synthesis failed: %s", llm_err)

        # 4. Fallback: assume literal formula was already valid or raise clear error
        self.validate_expression_ast(candidate_expr)
        return candidate_expr

    def apply_and_record_concept(
        self,
        df: pl.DataFrame,
        declaration: ConceptDeclaration,
        dag: Optional[TransformationDAG] = None,
    ) -> Tuple[pl.DataFrame, TransformationNode]:
        """
        Synthesizes the concept expression, evaluates it against the DataFrame,
        adds the column, and records the step into the TransformationDAG.
        """
        start_time = time.perf_counter()
        schema_dict = {col: str(df[col].dtype) for col in df.columns}
        expr_str = self.synthesize_polars_expression(schema_dict, declaration)

        # Evaluate expression safely in scoped namespace
        eval_scope = {"pl": pl}
        try:
            compiled_expr = eval(expr_str, {"__builtins__": self.SAFE_EVAL_BUILTINS}, eval_scope)
        except Exception as eval_err:
            raise RuntimeError(f"Failed to evaluate concept expression '{expr_str}': {eval_err}")

        input_rows = df.height
        input_cols = df.width
        input_hash = _compute_data_hash(df)

        try:
            new_df = df.with_columns(compiled_expr.alias(declaration.concept_name))
        except Exception as materialize_err:
            raise RuntimeError(f"Polars failed to compute derived column '{declaration.concept_name}': {materialize_err}")

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if dag is not None:
            node = dag.add_node(
                operation="concept_derivation",
                df_before=df,
                df_after=new_df,
                target_column=declaration.concept_name,
                parameters={
                    "expression": expr_str,
                    "formula_or_intent": declaration.formula_or_intent,
                    "description": declaration.description or "",
                },
                duration_ms=round(duration_ms, 2),
                values_changed=new_df.height,
            )
        else:
            input_rows = df.height
            input_cols = df.width
            input_hash = _compute_data_hash(df)
            output_rows = new_df.height
            output_cols = new_df.width
            output_hash = _compute_data_hash(new_df)

            node = TransformationNode(
                operation="concept_derivation",
                target_column=declaration.concept_name,
                parameters={
                    "expression": expr_str,
                    "formula_or_intent": declaration.formula_or_intent,
                    "description": declaration.description or "",
                },
                duration_ms=round(duration_ms, 2),
                input_rows=input_rows,
                input_cols=input_cols,
                output_rows=output_rows,
                output_cols=output_cols,
                input_hash=input_hash,
                output_hash=output_hash,
                rows_affected=output_rows,
                values_changed=output_rows,
                reversibility="full",
                reverse_hint=f"df = df.drop('{declaration.concept_name}')",
            )

        logger.info(
            "Synthesized virtual concept '%s' via '%s' (duration: %.2fms)",
            declaration.concept_name, expr_str, duration_ms
        )

        return new_df, node
