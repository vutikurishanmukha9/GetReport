from __future__ import annotations
import polars as pl

class EmptyDatasetError(ValueError): pass
class InsufficientDataError(ValueError): pass
class AnalysisError(RuntimeError): pass

def validate_input(df: pl.DataFrame) -> None:
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df)}")
    if df.height == 0 or df.width == 0:
        raise EmptyDatasetError("Empty dataset")
