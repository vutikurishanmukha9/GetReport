"""
remediation.py
~~~~~~~~~~~~~~
Functional Dispatch Strategy Pattern for Issue Ledger data cleaning rules.
Maps rule actions ('drop_rows', 'fill_median', etc.) to pure, testable transformation functions.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional
import polars as pl

from app.core.factories.registry import Registry

logger = logging.getLogger(__name__)

# Type definition for a functional remediation transformation
RemediationFunc = Callable[[pl.DataFrame, str, dict[str, Any]], tuple[pl.DataFrame, dict[str, Any], int]]

remediation_registry = Registry[RemediationFunc]("remediation_rules")


@remediation_registry.register("drop_rows")
def drop_null_rows(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Drops rows where the target column is null."""
    initial_len = df.height
    filtered_df = df.filter(pl.col(col).is_not_null())
    dropped = initial_len - filtered_df.height
    return filtered_df, {"rows_dropped": dropped}, dropped


@remediation_registry.register("fill_mean")
def fill_null_mean(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Imputes numeric nulls with column arithmetic mean."""
    if df[col].dtype in (pl.Int64, pl.Float64, pl.Float32, pl.Int32):
        mean_val = df[col].mean()
        null_cnt = df[col].null_count()
        if null_cnt > 0 and mean_val is not None:
            updated_df = df.with_columns(pl.col(col).fill_null(mean_val))
            return updated_df, {"fill_value": mean_val, "nulls_filled": null_cnt}, null_cnt
    return df, {}, 0


@remediation_registry.register("fill_median")
def fill_null_median(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Imputes numeric nulls with column median."""
    if df[col].dtype in (pl.Int64, pl.Float64, pl.Float32, pl.Int32):
        median_val = df[col].median()
        null_cnt = df[col].null_count()
        if null_cnt > 0 and median_val is not None:
            updated_df = df.with_columns(pl.col(col).fill_null(median_val))
            return updated_df, {"fill_value": median_val, "nulls_filled": null_cnt}, null_cnt
    return df, {}, 0


@remediation_registry.register("fill_mode")
def fill_null_mode(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Imputes categorical/text nulls with most frequent mode."""
    mode_s = df[col].mode()
    if mode_s.len() > 0:
        mode_val = mode_s[0]
        null_cnt = df[col].null_count()
        if null_cnt > 0 and mode_val is not None:
            updated_df = df.with_columns(pl.col(col).fill_null(mode_val))
            return updated_df, {"fill_value": mode_val, "nulls_filled": null_cnt}, null_cnt
    return df, {}, 0


@remediation_registry.register("fill_value")
def fill_null_value(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Imputes nulls with explicit user-provided constant."""
    val = params.get("value")
    if val is not None:
        null_cnt = df[col].null_count()
        if null_cnt > 0:
            updated_df = df.with_columns(pl.col(col).fill_null(val))
            return updated_df, {"fill_value": val, "nulls_filled": null_cnt}, null_cnt
    return df, {}, 0


@remediation_registry.register("replace_outliers_median")
def replace_outliers_iqr(
    df: pl.DataFrame,
    col: str,
    params: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """Clips outliers to skewness-adjusted Tukey fences using IQR."""
    if df[col].dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        if q1 is not None and q3 is not None:
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_replaced = df.filter(outlier_mask).height

            updated_df = df.with_columns(
                pl.when(pl.col(col) > upper_bound)
                .then(pl.lit(upper_bound, dtype=df[col].dtype))
                .when(pl.col(col) < lower_bound)
                .then(pl.lit(lower_bound, dtype=df[col].dtype))
                .otherwise(pl.col(col))
                .alias(col)
            )
            return (
                updated_df,
                {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outliers_replaced": outliers_replaced,
                },
                outliers_replaced,
            )
    return df, {}, 0


def apply_remediation(
    action: str,
    df: pl.DataFrame,
    col: str,
    params: Optional[dict[str, Any]] = None,
) -> tuple[pl.DataFrame, dict[str, Any], int]:
    """
    Public dispatcher for data remediation strategies.
    Looks up functional strategy in remediation_registry.
    """
    handler = remediation_registry.get(action)
    return handler(df, col, params or {})
