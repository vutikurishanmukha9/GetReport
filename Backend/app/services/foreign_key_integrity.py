from __future__ import annotations
import logging
import polars as pl

logger = logging.getLogger(__name__)

def detect_foreign_key_violations(
    primary_df: pl.DataFrame,
    foreign_df: pl.DataFrame,
    pk_col: str,
    fk_col: str,
    primary_name: str = "parent_table",
    foreign_name: str = "child_table"
) -> dict:
    """
    Evaluates foreign key referential integrity between a parent table and a child table.
    
    Identifies:
    1. Orphaned child records (child rows whose foreign key does not exist in parent primary key)
    2. Referential completeness percentage
    3. Suggested 1-click Polars remediation code
    """
    if pk_col not in primary_df.columns or fk_col not in foreign_df.columns:
        return {
            "has_issue": False,
            "reason": f"Column '{pk_col}' or '{fk_col}' not found in respective DataFrames",
        }

    total_child_rows = foreign_df.height
    if total_child_rows == 0:
        return {"has_issue": False, "reason": "Child DataFrame is empty"}

    # Distinct parent PKs
    parent_pks = primary_df.select(pl.col(pk_col).drop_nulls()).unique()

    # Join child with parent to find orphans
    orphans = foreign_df.filter(
        pl.col(fk_col).is_not_null()
    ).join(
        parent_pks, left_on=fk_col, right_on=pk_col, how="anti"
    )

    orphan_count = orphans.height
    orphan_pct = round(orphan_count / total_child_rows * 100, 2)

    if orphan_count > 0:
        sample_orphan_keys = orphans[fk_col].head(10).to_list()
        fix_code = (
            f"# Filter out orphaned records not in {primary_name} ({pk_col})\n"
            f"{foreign_name} = {foreign_name}.join({primary_name}.select('{pk_col}'), left_on='{fk_col}', right_on='{pk_col}', how='semi')"
        )

        return {
            "has_issue": True,
            "primary_table": primary_name,
            "foreign_table": foreign_name,
            "pk_col": pk_col,
            "fk_col": fk_col,
            "orphan_count": orphan_count,
            "orphan_pct": orphan_pct,
            "referential_completeness": round(100.0 - orphan_pct, 2),
            "sample_orphan_keys": sample_orphan_keys,
            "suggested_fix": f"Remove {orphan_count:,} orphaned child record(s) without valid '{pk_col}' parent",
            "fix_code": fix_code,
        }

    return {
        "has_issue": False,
        "primary_table": primary_name,
        "foreign_table": foreign_name,
        "referential_completeness": 100.0,
        "orphan_count": 0,
    }
