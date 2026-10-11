"""
Snowflake Security Guard and Query AST Validation.
Enforces SSRF prevention on account identifiers and strict AST inspection on SQL inputs.
"""
from __future__ import annotations

import re
from typing import Optional, Set, Tuple

import sqlglot
import sqlglot.expressions as exp

# Strict Regex: alphanumeric, hyphens, underscores, dots (e.g. "xy12345.us-east-1" or "org-account")
ACCOUNT_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+(\.[a-zA-Z0-9_\-]+)*$")
ALLOWED_AUTHENTICATORS: Set[str] = {"snowflake", "snowflake_jwt"}


def validate_account_identifier(account: str) -> bool:
    """
    Validates Snowflake account identifier syntax to eliminate SSRF attacks.
    Account names must strictly match valid host prefixes without URL schemes, paths, or query strings.
    """
    if not account or len(account) > 128:
        return False
    # Ensure no URI components or injection characters exist
    if "://" in account or "/" in account or "?" in account or "#" in account or "@" in account:
        return False
    return bool(ACCOUNT_REGEX.match(account.strip()))


def quote_ident(ident: str) -> str:
    """
    Safely quotes and escapes a Snowflake SQL object identifier.
    Guarantees special characters or reserved keywords do not break query boundaries.
    """
    cleaned = ident.replace('"', '""')
    return f'"{cleaned}"'


def validate_sql_query(sql_text: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates that a custom query is a single, non-destructive read query (SELECT / UNION / CTE).
    Returns (is_valid, clean_sql, error_message).
    """
    trimmed = sql_text.strip()
    if not trimmed:
        return False, "", "SQL query cannot be empty."

    try:
        parsed_expressions = sqlglot.parse(trimmed, read="snowflake")
    except Exception as parse_err:
        return False, "", f"SQL Syntax Error: {parse_err}"

    if not parsed_expressions:
        return False, "", "No valid SQL expressions detected."

    if len(parsed_expressions) > 1:
        return False, "", "Multiple SQL statements are not permitted. Please submit a single query."

    root_expr = parsed_expressions[0]

    # Prohibited dangerous statement types
    banned_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.TruncateTable,
        exp.Command,
        exp.Grant,
    )
    if isinstance(root_expr, banned_types):
        return False, "", f"Operation not allowed: {root_expr.key.upper()} statements are prohibited in read-only reports."

    # Must be a Query (Select, Union, Intersect, Except)
    if not isinstance(root_expr, exp.Query):
        return False, "", "Only analytical read queries (SELECT / CTE) are supported."

    # Clean query text: strips comments and normalizes
    clean_sql = root_expr.sql(dialect="snowflake", comments=False).rstrip(";").strip()
    return True, clean_sql, None
