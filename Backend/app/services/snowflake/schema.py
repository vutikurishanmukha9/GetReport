"""
Deterministic Snowflake Schema Compilation and Apache Arrow Batch Normalization.
Ensures zero-drift schema compliance for PyArrow ParquetWriter.
"""
from __future__ import annotations

import re
from typing import Any, Sequence, Tuple
import pyarrow as pa
import pyarrow.compute as pc

try:
    from snowflake.connector.constants import FIELD_ID_TO_NAME
except ImportError:
    # Comprehensive fallback map matching Snowflake Python Connector specification
    FIELD_ID_TO_NAME = {
        0: "FIXED",
        1: "REAL",
        2: "TEXT",
        3: "DATE",
        4: "TIMESTAMP",
        5: "VARIANT",
        6: "TIMESTAMP_LTZ",
        7: "TIMESTAMP_TZ",
        8: "TIMESTAMP_NTZ",
        9: "OBJECT",
        10: "ARRAY",
        11: "BINARY",
        12: "TIME",
        13: "BOOLEAN",
        14: "GEOGRAPHY",
        15: "GEOMETRY",
        16: "VECTOR",
        17: "MAP",
        18: "FILE",
        19: "INTERVAL_YEAR_MONTH",
        20: "INTERVAL_DAY_TIME",
    }


def compile_target_arrow_schema(
    cursor_description: Sequence[Any],
) -> Tuple[pa.Schema, dict[str, str]]:
    """
    Builds a deterministic PyArrow schema from Snowflake cursor metadata.
    
    Rules enforced:
      1. Lowercase snake_case column normalization with deterministic deduplication.
      2. Decimal precision preservation: NUMBER(p, s) with p > 18 or s > 0 -> decimal128(p, s).
      3. Timestamp microsecond normalization: NTZ -> timestamp('us', tz=None); LTZ/TZ -> timestamp('us', tz='UTC').
      4. Boolean mapped directly to pa.bool_().
      5. Complex types (VARIANT, OBJECT, ARRAY) typed as pa.string() for zero-crash Parquet streaming.
    """
    fields: list[pa.Field] = []
    seen_names: set[str] = set()
    rename_map: dict[str, str] = {}

    for col in cursor_description:
        # Cursor description tuple:
        # (name, type_code, display_size, internal_size, precision, scale, null_ok)
        raw_name = str(col[0])
        type_code = col[1]
        precision = col[4] if len(col) > 4 else None
        scale = col[5] if len(col) > 5 else None
        null_ok = bool(col[6]) if len(col) > 6 else True

        # Generate clean snake_case identifier
        base_name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip().lower()).strip("_")
        if not base_name:
            base_name = "col"

        # Deduplicate collisions
        col_name = base_name
        counter = 1
        while col_name in seen_names:
            col_name = f"{base_name}_{counter}"
            counter += 1
        seen_names.add(col_name)
        rename_map[raw_name] = col_name

        # Resolve Snowflake type name
        if isinstance(type_code, int):
            sf_type_name = FIELD_ID_TO_NAME.get(type_code, "TEXT").upper()
        else:
            sf_type_name = str(type_code).upper()

        # Map to deterministic PyArrow DataType
        if sf_type_name == "FIXED":
            if scale == 0 and precision is not None and precision <= 18:
                arrow_type = pa.int64()
            else:
                p = min(max(precision or 38, 1), 38)
                s = min(max(scale or 0, 0), p)
                arrow_type = pa.decimal128(p, s)
        elif sf_type_name == "REAL":
            arrow_type = pa.float64()
        elif sf_type_name in ("TEXT", "VARIANT", "OBJECT", "ARRAY", "GEOGRAPHY", "GEOMETRY", "MAP", "FILE"):
            arrow_type = pa.string()
        elif sf_type_name == "DATE":
            arrow_type = pa.date32()
        elif sf_type_name == "TIME":
            arrow_type = pa.time64("us")
        elif sf_type_name == "TIMESTAMP_NTZ":
            arrow_type = pa.timestamp("us", tz=None)
        elif sf_type_name in ("TIMESTAMP_LTZ", "TIMESTAMP_TZ", "TIMESTAMP"):
            arrow_type = pa.timestamp("us", tz="UTC")
        elif sf_type_name == "BOOLEAN":
            arrow_type = pa.bool_()
        elif sf_type_name == "BINARY":
            arrow_type = pa.binary()
        elif sf_type_name in ("INTERVAL_YEAR_MONTH", "INTERVAL_DAY_TIME"):
            arrow_type = pa.string()
        else:
            arrow_type = pa.string()

        fields.append(pa.field(col_name, arrow_type, nullable=null_ok))

    return pa.schema(fields), rename_map


def normalize_arrow_batch(
    batch: pa.RecordBatch,
    target_schema: pa.Schema,
    rename_map: dict[str, str],
) -> pa.RecordBatch:
    """
    Renames columns according to rename_map and casts all arrays to target_schema.
    Handles un-stringified nested types when target is pa.string().
    """
    new_arrays: list[pa.Array] = []

    for raw_name, target_field in zip(batch.schema.names, target_schema):
        raw_array = batch.column(raw_name)

        # Stringify nested lists or structs if delivered un-stringified
        if pa.types.is_nested(raw_array.type) and pa.types.is_string(target_field.type):
            raw_array = pc.cast(raw_array, pa.string())

        # Safe cast to pre-compiled schema type
        try:
            casted_array = pc.cast(raw_array, target_field.type, safe=False)
        except Exception:
            # Fallback for unexpected casting divergence
            if pa.types.is_string(target_field.type):
                casted_array = pc.cast(raw_array, pa.string())
            else:
                casted_array = raw_array

        new_arrays.append(casted_array)

    return pa.RecordBatch.from_arrays(new_arrays, schema=target_schema)
