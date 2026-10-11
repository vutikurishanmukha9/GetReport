"""
Snowflake Ingestion Service Package for GetReport.
Implements secure key-pair authentication, encrypted sliding session vault,
deterministic Arrow schema normalization, and out-of-core DuckDB profiling.
"""

from app.services.snowflake.schema import (
    compile_target_arrow_schema,
    normalize_arrow_batch,
)
from app.services.snowflake.vault import (
    EncryptedSessionVault,
    prepare_private_key_der,
    get_snowflake_vault,
)
from app.services.snowflake.query_guard import (
    validate_account_identifier,
    validate_sql_query,
    quote_ident,
)
from app.services.snowflake.driver import (
    SnowflakeIngestDriver,
    RealSnowflakeDriver,
    MockSnowflakeDriver,
    get_snowflake_driver,
)
from app.services.snowflake.profiler import profile_parquet_out_of_core
from app.services.snowflake.executor import execute_and_stream_snowflake

__all__ = [
    "compile_target_arrow_schema",
    "normalize_arrow_batch",
    "EncryptedSessionVault",
    "prepare_private_key_der",
    "get_snowflake_vault",
    "validate_account_identifier",
    "validate_sql_query",
    "quote_ident",
    "SnowflakeIngestDriver",
    "RealSnowflakeDriver",
    "MockSnowflakeDriver",
    "get_snowflake_driver",
    "profile_parquet_out_of_core",
    "execute_and_stream_snowflake",
]
