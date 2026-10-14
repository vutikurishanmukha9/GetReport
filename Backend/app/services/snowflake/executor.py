"""
Asynchronous Snowflake Execution and Memory-Safe Arrow Streaming Loop.
Enforces cancellation checks, query deadlines, cumulative byte budgets, and atomic Parquet file creation.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import pyarrow.parquet as pq

from app.services.snowflake.driver import SnowflakeIngestDriver
from app.services.snowflake.schema import compile_target_arrow_schema, normalize_arrow_batch

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BYTE_LIMIT = 150 * 1024 * 1024  # 150MB uncompressed byte ceiling
STATEMENT_TIMEOUT_SECONDS = 300  # 5 minutes statement timeout
TOTAL_DEADLINE_SECONDS = 420  # 7 minutes job deadline


class QueryCancelledException(Exception):
    """Raised when an interactive query cancellation is requested by the user."""
    pass


class QueryTimeoutException(Exception):
    """Raised when a query exceeds its statement or job execution deadline."""
    pass


def execute_and_stream_snowflake(
    driver: SnowflakeIngestDriver,
    session_config: Dict[str, Any],
    query: str,
    output_path: str,
    task_id: str,
    byte_limit: int = DEFAULT_BYTE_LIMIT,
    redis_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Submits query asynchronously, monitors query polling loop, streams Arrow batches,
    normalizes batches against pre-compiled schema, enforces byte limits, and writes local Parquet.
    """
    cancel_key = f"snowflake:cancel:{task_id}"
    active_key = f"snowflake:active_query:{task_id}"

    # 1. Asynchronous query submission
    logger.info("Submitting Snowflake query for task %s...", task_id)
    sfqid = driver.submit_query_async(session_config, query)
    logger.info("Snowflake query submitted successfully: sfqid=%s", sfqid)

    if redis_client:
        try:
            redis_client.set(active_key, sfqid, ex=600)
        except Exception as e:
            logger.debug("Could not store active query sfqid in Redis: %s", e)

    # 2. Polling loop with timeout, cancel checks, and backoff
    start_time = time.time()
    poll_interval = 1.0
    while driver.is_query_running(session_config, sfqid):
        # Check cancellation
        if redis_client:
            try:
                if redis_client.get(cancel_key):
                    logger.info("Cancellation requested for task %s (sfqid=%s). Aborting...", task_id, sfqid)
                    driver.abort_query(session_config, sfqid)
                    raise QueryCancelledException("Snowflake query was cancelled by the user.")
            except QueryCancelledException:
                raise
            except Exception as e:
                logger.debug("Error checking cancellation flag: %s", e)

        # Check statement timeout
        elapsed = time.time() - start_time
        if elapsed > STATEMENT_TIMEOUT_SECONDS:
            logger.warning("Query %s exceeded statement deadline of %ds. Aborting...", sfqid, STATEMENT_TIMEOUT_SECONDS)
            driver.abort_query(session_config, sfqid)
            raise QueryTimeoutException(f"Query exceeded statement deadline of {STATEMENT_TIMEOUT_SECONDS} seconds.")

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.25, 3.0)

    # 3. Stream Arrow batches to local Parquet
    logger.info("Query %s finished execution. Beginning batch stream to disk...", sfqid)
    batches, cursor_desc = driver.fetch_arrow_batches(session_config, sfqid)
    target_schema, rename_map = compile_target_arrow_schema(cursor_desc)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temp_path = f"{output_path}.tmp.parquet"
    writer: Optional[pq.ParquetWriter] = None

    bytes_accumulated = 0
    total_rows = 0
    truncated = False

    try:
        for batch in batches:
            # Check cancellation during download stream
            if redis_client:
                try:
                    if redis_client.get(cancel_key):
                        logger.info("Cancellation requested during batch download for task %s.", task_id)
                        driver.abort_query(session_config, sfqid)
                        raise QueryCancelledException("Download was aborted by the user.")
                except QueryCancelledException:
                    raise
                except Exception:
                    pass

            bytes_accumulated += batch.nbytes
            if bytes_accumulated > byte_limit:
                truncated = True
                logger.warning(
                    "Task %s reached memory safety ceiling (%d MB). Truncating stream at %d rows.",
                    task_id,
                    byte_limit // (1024 * 1024),
                    total_rows,
                )
                break

            normalized_batch = normalize_arrow_batch(batch, target_schema, rename_map)
            total_rows += normalized_batch.num_rows

            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    target_schema,
                    compression="SNAPPY",
                )
            writer.write_batch(normalized_batch)

        if writer is not None:
            writer.close()
            writer = None

        if os.path.exists(temp_path):
            # Atomic rename from temporary file to final target
            os.replace(temp_path, output_path)
            logger.info("Atomic write complete for %s (%d rows, %d bytes).", output_path, total_rows, bytes_accumulated)
        else:
            # Zero rows returned: create empty Parquet table with target_schema
            empty_table = target_schema.empty_table()
            pq.write_table(empty_table, output_path, compression="SNAPPY")
            logger.info("Created empty Parquet file at %s (0 rows).", output_path)

        return {
            "query_id": sfqid,
            "total_rows": total_rows,
            "total_bytes": bytes_accumulated,
            "truncated": truncated,
            "output_path": output_path,
            "columns": target_schema.names,
        }

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if redis_client:
            try:
                redis_client.delete(active_key)
                redis_client.delete(cancel_key)
            except Exception:
                pass
