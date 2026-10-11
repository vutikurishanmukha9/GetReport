"""
Comprehensive Test Suite for Snowflake Ingestion Integration.
Verifies cryptographic session vault, SSRF guard, AST safety, deterministic Arrow compilation,
batch normalization, out-of-core profiling, and mock driver streaming.
"""
from __future__ import annotations

import decimal
import os
import tempfile
import pyarrow as pa
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.services.snowflake.driver import MockSnowflakeDriver
from app.services.snowflake.executor import (
    QueryCancelledException,
    execute_and_stream_snowflake,
)
from app.services.snowflake.profiler import profile_parquet_out_of_core
from app.services.snowflake.query_guard import (
    quote_ident,
    validate_account_identifier,
    validate_sql_query,
)
from app.services.snowflake.schema import (
    compile_target_arrow_schema,
    normalize_arrow_batch,
)
from app.services.snowflake.error_mapper import sanitize_snowflake_error
from app.services.snowflake.vault import (
    EncryptedSessionVault,
    prepare_private_key_der,
)


# ─── 1. SSRF & Query Guard Tests ─────────────────────────────────────────────

def test_validate_account_identifier():
    assert validate_account_identifier("xy12345.us-east-1") is True
    assert validate_account_identifier("org-account_prod") is True
    assert validate_account_identifier("acct1.privatelink") is True

    # Rejection of SSRF vectors and malformed identifiers
    assert validate_account_identifier("https://malicious.com") is False
    assert validate_account_identifier("127.0.0.1:8000") is False
    assert validate_account_identifier("../../../etc/passwd") is False
    assert validate_account_identifier("acct; DROP TABLE users;") is False
    assert validate_account_identifier("") is False
    assert validate_account_identifier("a" * 150) is False


def test_quote_ident():
    assert quote_ident("normal_table") == '"normal_table"'
    assert quote_ident('table_with_"quote"') == '"table_with_""quote"""'
    assert quote_ident("SELECT") == '"SELECT"'


def test_validate_sql_query():
    # Valid read queries
    ok, clean, err = validate_sql_query("SELECT id, name FROM customers WHERE active = true")
    assert ok is True
    assert "SELECT" in clean
    assert err is None

    # Valid CTE
    ok, clean, err = validate_sql_query("WITH cte AS (SELECT * FROM sales) SELECT * FROM cte")
    assert ok is True
    assert err is None

    # Prohibited mutations
    for bad_sql in [
        "DROP TABLE customers;",
        "INSERT INTO audit_log VALUES (1, 'hack')",
        "UPDATE users SET is_admin = true",
        "DELETE FROM orders WHERE id = 1",
        "ALTER TABLE users ADD COLUMN compromised text",
        "TRUNCATE TABLE logs",
    ]:
        ok, _, err = validate_sql_query(bad_sql)
        assert ok is False
        assert "prohibited" in err.lower() or "not allowed" in err.lower()

    # Multiple statements
    ok, _, err = validate_sql_query("SELECT 1; SELECT 2;")
    assert ok is False
    assert "multiple" in err.lower()

    # Empty query
    ok, _, err = validate_sql_query("   ")
    assert ok is False


# ─── 2. Schema Compilation & Batch Normalization ─────────────────────────────

def test_compile_target_arrow_schema():
    # Synthetic cursor description: (name, type_code, display_size, internal_size, precision, scale, null_ok)
    desc = [
        ("TRANSACTION_ID", 0, None, None, 18, 0, False),  # FIXED int64
        ("ACCOUNT_BALANCE", 0, None, None, 38, 4, True),  # FIXED decimal128(38, 4)
        ("LATITUDE", 1, None, None, None, None, True),  # REAL float64
        ("CUSTOMER_NAME", 2, None, None, None, None, True),  # TEXT string
        ("IS_ACTIVE", 13, None, None, None, None, False),  # BOOLEAN bool
        ("BIRTH_DATE", 3, None, None, None, None, True),  # DATE date32
        ("LAST_LOGIN_NTZ", 8, None, None, None, None, True),  # TIMESTAMP_NTZ
        ("CREATED_AT_TZ", 7, None, None, None, None, True),  # TIMESTAMP_TZ
        ("RAW_METADATA", 5, None, None, None, None, True),  # VARIANT string
        ("Duplicate_Name", 2, None, None, None, None, True),
        ("DUPLICATE_NAME", 2, None, None, None, None, True),  # Collision test
    ]

    schema, rename_map = compile_target_arrow_schema(desc)

    # Verify column naming and deduplication
    assert schema.names[0] == "transaction_id"
    assert schema.names[9] == "duplicate_name"
    assert schema.names[10] == "duplicate_name_1"
    assert len(schema.names) == len(set(schema.names))  # All unique

    # Verify precision and type mapping
    assert schema.field("transaction_id").type == pa.int64()
    assert schema.field("account_balance").type == pa.decimal128(38, 4)
    assert schema.field("latitude").type == pa.float64()
    assert schema.field("customer_name").type == pa.string()
    assert schema.field("is_active").type == pa.bool_()
    assert schema.field("birth_date").type == pa.date32()
    assert schema.field("last_login_ntz").type == pa.timestamp("us", tz=None)
    assert schema.field("created_at_tz").type == pa.timestamp("us", tz="UTC")
    assert schema.field("raw_metadata").type == pa.string()


def test_normalize_arrow_batch():
    desc = [
        ("ID", 0, None, None, 10, 0, False),
        ("NAME", 2, None, None, None, None, True),
        ("AMOUNT", 0, None, None, 12, 2, False),
    ]
    target_schema, rename_map = compile_target_arrow_schema(desc)

    raw_batch = pa.RecordBatch.from_pydict({
        "ID": pa.array([1, 2, 3], type=pa.int64()),
        "NAME": pa.array(["Alice", "Bob", "Charlie"], type=pa.string()),
        "AMOUNT": pa.array([decimal.Decimal("10.50"), decimal.Decimal("20.75"), decimal.Decimal("30.00")], type=pa.decimal128(12, 2)),
    })

    normalized = normalize_arrow_batch(raw_batch, target_schema, rename_map)
    assert normalized.schema == target_schema
    assert normalized.schema.names == ["id", "name", "amount"]
    assert normalized.num_rows == 3


# ─── 3. Cryptographic Vault Tests ────────────────────────────────────────────

def test_private_key_der_serialization():
    # Generate temporary RSA private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_bytes = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    der_bytes = prepare_private_key_der(pem_bytes)
    assert isinstance(der_bytes, bytes)
    assert len(der_bytes) > 500


def test_encrypted_session_vault():
    vault = EncryptedSessionVault(secret_key="test-secret-key-for-unit-tests-only")
    user_id = "user_123"

    creds = {
        "account": "test-acct",
        "user": "test-user",
        "warehouse": "COMPUTE_WH",
        "database": "DEMO_DB",
        "schema": "PUBLIC",
        "password": "super-secret-password-xyz",
    }

    session_id = vault.store_session(user_id, creds, ttl_seconds=60)
    assert session_id is not None
    assert len(session_id) > 20

    # Retrieve and decrypt
    retrieved = vault.retrieve_session(user_id, session_id)
    assert retrieved is not None
    assert retrieved["account"] == "test-acct"
    assert retrieved["password"] == "super-secret-password-xyz"

    # AAD protection: wrong user cannot decrypt
    wrong_user_retrieved = vault.retrieve_session("impostor_user_456", session_id)
    assert wrong_user_retrieved is None

    # Invalidation
    vault.invalidate_session(session_id)
    assert vault.retrieve_session(user_id, session_id) is None


# ─── 4. Mock Driver & End-to-End Extraction ──────────────────────────────────

def test_mock_driver_metadata_discovery():
    driver = MockSnowflakeDriver()
    config = {"account": "test-acct", "user": "test-user"}

    assert driver.test_connection(config) is True
    dbs = driver.discover_databases(config)
    assert len(dbs) >= 2
    assert dbs[0]["name"] == "ANALYTICS_DB"

    schemas = driver.discover_schemas(config, "ANALYTICS_DB")
    assert len(schemas) >= 2

    objects = driver.discover_objects(config, "ANALYTICS_DB", "PUBLIC")
    assert any(obj["name"] == "TRANSACTIONS" and obj["type"] == "TABLE" for obj in objects)

    preview_rows, cols = driver.preview_query(config, "SELECT * FROM TRANSACTIONS", limit=5)
    assert len(preview_rows) == 5
    assert "customer_name" in cols


def test_execute_and_stream_snowflake_to_parquet():
    driver = MockSnowflakeDriver()
    config = {"account": "test-acct", "user": "test-user"}
    task_id = "test-task-12345"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_extract.parquet")
        result = execute_and_stream_snowflake(
            driver=driver,
            session_config=config,
            query="SELECT * FROM TRANSACTIONS",
            output_path=output_path,
            task_id=task_id,
            byte_limit=10 * 1024 * 1024,
        )

        assert os.path.exists(output_path)
        assert result["total_rows"] == 100  # 2 mock batches * 50 rows
        assert result["truncated"] is False
        assert "transaction_id" in result["columns"]
        assert "merchant_name" in result["columns"]

        # Run DuckDB out-of-core profiling over generated file
        profile = profile_parquet_out_of_core(output_path)
        assert profile["total_rows"] == 100
        assert profile["total_columns"] >= 5
        assert "transaction_id" in profile["columns"]
        assert profile["columns"]["transaction_id"]["non_null_count"] == 100


def test_execute_and_stream_cancellation():
    driver = MockSnowflakeDriver()
    config = {"account": "test-acct", "user": "test-user"}
    task_id = "test-cancel-task"

    # Mock Redis client with cancel flag set
    class MockRedis:
        def __init__(self):
            self.store = {f"snowflake:cancel:{task_id}": b"1"}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, val, **kwargs):
            self.store[key] = val

        def delete(self, *keys):
            for k in keys:
                self.store.pop(k, None)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cancelled_extract.parquet")
        with pytest.raises(QueryCancelledException):
            execute_and_stream_snowflake(
                driver=driver,
                session_config=config,
                query="SELECT * FROM LARGE_TABLE",
                output_path=output_path,
                task_id=task_id,
                redis_client=MockRedis(),
            )
        assert not os.path.exists(output_path)


# ─── 5. FastAPI Endpoints Integration Flow ───────────────────────────────────

def test_snowflake_api_endpoints_flow():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # 1. Connect (using mock sandbox driver)
    res = client.post("/api/snowflake/connect", json={
        "account": "test-account",
        "user": "test-user",
        "warehouse": "COMPUTE_WH",
        "database": "DEMO_DB",
        "schema": "PUBLIC",
        "auth_type": "password",
        "password": "mock-password-123",
        "use_mock": True,
    })
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    assert session_id

    # 2. List databases
    res = client.get(f"/api/snowflake/databases?session_id={session_id}")
    assert res.status_code == 200
    assert len(res.json()["databases"]) >= 2

    # 3. List schemas
    res = client.get(f"/api/snowflake/schemas?session_id={session_id}&database_name=ANALYTICS_DB")
    assert res.status_code == 200
    assert len(res.json()["schemas"]) >= 2

    # 4. List objects
    res = client.get(f"/api/snowflake/objects?session_id={session_id}&database_name=ANALYTICS_DB&schema_name=PUBLIC")
    assert res.status_code == 200
    assert len(res.json()["objects"]) >= 2

    # 5. Preview query
    res = client.post("/api/snowflake/preview", json={
        "session_id": session_id,
        "query": "SELECT * FROM TRANSACTIONS",
        "limit": 5,
    })
    assert res.status_code == 200
    preview_data = res.json()
    assert len(preview_data["rows"]) == 5

    # 6. SSRF / Banned query rejection
    bad_res = client.post("/api/snowflake/preview", json={
        "session_id": session_id,
        "query": "DROP TABLE TRANSACTIONS",
    })
    assert bad_res.status_code == 400

    # 7. Start import
    res = client.post("/api/snowflake/import", json={
        "session_id": session_id,
        "query": "SELECT * FROM TRANSACTIONS",
        "dataset_name": "sf_transactions",
    })
    assert res.status_code == 200
    task_id = res.json()["task_id"]
    assert task_id

    # 8. Cancel query
    cancel_res = client.post(f"/api/snowflake/cancel/{task_id}")
    assert cancel_res.status_code == 200
    assert cancel_res.json()["status"] == "CANCELLED"

    # 9. Disconnect
    disc_res = client.post(f"/api/snowflake/disconnect/{session_id}")
    assert disc_res.status_code == 200


def test_snowflake_all_type_codes():
    """Verifies that all 15 core Snowflake connector type codes (0-14) map deterministically to PyArrow types."""
    desc = [
        ("COL_FIXED_INT", 0, None, None, 10, 0, True),
        ("COL_FIXED_DEC", 0, None, None, 38, 4, True),
        ("COL_REAL", 1, None, None, None, None, True),
        ("COL_TEXT", 2, None, None, None, None, True),
        ("COL_DATE", 3, None, None, None, None, True),
        ("COL_TIMESTAMP", 4, None, None, None, None, True),
        ("COL_VARIANT", 5, None, None, None, None, True),
        ("COL_TIMESTAMP_LTZ", 6, None, None, None, None, True),
        ("COL_TIMESTAMP_TZ", 7, None, None, None, None, True),
        ("COL_TIMESTAMP_NTZ", 8, None, None, None, None, True),
        ("COL_OBJECT", 9, None, None, None, None, True),
        ("COL_ARRAY", 10, None, None, None, None, True),
        ("COL_BINARY", 11, None, None, None, None, True),
        ("COL_TIME", 12, None, None, None, None, True),
        ("COL_BOOLEAN", 13, None, None, None, None, True),
        ("COL_GEOGRAPHY", 14, None, None, None, None, True),
    ]
    schema, _ = compile_target_arrow_schema(desc)
    assert schema.field("col_fixed_int").type == pa.int64()
    assert schema.field("col_fixed_dec").type == pa.decimal128(38, 4)
    assert schema.field("col_real").type == pa.float64()
    assert schema.field("col_text").type == pa.string()
    assert schema.field("col_date").type == pa.date32()
    assert schema.field("col_timestamp").type == pa.timestamp("us", tz="UTC")
    assert schema.field("col_variant").type == pa.string()
    assert schema.field("col_timestamp_ltz").type == pa.timestamp("us", tz="UTC")
    assert schema.field("col_timestamp_tz").type == pa.timestamp("us", tz="UTC")
    assert schema.field("col_timestamp_ntz").type == pa.timestamp("us", tz=None)
    assert schema.field("col_object").type == pa.string()
    assert schema.field("col_array").type == pa.string()
    assert schema.field("col_binary").type == pa.binary()
    assert schema.field("col_time").type == pa.time64("us")
    assert schema.field("col_boolean").type == pa.bool_()
    assert schema.field("col_geography").type == pa.string()


def test_sanitize_snowflake_error():
    """Verifies that error codes map to human-readable explanations and scrub credentials."""
    # Auth failure 390100
    err1 = Exception("390100 (08004): Incorrect username or password was specified.")
    msg1 = sanitize_snowflake_error(err1)
    assert "Authentication Failed" in msg1

    # Network policy IP allowlist
    err2 = "000605 (08001): Connection blocked by Snowflake network rule policy."
    msg2 = sanitize_snowflake_error(err2)
    assert "Network Policy" in msg2 or "IP Allowlist" in msg2

    # Warehouse suspended
    err3 = "000606: Warehouse 'ANALYTICS_WH' is suspended or invalid."
    msg3 = sanitize_snowflake_error(err3)
    assert "Warehouse Unavailable" in msg3

    # Statement timeout
    err4 = "000630: Statement timeout expired."
    msg4 = sanitize_snowflake_error(err4)
    assert "Timeout" in msg4

    # Credential scrubbing: private key, password, token
    leak = (
        "Error in session: password=supersecretpass; token=eyJh123; "
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    )
    cleaned = sanitize_snowflake_error(leak)
    assert "supersecretpass" not in cleaned
    assert "[REDACTED]" in cleaned
    assert "MIIEowIBAAKCAQEA" not in cleaned
    assert "[REDACTED_KEY]" in cleaned


def test_snowflake_pagination_and_delete_extract():
    """Verifies pagination cursor on objects and GDPR extract deletion with tenant ownership check."""
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    # 1. Connect
    res = client.post("/api/snowflake/connect", json={
        "account": "test-acct",
        "user": "test-user",
        "warehouse": "ANALYTICS_WH",
        "database": "ANALYTICS_DB",
        "schema": "PUBLIC",
        "auth_type": "password",
        "password": "mock-password-123",
        "use_mock": True,
    })
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # 2. Test pagination with from_last on objects
    res_page1 = client.get(
        f"/api/snowflake/objects?session_id={session_id}&database_name=ANALYTICS_DB&schema_name=PUBLIC&limit=1"
    )
    assert res_page1.status_code == 200
    p1_data = res_page1.json()
    assert len(p1_data["objects"]) == 1
    assert p1_data["has_more"] is True
    first_name = p1_data["objects"][0]["name"]

    res_page2 = client.get(
        f"/api/snowflake/objects?session_id={session_id}&database_name=ANALYTICS_DB&schema_name=PUBLIC&limit=1&from_last={first_name}"
    )
    assert res_page2.status_code == 200
    p2_data = res_page2.json()
    assert len(p2_data["objects"]) == 1
    assert p2_data["objects"][0]["name"] != first_name

    # 3. Import and create task with owner header
    import_res = client.post(
        "/api/snowflake/import",
        json={
            "session_id": session_id,
            "query": "SELECT * FROM TRANSACTIONS",
            "dataset_name": "sf_compliance_test",
        },
        headers={"x-user-id": "compliance_user_1"},
    )
    assert import_res.status_code == 200
    task_id = import_res.json()["task_id"]

    # 4. Attempt delete from mismatched tenant owner -> 403 Forbidden
    bad_delete = client.post(
        f"/api/snowflake/delete-extract/{task_id}",
        headers={"x-user-id": "unauthorized_user_2"},
    )
    assert bad_delete.status_code == 403

    # 5. Delete with authorized owner -> 200 OK
    good_delete = client.post(
        f"/api/snowflake/delete-extract/{task_id}",
        headers={"x-user-id": "compliance_user_1"},
    )
    assert good_delete.status_code == 200
    assert good_delete.json()["status"] == "DELETED"


