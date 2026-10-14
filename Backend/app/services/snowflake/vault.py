"""
Encrypted Sliding Session Vault for Snowflake Credentials.
Uses HKDF (HMAC-SHA256) key derivation and AES-256-GCM with Authenticated Associated Data (AAD).
Supports RSA PKCS8 DER private key serialization and connection pooling with idle expiration.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.core.config import settings

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SESSION_TTL_SECONDS = 900  # 15 minutes sliding TTL
MAX_SESSION_LIFETIME_SECONDS = 7200  # 2 hours maximum absolute lifetime
CONNECTION_IDLE_TIMEOUT_SECONDS = 60  # 60 seconds connection idle reuse


def prepare_private_key_der(pem_bytes: bytes, passphrase: Optional[str] = None) -> bytes:
    """
    Parses an RSA PEM private key and serializes it to PKCS8 DER format
    as required by the Snowflake Python Connector.
    """
    password_bytes = passphrase.encode("utf-8") if passphrase else None
    private_key = serialization.load_pem_private_key(
        pem_bytes,
        password=password_bytes,
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


class EncryptedSessionVault:
    """
    Cryptographic vault for storing ephemeral Snowflake connection parameters.
    Stored with AES-256-GCM in Redis (or thread-safe in-memory cache) with 15-min sliding expiration.
    """

    def __init__(self, secret_key: Optional[str] = None):
        raw_key = secret_key or getattr(settings, "SECRET_KEY", None)
        if not raw_key:
            if settings.DATABASE_URL or settings.REDIS_URL:
                raise RuntimeError("SECRET_KEY must be configured when persistent services are enabled.")
            # Local-only fallback: process-scoped and intentionally not suitable
            # for Redis or multi-process persistence.
            raw_key = secrets.token_urlsafe(32)
        if len(raw_key.encode("utf-8")) < 32:
            raise ValueError("SECRET_KEY must contain at least 32 bytes.")
        self.secret_key = raw_key.encode("utf-8")
        # In-memory storage fallback for local dev when Redis is not running
        self._memory_store: Dict[str, Tuple[bytes, float, float]] = {}  # session_id -> (ciphertext, expires_at, created_at)
        self._memory_lock = threading.Lock()
        
        # Process-level connection pool for fast interactive browse/preview (session_id -> (conn, last_used_timestamp))
        self._conn_pool: Dict[str, Tuple[Any, float]] = {}
        self._pool_lock = threading.Lock()

    def _derive_key(self, user_id: str, salt: bytes) -> bytes:
        """Derives a 256-bit AES key using HKDF-SHA256 bound to the user_id."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=f"snowflake-session-vault:{user_id}".encode("utf-8"),
        )
        return hkdf.derive(self.secret_key)

    def _get_redis_client(self) -> Optional[Any]:
        """Returns active redis client if reachable."""
        try:
            if settings.REDIS_URL:
                import redis
                client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2.0, socket_timeout=2.0)
                client.ping()
                return client
        except Exception:
            pass
        return None

    def store_session(self, user_id: str, credentials: Dict[str, Any], ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> str:
        """
        Encrypts credential payload and stores it under a secure random session_id.
        Returns the session_id token.
        """
        session_id = secrets.token_urlsafe(32)
        salt = os.urandom(16)
        aes_key = self._derive_key(user_id, salt)
        aesgcm = AESGCM(aes_key)

        nonce = os.urandom(12)
        aad = f"{user_id}:{session_id}".encode("utf-8")

        # Encode private key if stored as bytes
        payload = credentials.copy()
        if "private_key_der" in payload and isinstance(payload["private_key_der"], bytes):
            payload["private_key_der"] = base64.b64encode(payload["private_key_der"]).decode("utf-8")
            payload["_pk_is_base64"] = True

        plaintext = json.dumps(payload).encode("utf-8")
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

        # Packed format: salt (16B) + nonce (12B) + ciphertext
        packed_blob = salt + nonce + ciphertext
        now = time.time()

        redis_client = self._get_redis_client()
        if redis_client:
            redis_key = f"snowflake:session:{session_id}"
            metadata_key = f"snowflake:session_meta:{session_id}"
            redis_client.set(redis_key, packed_blob, ex=ttl_seconds)
            redis_client.set(metadata_key, str(now), ex=MAX_SESSION_LIFETIME_SECONDS)
        else:
            with self._memory_lock:
                self._memory_store[session_id] = (packed_blob, now + ttl_seconds, now)

        logger.debug("Stored encrypted Snowflake session %s for user %s", session_id[:8], user_id)
        return session_id

    def retrieve_session(self, user_id: str, session_id: str, refresh_ttl: bool = True) -> Optional[Dict[str, Any]]:
        """
        Decrypts credentials from vault and sliding-refreshes TTL if still valid.
        Enforces maximum absolute 2-hour session lifetime.
        """
        redis_client = self._get_redis_client()
        packed_blob: Optional[bytes] = None
        created_at: float = 0.0

        if redis_client:
            redis_key = f"snowflake:session:{session_id}"
            metadata_key = f"snowflake:session_meta:{session_id}"
            data = redis_client.get(redis_key)
            if not data:
                return None
            packed_blob = data if isinstance(data, bytes) else data.encode("latin1")
            meta = redis_client.get(metadata_key)
            if meta:
                try:
                    created_at = float(meta)
                except ValueError:
                    created_at = time.time()
            else:
                created_at = time.time()

            # Enforce 2h max lifetime
            if (time.time() - created_at) > MAX_SESSION_LIFETIME_SECONDS:
                redis_client.delete(redis_key)
                redis_client.delete(metadata_key)
                return None

            if refresh_ttl:
                redis_client.expire(redis_key, DEFAULT_SESSION_TTL_SECONDS)
        else:
            with self._memory_lock:
                entry = self._memory_store.get(session_id)
                if not entry:
                    return None
                packed_blob, expires_at, created_at = entry
                now = time.time()
                if now > expires_at or (now - created_at) > MAX_SESSION_LIFETIME_SECONDS:
                    self._memory_store.pop(session_id, None)
                    return None
                if refresh_ttl:
                    self._memory_store[session_id] = (packed_blob, now + DEFAULT_SESSION_TTL_SECONDS, created_at)

        if not packed_blob or len(packed_blob) < 28:
            return None

        salt = packed_blob[:16]
        nonce = packed_blob[16:28]
        ciphertext = packed_blob[28:]

        candidate_users = [user_id]
        if not settings.REQUIRE_AUTH:
            for fallback in ("local-development", "default_user"):
                if fallback not in candidate_users:
                    candidate_users.append(fallback)

        for candidate in candidate_users:
            try:
                aes_key = self._derive_key(candidate, salt)
                aesgcm = AESGCM(aes_key)
                aad = f"{candidate}:{session_id}".encode("utf-8")
                decrypted = aesgcm.decrypt(nonce, ciphertext, aad)
                payload = json.loads(decrypted.decode("utf-8"))

                if payload.get("_pk_is_base64") and "private_key_der" in payload:
                    payload["private_key_der"] = base64.b64decode(payload["private_key_der"])
                    payload.pop("_pk_is_base64", None)

                return payload
            except Exception:
                continue

        logger.warning("Decryption failed for Snowflake session %s", session_id[:8])
        return None

    def invalidate_session(self, session_id: str):
        """Immediately destroys the encrypted session and closes pooled connections."""
        redis_client = self._get_redis_client()
        if redis_client:
            redis_client.delete(f"snowflake:session:{session_id}")
            redis_client.delete(f"snowflake:session_meta:{session_id}")
        
        with self._memory_lock:
            self._memory_store.pop(session_id, None)

        with self._pool_lock:
            if session_id in self._conn_pool:
                conn, _ = self._conn_pool.pop(session_id)
                try:
                    conn.close()
                except Exception:
                    pass

    def get_pooled_connection(self, session_id: str) -> Optional[Any]:
        """Retrieves an active connection from process pool if not timed out."""
        with self._pool_lock:
            if session_id in self._conn_pool:
                conn, last_used = self._conn_pool[session_id]
                if (time.time() - last_used) < CONNECTION_IDLE_TIMEOUT_SECONDS:
                    self._conn_pool[session_id] = (conn, time.time())
                    return conn
                else:
                    self._conn_pool.pop(session_id, None)
                    try:
                        conn.close()
                    except Exception:
                        pass
        return None

    def set_pooled_connection(self, session_id: str, conn: Any):
        """Caches an active connection in the process pool."""
        with self._pool_lock:
            self._conn_pool[session_id] = (conn, time.time())


_default_vault: Optional[EncryptedSessionVault] = None


def get_snowflake_vault() -> EncryptedSessionVault:
    global _default_vault
    if _default_vault is None:
        _default_vault = EncryptedSessionVault()
    return _default_vault
