"""
Snowflake Error Code Mapping and Sanitization.
Translates cryptic database errors into human-readable, zero-jargon operational alerts.
Ensures zero credentials or private keys leak into error strings.
"""
import re
from typing import Optional


# Mapping of Snowflake numeric/SQLSTATE error codes to friendly explanations
ERROR_CODE_MAP = {
    # Authentication & Credentials
    "390100": "Authentication Failed: Incorrect username, password, or RSA private key. Please check your credentials.",
    "390101": "User Account Locked: Your Snowflake account is temporarily locked due to multiple failed login attempts.",
    "390111": "IP Allowlist Blocked: Your Snowflake Network Policy does not allow connections from this IP address.",
    "390114": "Authentication Session Expired: Your Snowflake session has expired. Please re-authenticate.",
    "390144": "JWT Token Expired: The RSA key-pair token has expired or the system clock is skewed.",
    "390145": "JWT Token Invalid: Failed to verify RSA key-pair signature. Ensure public key is registered in Snowflake.",
    
    # Warehouse & Compute
    "000605": "Network Policy Blocked: Connection blocked by Snowflake network rule.",
    "000606": "Warehouse Unavailable: The warehouse is suspended, does not exist, or your role lacks USAGE permissions.",
    "000624": "Warehouse Out of Memory: Query exceeded Snowflake warehouse memory limits.",
    "000630": "Query Statement Timeout: The query took longer than the configured statement deadline.",
    
    # Network & Transport
    "08001": "Network Host Unreachable: Unable to connect to the specified Snowflake account hostname.",
    "08004": "Connection Rejected: Snowflake server rejected the connection handshake.",
    "08006": "Connection Terminated: The network socket connection to Snowflake was closed unexpectedly.",
    
    # Schema & Object Permissions
    "002003": "Database or Schema Not Found: The specified database or schema does not exist or your role lacks access.",
    "002043": "Object Not Found: The specified table or view does not exist in this database and schema.",
    "003001": "Permission Denied: Your active role does not have SELECT permission on this table or view.",
    
    # SQL Syntax
    "001003": "SQL Syntax Error: The submitted query contains a syntax error in the Snowflake dialect.",
}


def sanitize_snowflake_error(error: Exception | str) -> str:
    """
    Parses a Snowflake exception or error string, maps known error codes to
    clear user-friendly messages, and strips any sensitive server paths or credentials.
    """
    raw_msg = str(error)

    # 1. Strip any potential private keys, passwords, or tokens from error text
    cleaned_msg = re.sub(r"(-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----)", "[REDACTED_KEY]", raw_msg, flags=re.DOTALL)
    cleaned_msg = re.sub(r"(password\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", cleaned_msg, flags=re.IGNORECASE)
    cleaned_msg = re.sub(r"(token\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", cleaned_msg, flags=re.IGNORECASE)

    # 2. Extract error code (e.g. '390100 (08004):' or '000606')
    code_match = re.search(r"\b(\d{6})\b", cleaned_msg)
    if code_match:
        code = code_match.group(1)
        if code in ERROR_CODE_MAP:
            return ERROR_CODE_MAP[code]

    sqlstate_match = re.search(r"\(([0-9A-Z]{5})\)", cleaned_msg)
    if sqlstate_match:
        sqlstate = sqlstate_match.group(1)
        if sqlstate in ERROR_CODE_MAP:
            return ERROR_CODE_MAP[sqlstate]

    # 3. Handle common text patterns
    lower = cleaned_msg.lower()
    if "network policy" in lower or "ip" in lower and "not allowed" in lower:
        return "IP Allowlist Blocked: Your Snowflake Network Policy does not allow connections from this IP address."
    if "incorrect username or password" in lower:
        return "Authentication Failed: Incorrect username, password, or RSA private key."
    if "could not be resolved" in lower or "unknown host" in lower:
        return "Account Not Found: The Snowflake account identifier could not be reached. Check the account format."
    if "statement timeout" in lower or "timed out" in lower:
        return "Query Statement Timeout: The query took longer than the configured statement deadline."

    # Fallback to sanitized message with prefix
    return f"Snowflake Notice: {cleaned_msg[:200]}"
