"""
OpenMetadata-Inspired PII (Personally Identifiable Information) Guard for GetReport.
Detects, classifies, and masks sensitive data (Emails, Credit Cards with Luhn Checksums,
SSNs, Phone Numbers, IP Addresses) natively across dataset columns.
"""

from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
import polars as pl

logger = logging.getLogger(__name__)

# Precompiled regex patterns
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_REGEX = re.compile(r"^\+?[0-9]{1,3}?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$")
SSN_REGEX = re.compile(r"^(?:\d{3}-\d{2}-\d{4}|\d{9})$")
IPV4_REGEX = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
CREDIT_CARD_CANDIDATE = re.compile(r"^[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{1,7}$")


def verify_luhn_checksum(card_number_str: str) -> bool:
    """Verifies standard Luhn MOD-10 algorithm for valid credit cards."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


def mask_pii_value(val: str, pii_type: str) -> str:
    """Applies privacy-preserving redaction while keeping recognizable structure."""
    if not val:
        return val

    s = str(val).strip()

    if pii_type == "EMAIL":
        if "@" in s:
            parts = s.split("@", 1)
            user, domain = parts[0], parts[1]
            masked_user = (user[0] + "***") if len(user) > 0 else "***"
            return f"{masked_user}@{domain}"
        return "***@***"

    elif pii_type == "CREDIT_CARD":
        digits = [c for c in s if c.isdigit()]
        last4 = "".join(digits[-4:]) if len(digits) >= 4 else "xxxx"
        return f"****-****-****-{last4}"

    elif pii_type == "SSN":
        digits = [c for c in s if c.isdigit()]
        last4 = "".join(digits[-4:]) if len(digits) >= 4 else "xxxx"
        return f"***-**-{last4}"

    elif pii_type == "PHONE":
        digits = [c for c in s if c.isdigit()]
        last4 = "".join(digits[-4:]) if len(digits) >= 4 else "xxxx"
        return f"(***) ***-{last4}"

    elif pii_type == "IPV4":
        parts = s.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.***.***"
        return "***.***.***.***"

    return "[REDACTED]"


class PIIGuard:
    """
    High-performance PII scanner and redaction engine.
    """
    @classmethod
    def classify_value(cls, val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        if not s or len(s) > 120:
            return None

        if EMAIL_REGEX.match(s):
            return "EMAIL"
        if SSN_REGEX.match(s):
            return "SSN"
        if IPV4_REGEX.match(s):
            return "IPV4"
        if PHONE_REGEX.match(s):
            return "PHONE"
        if CREDIT_CARD_CANDIDATE.match(s) and verify_luhn_checksum(s):
            return "CREDIT_CARD"

        return None

    @classmethod
    def scan_dataframe(cls, df: pl.DataFrame, sample_limit: int = 300) -> Dict[str, Any]:
        """
        Scans all text and object columns for PII.
        Returns detailed compliance metrics and masked sample previews.
        """
        findings = {}
        total_pii_cols = 0

        for col in df.columns:
            s = df[col]
            if s.dtype not in (pl.String, pl.Categorical, pl.Object):
                continue

            non_nulls = s.drop_nulls()
            if non_nulls.len() == 0:
                continue

            sample = non_nulls.head(sample_limit).to_list()
            matched_types: Dict[str, int] = {}

            for val in sample:
                detected = cls.classify_value(val)
                if detected:
                    matched_types[detected] = matched_types.get(detected, 0) + 1

            if matched_types:
                # Primary detected type
                primary_type = max(matched_types, key=matched_types.get)
                count = matched_types[primary_type]
                ratio = count / len(sample)

                # Flag as PII if match ratio is >= 20% of sampled values
                if ratio >= 0.20 or count >= 5:
                    total_pii_cols += 1
                    sample_orig = str(sample[0])
                    sample_masked = mask_pii_value(sample_orig, primary_type)

                    findings[col] = {
                        "column": col,
                        "pii_type": primary_type,
                        "confidence": "HIGH" if ratio >= 0.50 else "MEDIUM",
                        "match_ratio": round(ratio, 4),
                        "matches_in_sample": count,
                        "sample_masked": sample_masked,
                        "action_required": "Mask or encrypt before sharing or production modeling"
                    }

        has_pii = total_pii_cols > 0

        return {
            "has_pii": has_pii,
            "total_pii_columns": total_pii_cols,
            "risk_level": "CRITICAL" if total_pii_cols >= 3 else ("HIGH" if total_pii_cols > 0 else "NONE"),
            "findings": findings
        }
