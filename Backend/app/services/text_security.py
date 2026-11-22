"""
text_security.py
~~~~~~~~~~~~~~~~
Scans dataset string columns for adversarial data security hazards:
- Formula Injection (CWE-1236 / CSV Injection)
- Cross-Site Scripting (XSS / HTML script tags)
- SQL Injection Strings
- Prompt Injection Phrases (LLM adversarial manipulation)
- Mojibake / Encoding Damage (UTF-8 decoded as Latin-1/Windows-1252)
"""
from __future__ import annotations

import logging
import re
from typing import Any
import polars as pl

logger = logging.getLogger(__name__)

# Patterns for each security vulnerability class
FORMULA_INJECTION_PATTERN = re.compile(r"^[\s]*[=+\-@|%]|HYPERLINK|cmd\||DDE", re.IGNORECASE)
XSS_PATTERN = re.compile(r"<script|<img[^>]+onerror|<svg[^>]+onload|javascript:|eval\(", re.IGNORECASE)
SQLI_PATTERN = re.compile(r"'\s*OR\s*['\"\d=]|;\s*--|--\s*$|UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO", re.IGNORECASE)
PROMPT_INJECTION_PATTERN = re.compile(
    r"ignore\s+(all\s+)?previous\s+instructions|system\s+prompt:|disregard\s+(all\s+)?prior|you\s+are\s+now\s+a",
    re.IGNORECASE
)
MOJIBAKE_PATTERN = re.compile(r"Ã[©¢£¥§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿À-ÿ]|â€[™œž]|â€”|â€™|\ufffd")


def scan_text_hazards(df: pl.DataFrame) -> dict[str, Any]:
    """
    Scan all string/text columns in the DataFrame for security threats and text integrity hazards.
    """
    findings: list[dict[str, Any]] = []
    text_cols = [c for c, t in df.schema.items() if t in (pl.Utf8, pl.String, pl.Object)]
    
    formula_count = 0
    formula_cols: list[str] = []
    
    xss_count = 0
    xss_cols: list[str] = []
    
    sqli_count = 0
    sqli_cols: list[str] = []
    
    prompt_count = 0
    prompt_cols: list[str] = []
    
    mojibake_count = 0
    mojibake_cols: list[str] = []
    
    for c in text_cols:
        series = df[c].drop_nulls()
        if series.len() == 0:
            continue
            
        sample = series.head(5000).to_list()
        
        # 1. Formula Injection (CWE-1236)
        c_formula = sum(1 for s in sample if FORMULA_INJECTION_PATTERN.search(str(s)))
        if c_formula > 0:
            extrapolated = int(round(c_formula / len(sample) * series.len()))
            formula_count += extrapolated
            formula_cols.append(c)
            
        # 2. XSS / Script Payloads
        c_xss = sum(1 for s in sample if XSS_PATTERN.search(str(s)))
        if c_xss > 0:
            extrapolated = int(round(c_xss / len(sample) * series.len()))
            xss_count += extrapolated
            xss_cols.append(c)
            
        # 3. SQL Injection Strings
        c_sqli = sum(1 for s in sample if SQLI_PATTERN.search(str(s)))
        if c_sqli > 0:
            extrapolated = int(round(c_sqli / len(sample) * series.len()))
            sqli_count += extrapolated
            sqli_cols.append(c)
            
        # 4. Prompt Injection Phrases
        c_prompt = sum(1 for s in sample if PROMPT_INJECTION_PATTERN.search(str(s)))
        if c_prompt > 0:
            extrapolated = int(round(c_prompt / len(sample) * series.len()))
            prompt_count += extrapolated
            prompt_cols.append(c)
            
        # 5. Mojibake / Encoding Damage
        c_mojibake = sum(1 for s in sample if MOJIBAKE_PATTERN.search(str(s)))
        if c_mojibake > 0:
            extrapolated = int(round(c_mojibake / len(sample) * series.len()))
            mojibake_count += extrapolated
            mojibake_cols.append(c)

    if formula_count > 0:
        findings.append({
            "hazard": "Formula Injection (CWE-1236)",
            "count": formula_count,
            "columns": formula_cols,
            "severity": "CRITICAL",
            "description": "Formulas starting with =, +, -, @, or pipe that execute malicious commands in Excel/CSV exports."
        })
        
    if xss_count > 0:
        findings.append({
            "hazard": "XSS / Script Payloads",
            "count": xss_count,
            "columns": xss_cols,
            "severity": "HIGH",
            "description": "Raw HTML tags (<script>, onload, javascript:) capable of executing stored script attacks in web UI dashboards."
        })
        
    if sqli_count > 0:
        findings.append({
            "hazard": "SQL Injection Strings",
            "count": sqli_count,
            "columns": sqli_cols,
            "severity": "HIGH",
            "description": "SQL injection fragments (' OR 1=1, UNION SELECT, DROP TABLE) embedded in unstructured notes."
        })
        
    if prompt_count > 0:
        findings.append({
            "hazard": "Prompt-Injection Phrases",
            "count": prompt_count,
            "columns": prompt_cols,
            "severity": "HIGH",
            "description": "Adversarial instructions ('ignore previous instructions') designed to hijack downstream LLM analyzers."
        })
        
    if mojibake_count > 0:
        findings.append({
            "hazard": "Mojibake / Encoding Damage",
            "count": mojibake_count,
            "columns": mojibake_cols,
            "severity": "MEDIUM",
            "description": "Corrupted multi-byte UTF-8 character sequences resulting from double-encoding or Windows-1252 misinterpretation."
        })

    return {
        "has_hazards": len(findings) > 0,
        "total_hazards_count": sum(f["count"] for f in findings),
        "findings": findings
    }
