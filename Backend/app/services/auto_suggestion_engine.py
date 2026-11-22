from __future__ import annotations
import datetime as _dt
import functools
import hashlib
import hmac
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

log = logging.getLogger("auto_suggestion_engine")

__version__ = "2.1.0"

__all__ = [
    "EngineConfig", "analyze", "clean", "generate_comprehensive_suggestions",
    "build_remediation_plan", "apply_suggestion", "apply_suggestions",
    "profile_columns", "to_json", "is_identifier_column", "compute_quality_scores",
]

# =============================================================================
# Constants
# =============================================================================

MASKED_NULL_TOKENS = {
    "n/a", "na", "null", "none", "nil", "-", "--", "---", "?", "??",
    "#n/a", "nan", "missing", "undefined", "n.a.", "n.a", "none.", "null.", ".",
    "", "not available", "not applicable", "#null!", "#value!", "#ref!",
    "#div/0!", "(blank)", "blank", "empty", "void", "nat", "<na>", "<null>",
    "tbd", "n/d", "nd",
}
_MASKED_LIST = sorted(MASKED_NULL_TOKENS)

# Kept for backwards compatibility with the v1 module.
MASKED_NULL_REGEX = re.compile(
    r"^(?:\s*|\s*N/?A\s*|\s*NULL\s*|\s*NONE\s*|\s*NIL\s*|\s*-+\s*|\s*\?+\s*"
    r"|\s*#N/A\s*|\s*NAN\s*|\s*\.\s*|\s*MISSING\s*|\s*UNDEFINED\s*)$",
    re.IGNORECASE,
)
DIRTY_NUMERIC_REGEX = re.compile(r"^\s*[$€£¥₹]?\s*-?[\d,]+(?:\.\d+)?%?\s*$")

ID_INDICATORS = [
    "id", "uuid", "guid", "pk", "code", "ssn", "phone", "email", "key",
    "card", "credit_card", "creditcard", "cc_num", "zip", "zip_code", "zipcode",
    "postal", "postal_code", "pin", "pincode", "api_key", "apikey", "token", "hash", "secret"
]

# Tokens that can be legitimate codes (Namibia = "NA", "None" as a category, ...).
AMBIGUOUS_NULL_TOKENS = {"na", "n.a.", "n.a", "none", "nd", "nat"}
AMBIGUOUS_COLUMN_HINT = re.compile(
    r"country|code|iso|region|abbr|state|province|lang|locale|continent|prefix|suffix|grade|class")

TRUE_TOKENS = {"true", "t", "yes", "y", "1", "on", "enabled", "active"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "off", "disabled", "inactive"}
_TRUE_LIST, _FALSE_LIST = sorted(TRUE_TOKENS), sorted(FALSE_TOKENS)

# Permissive-but-safe "looks like a number" pattern (ISO dates do NOT match).
NUMERIC_LIKE = (
    r"^\s*\(?\s*[+-]?\s*(?i:(?:rs\.?|inr|usd|eur|gbp)\s*)?[$€£¥₹]?\s*[+-]?"
    r"\d[\d,.]*\s*%?\s*\)?\s*-?\s*$"
)
PLAIN_NUMERIC = r"^\s*[+-]?\d+(?:\.\d+)?\s*$"

EMAIL_RE = r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
MOJIBAKE_RE = r"Ã[\x{0080}-\x{00BF}]|â€|Â[\x{00A0}-\x{00BF}]"
DATE_LIKE_RE = (
    r"^\s*(?:\d{1,4}[-/. ]\d{1,2}[-/. ]\d{1,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4})"
)

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d",
    "%d.%m.%Y", "%m.%d.%Y", "%b %d, %Y", "%d %b %Y", "%B %d, %Y", "%d %B %Y", "%d-%b-%Y",
    "%d/%m/%y", "%m/%d/%y",
]
DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
]
# (day-first, month-first) pairs used for day/month ambiguity checks.
_AMBIGUOUS_PAIRS = [("%d/%m/%Y", "%m/%d/%Y"), ("%d-%m-%Y", "%m-%d-%Y"),
                    ("%d.%m.%Y", "%m.%d.%Y"), ("%d/%m/%y", "%m/%d/%y")]

# (name regex, min, max, label, exclusion regex)
RANGE_RULES: List[Tuple[re.Pattern, Optional[float], Optional[float], str, Optional[re.Pattern]]] = [
    (re.compile(r"(^|_)age($|_)"), 0, 120, "human age", re.compile(r"stage|usage|image")),
    (re.compile(r"(^|_)(pct|percent|percentage)($|_)"), 0, 100, "percentage",
     re.compile(r"growth|change|delta|diff|increase|return|yoy|mom")),
    (re.compile(r"(^|_)(price|cost|amount|salary|revenue|quantity|qty|count|weight|height|"
                r"distance|duration|fare|income|sales)($|_)"), 0, None, "non-negative measure",
     re.compile(r"change|delta|diff|net|growth|adj")),
    (re.compile(r"(^|_)year($|_)"), 1900, 2100, "calendar year", None),
]

SEVERITY_WEIGHT = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
RISK_RANK = {"low": 0, "medium": 1, "high": 2}

PILLAR_WEIGHTS = {
    "Completeness": 0.25, "Validity": 0.25, "Consistency": 0.15,
    "Uniqueness": 0.15, "Stability": 0.20,
}

# Execution order. Detectors tag suggestions with these stages.
STAGE = {
    "masked_nulls": 1, "text_hygiene": 2, "categories": 3, "type_coercion": 4,
    "validity": 5, "dedupe": 6, "indicators": 7, "imputation": 8,
    "outliers": 9, "transforms": 10, "pruning": 11, "privacy": 12, "naming": 13,
}
PROJECT_GROUP_A = {1, 2, 3, 4}   # simulated before validity / downstream detectors
PROJECT_GROUP_B = {3, 5}

# Structural problems (constant / redundant columns) affect schema, not cells.
STRUCTURAL_WEIGHT = 0.25

_DERIVED_SUFFIXES = ("__was_missing", "__outlier", "__log1p")

_STR = getattr(pl, "String", pl.Utf8)
_INT_TYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
_FLOAT_TYPES = (pl.Float32, pl.Float64)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class EngineConfig:
    sample_size: int = 20_000           # max rows used for expensive statistics
    seed: int = 42
    min_rows_for_stats: int = 10
    numeric_parse_ratio: float = 0.80
    date_parse_ratio: float = 0.85
    bool_parse_ratio: float = 0.95
    quasi_constant_ratio: float = 0.995
    high_missing_drop_ratio: float = 0.70
    outlier_iqr_k: float = 1.5
    outlier_extreme_k: float = 3.0
    outlier_systemic_ratio: float = 0.10  # >10% "outliers" => natural heavy tail, flag don't cap
    skew_symmetric: float = 0.5
    skew_transform: float = 1.5
    redundancy_corr: float = 0.98
    group_impute_adj_eta2: float = 0.30
    max_fuzzy_uniques: int = 300
    fuzzy_threshold: float = 0.90
    max_corr_columns: int = 60
    auto_project_confidence: float = 90.0  # min confidence for a fix to be simulated downstream
    pii_salt: str = ""                     # set a secret to get keyed (HMAC) pseudonyms
    enable_pii: bool = True
    enable_fuzzy: bool = True
    enable_column_naming: bool = True


# =============================================================================
# Small helpers
# =============================================================================

def _is_str(dt) -> bool:
    return dt == _STR

def _is_int(dt) -> bool:
    return dt in _INT_TYPES

def _is_float(dt) -> bool:
    return dt in _FLOAT_TYPES

def _is_num(dt) -> bool:
    return _is_int(dt) or _is_float(dt)

def _is_temporal(dt) -> bool:
    return dt == pl.Date or dt == pl.Datetime

def _is_nested(dt) -> bool:
    try:
        return bool(dt.is_nested())
    except Exception:
        return dt in (pl.List, pl.Struct, pl.Object)

def _is_simple(dt) -> bool:
    """Types that support value_counts / unique / equality safely."""
    return not _is_nested(dt) and dt != pl.Object and dt != pl.Binary

def _snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    s = re.sub(r"\W+", "_", s).strip("_").lower()      # \W is unicode aware
    s = re.sub(r"_+", "_", s)
    return s or "col"

def is_identifier_column(col_name: str) -> bool:
    """Detect if a column acts as an identifier / unique key (camelCase aware)."""
    col_lower = _snake(col_name)
    return any(
        col_lower == ind or col_lower.startswith(f"{ind}_")
        or col_lower.endswith(f"_{ind}") or f"_{ind}_" in col_lower
        for ind in ID_INDICATORS
    )

def _is_pk_name(col_name: str) -> bool:
    n = _snake(col_name)
    return n in {"id", "uuid", "guid", "pk"} or n.endswith(("_id", "_uuid", "_guid", "_pk")) or n.startswith("id_")

def _severity(pct: float, *, critical: float = 50.0, high: float = 15.0, medium: float = 3.0) -> str:
    if pct >= critical:
        return "critical"
    if pct >= high:
        return "high"
    if pct >= medium:
        return "medium"
    return "low"

def wilson_lower(k: float, n: float, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a proportion."""
    if n <= 0:
        return 0.0
    p = min(1.0, max(0.0, k / n))
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - adj) / denom)

def _confidence(base: float, *, k: Optional[float] = None, n: Optional[float] = None) -> float:
    """Evidence-adjusted confidence: penalises small samples and weak match ratios."""
    c = base
    if n is not None and n < 30:
        c -= 4.0 * max(0.0, 1.0 - n / 30.0)
    if k is not None and n is not None and n > 0 and k < n:
        c -= (1.0 - wilson_lower(k, n)) * 10.0
    return round(max(40.0, min(99.9, c)), 1)

def _fmt_gain(gain: float, pillar: str) -> str:
    return f"+{gain:.2f}% {pillar}" if gain < 0.1 else f"+{gain:.1f}% {pillar}"

def _j(v: Any) -> Any:
    """Make a value JSON-safe."""
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if (math.isnan(v) or math.isinf(v)) else float(v)
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    return str(v)

def _jl(vals: Optional[Sequence[Any]]) -> List[Any]:
    return [_j(v) for v in (vals or [])]

def _eval(s: pl.Series, fn: Callable[[pl.Expr], pl.Expr]) -> pl.Series:
    """Evaluate an expression builder against a Series."""
    return s.to_frame("v").select(fn(pl.col("v"))).to_series()

def _sample_series(s: pl.Series, cfg: EngineConfig) -> pl.Series:
    return s.sample(n=cfg.sample_size, seed=cfg.seed) if s.len() > cfg.sample_size else s

def _clean_num(s: pl.Series) -> pl.Series:
    s = s.drop_nulls()
    if _is_float(s.dtype):
        s = s.filter(s.is_finite())
    return s

def _safe_skew(s: pl.Series) -> Optional[float]:
    try:
        v = s.skew()
        return None if v is None or math.isnan(v) else float(v)
    except Exception:
        return None

def _nonmasked_strings(s: pl.Series, strip: bool = True) -> pl.Series:
    """Non-null, non-placeholder strings (optionally stripped)."""
    nn = s.drop_nulls()
    if nn.len() == 0:
        return nn
    norm = nn.str.strip_chars().str.to_lowercase()
    nn = nn.filter(~norm.is_in(_MASKED_LIST))
    return nn.str.strip_chars() if strip else nn

def _to_polars(df: Any) -> pl.DataFrame:
    """Accept Polars (eager / lazy) and pandas frames."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if hasattr(df, "to_numpy") and hasattr(df, "columns") and hasattr(df, "dtypes"):
        return pl.from_pandas(df)
    raise TypeError(f"Unsupported frame type: {type(df).__name__}")

def _hash_value(v: str, salt: str) -> str:
    if salt:
        return hmac.new(salt.encode(), v.encode(), hashlib.sha256).hexdigest()[:16]
    return hashlib.sha256(v.encode()).hexdigest()[:16]


# =============================================================================
# Expression builders shared by detectors AND executors
# =============================================================================

def _clean_text_expr(c: pl.Expr, collapse: bool = True) -> pl.Expr:
    e = (c.str.replace_all(r"[\x{200B}-\x{200D}\x{FEFF}\x{2060}]", "")
          .str.replace_all(r"[\x00-\x08\x0E-\x1F\x7F]", ""))
    if collapse:
        e = e.str.replace_all(r"\s+", " ")
    return e.str.strip_chars()

def _numeric_parse_expr(c: pl.Expr, style: str = "us") -> pl.Expr:
    s = c.str.strip_chars().str.replace_all(r"(?i)\b(?:rs|inr|usd|eur|gbp)\b\.?", "")
    neg = (s.str.contains(r"^\s*\(.*\)\s*$") | s.str.contains(r"^[^\d]*-") | s.str.contains(r"-\s*$"))
    core = s.str.replace_all(r"[^0-9.,]", "")
    if style == "eu":
        core = core.str.replace_all(r"\.", "").str.replace(",", ".")
    else:
        core = core.str.replace_all(",", "")
    val = core.cast(pl.Float64, strict=False)
    return pl.when(neg).then(-val).otherwise(val)

def _infer_decimal_style(vals: pl.Series) -> Tuple[str, bool]:
    """Returns (style, ambiguous)."""
    eu = int(vals.str.contains(r"\d,\d{1,2}\s*[%)]?\s*-?$|\d\.\d{3},\d").sum() or 0)
    us = int(vals.str.contains(r"\d\.\d{1,2}\s*[%)]?\s*-?$|\d,\d{3}\.\d").sum() or 0)
    if eu == 0 and us == 0:
        return "us", True
    return ("eu" if eu > us else "us"), (min(eu, us) / max(eu, us, 1) > 0.2)

def _bool_expr(c: pl.Expr) -> pl.Expr:
    s = c.str.strip_chars().str.to_lowercase()
    return (pl.when(s.is_in(_TRUE_LIST)).then(True)
              .when(s.is_in(_FALSE_LIST)).then(False).otherwise(None))

def _date_expr(c: pl.Expr, kind: str, formats: Sequence[Optional[str]]) -> pl.Expr:
    c = c.str.strip_chars()
    parts = [c.str.to_date(f, strict=False) if kind == "date" else c.str.to_datetime(f, strict=False)
             for f in formats]
    return pl.coalesce(parts)

def _fix_mojibake_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = v.encode(enc).decode("utf-8")
            if fixed != v:
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return v

def _luhn_ok(num: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", num)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# =============================================================================
# Detection context
# =============================================================================

class _Context:
    """Holds the raw frame, the projected working frame and the suggestion factory."""

    def __init__(self, orig: pl.DataFrame, work: pl.DataFrame, cfg: EngineConfig):
        self.orig = orig
        self.work = work
        self.cfg = cfg
        self.n_rows = orig.height
        self.n_cols = orig.width
        self.total_cells = max(1, orig.height * orig.width)
        self.warnings: List[str] = []
        self._cats: Optional[List[str]] = None

    def cat_candidates(self) -> List[str]:
        """Low-cardinality grouping columns useful for group-wise imputation."""
        if self._cats is None:
            cands = []
            for col, dt in self.work.schema.items():
                if not (_is_str(dt) or dt == pl.Boolean or dt == pl.Categorical):
                    continue
                if is_identifier_column(col):
                    continue
                s = self.work[col]
                if self.n_rows and s.null_count() / self.n_rows > 0.3:
                    continue
                nu = s.n_unique()
                if 2 <= nu <= 30:
                    cands.append((nu, col))
            self._cats = [c for _, c in sorted(cands)[:8]]
        return self._cats

    def make(self, *, type: str, column: str, action: str, count: int, severity: str,
             suggestion: str, label: str, rationale: str, confidence: float,
             pillar: str, stage: int, affected_cells: float = 0.0, resolution: float = 0.9,
             before: Optional[Sequence[Any]] = None, after: Optional[Sequence[Any]] = None,
             params: Optional[Dict[str, Any]] = None, advisory: bool = False,
             risk: str = "low", reversible: bool = True,
             evidence: Optional[Dict[str, Any]] = None,
             impact_text: Optional[str] = None) -> Dict[str, Any]:
        penalty = max(0.0, float(affected_cells))
        gain = penalty * resolution / self.total_cells * 100.0
        return {
            "id": f"{type}:{column}:{action}",
            "type": type,
            "column": column,
            "count": int(count),
            "severity": severity,
            "suggestion": suggestion,
            "label": label,
            "action": action,
            "rationale": rationale,
            "confidence_score": round(float(confidence), 1),
            "target_pillar": pillar,
            "impact_estimate": impact_text or _fmt_gain(gain, pillar),
            "impact_dataset_pct": round(gain, 3),
            "before_sample": _jl(before),
            "after_sample": _jl(after),
            "stage": stage,
            "params": _j(params or {}),
            "advisory": advisory,
            "risk": risk,
            "reversible": reversible,
            "evidence": {k: _j(v) for k, v in (evidence or {}).items()},
            "affected_cells": int(penalty),
            "resolution_factor": resolution,
            "depends_on": [],
            "conflicts_with": [],
        }


def _per_column(fn: Callable[[_Context, str], List[Dict[str, Any]]]):
    """Run a column-level detector with per-column fault isolation."""
    @functools.wraps(fn)
    def wrapper(ctx: _Context) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for col in list(ctx.work.columns):
            try:
                out.extend(fn(ctx, col) or [])
            except Exception as e:
                msg = f"{fn.__name__}[{col}] failed: {type(e).__name__}: {e}"
                log.warning(msg)
                ctx.warnings.append(msg)
        return out
    return wrapper


# =============================================================================
# PHASE A detectors: NaN, masked nulls, text hygiene, categories, type inference
# =============================================================================

@_per_column
def detect_nan_inf(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Float NaN / +-inf are not nulls in Polars; they silently poison means and fills."""
    s = ctx.work[col]
    if not _is_float(s.dtype):
        return []
    nan = int(s.is_nan().sum() or 0)
    inf = int(s.is_infinite().sum() or 0)
    cnt = nan + inf
    if cnt == 0:
        return []
    pct = cnt / ctx.n_rows * 100
    return [ctx.make(
        type="nan_inf_values", column=col, action="nan_to_null", count=cnt,
        severity=_severity(pct, critical=40, high=10, medium=1),
        suggestion=f"Convert NaN/Inf to null ({cnt})", label="Convert NaN / Inf -> null",
        rationale=f"{nan} NaN and {inf} infinite value(s) are not recognised as missing by Polars; "
                  f"they turn means, sums and imputations into NaN. Converting them to real nulls lets "
                  f"every downstream step treat them correctly.",
        confidence=99.0, pillar="Completeness", stage=STAGE["masked_nulls"],
        affected_cells=cnt, resolution=1.0, before=[float("nan")] * min(cnt, 2) if nan else [float("inf")],
        after=[None] * (min(cnt, 2) if nan else 1), evidence={"nan": nan, "inf": inf},
    )]


@_per_column
def detect_masked_nulls(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    s = ctx.work[col]
    if not _is_str(s.dtype):
        return []
    nn = s.drop_nulls()
    if nn.len() == 0:
        return []
    norm = nn.str.strip_chars().str.to_lowercase()
    mask = norm.is_in(_MASKED_LIST)
    cnt = int(mask.sum() or 0)
    if cnt == 0:
        return []
    vc = norm.filter(mask).value_counts(sort=True)
    found = [(str(a), int(b)) for a, b in vc.rows()]
    found_tokens = [t for t, _ in found]
    samples = nn.filter(mask).unique(maintain_order=True).head(3).to_list()
    pct = cnt / ctx.n_rows * 100
    ambiguous = set(found_tokens) <= AMBIGUOUS_NULL_TOKENS and bool(AMBIGUOUS_COLUMN_HINT.search(col.lower()))
    conf = 72.0 if ambiguous else _confidence(98.5, n=nn.len())
    return [ctx.make(
        type="masked_nulls", column=col, action="clean_masked_nulls", count=cnt,
        severity=_severity(pct, critical=40, high=10, medium=1),
        suggestion=f"Clean Masked Nulls ({cnt} placeholders)",
        label="Clean Masked Nulls (-> null)",
        rationale=(f"Found {cnt} placeholder string(s) ({pct:.1f}%) masquerading as valid text "
                   f"({', '.join(repr(t) for t in found_tokens[:4])}); converting to real nulls prevents "
                   f"distorted downstream metrics."
                   + (" Warning: 'NA'/'None' may be a legitimate code in this column." if ambiguous else "")),
        confidence=conf, pillar="Completeness", stage=STAGE["masked_nulls"],
        affected_cells=cnt, resolution=1.0, before=samples, after=[None] * len(samples),
        params={"tokens": found_tokens}, risk="medium" if ambiguous else "low",
        evidence={"token_counts": dict(found), "pct": round(pct, 2)},
    )]


@_per_column
def detect_text_hygiene(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    s = ctx.work[col]
    if not _is_str(s.dtype):
        return []
    nn = _nonmasked_strings(s, strip=False)
    n = nn.len()
    if n < 5:
        return []
    out: List[Dict[str, Any]] = []
    avg_len = float(nn.str.len_chars().mean() or 0)
    collapse = avg_len <= 60                       # never flatten newlines in long free text
    cleaned = _eval(nn, lambda c: _clean_text_expr(c, collapse))
    changed = cleaned != nn
    cnt = int(changed.sum() or 0)
    if cnt > 0:
        pct = cnt / ctx.n_rows * 100
        pairs = (pl.DataFrame({"b": nn.filter(changed), "a": cleaned.filter(changed)})
                 .unique(subset="b", maintain_order=True).head(3))
        out.append(ctx.make(
            type="text_whitespace", column=col, action="clean_text", count=cnt,
            severity="medium" if pct > 30 else "low",
            suggestion=f"Clean Text Whitespace ({cnt} untrimmed)",
            label="Clean Text (Trim / Collapse / Strip invisibles)",
            rationale=(f"{cnt} values ({pct:.1f}%) contain leading/trailing or repeated whitespace, "
                       f"zero-width or control characters; cleaning prevents fragmented groupings in "
                       f"pivots, joins and charts."),
            confidence=_confidence(94.0, n=n), pillar="Consistency", stage=STAGE["text_hygiene"],
            affected_cells=cnt, resolution=1.0,
            before=[repr(x) for x in pairs["b"].to_list()], after=[repr(x) for x in pairs["a"].to_list()],
            params={"collapse": collapse}, evidence={"pct": round(pct, 2)},
        ))

    moji = nn.str.contains(MOJIBAKE_RE)
    mcnt = int(moji.sum() or 0)
    if mcnt > 0:
        flagged = nn.filter(moji).unique(maintain_order=True).head(3).to_list()
        fixed = [_fix_mojibake_str(v) for v in flagged]
        if sum(1 for a, b in zip(flagged, fixed) if a != b) >= max(1, len(flagged) // 2):
            out.append(ctx.make(
                type="mojibake", column=col, action="fix_mojibake", count=mcnt,
                severity="medium", suggestion=f"Repair Garbled Encoding ({mcnt} values)",
                label="Fix Mojibake (UTF-8 mis-decoded)",
                rationale=f"{mcnt} values show UTF-8 text decoded as Latin-1/CP1252 (e.g. 'Ã©'); "
                          f"re-decoding restores the original characters.",
                confidence=85.0, pillar="Validity", stage=STAGE["text_hygiene"],
                affected_cells=mcnt, resolution=0.95, before=flagged, after=fixed, risk="medium",
            ))
    return out


@_per_column
def detect_category_inconsistency(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Casing / spacing variants (high confidence) and fuzzy near-duplicates (lower)."""
    cfg = ctx.cfg
    s = ctx.work[col]
    if not _is_str(s.dtype) or is_identifier_column(col):
        return []
    nn = _nonmasked_strings(s)
    n = nn.len()
    if n < 20:
        return []
    nu = nn.n_unique()
    if nu < 2 or nu > 2000 or nu / n > 0.5 or float(nn.str.len_chars().mean() or 0) > 40:
        return []

    out: List[Dict[str, Any]] = []
    nn_eval = nn if nn.len() <= 30000 else nn.head(30000)
    raw_counts = [(str(a), int(b)) for a, b in nn_eval.value_counts(sort=True).rows()]
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for raw, c in raw_counts:
        groups.setdefault(re.sub(r"\s+", " ", raw).casefold(), []).append((raw, c))

    def canon_of(members: List[Tuple[str, int]]) -> str:
        return max(members, key=lambda rc: (rc[1], rc[0].istitle(), rc[0]))[0]

    # ---- casing / spacing variants -----------------------------------------
    case_map: Dict[str, str] = {}
    affected = 0
    for members in groups.values():
        if len(members) > 1:
            canon = canon_of(members)
            for raw, c in members:
                if raw != canon:
                    case_map[raw] = canon
                    affected += c
    if case_map:
        items = list(case_map.items())[:3]
        out.append(ctx.make(
            type="inconsistent_casing", column=col, action="standardize_categories", count=affected,
            severity=_severity(affected / ctx.n_rows * 100, critical=40, high=15, medium=2),
            suggestion=f"Standardize Category Casing ({len(case_map)} variants)",
            label="Standardize Categories (case / spacing)",
            rationale=(f"{len(case_map)} spelling variants differ only by case or spacing "
                       f"(e.g. {items[0][0]!r} vs {items[0][1]!r}) and split {affected} rows across "
                       f"duplicate groups."),
            confidence=_confidence(96.0, n=n), pillar="Consistency", stage=STAGE["categories"],
            affected_cells=affected, resolution=1.0,
            before=[a for a, _ in items], after=[b for _, b in items],
            params={"mapping": case_map}, evidence={"variants": len(case_map)},
        ))

    # ---- fuzzy near-duplicates ---------------------------------------------
    if not cfg.enable_fuzzy or len(groups) > cfg.max_fuzzy_uniques:
        return out
    info = []
    for k1, members in groups.items():
        info.append((sum(c for _, c in members), k1, re.sub(r"[^0-9a-z]+", "", k1), canon_of(members)))
    info.sort(reverse=True)
    heads: List[Tuple[str, str, str]] = []
    fuzzy_map: Dict[str, str] = {}
    f_affected = 0
    min_ratio = 1.0
    for _total, k1, k2, canon in info:
        target = None
        for _hk1, hk2, hcanon in heads:
            if k2 == hk2:
                ratio = 1.0
            elif len(k2) >= 5 and len(hk2) >= 5 and re.findall(r"\d+", k2) == re.findall(r"\d+", hk2):
                sm = SequenceMatcher(None, k2, hk2)
                if sm.real_quick_ratio() < cfg.fuzzy_threshold or sm.quick_ratio() < cfg.fuzzy_threshold:
                    continue
                ratio = sm.ratio()
            else:
                continue
            if ratio >= cfg.fuzzy_threshold:
                target, min_ratio = hcanon, min(min_ratio, ratio)
                break
        if target is None:
            heads.append((k1, k2, canon))
        else:
            for raw, c in groups[k1]:
                if raw != target:
                    fuzzy_map[raw] = target
                    f_affected += c
    norm_ws = lambda x: re.sub(r"\s+", " ", x).casefold()
    fuzzy_map = {a: b for a, b in fuzzy_map.items() if norm_ws(a) != norm_ws(b)}
    if fuzzy_map:
        items = list(fuzzy_map.items())[:3]
        out.append(ctx.make(
            type="near_duplicate_categories", column=col, action="standardize_categories",
            count=f_affected, severity="medium",
            suggestion=f"Merge Near-Duplicate Categories ({len(fuzzy_map)} variants)",
            label="Merge Near-Duplicate Categories (fuzzy)",
            rationale=(f"{len(fuzzy_map)} labels are near-identical to a more frequent label "
                       f"(e.g. {items[0][0]!r} -> {items[0][1]!r}, similarity >= {min_ratio:.2f}); "
                       f"likely typos or punctuation variants. Review before applying."),
            confidence=80.0 if min_ratio >= 0.95 else 74.0, pillar="Consistency",
            stage=STAGE["categories"], affected_cells=f_affected, resolution=0.9,
            before=[a for a, _ in items], after=[b for _, b in items],
            params={"mapping": fuzzy_map}, risk="medium",
            evidence={"min_similarity": round(min_ratio, 3)},
        ))
    return out


# ---- type inference helpers --------------------------------------------------

def _infer_boolean(ctx, col, vals, n, n_full):
    norm = vals.str.to_lowercase()
    uniq_norm = norm.unique().to_list()
    if len(uniq_norm) > 8:
        return None
    is_t, is_f = norm.is_in(_TRUE_LIST), norm.is_in(_FALSE_LIST)
    k = int((is_t | is_f).sum() or 0)
    if k / n < ctx.cfg.bool_parse_ratio or not (bool(is_t.any()) and bool(is_f.any())):
        return None
    zero_one_only = set(uniq_norm) <= {"0", "1"}
    uniq = vals.unique(maintain_order=True).head(3).to_list()
    return ctx.make(
        type="boolean_text", column=col, action="cast_boolean", count=n_full,
        severity="medium", suggestion="Cast to Boolean", label="Cast Text -> Boolean",
        rationale=(f"All {len(uniq_norm)} distinct values are boolean tokens "
                   f"({', '.join(sorted(map(str, uniq_norm))[:6])}); a true Boolean enables "
                   f"filtering and cuts memory."
                   + (" Values are only 0/1, which may equally be a numeric flag; verify." if zero_one_only else "")),
        confidence=min(_confidence(95.0, k=k, n=n), 85.0) if zero_one_only else _confidence(95.0, k=k, n=n),
        pillar="Validity", stage=STAGE["type_coercion"], affected_cells=n_full, resolution=0.9,
        before=uniq, after=_eval(pl.Series(uniq, dtype=_STR), _bool_expr).to_list(),
        risk="medium" if zero_one_only else "low",
    )


def _infer_numeric(ctx, col, vals, n, n_full):
    """Returns (handled, suggestion). handled=True stops further inference for the column."""
    cfg = ctx.cfg
    pat_ok = vals.str.contains(NUMERIC_LIKE)
    if int(pat_ok.sum() or 0) / n < cfg.numeric_parse_ratio:
        return False, None
    style, ambiguous = _infer_decimal_style(vals.filter(pat_ok))
    parsed = _eval(vals, lambda c: _numeric_parse_expr(c, style))
    k = int(parsed.is_not_null().sum() or 0)
    if k / n < cfg.numeric_parse_ratio:
        return False, None
    plain = vals.str.contains(PLAIN_NUMERIC)
    plain_ratio = int(plain.sum() or 0) / n
    lead0 = int(vals.str.contains(r"^\s*0\d").sum() or 0) / n
    if plain_ratio >= 0.98 and lead0 >= 0.05:
        return True, ctx.make(
            type="identifier_numeric_text", column=col, action="keep_as_text", count=n_full,
            severity="low", suggestion="Keep as Text (leading zeros)",
            label="Keep as Text (preserve leading zeros)",
            rationale=f"{lead0 * 100:.0f}% of values start with 0 (zip / phone / account style); "
                      f"casting to a number would silently destroy information.",
            confidence=92.0, pillar="Validity", stage=STAGE["type_coercion"],
            advisory=True, impact_text="Prevents data loss",
            before=vals.filter(vals.str.contains(r"^\s*0\d")).head(3).to_list(),
            after=["(unchanged)"] * 3,
        )
    if plain_ratio >= 0.98 and is_identifier_column(col):
        return True, None
    formatted = plain_ratio < 0.98
    preview_src = (vals.filter(~plain) if formatted else vals).unique(maintain_order=True).head(3)
    preview_vals = _eval(preview_src, lambda c: _numeric_parse_expr(c, style)).to_list()
    pct_flag = int(vals.str.contains("%").sum() or 0) / n >= 0.5
    symbols = sorted({ch for ch in "".join(vals.head(500).to_list()) if ch in "$€£¥₹%"})
    good = parsed.drop_nulls()
    integral = bool(good.len() > 0 and (good == good.round(0)).all() and float(good.abs().max()) < 9e15)
    total_est = int(round(k / n * n_full))
    fmt_cnt = int(round((1 - plain_ratio) * n_full))
    conf = _confidence(96.0 if formatted else 93.0, k=k, n=n)
    if ambiguous:
        conf = round(conf - 8, 1)
    return True, ctx.make(
        type="dirty_numeric" if formatted else "numeric_as_text", column=col,
        action="coerce_numeric", count=total_est, severity="high" if formatted else "medium",
        suggestion=(f"Coerce to Numeric ({fmt_cnt / ctx.n_rows * 100:.1f}% formatted)"
                    if formatted else "Cast Text -> Number"),
        label=("Coerce to Numeric (Strip currency & format)" if formatted else "Cast Text -> Number"),
        rationale=(f"{total_est} values parse cleanly as numbers"
                   + (f" after stripping {' '.join(symbols)} / separators" if symbols else "")
                   + f" (decimal style: {style.upper()}"
                   + (", ambiguous - verify" if ambiguous else "") + ")."
                   + (" Values are percentages (kept in percent points)." if pct_flag else "")
                   + " Unlocks aggregation, statistics and charting."),
        confidence=conf, pillar="Validity", stage=STAGE["type_coercion"],
        affected_cells=fmt_cnt if formatted else total_est, resolution=1.0,
        before=preview_src.to_list(), after=preview_vals,
        params={"style": style, "integral": integral, "percent_to_fraction": False},
        risk="medium" if ambiguous else "low",
        evidence={"parse_ratio": round(k / n, 4), "plain_ratio": round(plain_ratio, 4),
                  "percent": pct_flag, "style": style},
    )


def _infer_date(ctx, col, vals, n, n_full):
    cfg = ctx.cfg
    if int(vals.str.contains(DATE_LIKE_RE).sum() or 0) / n < 0.80:
        return None
    dsample = vals.head(2_000)
    dn = dsample.len()
    cov: Dict[Tuple[str, Optional[str]], pl.Series] = {}
    for kind, fmts in (("date", DATE_FORMATS), ("datetime", DATETIME_FORMATS + [None])):
        for f in fmts:
            try:
                p = (dsample.str.to_date(f, strict=False) if kind == "date"
                     else dsample.str.to_datetime(f, strict=False))
                ok = p.is_not_null()
                if int(ok.sum() or 0) / dn > 0.01:
                    cov[(kind, f)] = ok
            except Exception:
                continue
    if not cov:
        return None
    best_date = max((int(m.sum()) for (k_, _), m in cov.items() if k_ == "date"), default=0)
    best_dt = max((int(m.sum()) for (k_, _), m in cov.items() if k_ == "datetime"), default=0)
    kind = "datetime" if best_dt > best_date else "date"
    chosen = sorted(((f, int(m.sum())) for (k_, f), m in cov.items() if k_ == kind), key=lambda t: -t[1])
    union = None
    used: List[Optional[str]] = []
    for f, _ in chosen:
        m = cov[(kind, f)]
        gain = m if union is None else (m & ~union)
        if union is None or int(gain.sum() or 0) / dn > 0.005:
            used.append(f)
            union = m if union is None else (union | m)
    k = int(union.sum()) if union is not None else 0
    if k / dn < cfg.date_parse_ratio:
        return None

    # Day/month ambiguity: every row that parses day-first also parses month-first.
    ambiguous_dm = False
    if kind == "date":
        for f1, f2 in _AMBIGUOUS_PAIRS:
            a, b = cov.get(("date", f1)), cov.get(("date", f2))
            if a is not None and b is not None:
                both, either = int((a & b).sum()), int((a | b).sum())
                if either > 0 and both == either:
                    ambiguous_dm = True
    if ambiguous_dm:
        used.sort(key=lambda f: 0 if f and f.startswith("%d") else 1 if f and f.startswith("%m") else 2)
    real_fmts = [f for f in used if f is not None]
    mixed = len(real_fmts) > 1 and not ambiguous_dm
    prev_src = vals.unique(maintain_order=True).head(3)
    prev_after = _eval(prev_src, lambda c: _date_expr(c, kind, used)).cast(pl.String).to_list()
    conf = _confidence(95.0 if re.search(r"date|time|(^|_)dt($|_)|created|updated|_at$", col.lower()) else 92.0,
                       k=k, n=dn)
    if ambiguous_dm:
        conf = min(conf, 78.0)
    elif mixed:
        conf -= 4
    est = int(round(k / dn * n_full))
    return ctx.make(
        type="date_text", column=col, action="parse_date", count=est, severity="high",
        suggestion=f"Parse to {kind.title()}" + (" (mixed formats)" if mixed else ""),
        label=f"Parse Text -> {kind.title()}",
        rationale=(f"{k / dn * 100:.0f}% of values parse as {kind}s using "
                   f"{len(real_fmts)} format(s) ({', '.join(str(f) for f in used[:3])})."
                   + (" Mixed formats detected; values are parsed with the first matching format." if mixed else "")
                   + (" Day/month order is ambiguous in this sample; day-first assumed, verify." if ambiguous_dm else "")
                   + " Typed dates enable sorting, resampling and time-series analysis."),
        confidence=conf, pillar="Validity", stage=STAGE["type_coercion"],
        affected_cells=est, resolution=1.0, before=prev_src.to_list(), after=prev_after,
        params={"kind": kind, "formats": used}, risk="medium" if ambiguous_dm else "low",
        evidence={"parse_ratio": round(k / dn, 4), "mixed_formats": mixed, "dm_ambiguous": ambiguous_dm},
    )


@_per_column
def detect_type_inference(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Boolean-like text, numeric-as-text (dirty or plain) and date parsing."""
    s = ctx.work[col]
    if not _is_str(s.dtype):
        return []
    vals_full = _nonmasked_strings(s)
    n_full = vals_full.len()
    if n_full < 5:
        return []
    vals = _sample_series(vals_full, ctx.cfg)
    n = vals.len()

    sug = _infer_boolean(ctx, col, vals, n, n_full)
    if sug:
        return [sug]
    handled, sug = _infer_numeric(ctx, col, vals, n, n_full)
    if handled:
        return [sug] if sug else []
    sug = _infer_date(ctx, col, vals, n, n_full)
    return [sug] if sug else []


@_per_column
def detect_integral_floats(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Float columns that only hold whole numbers (typical after reading CSV with blanks)."""
    if not _is_float(ctx.work.schema[col]):
        return []
    s = _clean_num(ctx.work[col])
    n = s.len()
    if n < 10 or float(s.abs().max()) >= 9e15 or not bool((s == s.round(0)).all()):
        return []
    return [ctx.make(
        type="integral_float", column=col, action="cast_int", count=n, severity="low",
        suggestion="Cast Whole-Number Float -> Integer", label="Cast Float -> Int64",
        rationale="Every observed value is a whole number; Int64 (nullable) is exact, smaller and prints "
                  "without '.0'. Advisory because some columns are conceptually continuous.",
        confidence=85.0, pillar="Validity", stage=STAGE["type_coercion"], advisory=True,
        impact_text="Cleaner type, exact arithmetic", before=[float(s[0])], after=[int(s[0])],
    )]


# =============================================================================
# PHASE B detectors: validity (sentinels, ranges, formats)
# =============================================================================

@_per_column
def detect_sentinels(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Magic numbers (-999, 9999, ...) that sit far outside the rest of the distribution."""
    if not _is_num(ctx.work.schema[col]) or is_identifier_column(col):
        return []
    s = _clean_num(ctx.work[col])
    n = s.len()
    if n < max(20, ctx.cfg.min_rows_for_stats):
        return []
    found: List[Tuple[float, int]] = []
    # Scan every distinct value that *looks* like a sentinel (the old top-6-by-frequency scan
    # missed sentinels in flat distributions such as ages).
    vc = s.value_counts(sort=True)
    vcol = vc.columns[0]
    cand = vc.filter((pl.col(vcol) == -1) | (pl.col(vcol).cast(pl.Float64).abs() >= 99)).rows()
    for val, cnt in cand:
        v = float(val)
        is_int_val = v.is_integer()
        looks = v == -1 or (is_int_val and v != 0 and len(str(int(abs(v)))) >= 2
                            and set(str(int(abs(v)))) == {"9"})
        if not looks or cnt < max(3, int(0.002 * n)):
            continue
        rest = s.filter(s != val)
        if rest.len() < 10:
            continue
        q1, q3 = rest.quantile(0.25), rest.quantile(0.75)
        if q1 is None or q3 is None or q3 - q1 <= 0:
            continue
        iqr = q3 - q1
        if v < q1 - 3 * iqr or v > q3 + 3 * iqr:
            found.append((v, int(cnt)))
    if not found:
        return []
    total = sum(c for _, c in found)
    vals = [int(v) for v, _ in found]
    return [ctx.make(
        type="sentinel_values", column=col, action="nullify_sentinel", count=total,
        severity=_severity(total / ctx.n_rows * 100, critical=30, high=8, medium=1),
        suggestion=f"Null Out Sentinel Values {vals}", label="Convert Sentinel Codes -> null",
        rationale=(f"Value(s) {vals} occur {total} times and sit far outside the remaining distribution "
                   f"(> 3 IQR); they are almost certainly 'missing' codes that corrupt means, "
                   f"correlations and models."),
        confidence=_confidence(93.0, n=n), pillar="Validity", stage=STAGE["validity"],
        affected_cells=total, resolution=1.0, before=vals[:3], after=[None] * len(vals[:3]),
        params={"values": vals}, evidence={"counts": {str(int(v)): c for v, c in found}},
    )]


@_per_column
def detect_domain_validity(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Name-driven range rules (age, %, non-negative measures, year) + implausible dates."""
    dt = ctx.work.schema[col]
    name = _snake(col)
    if _is_num(dt):
        s = _clean_num(ctx.work[col])
        if s.len() < 5:
            return []
        for rx, lo, hi, label, excl in RANGE_RULES:
            if not rx.search(name) or (excl and excl.search(name)):
                continue
            mask = pl.Series([False] * s.len())
            if lo is not None:
                mask = mask | (s < lo)
            if hi is not None:
                mask = mask | (s > hi)
            cnt = int(mask.sum() or 0)
            if cnt == 0 or cnt / s.len() > 0.30:        # >30% => rule probably wrong for this column
                return []
            bounds = f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else '+inf'}]"
            bad_vals = s.filter(mask).unique(maintain_order=True).head(3).to_list()
            return [ctx.make(
                type="out_of_range", column=col, action="nullify_out_of_range", count=cnt,
                severity=_severity(cnt / ctx.n_rows * 100, critical=20, high=5, medium=0.5),
                suggestion=f"Null Out Impossible Values ({label})", label=f"Null Values Outside {bounds}",
                rationale=f"{cnt} value(s) violate the plausible range {bounds} for a {label} "
                          f"(column name heuristic); they are data-entry or unit errors.",
                confidence=_confidence(89.0, n=s.len()), pillar="Validity", stage=STAGE["validity"],
                affected_cells=cnt, resolution=1.0, before=bad_vals, after=[None] * len(bad_vals),
                params={"min": lo, "max": hi, "kind": "numeric"}, risk="medium",
                evidence={"rule": label},
            )]
        return []
    if _is_temporal(dt):
        if getattr(dt, "time_zone", None):               # tz-aware: skip naive bounds
            return []
        s = ctx.work[col].drop_nulls()
        if s.len() < 5:
            return []
        if dt == pl.Date:
            lo, hi = _dt.date(1900, 1, 1), _dt.date(2100, 12, 31)
        else:
            lo, hi = _dt.datetime(1900, 1, 1), _dt.datetime(2100, 12, 31)
        mask = (s < lo) | (s > hi)
        cnt = int(mask.sum() or 0)
        if cnt:
            bad = s.filter(mask).unique(maintain_order=True).head(3).to_list()
            return [ctx.make(
                type="out_of_range", column=col, action="nullify_out_of_range", count=cnt,
                severity="medium", suggestion="Null Out Implausible Dates",
                label="Null Dates Outside 1900-2100",
                rationale=f"{cnt} timestamps fall outside 1900-2100, typically parse errors or default epochs.",
                confidence=90.0, pillar="Validity", stage=STAGE["validity"], affected_cells=cnt,
                resolution=1.0, before=bad, after=[None] * len(bad),
                params={"kind": "date", "min": "1900-01-01", "max": "2100-12-31"}, risk="medium",
            )]
    return []


@_per_column
def detect_invalid_formats(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Malformed e-mail addresses and mixed-case e-mail duplicates."""
    s = ctx.work[col]
    if not _is_str(s.dtype):
        return []
    vals = _nonmasked_strings(s)
    n = vals.len()
    if n < 5:
        return []
    sample = _sample_series(vals, ctx.cfg)
    ok_ratio = int(sample.str.contains(EMAIL_RE).sum() or 0) / sample.len()
    if not ("email" in col.lower() or ok_ratio >= 0.6):
        return []
    out: List[Dict[str, Any]] = []
    valid = vals.str.contains(EMAIL_RE)
    bad = int((~valid).sum() or 0)
    if 0 < bad <= n * 0.4:
        samples = vals.filter(~valid).unique(maintain_order=True).head(3).to_list()
        out.append(ctx.make(
            type="invalid_email", column=col, action="nullify_invalid_regex", count=bad,
            severity=_severity(bad / ctx.n_rows * 100, critical=30, high=10, medium=1),
            suggestion=f"Null Out Malformed E-mails ({bad})", label="Null Invalid E-mails",
            rationale=f"{bad} of {n} values do not look like e-mail addresses; keeping them breaks "
                      f"outreach, joins and dedup.",
            confidence=_confidence(91.0, n=n), pillar="Validity", stage=STAGE["validity"],
            affected_cells=bad, resolution=1.0, before=samples, after=[None] * len(samples),
            params={"regex": EMAIL_RE}, risk="medium",
        ))
    good = vals.filter(valid)
    upper = good.str.contains(r"[A-Z]")
    ucnt = int(upper.sum() or 0)
    if ucnt > 0 and good.len() > 0 and ok_ratio >= 0.6:
        ex = good.filter(upper).unique(maintain_order=True).head(3).to_list()
        out.append(ctx.make(
            type="email_case", column=col, action="lowercase_text", count=ucnt,
            severity="low", suggestion=f"Lowercase E-mails ({ucnt} mixed-case)",
            label="Lowercase E-mail Addresses",
            rationale=f"{ucnt} addresses contain upper-case letters; e-mail is case-insensitive in practice, "
                      f"and mixed case hides duplicates and breaks joins.",
            confidence=92.0, pillar="Consistency", stage=STAGE["categories"],
            affected_cells=ucnt, resolution=1.0, before=ex, after=[x.lower() for x in ex],
        ))
    return out


# =============================================================================
# PHASE C detectors (run on the projected working frame)
# =============================================================================

def detect_empty_rows(ctx: _Context) -> List[Dict[str, Any]]:
    df = ctx.work
    if df.height == 0 or df.width == 0:
        return []
    cols = [c for c, dt in df.schema.items() if dt != pl.Object]
    if not cols:
        return []
    cnt = int(df.select(pl.all_horizontal([pl.col(c).is_null() for c in cols]).sum()).item() or 0)
    if cnt == 0:
        return []
    return [ctx.make(
        type="empty_rows", column="*", action="drop_empty_rows", count=cnt,
        severity=_severity(cnt / ctx.n_rows * 100, critical=30, high=10, medium=1),
        suggestion=f"Drop Fully Empty Rows ({cnt})", label="Drop Empty Rows",
        rationale=f"{cnt} rows have no data in any column (after placeholder cleaning); they carry no information.",
        confidence=99.0, pillar="Completeness", stage=STAGE["dedupe"],
        affected_cells=cnt * ctx.n_cols, resolution=1.0, before=["<all null>"] * min(cnt, 2),
        after=["<row dropped>"] * min(cnt, 2),
    )]


def detect_duplicates(ctx: _Context) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    df, n = ctx.work, ctx.work.height
    if n < 2:
        return out
    cols = [c for c, dt in df.schema.items() if _is_simple(dt)]
    if not cols:
        return out
    dup_full = n - df.select(cols).unique().height

    if dup_full > 0:
        pct = dup_full / n * 100
        out.append(ctx.make(
            type="duplicate_rows", column="*", action="drop_duplicate_rows", count=dup_full,
            severity=_severity(pct, critical=30, high=10, medium=1),
            suggestion=f"Drop Exact Duplicate Rows ({dup_full})", label="Drop Duplicate Rows (keep first)",
            rationale=f"{dup_full} rows ({pct:.1f}%) are exact copies of an earlier row, inflating counts and sums.",
            confidence=_confidence(97.0, n=n), pillar="Uniqueness", stage=STAGE["dedupe"],
            affected_cells=dup_full * ctx.n_cols, resolution=1.0,
            before=[f"{dup_full} repeated rows"], after=["<removed>"], params={"subset": None},
            evidence={"pct": round(pct, 2)},
        ))
    else:
        # Same payload, different surrogate key?
        id_like = []
        for c in cols:
            if not _is_pk_name(c):
                continue
            nn = df[c].drop_nulls()
            if nn.len() > 0 and nn.n_unique() / nn.len() >= 0.99:
                id_like.append(c)
        subset = [c for c in cols if c not in id_like]
        if id_like and len(subset) >= 2:
            d2 = n - df.select(subset).unique().height
            if d2 > 0:
                out.append(ctx.make(
                    type="duplicate_rows", column="*", action="drop_duplicate_rows", count=d2,
                    severity=_severity(d2 / n * 100, critical=30, high=10, medium=1),
                    suggestion=f"Drop Duplicates Ignoring Surrogate Key ({d2})",
                    label="Drop Duplicate Records (ignore ID)",
                    rationale=f"{d2} rows are identical in every column except the unique key "
                              f"{', '.join(id_like)}; they are probably re-ingested records.",
                    confidence=75.0, pillar="Uniqueness", stage=STAGE["dedupe"],
                    affected_cells=d2 * ctx.n_cols, resolution=0.9, params={"subset": subset},
                    before=[f"{d2} repeated records"], after=["<removed>"], risk="medium",
                ))

    for c in cols:
        if not _is_pk_name(c):
            continue
        s = df[c].drop_nulls()
        nn = s.len()
        if nn < 10:
            continue
        nu = s.n_unique()
        if nu == nn or nu / nn < 0.95:
            continue
        dups = nn - nu
        vc = s.value_counts(sort=True).head(2)
        out.append(ctx.make(
            type="duplicate_keys", column=c, action="drop_duplicate_keys", count=dups,
            severity=_severity(dups / n * 100, critical=20, high=5, medium=0.5),
            suggestion=f"Resolve Duplicate Keys ({dups})", label="Keep First Row per Key",
            rationale=f"'{c}' is {nu / nn * 100:.1f}% unique - it looks like a primary key but {dups} values repeat, "
                      f"which breaks joins and lookups.",
            confidence=82.0, pillar="Uniqueness", stage=STAGE["dedupe"],
            affected_cells=dups * ctx.n_cols, resolution=0.9, params={"subset": [c]},
            before=[v for v, _ in vc.rows()], after=["<first kept>"] * vc.height, risk="medium",
        ))
    return out


def _numeric_matrix(ctx: _Context, cols: List[str]) -> np.ndarray:
    sub = ctx.work.select([pl.col(c).cast(pl.Float64, strict=False).fill_null(float("nan")) for c in cols])
    if sub.height > ctx.cfg.sample_size:
        sub = sub.sample(n=ctx.cfg.sample_size, seed=ctx.cfg.seed)
    X = sub.to_numpy()
    with np.errstate(all="ignore"):
        invalid_mask = (~np.isfinite(X)) | (np.abs(X) > 1e12)
        X[invalid_mask] = np.nan          # +-inf and extreme overflow values behave as missing
    return X


def detect_informative_missingness(ctx: _Context) -> List[Dict[str, Any]]:
    """Missingness that correlates with other numeric columns (likely MNAR / MAR)."""
    out: List[Dict[str, Any]] = []
    if ctx.n_rows < 100:
        return out
    num_cols = [c for c, dt in ctx.work.schema.items() if _is_num(dt)][:40]
    if len(num_cols) < 2:
        return out
    X = _numeric_matrix(ctx, num_cols)
    if len(X) > 10_000:
        rng = np.random.default_rng(ctx.cfg.seed)
        sample_idx = rng.choice(len(X), size=10_000, replace=False)
        X = X[sample_idx]
    for j, col in enumerate(num_cols):
        ind = np.isnan(X[:, j])
        rate = ind.mean()
        if not 0.02 <= rate <= 0.98 or ind.sum() < 10:
            continue
        best, best_r = None, 0.0
        for k, other in enumerate(num_cols):
            if k == j:
                continue
            valid = ~np.isnan(X[:, k])
            if valid.sum() < 30:
                continue
            a, b = ind[valid].astype(float), X[valid, k]
            with np.errstate(all="ignore"):
                if a.std() == 0 or b.std() == 0:
                    continue
                r = float(np.corrcoef(a, b)[0, 1])
            if not math.isnan(r) and abs(r) > abs(best_r):
                best, best_r = other, r
        if best and abs(best_r) >= 0.30:
            out.append(ctx.make(
                type="informative_missingness", column=col, action="add_missing_indicator",
                count=int(ind.sum()), severity="medium",
                suggestion=f"Add Missing-Indicator Flag (r={best_r:+.2f} with '{best}')",
                label="Add 'was_missing' Indicator",
                rationale=f"Missingness in '{col}' correlates with '{best}' (r={best_r:+.2f}); it is not random, so "
                          f"imputing silently would hide a signal. Keep a flag column before filling.",
                confidence=_confidence(82.0, n=int(ind.sum())), pillar="Completeness",
                stage=STAGE["indicators"], affected_cells=0, impact_text="Preserves missingness signal",
                before=["<null>"], after=[True], params={"correlated_with": best, "r": round(best_r, 3)},
            ))
    return out


def _best_group_column(ctx: _Context, target: str) -> Tuple[Optional[str], float]:
    """Group column that explains the most variance of `target` (adjusted eta squared)."""
    best, best_eta = None, 0.0
    for g in ctx.cat_candidates():
        if g == target:
            continue
        try:
            sub = ctx.work.select([pl.col(target).cast(pl.Float64, strict=False).alias("y"),
                                   pl.col(g).cast(pl.String).alias("g")]).drop_nulls()
            sub = sub.filter(pl.col("y").is_finite() & (pl.col("y").abs() < 1e12))
            N = sub.height
            if N < 30:
                continue
            stats = sub.group_by("g").agg(pl.len().alias("n"), pl.col("y").mean().alias("m"))
            k = stats.height
            if k < 2 or N <= k:
                continue
            with np.errstate(all="ignore"):
                mu = float(sub["y"].mean())
                sst = float(((sub["y"] - mu) ** 2).sum())
                if not sst > 0:
                    continue
                ssb = float((stats["n"].cast(pl.Float64) * (stats["m"] - mu) ** 2).sum())
                eta = ssb / sst
                adj = 1 - (1 - eta) * (N - 1) / (N - k)
            if adj > best_eta:
                best, best_eta = g, adj
        except Exception:
            continue
    return best, best_eta


@_per_column
def detect_missing_values(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Distribution-aware imputation strategy per column."""
    cfg, df, n = ctx.cfg, ctx.work, ctx.n_rows
    s = df[col]
    nulls = s.null_count()
    if nulls == 0 or n == 0:
        return []
    dt = s.dtype
    pct = nulls / n * 100
    sev = _severity(pct, critical=60, high=20, medium=0)
    raw_nulls = ctx.orig[col].null_count() if col in ctx.orig.columns else 0
    row_risk = "high" if pct > 20 else "medium" if pct > 5 else "low"

    def emit(**kw):
        base = dict(type="missing_values", column=col, count=nulls, severity=sev,
                    pillar="Completeness", affected_cells=raw_nulls)
        base.update(kw)
        return [ctx.make(**base)]

    if n - nulls == 0:
        return emit(suggestion="Drop Column (All null)", label="Drop Column (All null)", action="drop_column",
                    rationale="All values in the column are missing; it provides no analytical utility.",
                    confidence=99.0, stage=STAGE["pruning"], resolution=1.0, severity="high",
                    before=[None, None], after=["<dropped>", "<dropped>"], reversible=False)

    if pct >= 50.0:
        return emit(suggestion=f"Drop Column ({pct:.0f}% missing)", label="Drop Mostly-Empty Column",
                    action="drop_column",
                    rationale=f"{pct:.1f}% of values are missing; any imputation would be mostly fabricated. "
                              f"Consider dropping or sourcing the data again.",
                    confidence=85.0, stage=STAGE["pruning"], resolution=0.8, risk="medium", reversible=False,
                    before=[None, None], after=["<dropped>", "<dropped>"])

    if is_identifier_column(col):
        return emit(suggestion="Drop Rows (Key integrity)", label="Drop Missing Rows", action="drop_rows",
                    rationale="Identifier or key attribute; imputing synthetic values corrupts entity uniqueness "
                              "and foreign relationships.",
                    confidence=97.0, stage=STAGE["imputation"], resolution=1.0, risk=row_risk,
                    before=[None, None], after=["<row dropped>"] * 2)

    if _is_temporal(dt):
        return emit(suggestion="Drop Rows (Temporal chronology)", label="Drop Missing Rows", action="drop_rows",
                    rationale="Timestamps cannot be safely imputed with mean or median without distorting "
                              "chronological ordering.",
                    confidence=95.0, stage=STAGE["imputation"], resolution=1.0, risk=row_risk,
                    before=[None, None], after=["<row dropped>"] * 2)

    if dt == pl.Boolean:
        mode_val = s.drop_nulls().mode().sort()[0]
        return emit(suggestion=f"Fill with Mode ({mode_val})", label="Fill with Most Frequent", action="fill_mode",
                    rationale=f"Boolean flag; most frequent value is {mode_val}.", confidence=85.0,
                    stage=STAGE["imputation"], before=[None, None], after=[mode_val, mode_val])

    if _is_num(dt):
        series = _clean_num(s)
        if series.len() < 3:
            return emit(suggestion="Fill with Median (few observations)", label="Fill with Median",
                        action="fill_median", rationale="Very few observed values; median is the safest simple fill.",
                        confidence=70.0, stage=STAGE["imputation"], before=[None],
                        after=[series.median() if series.len() else None], risk="medium")
        nu = series.n_unique()
        if _is_int(dt) and nu <= 7:
            top = series.mode().sort()[0]
            return emit(suggestion=f"Fill with Mode ({top})", label="Fill with Most Frequent", action="fill_mode",
                        rationale=f"Discrete integer scale ({nu} distinct values, mode {top}); avoids invalid "
                                  f"fractional counts.",
                        confidence=95.0, stage=STAGE["imputation"], before=[None, None], after=[top, top])

        grp, eta = _best_group_column(ctx, col)
        skew = _safe_skew(series)
        has_out = False
        if series.len() >= cfg.min_rows_for_stats:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            if q1 is not None and q3 is not None and q3 - q1 > 0:
                iqr = q3 - q1
                has_out = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum() or 0) > 0

        if grp and eta >= cfg.group_impute_adj_eta2:
            med = round(float(series.median()), 2)
            return emit(suggestion=f"Fill with Median by '{grp}' (explains {eta * 100:.0f}% of variance)",
                        label=f"Fill with Median per '{grp}'", action="fill_group_median",
                        rationale=f"'{grp}' explains {eta * 100:.0f}% of the variance of '{col}' (adjusted eta squared); "
                                  f"group-wise medians preserve structure that a global fill would flatten.",
                        confidence=round(88 + min(10.0, eta * 12), 1), stage=STAGE["imputation"],
                        before=[None, None], after=[f"<median of {grp}>"] * 2,
                        params={"group_by": grp}, evidence={"adj_eta2": round(eta, 3), "global_median": med})
        if skew is not None and abs(skew) < cfg.skew_symmetric and not has_out:
            mean_val = round(float(series.mean()), 2)
            return emit(suggestion=f"Fill with Average (skew {skew:.2f})", label="Fill with Average",
                        action="fill_mean",
                        rationale=f"Near-symmetric distribution (skewness {skew:.2f}, no Tukey outliers); the mean "
                                  f"preserves sample mean and parametric variance.",
                        confidence=92.0, stage=STAGE["imputation"], before=[None, None], after=[mean_val] * 2,
                        evidence={"skew": skew})
        med = round(float(series.median()), 2)
        skew_s = f"skew: {skew:.2f}" if skew is not None else "asymmetric"
        return emit(suggestion=f"Fill with Median ({skew_s})", label="Fill with Median", action="fill_median",
                    rationale=f"Skewed or outlier-prone distribution ({skew_s}"
                              f"{'; outliers present' if has_out else ''}); the median is robust to tail values.",
                    confidence=96.0, stage=STAGE["imputation"], before=[None, None], after=[med] * 2,
                    evidence={"skew": skew, "outliers": has_out})

    if _is_str(dt) or dt == pl.Categorical:
        nn = s.drop_nulls().cast(pl.String)
        nu = nn.n_unique()
        ratio = nu / nn.len()
        if nu <= 20 or ratio < 0.15:
            mode_val = str(nn.mode().sort()[0])
            disp = mode_val if len(mode_val) <= 15 else f"{mode_val[:12]}..."
            return emit(suggestion=f"Fill with Most Frequent ('{disp}')", label="Fill with Most Frequent",
                        action="fill_mode",
                        rationale=f"Categorical attribute with {nu} classes; most frequent category is '{mode_val}'.",
                        confidence=_confidence(91.0, n=nn.len()), stage=STAGE["imputation"],
                        before=[None, None], after=[mode_val] * 2)
        return emit(suggestion="Fill with 'Unknown' (Placeholder)", label="Fill with 'Unknown'",
                    action="fill_value",
                    rationale=f"High-cardinality text ({nu} unique entries); an explicit placeholder avoids "
                              f"falsifying categories.",
                    confidence=88.0, stage=STAGE["imputation"], before=[None, None],
                    after=["Unknown"] * 2, params={"value": "Unknown"})
    return []


@_per_column
def detect_outliers(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """Tukey fences (log-scale for heavy right tails); systemic tails are flagged, not capped."""
    cfg = ctx.cfg
    dt = ctx.work.schema[col]
    if not _is_num(dt) or is_identifier_column(col):
        return []
    s = _clean_num(ctx.work[col])
    n = s.len()
    if n < cfg.min_rows_for_stats or s.n_unique() < 10:
        return []
    skew = _safe_skew(s)
    use_log = bool(skew is not None and abs(skew) > 1.5 and float(s.min()) >= 0)
    z = s.cast(pl.Float64).log1p() if use_log else s
    q1, q3 = z.quantile(0.25), z.quantile(0.75)
    if q1 is None or q3 is None or q3 - q1 <= 0:
        return []
    iqr = q3 - q1
    lo_z, hi_z = q1 - cfg.outlier_iqr_k * iqr, q3 + cfg.outlier_iqr_k * iqr
    ex_lo, ex_hi = q1 - cfg.outlier_extreme_k * iqr, q3 + cfg.outlier_extreme_k * iqr
    lower, upper = (math.expm1(lo_z), math.expm1(hi_z)) if use_log else (lo_z, hi_z)
    if use_log:
        lower = max(lower, 0.0)
    out_mask = (z < lo_z) | (z > hi_z)
    cnt = int(out_mask.sum() or 0)
    if cnt == 0:
        return []
    extreme = int(((z < ex_lo) | (z > ex_hi)).sum() or 0)
    pct = cnt / n * 100
    if _is_int(dt):
        lower, upper = math.ceil(lower), math.floor(upper)
    else:
        lower, upper = float(lower), float(upper)

    vals = s.filter(out_mask)
    dist = (vals - (lower + upper) / 2).abs()
    top_idx = dist.arg_sort(descending=True).head(3)
    before = [round(float(vals[int(i)]), 2) for i in top_idx]
    after = [round(upper if v > upper else lower, 2) for v in before]
    sev = "high" if pct > 5 or extreme / n > 0.01 else ("medium" if extreme or pct > 0.5 else "low")
    method = "Tukey IQR on log1p scale" if use_log else "Tukey IQR"

    if pct > cfg.outlier_systemic_ratio * 100:
        return [ctx.make(
            type="outliers", column=col, action="flag_outliers", count=cnt, severity="medium",
            suggestion=f"Flag Heavy Tail ({pct:.1f}% outside fences)", label="Add Outlier Flag Column",
            rationale=f"{pct:.1f}% of values fall outside [{lower:.1f}, {upper:.1f}] ({method}); that is too "
                      f"many to be errors; the distribution is naturally heavy-tailed, so flag instead of capping.",
            confidence=75.0, pillar="Stability", stage=STAGE["outliers"], affected_cells=cnt * 0.3,
            resolution=0.8, before=before, after=[True] * len(before),
            params={"lower": lower, "upper": upper}, evidence={"pct": round(pct, 2), "extreme": extreme},
        )]
    conf = 95.0 if extreme else (85.0 if use_log else 92.0)
    return [ctx.make(
        type="outliers", column=col, action="cap_outliers", count=cnt, severity=sev,
        suggestion=f"Cap Outliers ({pct:.1f}% outside fences)", label=f"Cap Outliers ({method})",
        rationale=f"{cnt} values ({pct:.1f}%) fall outside [{lower:.1f}, {upper:.1f}] ({method}"
                  f"{', ' + str(extreme) + ' extreme' if extreme else ''}); winsorising to the fences "
                  f"stabilises variance while keeping every row.",
        confidence=_confidence(conf, n=n), pillar="Stability", stage=STAGE["outliers"],
        affected_cells=cnt, resolution=0.85, before=before, after=after,
        params={"lower": lower, "upper": upper},
        evidence={"pct": round(pct, 2), "extreme": extreme, "log_scale": use_log, "skew": skew},
    )]


@_per_column
def detect_skew_transforms(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    dt = ctx.work.schema[col]
    if not _is_num(dt) or is_identifier_column(col):
        return []
    s = _clean_num(ctx.work[col])
    if s.len() < 30 or s.n_unique() < 20 or float(s.min()) < 0:
        return []
    skew = _safe_skew(s)
    if skew is None or skew <= ctx.cfg.skew_transform:
        return []
    new_skew = _safe_skew(s.cast(pl.Float64).log1p())
    if new_skew is None or abs(new_skew) >= abs(skew) * 0.6:
        return []
    return [ctx.make(
        type="skewed_distribution", column=col, action="log1p_transform", count=s.len(),
        severity="low", suggestion=f"Add log1p Feature (skew {skew:.2f} -> {new_skew:.2f})",
        label="Add log1p Column",
        rationale=f"Right-skewed (skewness {skew:.2f}); log1p reduces it to {new_skew:.2f}, which helps "
                  f"linear models and charts. Original column is kept.",
        confidence=88.0, pillar="Stability", stage=STAGE["transforms"], affected_cells=0,
        impact_text=f"Skew {skew:.2f} -> {new_skew:.2f}", before=[round(float(s[0]), 2)],
        after=[round(math.log1p(float(s[0])), 3)], evidence={"skew": skew, "skew_after": new_skew},
    )]


@_per_column
def detect_constant_columns(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    cfg, n = ctx.cfg, ctx.n_rows
    s = ctx.work[col]
    if n < 10 or not _is_simple(s.dtype):
        return []
    nn = s.drop_nulls()
    if nn.len() == 0:
        return []
    vc = nn.value_counts(sort=True)
    top_val, top_cnt = vc.row(0)[0], int(vc.row(0)[1])
    nu = vc.height
    nulls = s.null_count()
    dom = top_cnt / nn.len() * 100
    cells = n * STRUCTURAL_WEIGHT
    if nu == 1:
        informative_nulls = nulls / n >= 0.05
        return [ctx.make(
            type="constant_column", column=col, action="drop_column", count=n,
            severity="medium", suggestion="Drop Constant Column (Zero Variance)",
            label="Drop Constant Column (Zero Variance)",
            rationale=(f"All {nn.len()} non-null entries have the identical value ('{top_val}'); dropping "
                       f"uninformative dimensions removes collinearity and memory overhead."
                       + (f" Note: {nulls} nulls exist; if presence vs. absence matters, keep or convert "
                          f"to a flag first." if informative_nulls else "")),
            confidence=85.0 if informative_nulls else 99.0, pillar="Stability", stage=STAGE["pruning"],
            affected_cells=cells, resolution=1.0, before=[top_val] * 3, after=["<dropped>"] * 3,
            risk="medium" if informative_nulls else "low", reversible=False,
        )]
    if nn.len() >= 20 and dom >= cfg.quasi_constant_ratio * 100:
        return [ctx.make(
            type="constant_column", column=col, action="drop_column", count=top_cnt,
            severity="medium", suggestion=f"Drop Quasi-Constant Column ({dom:.1f}% single value)",
            label="Drop Quasi-Constant Column",
            rationale=f"Dominant value '{top_val}' constitutes {dom:.1f}% of non-null rows; near-zero "
                      f"variance adds no signal. Rare-event flags (fraud, churn) can look like this, so verify.",
            confidence=_confidence(95.0, n=nn.len()), pillar="Stability", stage=STAGE["pruning"],
            affected_cells=cells, resolution=1.0, before=[str(top_val)] * 3, after=["<dropped>"] * 3,
            risk="medium", reversible=False,
        )]
    return []


def detect_redundant_columns(ctx: _Context) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    df, n = ctx.work, ctx.work.height
    if n < 10:
        return out
    dropped: set = set()

    # exact duplicates (any simple dtype) - bucket by cheap signature first
    sig: Dict[Tuple, List[str]] = {}
    for c, dt in df.schema.items():
        if not _is_simple(dt):
            continue
        s = df[c]
        sig.setdefault((str(dt), s.null_count(), s.n_unique()), []).append(c)
    for cols in sig.values():
        for i, a in enumerate(cols):
            if a in dropped:
                continue
            for b in cols[i + 1:]:
                if b in dropped:
                    continue
                try:
                    same = bool(df[a].eq_missing(df[b]).all())
                except Exception:
                    same = False
                if same:
                    drop, keep = (a, b) if (_is_pk_name(b) and not _is_pk_name(a)) else (b, a)
                    dropped.add(drop)
                    out.append(ctx.make(
                        type="duplicate_column", column=drop, action="drop_column", count=n,
                        severity="medium", suggestion=f"Drop Duplicate Column (copy of '{keep}')",
                        label="Drop Duplicate Column",
                        rationale=f"'{drop}' is value-for-value identical to '{keep}'.",
                        confidence=99.0, pillar="Stability", stage=STAGE["pruning"],
                        affected_cells=n * STRUCTURAL_WEIGHT, resolution=1.0, before=[drop], after=["<dropped>"],
                        reversible=False, params={"duplicate_of": keep},
                    ))

    # near-perfect linear redundancy
    num_cols = [c for c, dt in df.schema.items()
                if _is_num(dt) and not is_identifier_column(c) and c not in dropped and df[c].n_unique() > 2]
    num_cols = sorted(num_cols, key=lambda c: df[c].null_count())[: ctx.cfg.max_corr_columns]
    if len(num_cols) >= 2:
        X = _numeric_matrix(ctx, num_cols)
        with np.errstate(all="ignore"):
            means = np.nanmean(X, axis=0)
            Xf = np.where(np.isnan(X), means, X)
            stds = np.nanstd(Xf, axis=0)
            ok = np.isfinite(stds) & (stds > 0)
        if ok.sum() >= 2:
            idx = np.where(ok)[0]
            with np.errstate(all="ignore"):
                C = np.corrcoef(Xf[:, idx], rowvar=False)
            for ii in range(len(idx)):
                for jj in range(ii + 1, len(idx)):
                    r = float(C[ii, jj])
                    if math.isnan(r) or abs(r) < ctx.cfg.redundancy_corr:
                        continue
                    a, b = num_cols[idx[ii]], num_cols[idx[jj]]
                    if a in dropped or b in dropped:
                        continue
                    na, nb = df[a].null_count(), df[b].null_count()
                    drop, keep = (a, b) if na > nb else (b, a)
                    dropped.add(drop)
                    out.append(ctx.make(
                        type="collinear_columns", column=drop, action="drop_column", count=n,
                        severity="low", suggestion=f"Drop Collinear Column (r={r:+.3f} with '{keep}')",
                        label="Drop Redundant Column",
                        rationale=f"'{drop}' is almost perfectly linearly related to '{keep}' (r={r:+.3f}); "
                                  f"keeping both causes multicollinearity.",
                        confidence=85.0, pillar="Stability", stage=STAGE["pruning"],
                        affected_cells=n * STRUCTURAL_WEIGHT * 0.5, resolution=1.0, before=[drop],
                        after=["<dropped>"], risk="medium", reversible=False,
                        params={"correlated_with": keep, "r": round(r, 4)},
                    ))
    return out


_PII_DETECTORS = [
    ("email address", EMAIL_RE, None),
    ("US SSN", r"^\d{3}-\d{2}-\d{4}$", None),
    ("Indian PAN", r"^[A-Z]{5}\d{4}[A-Z]$", None),
    ("Aadhaar number", r"^\d{4}\s\d{4}\s\d{4}$", None),
    ("payment card", r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,7}$", _luhn_ok),
    ("phone number", r"^(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.\-]\d{3,4}[\s.\-]\d{3,4}$|^\+\d{10,15}$", None),
    ("Indian mobile number", r"^(?:\+?91[\s\-]?)?[6-9]\d{9}$", None),
    ("IPv4 address", r"^(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)$", None),
]
_PII_NAME_HINT = re.compile(r"email|phone|mobile|ssn|pan|aadhaa?r|card|ip_?addr|contact")
# Only trust these when the column name also hints at them (too many numeric IDs look alike).
_PII_NAME_REQUIRED = {"Indian mobile number", "payment card"}


@_per_column
def detect_pii(ctx: _Context, col: str) -> List[Dict[str, Any]]:
    """PII (compliance, not score-affecting). Scans the ORIGINAL text so numeric-looking
    values (card numbers, mobiles) are not hidden by a projected type cast."""
    if not ctx.cfg.enable_pii or col not in ctx.orig.columns:
        return []
    s = ctx.orig[col]
    hinted = bool(_PII_NAME_HINT.search(_snake(col)))
    if _is_int(s.dtype) and hinted:
        s = s.cast(pl.String)
    if not _is_str(s.dtype):
        return []
    vals = _nonmasked_strings(s)
    if vals.len() < 5:
        return []
    sample = _sample_series(vals, ctx.cfg)
    for kind, rx, validator in _PII_DETECTORS:
        if kind in _PII_NAME_REQUIRED and not hinted:
            continue
        m = sample.str.contains(rx)
        hits = int(m.sum() or 0)
        ratio = hits / sample.len()
        if ratio < (0.3 if hinted else 0.6):
            continue
        if validator is not None:
            matched = sample.filter(m).head(200).to_list()
            if sum(1 for v in matched if validator(v)) / max(1, len(matched)) < 0.8:
                continue
        total = int(round(ratio * vals.len()))
        ex = sample.filter(m).head(2).to_list()
        salt = ctx.cfg.pii_salt
        return [ctx.make(
            type="pii_detected", column=col, action="hash_pii", count=total, severity="high",
            suggestion=f"Protect PII: {kind} ({total} values)", label="Hash / Pseudonymise PII",
            rationale=f"{ratio * 100:.0f}% of values look like {kind}s. Storing or sharing them in clear text "
                      f"is a privacy / compliance risk (GDPR, DPDP Act, HIPAA); hash or drop before sharing."
                      + ("" if salt else " No pii_salt is configured, so hashes are unsalted and low-entropy "
                                         "values (phones) can be brute-forced; set EngineConfig.pii_salt."),
            confidence=_confidence(90.0, k=hits, n=sample.len()), pillar="Compliance",
            stage=STAGE["privacy"], affected_cells=0, impact_text="Reduces privacy risk",
            before=ex, after=[_hash_value(v, salt) for v in ex],
            params={"salt": salt, "kind": kind, "cast_text": not _is_str(ctx.orig[col].dtype)},
            risk="medium", reversible=False,
        )]
    return []


def detect_column_names(ctx: _Context) -> List[Dict[str, Any]]:
    if not ctx.cfg.enable_column_naming:
        return []
    cols = ctx.orig.columns
    taken = {c for c in cols if _snake(c) == c}
    mapping: Dict[str, str] = {}
    collisions = 0
    for c in cols:
        t = _snake(c)
        if t == c:
            continue
        cand, i = t, 2
        while cand in taken:
            cand = f"{t}_{i}"
            i += 1
            collisions += 1
        taken.add(cand)
        mapping[c] = cand
    if not mapping:
        return []
    items = list(mapping.items())[:3]
    return [ctx.make(
        type="column_names", column="*", action="rename_columns", count=len(mapping),
        severity="low", suggestion=f"Normalize {len(mapping)} Column Names to snake_case",
        label="Normalize Column Names",
        rationale="Column names contain spaces, mixed case or special characters, which complicates SQL, "
                  "attribute access and joins."
                  + (f" {collisions} name collision(s) resolved with numeric suffixes." if collisions else ""),
        confidence=90.0, pillar="Consistency", stage=STAGE["naming"], affected_cells=len(mapping),
        resolution=1.0, before=[a for a, _ in items], after=[b for _, b in items],
        params={"mapping": mapping}, risk="medium", evidence={"collisions": collisions},
    )]


# =============================================================================
# Executors  (one per action; shared by apply_suggestion)
# =============================================================================

def _ex_nan_to_null(df, s):
    c = s["column"]
    bad = pl.col(c).is_nan() | pl.col(c).is_infinite()
    return df.with_columns(pl.when(bad).then(None).otherwise(pl.col(c)).alias(c))

def _ex_clean_masked(df, s):
    c = s["column"]
    p = s.get("params") or {}
    tokens = p.get("tokens") or _MASKED_LIST
    norm = pl.col(c).str.strip_chars().str.to_lowercase()
    return df.with_columns(pl.when(norm.is_in(tokens)).then(None).otherwise(pl.col(c)).alias(c))

def _ex_clean_text(df, s):
    c = s["column"]
    p = s.get("params") or {}
    return df.with_columns(_clean_text_expr(pl.col(c), p.get("collapse", True)).alias(c))

def _ex_mojibake(df, s):
    c = s["column"]
    return df.with_columns(pl.col(c).map_elements(_fix_mojibake_str, return_dtype=_STR).alias(c))

def _ex_standardize(df, s):
    c = s["column"]
    p = s.get("params") or {}
    mapping = p.get("mapping")
    if not mapping:
        return df.with_columns(pl.col(c).cast(pl.String).str.strip_chars().alias(c))
    base = pl.col(c).cast(pl.String).str.strip_chars()
    try:
        expr = base.replace(mapping)
    except Exception:
        expr = base.map_elements(lambda v: mapping.get(v, v), return_dtype=_STR)
    return df.with_columns(expr.alias(c))

def _ex_lowercase(df, s):
    c = s["column"]
    return df.with_columns(pl.col(c).str.strip_chars().str.to_lowercase().alias(c))

def _ex_cast_bool(df, s):
    c = s["column"]
    return df.with_columns(_bool_expr(pl.col(c)).alias(c))

def _ex_coerce_numeric(df, s):
    c = s["column"]
    p = s.get("params") or {}
    e = _numeric_parse_expr(pl.col(c), p.get("style", "us"))
    if p.get("percent_to_fraction"):
        e = e / 100.0
    elif p.get("integral"):
        e = e.round(0).cast(pl.Int64)
    return df.with_columns(e.alias(c))

def _ex_cast_int(df, s):
    c = s["column"]
    return df.with_columns(pl.col(c).cast(pl.Int64, strict=False).alias(c))

def _ex_parse_date(df, s):
    c = s["column"]
    p = s.get("params") or {}
    kind = p.get("kind", "date")
    fmts = p.get("formats") or (DATETIME_FORMATS if kind == "datetime" else DATE_FORMATS)
    return df.with_columns(_date_expr(pl.col(c), kind, fmts).alias(c))

def _ex_nullify_sentinel(df, s):
    c = s["column"]
    p = s.get("params") or {}
    values = p.get("values") or [-999, -9999, 9999, 99999, "-999", "-9999"]
    vals = pl.Series(values).cast(df.schema[c], strict=False).drop_nulls()
    return df.with_columns(pl.when(pl.col(c).is_in(vals)).then(None).otherwise(pl.col(c)).alias(c))

def _ex_nullify_range(df, s):
    c = s["column"]
    p = s.get("params") or {}
    lo, hi = p.get("min"), p.get("max")
    if lo is None and hi is None:
        col_lower = c.lower()
        for rx, r_min, r_max, _, excl in RANGE_RULES:
            if rx.search(col_lower) and not (excl and excl.search(col_lower)):
                lo, hi = r_min, r_max
                break
    if p.get("kind") == "date" and (lo is not None or hi is not None):
        conv = _dt.date.fromisoformat if df.schema[c] == pl.Date else _dt.datetime.fromisoformat
        lo = conv(lo) if lo is not None else None
        hi = conv(hi) if hi is not None else None
    cond = pl.lit(False)
    if lo is not None:
        cond = cond | (pl.col(c) < lo)
    if hi is not None:
        cond = cond | (pl.col(c) > hi)
    return df.with_columns(pl.when(cond).then(None).otherwise(pl.col(c)).alias(c))

def _ex_nullify_regex(df, s):
    c = s["column"]
    p = s.get("params") or {}
    rx = p.get("regex")
    if not rx:
        return df
    stripped = pl.col(c).str.strip_chars()
    return df.with_columns(pl.when(stripped.str.contains(rx)).then(stripped)
                           .otherwise(None).alias(c))

def _ex_drop_empty_rows(df, s):
    cols = [c for c in df.columns if df.schema[c] != pl.Object]
    return df.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols])) if cols else df

def _ex_drop_dup_rows(df, s):
    p = s.get("params") or {}
    subset = p.get("subset")
    if subset is None:
        subset = [c for c, dt in df.schema.items() if _is_simple(dt)]
    else:
        subset = [c for c in subset if c in df.columns]
    return df.unique(subset=subset or None, keep="first", maintain_order=True)

def _ex_drop_dup_keys(df, s):
    """Keep the first row per key; rows with a null key are never treated as duplicates."""
    c = s.get("column")
    p = s.get("params") or {}
    subset = p.get("subset") or ([c] if c else [])
    sub = [col for col in subset if col in df.columns]
    if not sub:
        return df
    key = pl.struct(sub) if len(sub) > 1 else pl.col(sub[0])
    any_null = pl.any_horizontal([pl.col(col).is_null() for col in sub])
    return df.filter(key.is_first_distinct() | any_null)

def _ex_missing_indicator(df, s):
    c = s["column"]
    return df.with_columns(pl.col(c).is_null().alias(f"{c}__was_missing"))

def _ex_drop_rows(df, s):
    return df.drop_nulls(subset=[s["column"]])

def _fill_expr(df, c, stat):
    e = pl.col(c)
    if _is_float(df.schema[c]):
        e = e.fill_nan(None)                           # NaN must not poison the statistic
    fill = e.mean() if stat == "mean" else e.median()
    if _is_int(df.schema[c]):
        fill = fill.round(0).cast(df.schema[c])
    return e.fill_null(fill).alias(c)

def _ex_fill_mean(df, s):
    return df.with_columns(_fill_expr(df, s["column"], "mean"))

def _ex_fill_median(df, s):
    return df.with_columns(_fill_expr(df, s["column"], "median"))

def _ex_fill_group_median(df, s):
    c = s["column"]
    p = s.get("params") or {}
    g = p.get("group_by")
    if not g or g not in df.columns:
        return df.with_columns(_fill_expr(df, c, "median"))
    base = pl.col(c).fill_nan(None) if _is_float(df.schema[c]) else pl.col(c)
    grp, glob = base.median().over(g), base.median()
    if _is_int(df.schema[c]):
        grp, glob = grp.round(0).cast(df.schema[c]), glob.round(0).cast(df.schema[c])
    return df.with_columns(base.fill_null(grp).fill_null(glob).alias(c))

def _ex_fill_mode(df, s):
    c = s["column"]
    nn = df[c].drop_nulls()
    if nn.len() == 0:
        return df
    val = nn.mode().sort()[0]
    try:
        return df.with_columns(pl.col(c).fill_null(val))
    except Exception:
        return df.with_columns(pl.col(c).cast(pl.String).fill_null(str(val)))

def _ex_fill_value(df, s):
    c = s["column"]
    p = s.get("params") or {}
    val = s.get("value") if s.get("value") is not None else p.get("value", "Unknown")
    return df.with_columns(pl.col(c).fill_null(val))

def _ex_cap(df, s):
    c = s["column"]
    p = s.get("params") or {}
    lo, hi = p.get("lower"), p.get("upper")
    if lo is None or hi is None:
        if df[c].dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lo = lo if lo is not None else (q1 - 1.5 * iqr)
                hi = hi if hi is not None else (q3 + 1.5 * iqr)
    return df.with_columns(pl.col(c).clip(lo, hi).alias(c))

def _ex_flag_outliers(df, s):
    c = s["column"]
    p = s.get("params") or {}
    lo, hi = p.get("lower"), p.get("upper")
    if lo is None or hi is None:
        if df[c].dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32):
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            if q1 is not None and q3 is not None:
                iqr = q3 - q1
                lo = lo if lo is not None else (q1 - 1.5 * iqr)
                hi = hi if hi is not None else (q3 + 1.5 * iqr)
    if lo is not None and hi is not None:
        flag = ((pl.col(c) < lo) | (pl.col(c) > hi)).fill_null(False)
        return df.with_columns(flag.alias(f"{c}__outlier"))
    return df

def _ex_log1p(df, s):
    c = s["column"]
    clean_float = pl.col(c).cast(pl.Float64, strict=False)
    valid_log = pl.when(clean_float > -1.0).then(clean_float.log1p()).otherwise(None)
    return df.with_columns(valid_log.alias(f"{c}__log1p"))

def _ex_drop_column(df, s):
    return df.drop([c for c in [s["column"]] if c in df.columns])

def _ex_hash_pii(df, s):
    c = s["column"]
    p = s.get("params") or {}
    salt = p.get("salt", "")
    col = pl.col(c).cast(pl.String)
    return df.with_columns(col.map_elements(lambda v: _hash_value(v, salt), return_dtype=_STR).alias(c))

def _ex_rename(df, s):
    p = s.get("params") or {}
    m = dict(p.get("mapping", {}))
    if not m:
        return df
    for col in df.columns:                              # keep derived columns in sync
        if col in m:
            continue
        for suf in _DERIVED_SUFFIXES:
            if col.endswith(suf) and col[: -len(suf)] in m:
                m[col] = m[col[: -len(suf)]] + suf
    return df.rename({a: b for a, b in m.items() if a in df.columns})

def _ex_noop(df, s):
    return df

_EXECUTORS: Dict[str, Callable[[pl.DataFrame, Dict[str, Any]], pl.DataFrame]] = {
    "nan_to_null": _ex_nan_to_null,
    "clean_masked_nulls": _ex_clean_masked, "clean_text": _ex_clean_text, "fix_mojibake": _ex_mojibake,
    "standardize_categories": _ex_standardize, "lowercase_text": _ex_lowercase, "cast_boolean": _ex_cast_bool,
    "coerce_numeric": _ex_coerce_numeric, "cast_int": _ex_cast_int, "parse_date": _ex_parse_date,
    "nullify_sentinel": _ex_nullify_sentinel, "nullify_out_of_range": _ex_nullify_range,
    "nullify_invalid_regex": _ex_nullify_regex, "drop_empty_rows": _ex_drop_empty_rows,
    "drop_duplicate_rows": _ex_drop_dup_rows, "drop_duplicate_keys": _ex_drop_dup_keys,
    "add_missing_indicator": _ex_missing_indicator, "drop_rows": _ex_drop_rows,
    "fill_mean": _ex_fill_mean, "fill_median": _ex_fill_median, "fill_group_median": _ex_fill_group_median,
    "fill_mode": _ex_fill_mode, "fill_value": _ex_fill_value, "cap_outliers": _ex_cap,
    "replace_outliers_median": _ex_cap,                # v1 action alias
    "flag_outliers": _ex_flag_outliers, "log1p_transform": _ex_log1p, "drop_column": _ex_drop_column,
    "hash_pii": _ex_hash_pii, "rename_columns": _ex_rename, "keep_as_text": _ex_noop,
}


def apply_suggestion(df: pl.DataFrame, suggestion: Dict[str, Any]) -> pl.DataFrame:
    """Apply one suggestion and return the new DataFrame (raises on failure)."""
    action = suggestion["action"]
    if action not in _EXECUTORS:
        raise ValueError(f"Unknown action: {action}")
    col = suggestion.get("column")
    if col not in (None, "*") and col not in df.columns:
        raise KeyError(f"Column '{col}' no longer exists")
    return _EXECUTORS[action](df, suggestion)


def apply_suggestions(df: pl.DataFrame, suggestions: Sequence[Dict[str, Any]], *,
                      skip_advisory: bool = True, strict: bool = False
                      ) -> Tuple[pl.DataFrame, List[Dict[str, Any]]]:
    """
    Apply suggestions in pipeline order (stage, then input order).
    Returns (new_df, log). A failing step is logged and skipped unless strict=True.
    """
    df = _to_polars(df)
    ordered = sorted(enumerate(suggestions), key=lambda t: (t[1].get("stage", 99), t[0]))
    log_rows: List[Dict[str, Any]] = []
    for _, s in ordered:
        if skip_advisory and s.get("advisory"):
            continue
        entry = {"id": s.get("id"), "action": s["action"], "column": s.get("column"),
                 "rows_before": df.height, "cols_before": df.width,
                 "nulls_before": int(sum(df.null_count().row(0)))}
        try:
            df = apply_suggestion(df, s)
            entry.update(ok=True, rows_after=df.height, cols_after=df.width,
                         nulls_after=int(sum(df.null_count().row(0))))
        except Exception as e:
            if strict:
                raise
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
        log_rows.append(entry)
    return df, log_rows


def build_remediation_plan(suggestions: Sequence[Dict[str, Any]], *, min_confidence: float = 90.0,
                           max_risk: str = "low", include_advisory: bool = False) -> List[Dict[str, Any]]:
    """Ordered, conflict-free subset that is safe to auto-apply."""
    if max_risk not in RISK_RANK:
        raise ValueError(f"max_risk must be one of {sorted(RISK_RANK)}")
    limit = RISK_RANK[max_risk]
    candidates = [
        s for s in suggestions
        if s["confidence_score"] >= min_confidence
        and RISK_RANK.get(s.get("risk", "low"), 1) <= limit
        and (include_advisory or not s.get("advisory"))
    ]
    candidates.sort(key=lambda s: (-s.get("priority_score", 0.0), s.get("stage", 99)))

    plan: List[Dict[str, Any]] = []
    accepted_ids: set = set()
    conflicting_ids: set = set()
    for s in candidates:
        s_id = s["id"]
        if s_id in conflicting_ids:
            continue
        s_conflicts = set(s.get("conflicts_with", []))
        if s_conflicts & accepted_ids:
            continue
        accepted_ids.add(s_id)
        conflicting_ids.update(s_conflicts)
        plan.append(s)
    return sorted(plan, key=lambda s: (s["stage"], -s.get("priority_score", 0.0)))


# =============================================================================
# Projection, scoring, orchestration
# =============================================================================

def _project(df: pl.DataFrame, suggestions: Sequence[Dict[str, Any]], stages: set,
             cfg: EngineConfig, warnings: List[str]) -> pl.DataFrame:
    """Simulate high-confidence upstream fixes so downstream detectors see clean data."""
    chosen = [s for s in suggestions if s["stage"] in stages and not s["advisory"]
              and s["confidence_score"] >= cfg.auto_project_confidence]
    for s in sorted(chosen, key=lambda x: x["stage"]):
        try:
            df = apply_suggestion(df, s)
        except Exception as e:
            warnings.append(f"projection of {s['id']} skipped: {type(e).__name__}: {e}")
    return df


def _run(detectors: Sequence[Callable[[_Context], List[Dict[str, Any]]]], ctx: _Context) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    for det in detectors:
        try:
            found.extend(det(ctx))
        except Exception as e:                       # one broken detector must never kill the run
            msg = f"{det.__name__} failed: {type(e).__name__}: {e}"
            log.warning(msg)
            ctx.warnings.append(msg)
    return found


def _priority(s: Dict[str, Any]) -> float:
    sev = SEVERITY_WEIGHT.get(s["severity"], 1) / 4 * 100
    impact = min(100.0, s["impact_dataset_pct"] * 10)
    score = 0.40 * sev + 0.35 * s["confidence_score"] + 0.25 * impact
    score -= {"low": 0, "medium": 5, "high": 15}.get(s["risk"], 5)
    return round(max(0.0, score), 2)


def _link_suggestions(items: List[Dict[str, Any]]) -> None:
    by_col: Dict[str, List[Dict[str, Any]]] = {}
    for s in items:
        by_col.setdefault(s["column"], []).append(s)
    for col, group in by_col.items():
        if col == "*":
            continue
        drops = [s for s in group if s["action"] == "drop_column"]
        for s in group:
            s["depends_on"] = [t["id"] for t in group
                               if t is not s and t["stage"] < s["stage"] and not t["advisory"]
                               and t["stage"] in (PROJECT_GROUP_A | PROJECT_GROUP_B)
                               and t["confidence_score"] >= 90.0]
            if drops and s["action"] != "drop_column":
                s["conflicts_with"] = [d["id"] for d in drops]


def compute_quality_scores(ctx: _Context, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pen = {p: 0.0 for p in PILLAR_WEIGHTS}
    res = {p: 0.0 for p in PILLAR_WEIGHTS}
    for s in items:
        p = s["target_pillar"]
        if p in pen and not s["advisory"]:
            pen[p] += s["affected_cells"]
            res[p] += s["affected_cells"] * s["resolution_factor"]
    before = {p: round(100 * (1 - min(1.0, pen[p] / ctx.total_cells)), 1) for p in pen}
    after = {p: round(100 * (1 - min(1.0, max(0.0, pen[p] - res[p]) / ctx.total_cells)), 1) for p in pen}
    overall = lambda d: round(sum(d[p] * w for p, w in PILLAR_WEIGHTS.items()), 1)
    return {"before": before, "projected_after": after,
            "overall_before": overall(before), "overall_after": overall(after)}


def profile_columns(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Lightweight per-column profile with a semantic-type guess."""
    rows = []
    n = df.height
    for col, dt in df.schema.items():
        s = df[col]
        nulls = s.null_count()
        nu = s.n_unique() if _is_simple(dt) else None
        nn_len = n - nulls
        if _is_str(dt) and nn_len > 0 and re.search(r"e_?mail", col.lower()):
            sem = "email"
        elif is_identifier_column(col):
            sem = "identifier"
        elif _is_temporal(dt):
            sem = "datetime"
        elif dt == pl.Boolean:
            sem = "boolean"
        elif _is_num(dt):
            sem = "numeric_discrete" if (_is_int(dt) and nu is not None and nu <= 10) else "numeric_continuous"
        elif _is_str(dt):
            sem = "categorical" if (nu is not None and nn_len and (nu <= 30 or nu / nn_len < 0.05)) else "text"
        else:
            sem = "other"
        rows.append({"column": col, "dtype": str(dt), "null_pct": round(nulls / n * 100, 2) if n else 0.0,
                     "n_unique": nu, "semantic_type": sem})
    return rows


def analyze(df: Any, config: Optional[EngineConfig] = None) -> Dict[str, Any]:
    """Run the full engine and return a report dict."""
    t0 = time.perf_counter()
    cfg = config or EngineConfig()
    df = _to_polars(df)
    if df.height == 0 or df.width == 0:
        return {"summary": {"rows": df.height, "columns": df.width, "suggestions": 0},
                "quality": {}, "suggestions": [], "plan": [], "profile": [], "warnings": [],
                "meta": {"engine_version": __version__, "elapsed_s": 0.0}}

    ctx = _Context(df, df, cfg)

    # Phase A: cleaning + typing on raw data
    a = _run([detect_nan_inf, detect_masked_nulls, detect_text_hygiene, detect_category_inconsistency,
              detect_type_inference], ctx)
    ctx.work = _project(df, a, PROJECT_GROUP_A, cfg, ctx.warnings)

    # Phase B: validity on typed data
    b = _run([detect_sentinels, detect_domain_validity, detect_invalid_formats], ctx)
    ctx.work = _project(ctx.work, b, PROJECT_GROUP_B, cfg, ctx.warnings)
    ctx._cats = None

    # Phase C: structure, completeness, distribution
    c = _run([detect_empty_rows, detect_duplicates, detect_informative_missingness, detect_missing_values,
              detect_outliers, detect_skew_transforms, detect_integral_floats, detect_constant_columns,
              detect_redundant_columns, detect_pii, detect_column_names], ctx)

    items = a + b + c
    _link_suggestions(items)
    for s in items:
        s["priority_score"] = _priority(s)
    items.sort(key=lambda s: (-s["priority_score"], s["stage"]))

    plan = build_remediation_plan(items)
    by_sev: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    for s in items:
        by_sev[s["severity"]] = by_sev.get(s["severity"], 0) + 1
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1

    return {
        "summary": {"rows": df.height, "columns": df.width, "suggestions": len(items),
                    "auto_applicable": len(plan), "by_severity": by_sev, "by_type": by_type},
        "quality": compute_quality_scores(ctx, items),
        "suggestions": items,
        "plan": [s["id"] for s in plan],
        "profile": profile_columns(df),
        "warnings": ctx.warnings,
        "meta": {"engine_version": __version__, "elapsed_s": round(time.perf_counter() - t0, 3),
                 "sampled": df.height > cfg.sample_size},
    }


def generate_comprehensive_suggestions(df: Any, config: Optional[EngineConfig] = None
                                       ) -> List[Dict[str, Any]]:
    """Main entry point (v1-compatible): prioritized list of remediation suggestions."""
    return analyze(df, config)["suggestions"]


def clean(df: Any, config: Optional[EngineConfig] = None, *, min_confidence: float = 90.0,
          max_risk: str = "low") -> Tuple[pl.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """One call: analyze, build the safe plan, apply it. Returns (clean_df, report, execution_log)."""
    df = _to_polars(df)
    report = analyze(df, config)
    plan = build_remediation_plan(report["suggestions"], min_confidence=min_confidence, max_risk=max_risk)
    out, steps = apply_suggestions(df, plan)
    return out, report, steps


def to_json(report: Any, indent: int = 2) -> str:
    return json.dumps(report, indent=indent, default=str, ensure_ascii=False)


# =============================================================================
# Backwards-compatible v1 entry points
# =============================================================================

def _legacy(detectors: Sequence[Callable], types: Optional[set] = None) -> Callable:
    def inner(df: pl.DataFrame) -> List[Dict[str, Any]]:
        df = _to_polars(df)
        ctx = _Context(df, df, EngineConfig())
        res = _run(detectors, ctx)
        for s in res:
            s["priority_score"] = _priority(s)
        return [s for s in res if types is None or s["type"] in types]
    return inner

detect_masked_null_suggestions = _legacy([detect_masked_nulls])
detect_dirty_numeric_suggestions = _legacy([detect_type_inference], {"dirty_numeric", "numeric_as_text"})
detect_constant_column_suggestions = _legacy([detect_constant_columns])
detect_text_hygiene_suggestions = _legacy([detect_text_hygiene], {"text_whitespace"})
detect_dynamic_missing_suggestions = _legacy([detect_missing_values])
detect_outlier_suggestions = _legacy([detect_outliers])


# =============================================================================
# Demo
# =============================================================================

def _demo() -> None:                                                   # pragma: no cover
    rng = np.random.default_rng(7)
    n = 600
    city = rng.choice(["Mumbai", "Delhi", "Chennai", "Hyderabad"], n)
    base = {"Mumbai": 90, "Delhi": 70, "Chennai": 50, "Hyderabad": 60}
    income = np.array([base[c] + rng.normal(0, 4) for c in city]).round(1).tolist()
    for i in rng.choice(n, 40, replace=False):
        income[i] = None
    income[5] = float("nan")
    df = pl.DataFrame({
        "Customer ID": list(range(n)),
        "Full Name": [f"  User {i} " if i % 7 == 0 else f"User {i}" for i in range(n)],
        "city": [c.upper() if i % 9 == 0 else (c + " " if i % 11 == 0 else c) for i, c in enumerate(city)],
        "income_k": income,
        "price": [f"${v:,.2f}" for v in rng.lognormal(3, 1, n)],
        "signup": [f"{1 + i % 28:02d}/{1 + i % 12:02d}/2023" for i in range(n)],
        "active": rng.choice(["yes", "no"], n).tolist(),
        "email": [f"User{i}@mail.com" if i % 15 else "not-an-email" for i in range(n)],
        "notes": rng.choice(["ok", "N/A", "-", "fine", "null"], n).tolist(),
        "age": [int(a) if i % 50 else -999 for i, a in enumerate(rng.integers(18, 80, n))],
        "region": ["IN"] * n,
    })
    df = pl.concat([df, df.head(10)])
    report = analyze(df)
    print(to_json({k: report[k] for k in ("summary", "quality", "meta", "warnings")}))
    for s in report["suggestions"][:14]:
        print(f"[{s['priority_score']:>5}] {s['severity']:<8} {s['column']:<12} {s['suggestion']}")
    cleaned, _, steps = clean(df)
    print(f"\nrows {df.height} -> {cleaned.height}, cols {df.width} -> {cleaned.width}")
    print("failed steps:", [x for x in steps if not x["ok"]])


if __name__ == "__main__":
    _demo()