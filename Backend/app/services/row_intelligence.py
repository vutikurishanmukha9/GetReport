""" Row Intelligence """
from __future__ import annotations
import logging
import math
import os
import re

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

ROW_ID = "__row_id__"

__all__ = [
    "ROW_ID", "RowVerdict", "RowTriageConfig", "RowFlag", "RowTriageResult",
    "RepairConfig", "RepairResult", "triage_rows", "repair_rows", "triage_and_repair", "sparse_row_mask",
]


class RowVerdict(str, Enum):
    PROTECT = "PROTECT"
    REVIEW = "REVIEW"
    FIX = "FIX"
    EXCLUDE = "EXCLUDE"


_SEVERITY: dict[RowVerdict, int] = {
    RowVerdict.PROTECT: 0,
    RowVerdict.REVIEW: 1,
    RowVerdict.FIX: 2,
    RowVerdict.EXCLUDE: 3,
}
_BY_SEVERITY: dict[int, RowVerdict] = {v: k for k, v in _SEVERITY.items()}

_ID_TOKENS = frozenset({"id", "uuid", "guid", "code", "sku", "zip", "zipcode", "pin", "phone", "mobile", "key"})
_NUMERIC_JUNK = r"[,\s$\u20ac\u00a3\u20b9%]"
_CURRENCY_WORDS = r"(?i)\b(?:rs|inr|usd|eur|gbp|aud|cad)\b\.?"
_INVISIBLE = r"[\x{200B}-\x{200D}\x{2060}\x{FEFF}]"
_MOJIBAKE = r"\x{00C3}.|\x{00E2}\x{20AC}|\x{00C2}[\x{00A0}-\x{00BF}]|\x{00EF}\x{00BB}\x{00BF}"

_SUMMARY_LABEL = re.compile(
    r"^\s*(grand\s+total|sub\s*-?\s*total|total|sum|average|avg|mean|median|summary|count|overall)\b",
    re.IGNORECASE,
)
_NOTE_LABEL = re.compile(
    r"^\s*(source|sources|note|notes|generated|printed|report|page|confidential|disclaimer)\b",
    re.IGNORECASE,
)

# Tried in order; the first format that parses the most values wins ties, so day-first beats month-first.
_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%y", "%m/%d/%y",
]
_BOOL_TRUE = frozenset({"true", "t", "yes", "y"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n"})

_START_TOKENS = frozenset({"start", "order", "create", "created", "begin", "open", "opened", "signup"})
_END_TOKENS = frozenset({"end", "ship", "shipped", "close", "closed", "finish", "deliver", "delivered", "expire", "expiry"})
_BIRTH_TOKENS = frozenset({"birth", "dob"})
_AFTER_BIRTH_TOKENS = frozenset({"hire", "hired", "join", "joined", "order", "signup", "registration", "admission", "enrolled"})
_NONNEG_TOKENS = frozenset({
    "qty", "quantity", "price", "amount", "count", "salary", "revenue", "cost", "distance", "weight", "height",
    "units", "stock", "sales", "population", "duration",
})
_PCT_TOKENS = frozenset({"pct", "percent", "percentage"})
_PHONE_TOKENS = frozenset({"phone", "mobile", "tel", "telephone", "cell"})
_URL_TOKENS = frozenset({"url", "link", "website", "site"})
_PAIR_LOW = {"min", "minimum", "low", "lower"}
_PAIR_HIGH = {"max", "maximum", "high", "upper"}

_DEFAULT_RISK = {
    "blank_row": 1.0, "footer_note": 1.0, "repeated_header": 1.0, "summary_row": 1.0, "exact_duplicate": 1.0,
    "compound_risk": 1.0, "normalized_duplicate": 0.6, "duplicate_key": 0.5, "sparse_row": 0.6,
    "type_violation": 0.5, "masked_null": 0.15, "impute": 0.10, "outlier": 0.35, "sentinel_value": 0.40,
    "invalid_email": 0.35, "invalid_phone": 0.30, "invalid_url": 0.30, "invalid_date": 0.50,
    "mixed_date_format": 0.20, "age_out_of_range": 0.70, "percent_out_of_range": 0.60,
    "year_out_of_range": 0.60, "negative_value": 0.30, "future_date": 0.50, "chronology_violation": 0.60,
    "pair_violation": 0.60, "relationship_violation": 0.50, "joint_anomaly": 0.40, "encoding_damage": 0.30,
    "whitespace_noise": 0.05, "invisible_chars": 0.05, "category_variant": 0.10,
}

_FLAG_NOTES = {
    "blank_row": "row has no data",
    "footer_note": "looks like a report footer or note, not a record",
    "repeated_header": "repeats the column headers",
    "summary_row": "looks like a total or summary line",
    "exact_duplicate": "identical copy of an earlier row",
    "normalized_duplicate": "same as another row once case, spacing and rounding are ignored",
    "duplicate_key": "{col} is not unique but the rest of the row differs",
    "sparse_row": "almost every field is empty",
    "type_violation": "{col} holds text in a numeric column",
    "masked_null": "{col} holds a placeholder standing in for a missing value",
    "impute": "{col} is empty",
    "outlier": "{col} is far outside the normal range",
    "sentinel_value": "{col} holds a code value used in place of missing data",
    "invalid_email": "{col} is not a valid email address",
    "invalid_phone": "{col} is not a plausible phone number",
    "invalid_url": "{col} is not a valid web address",
    "invalid_date": "{col} cannot be read as a date",
    "mixed_date_format": "{col} uses a different date format from the rest of the column",
    "age_out_of_range": "{col} is outside 0 to 120",
    "percent_out_of_range": "{col} is outside 0 to 100",
    "year_out_of_range": "{col} is not a plausible year",
    "negative_value": "{col} is negative where negatives are unusual",
    "future_date": "{col} is in the future",
    "chronology_violation": "{col} comes before the date it should follow",
    "pair_violation": "{col} breaks its min/max or low/high partner",
    "encoding_damage": "{col} contains garbled characters",
    "whitespace_noise": "{col} has stray spaces",
    "invisible_chars": "{col} has hidden characters",
    "category_variant": "{col} is a spelling variant of a more common value",
    "compound_risk": "several independent problems in the same row",
}


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class RowTriageConfig:
    protected_columns: frozenset[str] = frozenset()
    placeholders: frozenset[str] = frozenset({"n/a", "na", "-999", "null", "?", "-", "missing", "unknown", "none", "nil", "nan", "#n/a", "--"})
    sparse_fill_ratio: float = 0.10
    sparse_min_columns: int = 5
    footer_scan_rows: int = 5
    type_majority: float = 0.90
    max_impute_null_rate: float = 0.30
    outlier_iqr_k: float = 3.0
    min_rows_for_stats: int = 20
    summary_column_scan: int = 5
    summary_max_share: float = 0.05
    dedupe_exact: bool = True
    # learned layer: each numeric column is predicted from the others (self-supervised, per dataset)
    learned_enabled: bool = True
    learned_min_rows: int = 300
    learned_z_threshold: float = 6.0
    learned_min_r2: float = 0.30
    learned_folds: int = 3
    learned_fit_sample: int = 50_000
    learned_max_targets: int = 15
    learned_max_predictors: int = 30
    learned_max_categories: int = 50
    learned_max_flag_share: float = 0.01
    learned_tail_multiple: float = 2.5
    learned_min_group: int = 50
    # text layer (see text_intelligence.py)
    text_enabled: bool = True
    text_conflicts_enabled: bool = True
    text_min_rows: int = 100
    text_max_classes: int = 20
    text_max_targets: int = 10
    text_min_accuracy: float = 0.97
    text_min_lift: float = 0.15
    text_p_low: float = 0.05
    text_confident: float = 0.90
    text_shape_coverage: float = 0.95
    text_shape_max_established: int = 3
    text_shape_established: float = 0.05
    text_fixed_width_share: float = 0.90
    text_coherent_group: float = 0.80
    text_rare_class_share: float = 0.02
    text_free_text_len: float = 60.0
    text_max_distinct: int = 5000
    text_variant_similarity: float = 0.84
    random_state: int = 0
    # ---- v2: hygiene checks -------------------------------------------------
    hygiene_enabled: bool = True              # whitespace, hidden characters, garbled encoding
    variants_enabled: bool = True             # case / spacing / punctuation spelling variants of one category
    variant_max_distinct: int = 5000
    # ---- v2: typed text columns --------------------------------------------
    date_text_enabled: bool = True
    date_text_majority: float = 0.90
    date_probe_rows: int = 500
    bool_text_enabled: bool = True
    # ---- v2: numeric checks -------------------------------------------------
    numeric_sentinels: frozenset[float] = frozenset({-999.0, -9999.0, -99999.0, 9999.0, 99999.0, 999999.0})
    sentinel_max_share: float = 0.30
    skew_aware_outliers: bool = True          # strictly positive, heavily skewed columns are judged on the log scale
    skew_threshold: float = 1.5
    mad_fallback_z: float = 6.0               # used when the IQR is zero (many tied values)
    semantic_rules_enabled: bool = True       # age, percent, year, non-negative quantities, phone, url, future birth dates
    pair_rules_enabled: bool = True           # min/max, low/high, start/end, birth/after-birth
    # ---- v2: duplicates -----------------------------------------------------
    normalized_dupes_enabled: bool = True
    normalized_dupe_verdict: RowVerdict = RowVerdict.REVIEW
    key_duplicates_enabled: bool = True
    # ---- v2: joint anomaly (rows no single column explains) ----------------
    isolation_enabled: bool = True
    isolation_z: float = 8.0
    isolation_max_share: float = 0.005
    # ---- v2: risk scoring and escalation -----------------------------------
    risk_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_RISK))
    default_risk_weight: float = 0.30
    escalate_risk: float | None = None        # REVIEW rows at or above this risk with enough flags become EXCLUDE
    escalate_min_flags: int = 3

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "RowTriageConfig":
        """conservative: flag less, change less. balanced: defaults. aggressive: flag more, escalate compound risk."""
        base = cls()
        if name == "conservative":
            base = replace(
                base, outlier_iqr_k=4.5, learned_z_threshold=8.0, text_min_accuracy=0.985, max_impute_null_rate=0.15,
                isolation_z=10.0, normalized_dupe_verdict=RowVerdict.REVIEW, sentinel_max_share=0.15,
            )
        elif name == "aggressive":
            base = replace(
                base, outlier_iqr_k=2.0, learned_z_threshold=4.5, max_impute_null_rate=0.50, isolation_z=6.0,
                normalized_dupe_verdict=RowVerdict.EXCLUDE, escalate_risk=0.85, escalate_min_flags=3,
            )
        elif name != "balanced":
            raise ValueError(f"unknown preset {name!r}; use conservative, balanced or aggressive")
        return replace(base, **overrides) if overrides else base


# ─── Flag / Result ───────────────────────────────────────────────────────────
@dataclass
class RowFlag:
    verdict: RowVerdict
    code: str
    column: str | None
    mask: np.ndarray  # bool, length n_rows

    @property
    def label(self) -> str:
        return f"{self.code}:{self.column}" if self.column else self.code

    @property
    def count(self) -> int:
        return int(self.mask.sum())


@dataclass
class RowTriageResult:
    row_ids: np.ndarray
    flags: list[RowFlag]
    severity: np.ndarray                      # int8 per row
    non_imputable_columns: dict[str, float]   # column -> null rate among valid rows
    column_roles: dict[str, str]              # id | protected | numeric | numeric_text | date_text | bool_text | text | other
    explanations: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    learned: dict[str, Any] = field(default_factory=lambda: {"ran": False, "reason": "not_run"})
    # ---- v2 ----
    risk: np.ndarray | None = None                                   # 0..1 per row, compounded over all flags
    null_rates: dict[str, float] = field(default_factory=dict)
    canonical_maps: dict[str, dict[str, str]] = field(default_factory=dict)       # variant -> canonical spelling
    fences: dict[str, tuple[float, float]] = field(default_factory=dict)          # outlier fences, original units
    learned_predictions: dict[str, dict[int, float]] = field(default_factory=dict)  # column -> position -> expected
    date_formats: dict[str, str] = field(default_factory=dict)
    bool_maps: dict[str, dict[str, bool]] = field(default_factory=dict)
    _label_cache: dict[int, list[str]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.risk is None:
            self.risk = np.zeros(self.row_ids.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.row_ids.shape[0])

    @property
    def quality_score(self) -> np.ndarray:
        return 1.0 - self.risk

    # ---- masks -------------------------------------------------------------
    def mask(self, verdict: RowVerdict) -> np.ndarray:
        return self.severity == _SEVERITY[verdict]

    @property
    def exclude_mask(self) -> np.ndarray:
        return self.mask(RowVerdict.EXCLUDE)

    def exclude_mask_without(self, *codes: str) -> np.ndarray:
        """Rows excluded for any reason other than ``codes`` (used to leave dedupe to its own step)."""
        out = np.zeros(self.n_rows, dtype=bool)
        for f in self.flags:
            if f.verdict is RowVerdict.EXCLUDE and f.code not in codes:
                out |= f.mask
        return out

    @property
    def protect_mask(self) -> np.ndarray:
        return self.mask(RowVerdict.PROTECT)

    def cell_mask(self, code: str, column: str | None = None) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=bool)
        for f in self.flags:
            if f.code == code and (column is None or f.column == column):
                out |= f.mask
        return out

    def columns_with(self, code: str) -> list[str]:
        return sorted({f.column for f in self.flags if f.code == code and f.column})

    def ids_with(self, code: str) -> np.ndarray:
        return self.row_ids[self.cell_mask(code)]

    # ---- reporting ---------------------------------------------------------
    def reasons_by_row(self) -> dict[int, list[str]]:
        if self._label_cache is None:
            reasons: dict[int, list[str]] = {}
            for f in self.flags:
                for pos in np.flatnonzero(f.mask):
                    reasons.setdefault(int(pos), []).append(f.label)
            self._label_cache = reasons
        return self._label_cache

    def audit_frame(self, include_protected: bool = False) -> pl.DataFrame:
        reasons = self.reasons_by_row()
        positions = np.arange(self.n_rows) if include_protected else np.array(sorted(reasons), dtype=int)
        return pl.DataFrame(
            {
                "row_id": [int(self.row_ids[p]) for p in positions],
                "verdict": [_BY_SEVERITY[int(self.severity[p])].value for p in positions],
                "risk": [round(float(self.risk[p]), 3) for p in positions],
                "reasons": [reasons.get(int(p), []) for p in positions],
                "why": [self.explain(int(p)) for p in positions],
            },
            schema={"row_id": pl.Int64, "verdict": pl.Utf8, "risk": pl.Float64, "reasons": pl.List(pl.Utf8), "why": pl.Utf8},
        )

    def explain(self, position: int) -> str | None:
        parts: list[str] = []
        for d in self.explanations.get(position, []):
            if d.get("column") == "*":
                parts.append(str(d["note"]))
            elif "note" in d:
                parts.append(f"{d['column']}={d['value']} ({d['note']})")
            else:
                parts.append(f"{d['column']}={d['value']} (expected about {d['expected']} given the rest of the row)")
        seen_codes = {d.get("column") for d in self.explanations.get(position, [])}
        for label in self.reasons_by_row().get(position, []):
            code, _, col = label.partition(":")
            if code in ("relationship_violation", "joint_anomaly") or code not in _FLAG_NOTES:
                continue
            if col and col in seen_codes and code in ("outlier",):
                continue
            parts.append(_FLAG_NOTES[code].format(col=col or "row"))
        return "; ".join(parts) if parts else None

    def column_report(self) -> pl.DataFrame:
        rows = []
        for c, role in self.column_roles.items():
            codes: dict[str, int] = {}
            for f in self.flags:
                if f.column == c:
                    codes[f.code] = codes.get(f.code, 0) + f.count
            rows.append({
                "column": c,
                "role": role,
                "null_rate": round(self.null_rates.get(c, 0.0), 4),
                "imputable": c not in self.non_imputable_columns,
                "flagged_cells": sum(codes.values()),
                "issues": "; ".join(f"{k}={v}" for k, v in sorted(codes.items(), key=lambda kv: -kv[1])),
            })
        return pl.DataFrame(rows, schema={
            "column": pl.Utf8, "role": pl.Utf8, "null_rate": pl.Float64, "imputable": pl.Boolean,
            "flagged_cells": pl.Int64, "issues": pl.Utf8,
        })

    def summary(self) -> dict[str, Any]:
        n = self.n_rows
        counts = {v.value: int((self.severity == s).sum()) for v, s in _SEVERITY.items()}
        by_reason: dict[str, int] = {}
        for f in self.flags:
            by_reason[f.code] = by_reason.get(f.code, 0) + f.count
        return {
            "rows_scanned": n,
            "verdicts": counts,
            "untouched_pct": round(counts["PROTECT"] / n * 100, 2) if n else 100.0,
            "mean_quality": round(float(1.0 - self.risk.mean()), 4) if n else 1.0,
            "by_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
            "non_imputable_columns": {k: round(v, 4) for k, v in self.non_imputable_columns.items()},
            "typed_text_columns": {
                "numeric_text": [c for c, r in self.column_roles.items() if r == "numeric_text"],
                "date_text": list(self.date_formats),
                "bool_text": list(self.bool_maps),
            },
            "learned": self.learned,
        }

    def to_dict(self, max_examples: int = 25) -> dict[str, Any]:
        audit = self.audit_frame()
        out = self.summary()
        out["examples"] = audit.sort("risk", descending=True).head(max_examples).to_dicts()
        out["columns"] = self.column_report().to_dicts()
        return out


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_id_column(name: str) -> bool:
    return bool(_tokens(name) & _ID_TOKENS)


def _is_text(dtype: pl.DataType) -> bool:
    return dtype in (pl.Utf8, pl.String)


def _blank_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    c = pl.col(col)
    if _is_text(dtype):
        return c.is_null() | (c.str.replace_all(_INVISIBLE, "").str.strip_chars() == "")
    if dtype in (pl.Float32, pl.Float64):
        return c.is_null() | c.is_nan()
    return c.is_null()


def _numeric_from_text(s: pl.Series) -> pl.Series:
    """Text to Float64. Handles currency symbols and words, thousands commas, percent signs and (123) negatives."""
    t = s.str.strip_chars()
    neg = t.str.contains(r"^\(.*\)$").fill_null(False)
    cleaned = t.str.replace_all(_CURRENCY_WORDS, "").str.replace_all(_NUMERIC_JUNK, "").str.replace_all(r"[()]", "")
    v = cleaned.cast(pl.Float64, strict=False)
    v = v.zip_with(~neg, -v).fill_nan(None)
    bad = v.is_infinite().fill_null(False)
    if bad.any():
        v = v.set(bad, None)
    return v


def _parse_date_series(s: pl.Series, fmt: str) -> pl.Series:
    t = s.str.strip_chars()
    if "%H" in fmt:
        return t.str.to_datetime(fmt, strict=False).dt.date()
    return t.str.to_date(fmt, strict=False)


def _parse_dates_multi(s: pl.Series, primary: str) -> tuple[pl.Series, np.ndarray]:
    """Primary format first, then every other known format. Returns (dates, primary_ok mask)."""
    first = _parse_date_series(s, primary)
    primary_ok = first.is_not_null().to_numpy().astype(bool)
    out = first
    for fmt in _DATE_FORMATS:
        if fmt != primary and out.null_count() > 0:
            out = out.fill_null(_parse_date_series(s, fmt))
    return out, primary_ok


def _detect_date_format(s: pl.Series, cand: np.ndarray, cfg: RowTriageConfig) -> str | None:
    sub = s.filter(pl.Series(cand)).head(cfg.date_probe_rows)
    if sub.len() < 5:
        return None
    best, best_rate = None, 0.0
    for fmt in _DATE_FORMATS:
        try:
            rate = int(_parse_date_series(sub, fmt).is_not_null().sum()) / sub.len()
        except Exception:
            continue
        if rate > best_rate + 1e-9:
            best, best_rate = fmt, rate
    return best if best is not None and best_rate >= cfg.date_text_majority * 0.6 else None


def _skew(v: np.ndarray) -> float:
    sd = float(np.std(v))
    return float(np.mean(((v - np.mean(v)) / sd) ** 3)) if sd > 0 else 0.0


def _sparse_limit(ncols: int, cfg: "RowTriageConfig") -> int:
    """Max filled cells for a row to count as sparse. Never below 1, so an id-only row is always sparse."""
    return max(1, int(cfg.sparse_fill_ratio * ncols))


# ─── Learned layer ───────────────────────────────────────────────────────────
def _model_matrix(
    data: pl.DataFrame, roles: dict[str, str], valid: np.ndarray, cfg: RowTriageConfig,
    date_values: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Returns (numeric_features, categorical_codes). NaN marks missing; the model handles NaN natively."""
    numeric: dict[str, np.ndarray] = {}
    categorical: dict[str, np.ndarray] = {}
    for c in data.columns:
        role = roles.get(c)
        s = data[c]
        if role in ("id", "protected"):
            continue
        if role == "numeric":
            numeric[c] = s.cast(pl.Float64, strict=False).to_numpy().astype(float)
        elif role == "numeric_text":
            numeric[c] = _numeric_from_text(s).to_numpy().astype(float)
        elif role == "date_text" and date_values and c in date_values:
            numeric[c] = date_values[c]
        elif s.dtype in (pl.Date, pl.Datetime):
            numeric[c] = s.cast(pl.Datetime("us")).cast(pl.Int64).cast(pl.Float64).to_numpy().astype(float) / 86_400_000_000.0
        elif role in ("text", "bool_text"):
            sub = s.filter(pl.Series(valid)).drop_nulls()
            if 2 <= sub.n_unique() <= cfg.learned_max_categories:
                cats = sub.unique().sort().to_list()
                lookup = {v: i for i, v in enumerate(cats)}
                categorical[c] = np.array([lookup.get(v, -1) if v is not None else -1 for v in s.to_list()], dtype=float)
    return numeric, categorical


def _oof_predictions(
    F: np.ndarray, y: np.ndarray, cat_mask: np.ndarray, valid_pos: np.ndarray, cfg: RowTriageConfig,
    exclude_from_training: np.ndarray | None = None,
):
    """Out-of-fold predictions so a row is never judged by a model that memorised it."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng([cfg.random_state, 0x5EED])
    known = valid_pos[~np.isnan(y[valid_pos])]
    folds = rng.integers(0, cfg.learned_folds, known.size)
    pred = np.full(y.shape[0], np.nan)
    for k in range(cfg.learned_folds):
        train = known[folds != k]
        if exclude_from_training is not None:
            train = train[~exclude_from_training[train]]
        test = known[folds == k]
        if train.size > cfg.learned_fit_sample:
            train = rng.choice(train, cfg.learned_fit_sample, replace=False)
        if train.size < 30 or test.size == 0:
            continue
        model = HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.1, max_leaf_nodes=15, categorical_features=cat_mask if cat_mask.any() else None,
            random_state=cfg.random_state,
        )
        model.fit(F[train], y[train])
        pred[test] = model.predict(F[test])
    return pred, known


def _robust_sigma(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    sig = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    return sig if sig > 0 else float(np.std(r))


def _scaled_residuals(resid: np.ndarray, F_rows: np.ndarray, base_sigma: float, cfg: RowTriageConfig) -> np.ndarray:
    """|residual| / error scale, where the scale depends on which predictors were missing for that row."""
    nan = np.isnan(F_rows)
    cols = np.flatnonzero(nan.any(axis=0))
    if cols.size == 0:
        return np.abs(resid) / base_sigma
    cols = cols[np.argsort(-nan[:, cols].sum(axis=0))][:20]
    pattern = (nan[:, cols].astype(np.int64) * (1 << np.arange(cols.size, dtype=np.int64))).sum(axis=1)
    sigma = np.full(resid.shape[0], base_sigma)
    for pid in np.unique(pattern):
        m = pattern == pid
        if m.sum() >= cfg.learned_min_group:
            g = _robust_sigma(resid[m])
            if g > 0:
                sigma[m] = g
    return np.abs(resid) / sigma


def _joint_anomaly_layer(
    numeric: dict[str, np.ndarray], categorical: dict[str, np.ndarray], valid: np.ndarray, cfg: RowTriageConfig,
    flags: list[RowFlag], explanations: dict[int, list[dict[str, Any]]], skip_mask: np.ndarray, n: int,
) -> int:
    """Rows that look strange as a whole even though no single column or pair of columns gives them away."""
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return 0
    mats: list[np.ndarray] = []
    for c in list(numeric)[: cfg.learned_max_predictors]:
        a = numeric[c].copy()
        ok = valid & np.isfinite(a)
        med = float(np.median(a[ok])) if ok.any() else 0.0
        mats.append(np.where(np.isfinite(a), a, med))
    for c in list(categorical)[: cfg.learned_max_predictors]:
        mats.append(categorical[c])
    if len(mats) < 3:
        return 0
    X = np.column_stack(mats)
    vp = np.flatnonzero(valid)
    rng = np.random.default_rng([cfg.random_state, 0x150])
    fit = vp if vp.size <= cfg.learned_fit_sample else rng.choice(vp, cfg.learned_fit_sample, replace=False)
    iso = IsolationForest(n_estimators=50, max_samples=min(256, fit.size), n_jobs=1, random_state=cfg.random_state).fit(X[fit])
    s = -iso.score_samples(X[vp])
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    if mad <= 0:
        return 0
    z = (s - med) / mad
    hit = z >= cfg.isolation_z
    cap = max(1, int(cfg.isolation_max_share * vp.size))
    if hit.sum() > cap:
        hit = z >= np.sort(z)[-cap]
    pick = hit & ~skip_mask[vp]
    if not pick.any():
        return 0
    mask = np.zeros(n, dtype=bool)
    mask[vp[pick]] = True
    flags.append(RowFlag(RowVerdict.REVIEW, "joint_anomaly", None, mask))
    for pos, zz in zip(vp[pick], z[pick]):
        explanations.setdefault(int(pos), []).append(
            {"column": "*", "value": "", "note": "unusual combination of values across the whole row", "z": round(float(zz), 1)}
        )
    return int(pick.sum())


def _learned_layer(
    data: pl.DataFrame,
    roles: dict[str, str],
    valid: np.ndarray,
    cfg: RowTriageConfig,
    flags: list[RowFlag],
    explanations: dict[int, list[dict[str, Any]]],
    predictions: dict[str, dict[int, float]],
    date_values: dict[str, np.ndarray],
    masked_cells: dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Cross-column consistency. Each numeric column is predicted from the others; a cell is flagged when its
    actual value is far from what the rest of the row implies. REVIEW flags only. Columns the others cannot
    explain (R2 below the gate) are skipped, so ordinary spread is not treated as error. A joint anomaly pass
    then catches rows that are odd as a whole.
    """
    n_valid = int(valid.sum())
    if n_valid < cfg.learned_min_rows:
        return {"ran": False, "reason": f"too_few_rows ({n_valid} < {cfg.learned_min_rows})"}
    try:
        import sklearn  # noqa: F401
    except ImportError:
        logger.warning("scikit-learn not installed; learned row layer skipped")
        return {"ran": False, "reason": "scikit-learn_not_installed"}

    numeric, categorical = _model_matrix(data, roles, valid, cfg, date_values)
    for mc, mm in masked_cells.items():  # sentinel codes are not data: neither judge them nor learn from them
        if mc in numeric:
            numeric[mc] = np.where(mm, np.nan, numeric[mc])
    if not numeric or len(numeric) + len(categorical) < 2:
        return {"ran": False, "reason": "too_few_usable_columns"}

    names = list(numeric)[: cfg.learned_max_predictors] + list(categorical)[: max(0, cfg.learned_max_predictors - len(numeric))]
    full = np.column_stack([numeric[c] if c in numeric else categorical[c] for c in names])
    is_cat = np.array([c in categorical for c in names])
    valid_pos = np.flatnonzero(valid)

    targets = sorted(
        (c for c in numeric if c in names),
        key=lambda c: float(np.isnan(numeric[c][valid_pos]).mean()),
    )[: cfg.learned_max_targets]

    judged: dict[str, dict[str, float]] = {}
    skipped: dict[str, str] = {}
    total_mask = np.zeros(data.height, dtype=bool)
    for c in targets:
        j = names.index(c)
        keep_cols = [i for i in range(len(names)) if i != j]
        F = full[:, keep_cols]
        y = numeric[c]
        pred, known = _oof_predictions(F, y, is_cat[keep_cols], valid_pos, cfg)
        ok = known[~np.isnan(pred[known])]
        if ok.size < cfg.learned_min_rows:
            skipped[c] = "too_few_known_values"
            continue
        resid = y[ok] - pred[ok]
        var_y = float(np.var(y[ok]))
        r2 = 1.0 - float(np.var(resid)) / var_y if var_y > 0 else 0.0
        if r2 < cfg.learned_min_r2:
            skipped[c] = f"not_explained_by_other_columns (r2={r2:.2f})"
            continue
        sigma = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
        if sigma <= 0:
            sigma = float(np.std(resid))
        if sigma <= 0:
            skipped[c] = "zero_residual_spread"
            continue

        # Refit once without the rows that look wrong, so bad rows cannot teach the model that bad is normal.
        first = np.zeros(data.height, dtype=bool)
        first[ok[np.abs(resid) / sigma >= cfg.learned_z_threshold]] = True
        if first.any():
            pred2, _ = _oof_predictions(F, y, is_cat[keep_cols], valid_pos, cfg, exclude_from_training=first)
            ok2 = known[~np.isnan(pred2[known])]
            if ok2.size >= cfg.learned_min_rows:
                pred, ok = pred2, ok2
                resid = y[ok] - pred[ok]
                clean_resid = resid[~first[ok]]
                sig2 = float(np.median(np.abs(clean_resid - np.median(clean_resid)))) * 1.4826 if clean_resid.size else 0.0
                sigma = sig2 if sig2 > 0 else sigma

        z = _scaled_residuals(y[ok] - pred[ok], F[ok], sigma, cfg)
        threshold = max(cfg.learned_z_threshold, cfg.learned_tail_multiple * float(np.percentile(z, 99)))
        hit = z >= threshold
        cap = max(1, int(cfg.learned_max_flag_share * ok.size))
        if hit.sum() > cap:
            hit = z >= max(np.sort(z)[-cap], threshold)
        judged[c] = {"r2": round(r2, 3), "flagged": int(hit.sum()), "threshold": round(float(threshold), 1)}
        if not hit.any():
            continue
        mask = np.zeros(data.height, dtype=bool)
        mask[ok[hit]] = True
        flags.append(RowFlag(RowVerdict.REVIEW, "relationship_violation", c, mask))
        total_mask |= mask
        for pos, zz in zip(ok[hit], z[hit]):
            exp = pred[pos]
            predictions.setdefault(c, {})[int(pos)] = float(exp)
            exp_txt = int(round(exp)) if float(exp).is_integer() or abs(exp) >= 100 else round(float(exp), 2)
            explanations.setdefault(int(pos), []).append(
                {"column": c, "value": data[c][int(pos)], "expected": exp_txt, "z": round(float(zz), 1)}
            )

    joint = 0
    if cfg.isolation_enabled:
        try:
            joint = _joint_anomaly_layer(numeric, categorical, valid, cfg, flags, explanations, total_mask, data.height)
        except Exception as exc:  # the layer is advisory; never let it break triage
            logger.warning("Joint anomaly pass skipped: %s", exc)

    for pos in explanations:
        explanations[pos].sort(key=lambda d: -d.get("z", 0.0))
        del explanations[pos][3:]

    logger.info("Learned layer: judged %d columns, %d cells flagged, %d joint anomalies", len(judged), int(total_mask.sum()), joint)
    return {
        "ran": True,
        "reason": "ok",
        "method": "cross_column_prediction",
        "judged_columns": judged,
        "skipped_columns": skipped,
        "rows_flagged": int(total_mask.sum()),
        "joint_anomalies": joint,
    }


def sparse_row_mask(df: pl.DataFrame, config: RowTriageConfig | None = None) -> np.ndarray:
    """Rows with almost nothing filled. The pipeline must never impute these (it would invent a record)."""
    cfg = config or RowTriageConfig()
    cols = [c for c in df.columns if c != ROW_ID]
    if df.height == 0 or len(cols) < cfg.sparse_min_columns:
        return np.zeros(df.height, dtype=bool)
    filled = np.zeros(df.height, dtype=np.int32)
    for c in cols:
        filled += (~df.select(_blank_expr(c, df[c].dtype)).to_series().to_numpy().astype(bool)).astype(np.int32)
    return (filled > 0) & (filled <= _sparse_limit(len(cols), cfg))


# ─── Triage ──────────────────────────────────────────────────────────────────
def triage_rows(df: pl.DataFrame, config: RowTriageConfig | None = None) -> RowTriageResult:
    """Classify every row of ``df``. ``df`` may carry a ROW_ID column to keep ids stable across steps."""
    cfg = config or RowTriageConfig()
    n = df.height
    ids = df[ROW_ID].to_numpy().astype(np.int64) if ROW_ID in df.columns else np.arange(n, dtype=np.int64)
    data = df.drop(ROW_ID) if ROW_ID in df.columns else df
    cols = data.columns
    flags: list[RowFlag] = []
    roles: dict[str, str] = {}

    def add(verdict: RowVerdict, code: str, column: str | None, mask: np.ndarray) -> None:
        if mask.any():
            flags.append(RowFlag(verdict, code, column, mask.astype(bool)))

    if n == 0 or not cols:
        return RowTriageResult(ids, [], np.zeros(n, dtype=np.int8), {}, {})

    text_cols = [c for c in cols if _is_text(data[c].dtype)]
    for c in cols:
        if c in cfg.protected_columns:
            roles[c] = "protected"
        elif _is_id_column(c):
            roles[c] = "id"
        elif data[c].dtype.is_numeric():
            roles[c] = "numeric"
        elif _is_text(data[c].dtype):
            roles[c] = "text"
        else:
            roles[c] = "other"

    blanks: dict[str, np.ndarray] = {
        c: data.select(_blank_expr(c, data[c].dtype)).to_series().to_numpy().astype(bool) for c in cols
    }
    filled = np.zeros(n, dtype=np.int32)
    for c in cols:
        filled += (~blanks[c]).astype(np.int32)
    ncols = len(cols)

    parsed: dict[str, pl.Series] = {}
    parsed_ok: dict[str, np.ndarray] = {}
    placeholder_np: dict[str, np.ndarray] = {}
    for c in text_cols:
        p = _numeric_from_text(data[c])
        parsed[c] = p
        parsed_ok[c] = p.is_not_null().to_numpy().astype(bool)
        placeholder_np[c] = (
            data[c].str.strip_chars().str.to_lowercase().is_in(list(cfg.placeholders)).fill_null(False).to_numpy().astype(bool)
            if cfg.placeholders else np.zeros(n, dtype=bool)
        )

    # 1. Structural: blank rows
    add(RowVerdict.EXCLUDE, "blank_row", None, filled == 0)

    # 2. Structural: footer notes
    if ncols >= 3:
        tail = np.zeros(n, dtype=bool)
        tail[max(0, n - cfg.footer_scan_rows):] = True
        single = filled == 1
        for c in text_cols:
            s = data[c]
            looks_like_note = (
                s.str.len_chars().fill_null(0).gt(25)
                | s.str.contains(r"[:\*]").fill_null(False)
                | s.str.contains("(?i)" + _NOTE_LABEL.pattern).fill_null(False)
            ).to_numpy()
            add(RowVerdict.EXCLUDE, "footer_note", c, single & tail & ~blanks[c] & looks_like_note)

    # 3. Structural: repeated header rows
    if len(text_cols) >= 2:
        needed = max(2, math.ceil(len(text_cols) / 2))
        hits = np.zeros(n, dtype=np.int32)
        for c in text_cols:
            target = _norm(c)
            if not target:
                continue
            m = (data[c].str.to_lowercase().str.replace_all(r"[^a-z0-9]+", "").fill_null("") == target).to_numpy()
            hits += m.astype(np.int32)
        add(RowVerdict.EXCLUDE, "repeated_header", None, hits >= needed)

    # 4. Structural: summary / total rows
    for c in text_cols[: cfg.summary_column_scan]:
        s = data[c]
        label = s.str.contains("(?i)" + _SUMMARY_LABEL.pattern).fill_null(False)
        label = label & s.str.len_chars().fill_null(0).le(40)
        label_np = label.to_numpy().astype(bool)
        if not label_np.any():
            continue
        if n >= cfg.min_rows_for_stats and label_np.sum() > cfg.summary_max_share * n:
            continue
        other_text_filled = np.zeros(n, dtype=bool)
        for o in text_cols:
            if o != c:
                other_text_filled |= ~blanks[o] & ~parsed_ok[o]
        others_filled = filled - (~blanks[c]).astype(np.int32)
        add(RowVerdict.EXCLUDE, "summary_row", c, label_np & ~other_text_filled & (others_filled >= 1))

    # 5. Structural: duplicates (later copies only)
    exact_dup = np.zeros(n, dtype=bool)
    if cfg.dedupe_exact:
        try:
            exact_dup = ~data.select(pl.struct(pl.all()).is_first_distinct()).to_series().to_numpy().astype(bool)
            add(RowVerdict.EXCLUDE, "exact_duplicate", None, exact_dup & (filled > 0))
        except Exception as exc:
            logger.warning("Duplicate check skipped: %s", exc)
    if cfg.normalized_dupes_enabled:
        try:
            exprs = []
            for c in cols:
                dt = data[c].dtype
                if _is_text(dt):
                    exprs.append(pl.col(c).str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias(c))
                elif dt in (pl.Float32, pl.Float64):
                    exprs.append(pl.col(c).round(6).alias(c))
                else:
                    exprs.append(pl.col(c))
            norm_dup = ~data.select(exprs).select(pl.struct(pl.all()).is_first_distinct()).to_series().to_numpy().astype(bool)
            add(cfg.normalized_dupe_verdict, "normalized_duplicate", None, norm_dup & ~exact_dup & (filled > 0))
        except Exception as exc:
            logger.warning("Normalized duplicate check skipped: %s", exc)

    # 6. Sparse rows
    if ncols >= cfg.sparse_min_columns:
        already_footer = np.zeros(n, dtype=bool)
        for f in flags:
            if f.code == "footer_note":
                already_footer |= f.mask
        add(RowVerdict.REVIEW, "sparse_row", None, (filled > 0) & (filled <= _sparse_limit(ncols, cfg)) & ~already_footer)

    excluded_so_far = np.zeros(n, dtype=bool)
    for f in flags:
        if f.verdict is RowVerdict.EXCLUDE:
            excluded_so_far |= f.mask
    valid = ~excluded_so_far
    n_valid = int(valid.sum())
    non_imputable: dict[str, float] = {}
    null_rates: dict[str, float] = {}
    numeric_np: dict[str, np.ndarray] = {}
    parsed_dates: dict[str, pl.Series] = {}
    date_values: dict[str, np.ndarray] = {}
    date_formats: dict[str, str] = {}
    bool_maps: dict[str, dict[str, bool]] = {}
    canonical_maps: dict[str, dict[str, str]] = {}
    fences: dict[str, tuple[float, float]] = {}
    sentinel_cells: dict[str, np.ndarray] = {}
    now = datetime.now()

    # 6b. Hygiene: stray whitespace, hidden characters, garbled encoding
    if cfg.hygiene_enabled:
        for c in text_cols:
            if roles[c] == "protected":
                continue
            s = data[c]
            present = valid & ~blanks[c]
            ws = s.str.contains(r"^\s|\s$|\s{2,}|\x{00A0}").fill_null(False).to_numpy().astype(bool)
            hid = s.str.contains(_INVISIBLE).fill_null(False).to_numpy().astype(bool)
            add(RowVerdict.FIX, "whitespace_noise", c, ws & present)
            add(RowVerdict.FIX, "invisible_chars", c, hid & valid & ~(blanks[c] & ~hid))
            if roles[c] != "id":
                moj = s.str.contains(_MOJIBAKE).fill_null(False).to_numpy().astype(bool)
                add(RowVerdict.REVIEW, "encoding_damage", c, moj & present)

    # 7. Column-level checks
    for c in cols:
        role = roles[c]
        s = data[c]
        numeric_text = False
        date_text = False

        if role == "text":
            cand = valid & ~blanks[c] & ~placeholder_np[c]
            total = int(cand.sum())
            if total >= 5:
                rate = float((parsed_ok[c] & cand).sum()) / total
                if rate >= cfg.type_majority:
                    numeric_text = True
                    roles[c] = "numeric_text"
                    if rate < 1.0:
                        add(RowVerdict.REVIEW, "type_violation", c, cand & ~parsed_ok[c])
                elif cfg.date_text_enabled and rate < 0.5:
                    fmt = _detect_date_format(s, cand, cfg)
                    if fmt is not None:
                        dates, primary_ok = _parse_dates_multi(s, fmt)
                        any_ok = dates.is_not_null().to_numpy().astype(bool)
                        if float((any_ok & cand).sum()) / total >= cfg.date_text_majority:
                            date_text = True
                            roles[c] = "date_text"
                            date_formats[c] = fmt
                            parsed_dates[c] = dates
                            date_values[c] = dates.cast(pl.Int32).cast(pl.Float64).to_numpy().astype(float)
                            add(RowVerdict.REVIEW, "invalid_date", c, cand & ~any_ok)
                            add(RowVerdict.REVIEW, "mixed_date_format", c, cand & any_ok & ~primary_ok)
                if not numeric_text and not date_text and cfg.bool_text_enabled and role != "id":
                    vals = s.filter(pl.Series(cand)).str.strip_chars().str.to_lowercase().unique().to_list()
                    if 2 <= len(vals) <= 8 and all(v in _BOOL_TRUE or v in _BOOL_FALSE for v in vals):
                        if any(v in _BOOL_TRUE for v in vals) and any(v in _BOOL_FALSE for v in vals):
                            roles[c] = "bool_text"
                            bool_maps[c] = {**{v: True for v in _BOOL_TRUE}, **{v: False for v in _BOOL_FALSE}}

        is_num = role == "numeric" or numeric_text
        if role == "numeric":
            numeric_np[c] = s.cast(pl.Float64, strict=False).to_numpy().astype(float)
        elif numeric_text:
            numeric_np[c] = parsed[c].to_numpy().astype(float)
        arr = numeric_np.get(c)

        # 7a. masked null placeholders in text columns
        if role == "text" and cfg.placeholders:
            add(RowVerdict.FIX, "masked_null", c, placeholder_np[c] & valid)

        # 7b. missing values
        if (is_num or role == "text") and not date_text and n_valid:
            missing = blanks[c] | placeholder_np[c] if role == "text" else blanks[c]
            null_rate = float((missing & valid).sum()) / n_valid
            null_rates[c] = null_rate
            if null_rate > 0:
                if null_rate <= cfg.max_impute_null_rate:
                    add(RowVerdict.FIX, "impute", c, blanks[c] & valid)
                else:
                    non_imputable[c] = null_rate
        elif n_valid:
            null_rates[c] = float((blanks[c] & valid).sum()) / n_valid

        # 7c. sentinel codes, then outliers (judged without the sentinels so they cannot distort the fences)
        sentinel_mask = np.zeros(n, dtype=bool)
        if is_num and arr is not None and cfg.numeric_sentinels and n_valid >= cfg.min_rows_for_stats:
            for sv in cfg.numeric_sentinels:
                m = valid & (arr == sv)
                if not m.any() or m.sum() > cfg.sentinel_max_share * n_valid:
                    continue
                others = arr[valid & ~m & np.isfinite(arr)]
                if others.size >= cfg.min_rows_for_stats and (sv > others.max() or sv < others.min()):
                    sentinel_mask |= m
            add(RowVerdict.REVIEW, "sentinel_value", c, sentinel_mask)
            if sentinel_mask.any():
                sentinel_cells[c] = sentinel_mask

        if is_num and arr is not None and n_valid >= cfg.min_rows_for_stats:
            work = arr.copy()
            work[sentinel_mask] = np.nan
            usable = valid & np.isfinite(work)
            v = work[usable]
            if v.size >= cfg.min_rows_for_stats:
                with np.errstate(all="ignore"):
                    use_log = cfg.skew_aware_outliers and float(v.min()) > 0 and _skew(v) > cfg.skew_threshold
                    basis = np.log1p(v) if use_log else v
                    q1, q3 = np.percentile(basis, [25, 75])
                    iqr = q3 - q1
                    if iqr > 0:
                        lo, hi = q1 - cfg.outlier_iqr_k * iqr, q3 + cfg.outlier_iqr_k * iqr
                    else:
                        med = float(np.median(basis))
                        mad = float(np.median(np.abs(basis - med))) * 1.4826
                        lo, hi = (med - cfg.mad_fallback_z * mad, med + cfg.mad_fallback_z * mad) if mad > 0 else (-np.inf, np.inf)
                    if np.isfinite(lo) or np.isfinite(hi):
                        view = np.log1p(np.where(work > -1, work, np.nan)) if use_log else work
                        out = ((view < lo) | (view > hi)) & usable
                        if use_log:
                            out |= usable & (work <= -1)
                        add(RowVerdict.REVIEW, "outlier", c, out)
                        fences[c] = (float(np.expm1(lo)) if use_log else float(lo), float(np.expm1(hi)) if use_log else float(hi))

        # 7d. business rules (flag only)
        toks = _tokens(c)
        if role == "text" and not numeric_text and not date_text and "email" in toks:
            bad = (~s.str.strip_chars().str.contains(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").fill_null(True)).to_numpy().astype(bool)
            add(RowVerdict.REVIEW, "invalid_email", c, bad & ~blanks[c] & ~placeholder_np[c] & valid)
        if cfg.semantic_rules_enabled:
            if _is_text(s.dtype) and role != "protected" and not numeric_text and not date_text:
                if toks & _PHONE_TOKENS:
                    digits = s.str.replace_all(r"\D", "").str.len_chars().fill_null(0).to_numpy()
                    add(RowVerdict.REVIEW, "invalid_phone", c, ((digits < 7) | (digits > 15)) & ~blanks[c] & ~placeholder_np[c] & valid)
                if toks & _URL_TOKENS:
                    ok_url = s.str.strip_chars().str.contains(r"(?i)^(?:https?://|www\.)\S+\.\S+$").fill_null(False).to_numpy().astype(bool)
                    add(RowVerdict.REVIEW, "invalid_url", c, ~ok_url & ~blanks[c] & ~placeholder_np[c] & valid)
            if is_num and arr is not None and role != "protected":
                with np.errstate(invalid="ignore"):
                    finite = np.isfinite(arr) & ~sentinel_mask
                    if "age" in toks:
                        add(RowVerdict.REVIEW, "age_out_of_range", c, finite & ((arr < 0) | (arr > 120)) & valid)
                    if toks & _PCT_TOKENS:
                        add(RowVerdict.REVIEW, "percent_out_of_range", c, finite & ((arr < 0) | (arr > 100)) & valid)
                    if "year" in toks:
                        add(RowVerdict.REVIEW, "year_out_of_range", c, finite & ((arr < 1800) | (arr > now.year + 100)) & valid)
                    if toks & _NONNEG_TOKENS and "age" not in toks:
                        add(RowVerdict.REVIEW, "negative_value", c, finite & (arr < 0) & valid)

    # 8. Cross-column rules
    date_series: dict[str, pl.Series] = {}
    for c in cols:
        if data[c].dtype in (pl.Date, pl.Datetime):
            date_series[c] = data[c].cast(pl.Datetime("us"))
        elif c in parsed_dates:
            date_series[c] = parsed_dates[c].cast(pl.Datetime("us"))

    if cfg.pair_rules_enabled and len(date_series) >= 2:
        starts = [c for c in date_series if _tokens(c) & _START_TOKENS][:4]
        ends = [c for c in date_series if _tokens(c) & _END_TOKENS][:4]
        chron: dict[str, np.ndarray] = {}
        for e in ends:
            for st in starts:
                if st != e:
                    bad = (date_series[e] < date_series[st]).fill_null(False).to_numpy().astype(bool)
                    chron[e] = chron.get(e, np.zeros(n, dtype=bool)) | bad
        for b in [c for c in date_series if _tokens(c) & _BIRTH_TOKENS][:2]:
            for later in [c for c in date_series if _tokens(c) & _AFTER_BIRTH_TOKENS and c != b][:4]:
                bad = (date_series[b] > date_series[later]).fill_null(False).to_numpy().astype(bool)
                chron[b] = chron.get(b, np.zeros(n, dtype=bool)) | bad
        for col, bad in chron.items():
            add(RowVerdict.REVIEW, "chronology_violation", col, bad & valid)

    if cfg.semantic_rules_enabled:
        for c, ds in date_series.items():
            if _tokens(c) & _BIRTH_TOKENS:
                fut = (ds > pl.lit(now).cast(pl.Datetime("us"))).fill_null(False).to_numpy().astype(bool)
                add(RowVerdict.REVIEW, "future_date", c, fut & valid)

    if cfg.pair_rules_enabled:
        num_cols = list(numeric_np)
        for lo_c in num_cols:
            lo_toks = _tokens(lo_c)
            if not lo_toks & _PAIR_LOW:
                continue
            rest = lo_toks - _PAIR_LOW
            for hi_c in num_cols:
                hi_toks = _tokens(hi_c)
                if hi_c != lo_c and hi_toks & _PAIR_HIGH and (hi_toks - _PAIR_HIGH) == rest:
                    with np.errstate(invalid="ignore"):
                        bad = np.isfinite(numeric_np[lo_c]) & np.isfinite(numeric_np[hi_c]) & (numeric_np[lo_c] > numeric_np[hi_c])
                    add(RowVerdict.REVIEW, "pair_violation", hi_c, bad & valid)

    # 8b. Key duplicates: an id that should be unique appears on rows that differ in content
    if cfg.key_duplicates_enabled:
        for c in cols:
            if roles[c] == "id":
                nn = data[c].drop_nulls()
                if nn.len() >= cfg.min_rows_for_stats and 0.9 <= nn.n_unique() / nn.len() < 1.0:
                    dup = data[c].is_duplicated().to_numpy().astype(bool) & ~blanks[c]
                    add(RowVerdict.REVIEW, "duplicate_key", c, dup & valid & ~exact_dup)
                    break

    # 8c. Category spelling variants (case, spacing, punctuation) inside one column
    if cfg.variants_enabled:
        for c in text_cols:
            if roles[c] not in ("text",):
                continue
            present = valid & ~blanks[c] & ~placeholder_np[c]
            sub = data[c].filter(pl.Series(present))
            if sub.len() < 5 or sub.n_unique() > cfg.variant_max_distinct or sub.n_unique() > 0.5 * sub.len():
                continue
            vc = sub.value_counts()
            values, counts = vc[:, 0].to_list(), vc[:, 1].to_list()
            groups: dict[str, list[tuple[str, int]]] = {}
            for v_, k_ in zip(values, counts):
                key = _norm(str(v_))
                if key:
                    groups.setdefault(key, []).append((str(v_), int(k_)))
            cmap: dict[str, str] = {}
            for members in groups.values():
                if len(members) > 1:
                    canon = sorted(members, key=lambda t: (-t[1], t[0]))[0][0]
                    cmap.update({m: canon for m, _ in members if m != canon})
            if cmap:
                canonical_maps[c] = cmap
                mask = data[c].is_in(list(cmap)).fill_null(False).to_numpy().astype(bool) & present
                add(RowVerdict.FIX, "category_variant", c, mask)

    explanations: dict[int, list[dict[str, Any]]] = {}
    predictions: dict[str, dict[int, float]] = {}
    learned_info: dict[str, Any] = {"ran": False, "reason": "disabled"}
    if cfg.learned_enabled:
        learned_info = _learned_layer(data, roles, valid, cfg, flags, explanations, predictions, date_values, sentinel_cells)

    if cfg.text_enabled:
        try:
            from app.services.text_intelligence import run_text_layer  # local import: text layer imports this module

            learned_info["text"] = run_text_layer(data, roles, valid, cfg, flags, explanations)
        except ImportError:
            learned_info["text"] = {"ran": False, "reason": "text_layer_unavailable"}

    severity = np.zeros(n, dtype=np.int8)
    for f in flags:
        np.maximum(severity, np.where(f.mask, _SEVERITY[f.verdict], 0).astype(np.int8), out=severity)

    # 9. Risk: probability-style compounding over every flag that touches the row
    survive = np.ones(n)
    review_count = np.zeros(n, dtype=np.int32)
    for f in flags:
        survive[f.mask] *= 1.0 - min(0.999, cfg.risk_weights.get(f.code, cfg.default_risk_weight))
        if f.verdict is RowVerdict.REVIEW:
            review_count[f.mask] += 1
    risk = 1.0 - survive

    if cfg.escalate_risk is not None:
        esc = (severity < _SEVERITY[RowVerdict.EXCLUDE]) & (risk >= cfg.escalate_risk) & (review_count >= cfg.escalate_min_flags)
        if esc.any():
            flags.append(RowFlag(RowVerdict.EXCLUDE, "compound_risk", None, esc))
            severity[esc] = _SEVERITY[RowVerdict.EXCLUDE]
            risk[esc] = 1.0

    result = RowTriageResult(
        ids, flags, severity, non_imputable, roles, explanations, learned_info, risk, null_rates,
        canonical_maps, fences, predictions, date_formats, bool_maps,
    )
    logger.info("Row triage: %s", result.summary()["verdicts"])
    return result


# ─── Repair engine ───────────────────────────────────────────────────────────
@dataclass
class RepairConfig:
    # rows
    drop_verdicts: frozenset[RowVerdict] = frozenset({RowVerdict.EXCLUDE})
    drop_exact_duplicates: bool = True
    drop_sparse_rows: bool = False
    # text hygiene
    normalize_text: bool = True
    fix_encoding: bool = True
    canonicalize_variants: bool = True
    nullify_placeholders: bool = True
    # typing
    coerce_numeric_text: bool = True
    parse_dates: bool = True
    normalize_booleans: bool = True
    invalid_to_null: bool = True              # unparseable numbers or dates become null instead of blocking the conversion
    # values
    nullify_sentinels: bool = True
    null_rule_violations: frozenset[str] = frozenset({"age_out_of_range", "percent_out_of_range", "year_out_of_range"})
    outlier_action: str = "none"              # none | winsorize | null
    relationship_action: str = "none"         # none | predict | null
    # imputation
    impute: bool = True
    numeric_strategy: str = "median"          # median | mean | model
    categorical_strategy: str = "mode"        # mode | constant
    categorical_constant: str = "Unknown"
    max_categorical_distinct: int = 200
    impute_repaired_nulls: bool = True        # also fill cells that this run itself turned into nulls
    max_impute_null_rate: float | None = None  # None means use the triage config's limit
    # safety and audit
    never_touch_columns: frozenset[str] = frozenset()
    verify_after: bool = True
    max_log_rows: int = 500_000

    def __post_init__(self) -> None:
        if self.outlier_action not in ("none", "winsorize", "null"):
            raise ValueError("outlier_action must be none, winsorize or null")
        if self.relationship_action not in ("none", "predict", "null"):
            raise ValueError("relationship_action must be none, predict or null")
        if self.numeric_strategy not in ("median", "mean", "model"):
            raise ValueError("numeric_strategy must be median, mean or model")
        if self.categorical_strategy not in ("mode", "constant"):
            raise ValueError("categorical_strategy must be mode or constant")


@dataclass
class RepairResult:
    data: pl.DataFrame
    changes: pl.DataFrame            # one row per changed cell: row_id, column, code, old, new, strategy
    dropped: pl.DataFrame            # one row per removed row: row_id, verdict, reasons
    summary: dict[str, Any]
    skipped: dict[str, str]          # column -> why a repair was not applied

    def changes_for(self, row_id: int) -> pl.DataFrame:
        return self.changes.filter(pl.col("row_id") == row_id)


class _Ledger:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.cols: dict[str, list] = {k: [] for k in ("row_id", "column", "code", "old", "new", "strategy")}
        self.counts: dict[str, int] = {}
        self.total = 0

    def add(self, ids: np.ndarray, column: str, code: str, old: list, new: list, strategy: str) -> None:
        k = len(ids)
        self.counts[code] = self.counts.get(code, 0) + k
        self.total += k
        room = max(0, self.cap - len(self.cols["row_id"]))
        take = min(k, room)
        if take:
            self.cols["row_id"].extend(int(i) for i in ids[:take])
            self.cols["column"].extend([column] * take)
            self.cols["code"].extend([code] * take)
            self.cols["old"].extend(None if v is None else str(v) for v in old[:take])
            self.cols["new"].extend(None if v is None else str(v) for v in new[:take])
            self.cols["strategy"].extend([strategy] * take)

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.cols, schema={
            "row_id": pl.Int64, "column": pl.Utf8, "code": pl.Utf8, "old": pl.Utf8, "new": pl.Utf8, "strategy": pl.Utf8,
        })


def _null_series(dtype: pl.DataType, n: int) -> pl.Series:
    return pl.Series("x", [None] * n, dtype=dtype)


def _apply(
    df: pl.DataFrame, col: str, new: pl.Series, mask: np.ndarray, ids: np.ndarray, code: str, strategy: str, ledger: _Ledger,
) -> pl.DataFrame:
    """Write ``new`` into ``col`` only where ``mask`` is True, logging each cell that actually changed."""
    if not mask.any():
        return df
    old = df[col]
    new = new.cast(old.dtype, strict=False)
    keep_old = pl.Series(~mask)
    merged = old.zip_with(keep_old, new)
    same = old.eq_missing(merged).to_numpy().astype(bool)
    changed = mask & ~same
    if changed.any():
        pos = np.flatnonzero(changed)
        ledger.add(ids[pos], col, code, old.gather(pos).to_list(), merged.gather(pos).to_list(), strategy)
        return df.with_columns(merged.alias(col))
    return df


def _scatter(old: pl.Series, mask: np.ndarray, fn: Callable[[Any], Any]) -> pl.Series:
    vals = old.to_list()
    for i in np.flatnonzero(mask):
        vals[i] = fn(vals[i])
    return pl.Series(old.name, vals, dtype=old.dtype)


def _fix_mojibake(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    for enc in ("cp1252", "latin-1"):
        try:
            fixed = v.encode(enc).decode("utf-8")
            if fixed != v and not re.search(_MOJIBAKE.replace(r"\x{", r"\u").replace("}", ""), fixed):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return v


def _clean_text(s: pl.Series) -> pl.Series:
    out = (
        s.str.replace_all(r"\x{00A0}", " ").str.replace_all(_INVISIBLE, "")
        .str.replace_all(r"\s+", " ").str.strip_chars()
    )
    return out.zip_with(out != "", _null_series(out.dtype, out.len()))


def _model_impute(
    df: pl.DataFrame, col: str, targets: np.ndarray, roles: dict[str, str], tcfg: RowTriageConfig,
) -> tuple[np.ndarray | None, str]:
    """Predict missing numeric cells from the rest of the row. Returns (values or None, note)."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None, "scikit-learn_not_installed"
    valid = np.ones(df.height, dtype=bool)
    numeric, categorical = _model_matrix(df, roles, valid, tcfg)
    if col not in numeric:
        return None, "target_not_numeric"
    y = numeric[col]
    others_n = [c for c in numeric if c != col][: tcfg.learned_max_predictors]
    others_c = list(categorical)[: tcfg.learned_max_predictors]
    if len(others_n) + len(others_c) < 1:
        return None, "no_predictors"
    F = np.column_stack([numeric[c] for c in others_n] + [categorical[c] for c in others_c])
    cat_mask = np.array([False] * len(others_n) + [True] * len(others_c))
    known = np.flatnonzero(~np.isnan(y))
    if known.size < tcfg.learned_min_rows:
        return None, "too_few_known_values"
    pred, kn = _oof_predictions(F, y, cat_mask, np.arange(df.height), tcfg)
    ok = kn[~np.isnan(pred[kn])]
    if ok.size < 30:
        return None, "model_not_validated"
    resid = y[ok] - pred[ok]
    var_y = float(np.var(y[ok]))
    r2 = 1.0 - float(np.var(resid)) / var_y if var_y > 0 else 0.0
    if r2 < tcfg.learned_min_r2:
        return None, f"weak_model_r2_{r2:.2f}"
    rng = np.random.default_rng([tcfg.random_state, 0xF111])
    train = known if known.size <= tcfg.learned_fit_sample else rng.choice(known, tcfg.learned_fit_sample, replace=False)
    model = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.1, max_leaf_nodes=15,
        categorical_features=cat_mask if cat_mask.any() else None, random_state=tcfg.random_state,
    ).fit(F[train], y[train])
    out = np.full(df.height, np.nan)
    pos = np.flatnonzero(targets)
    out[pos] = model.predict(F[pos])
    return out, f"model_r2_{r2:.2f}"


def repair_rows(
    df: pl.DataFrame,
    result: RowTriageResult,
    config: RepairConfig | None = None,
    triage_config: RowTriageConfig | None = None,
) -> RepairResult:
    """Apply the triage verdicts. Returns cleaned data plus a full audit ledger of every change."""
    rc = config or RepairConfig()
    tcfg = triage_config or RowTriageConfig()
    if df.height != result.n_rows:
        raise ValueError(f"df has {df.height} rows but the triage result covers {result.n_rows}")

    ledger = _Ledger(rc.max_log_rows)
    skipped: dict[str, str] = {}
    coerced: dict[str, str] = {}
    roles = dict(result.column_roles)
    off_limits = set(rc.never_touch_columns) | {c for c, r in roles.items() if r in ("protected",)}
    id_cols = {c for c, r in roles.items() if r == "id"}

    # ---- 1. drop rows -------------------------------------------------------
    drop = np.zeros(result.n_rows, dtype=bool)
    if RowVerdict.EXCLUDE in rc.drop_verdicts:
        drop |= result.exclude_mask if rc.drop_exact_duplicates else result.exclude_mask_without("exact_duplicate")
    if RowVerdict.REVIEW in rc.drop_verdicts:
        drop |= result.mask(RowVerdict.REVIEW)
    if RowVerdict.FIX in rc.drop_verdicts:
        drop |= result.mask(RowVerdict.FIX)
    if rc.drop_sparse_rows:
        drop |= result.cell_mask("sparse_row")
    reasons = result.reasons_by_row()
    dropped = pl.DataFrame(
        {
            "row_id": [int(result.row_ids[p]) for p in np.flatnonzero(drop)],
            "verdict": [_BY_SEVERITY[int(result.severity[p])].value for p in np.flatnonzero(drop)],
            "reasons": [reasons.get(int(p), []) for p in np.flatnonzero(drop)],
        },
        schema={"row_id": pl.Int64, "verdict": pl.Utf8, "reasons": pl.List(pl.Utf8)},
    )
    keep = ~drop
    out = df.filter(pl.Series(keep))
    ids = result.row_ids[keep]

    def M(code: str, col: str | None = None) -> np.ndarray:
        return result.cell_mask(code, col)[keep]

    def editable(c: str) -> bool:
        return c in out.columns and c not in off_limits

    # ---- 2. text hygiene ----------------------------------------------------
    if rc.normalize_text:
        for c in sorted(set(result.columns_with("whitespace_noise")) | set(result.columns_with("invisible_chars"))):
            if editable(c) and _is_text(out[c].dtype):
                m = M("whitespace_noise", c) | M("invisible_chars", c)
                out = _apply(out, c, _clean_text(out[c]), m, ids, "whitespace_noise", "trim_collapse_strip_hidden", ledger)
    if rc.fix_encoding:
        for c in result.columns_with("encoding_damage"):
            if editable(c) and _is_text(out[c].dtype):
                m = M("encoding_damage", c)
                out = _apply(out, c, _scatter(out[c], m, _fix_mojibake), m, ids, "encoding_damage", "recode_utf8", ledger)
    if rc.canonicalize_variants:
        for c, cmap in result.canonical_maps.items():
            if editable(c) and c not in id_cols and _is_text(out[c].dtype):
                m = M("category_variant", c)
                out = _apply(out, c, _scatter(out[c], m, lambda v, cm=cmap: cm.get(v, v)), m, ids, "category_variant", "most_common_spelling", ledger)
    if rc.nullify_placeholders:
        for c in result.columns_with("masked_null"):
            if editable(c) and _is_text(out[c].dtype):
                out = _apply(out, c, _null_series(out[c].dtype, out.height), M("masked_null", c), ids, "masked_null", "placeholder_to_null", ledger)

    # ---- 3. type coercion ---------------------------------------------------
    for c, role in roles.items():
        if not editable(c) or c not in out.columns or not _is_text(out[c].dtype):
            continue
        if role == "numeric_text" and rc.coerce_numeric_text:
            vals = _numeric_from_text(out[c])
            bad = M("type_violation", c)
            if bad.any() and not rc.invalid_to_null:
                skipped[c] = "type_violation_present_and_invalid_to_null_disabled"
                continue
            if bad.any():
                bad_pos = np.flatnonzero(bad)
                ledger.add(ids[bad_pos], c, "type_violation", out[c].gather(bad_pos).to_list(), [None] * len(bad_pos), "unparseable_to_null")
            arr = vals.drop_nulls()
            as_int = arr.len() > 0 and bool((arr == arr.round(0)).all()) and float(arr.abs().max()) < 9e15
            out = out.with_columns((vals.cast(pl.Int64) if as_int else vals).alias(c))
            coerced[c] = "Int64" if as_int else "Float64"
        elif role == "date_text" and rc.parse_dates and c in result.date_formats:
            dates, _ = _parse_dates_multi(out[c], result.date_formats[c])
            bad = M("invalid_date", c)
            if bad.any() and not rc.invalid_to_null:
                skipped[c] = "invalid_date_present_and_invalid_to_null_disabled"
                continue
            if bad.any():
                bad_pos = np.flatnonzero(bad)
                ledger.add(ids[bad_pos], c, "invalid_date", out[c].gather(bad_pos).to_list(), [None] * len(bad_pos), "unparseable_to_null")
            mixed = M("mixed_date_format", c) & dates.is_not_null().to_numpy().astype(bool)
            if mixed.any():
                mp = np.flatnonzero(mixed)
                ledger.add(ids[mp], c, "mixed_date_format", out[c].gather(mp).to_list(), dates.gather(mp).to_list(), "reparsed_alternate_format")
            out = out.with_columns(dates.alias(c))
            coerced[c] = "Date"
        elif role == "bool_text" and rc.normalize_booleans and c in result.bool_maps:
            bm = result.bool_maps[c]
            keyed = out[c].str.strip_chars().str.to_lowercase().to_list()
            out = out.with_columns(pl.Series(c, [bm.get(v) if v is not None else None for v in keyed], dtype=pl.Boolean))
            coerced[c] = "Boolean"

    # ---- 4. value level repairs --------------------------------------------
    def null_cells(code: str, col: str, strategy: str) -> None:
        nonlocal out
        out = _apply(out, col, _null_series(out[col].dtype, out.height), M(code, col), ids, code, strategy, ledger)

    if rc.nullify_sentinels:
        for c in result.columns_with("sentinel_value"):
            if editable(c) and out[c].dtype.is_numeric():
                null_cells("sentinel_value", c, "sentinel_to_null")
    for code in sorted(rc.null_rule_violations):
        for c in result.columns_with(code):
            if editable(c) and c not in id_cols:
                null_cells(code, c, "rule_violation_to_null")

    if rc.relationship_action != "none":
        for c in result.columns_with("relationship_violation"):
            if not editable(c) or not out[c].dtype.is_numeric():
                continue
            if rc.relationship_action == "null":
                null_cells("relationship_violation", c, "relationship_to_null")
            else:
                exp = np.full(result.n_rows, np.nan)
                for pos, val in result.learned_predictions.get(c, {}).items():
                    exp[pos] = val
                exp = exp[keep]
                m = M("relationship_violation", c) & ~np.isnan(exp)
                pred = pl.Series(c, exp, dtype=pl.Float64)
                if not out[c].dtype.is_float():
                    pred = pred.round(0)
                out = _apply(out, c, pred, m, ids, "relationship_violation", "cross_column_prediction", ledger)

    if rc.outlier_action != "none":
        for c in result.columns_with("outlier"):
            if not editable(c) or not out[c].dtype.is_numeric():
                continue
            m = M("outlier", c)
            if rc.relationship_action != "none":
                m = m & ~M("relationship_violation", c)
            if rc.outlier_action == "null":
                out = _apply(out, c, _null_series(out[c].dtype, out.height), m, ids, "outlier", "outlier_to_null", ledger)
            elif c in result.fences:
                lo, hi = result.fences[c]
                clipped = out[c].cast(pl.Float64).clip(lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None)
                if not out[c].dtype.is_float():
                    clipped = clipped.round(0)
                out = _apply(out, c, clipped, m, ids, "outlier", "winsorize_to_fence", ledger)

    # ---- 5. imputation ------------------------------------------------------
    if rc.impute:
        limit = rc.max_impute_null_rate if rc.max_impute_null_rate is not None else tcfg.max_impute_null_rate
        eligible_rows = ~M("sparse_row")
        outlier_any = np.zeros(out.height, dtype=bool)
        for f in result.flags:
            if f.code == "outlier":
                outlier_any |= f.mask[keep]
        live_roles = {
            c: ("id" if c in id_cols else "protected" if c in off_limits
                else "numeric" if out[c].dtype.is_numeric()
                else "text" if _is_text(out[c].dtype) else "other")
            for c in out.columns if c != ROW_ID
        }
        for c in [x for x in out.columns if x != ROW_ID]:
            if not editable(c) or c in id_cols:
                continue
            s = out[c]
            blank = out.select(_blank_expr(c, s.dtype)).to_series().to_numpy().astype(bool)
            if not blank.any():
                continue
            rate = float(blank.mean())
            if rate > limit:
                skipped[c] = f"null_rate_{rate:.2f}_above_limit_{limit:.2f}"
                continue
            targets = blank & eligible_rows
            if not rc.impute_repaired_nulls:
                targets &= M("impute", c)
            if not targets.any():
                continue
            if s.dtype.is_numeric():
                strategy = rc.numeric_strategy
                note = strategy
                fill_vals: np.ndarray | None = None
                if strategy == "model":
                    fill_vals, note = _model_impute(out, c, targets, live_roles, tcfg)
                    if fill_vals is None:
                        strategy, note = "median", f"median_fallback({note})"
                if fill_vals is None:
                    base = s.filter(pl.Series(~blank & ~outlier_any)) if (~blank & ~outlier_any).any() else s.drop_nulls()
                    stat = base.mean() if strategy == "mean" else base.median()
                    if stat is None:
                        skipped[c] = "no_observed_values"
                        continue
                    fill_vals = np.full(out.height, float(stat))
                    note = strategy
                new = pl.Series(c, fill_vals, dtype=pl.Float64)
                if not s.dtype.is_float():
                    new = new.round(0)
                out = _apply(out, c, new, targets, ids, "impute", note, ledger)
            elif _is_text(s.dtype) or s.dtype == pl.Boolean:
                present = s.drop_nulls() if s.dtype == pl.Boolean else s.filter(pl.Series(~blank))
                nd = present.n_unique()
                if nd == 0:
                    skipped[c] = "no_observed_values"
                    continue
                if nd > rc.max_categorical_distinct or nd > 0.5 * present.len():
                    skipped[c] = "high_cardinality_text_not_imputed"
                    continue
                if rc.categorical_strategy == "constant" and _is_text(s.dtype):
                    fill, note = rc.categorical_constant, "constant"
                else:
                    fill, note = present.mode().sort()[0], "mode"
                out = _apply(out, c, pl.Series(c, [fill] * out.height, dtype=s.dtype), targets, ids, "impute", note, ledger)

    # ---- 5b. duplicates that only became visible after cleaning -------------
    if rc.drop_exact_duplicates and out.height:
        body = out.drop(ROW_ID) if ROW_ID in out.columns else out
        try:
            later = ~body.select(pl.struct(pl.all()).is_first_distinct()).to_series().to_numpy().astype(bool)
        except Exception:
            later = np.zeros(out.height, dtype=bool)
        if later.any():
            new_drop = pl.DataFrame(
                {"row_id": [int(i) for i in ids[later]], "verdict": ["EXCLUDE"] * int(later.sum()),
                 "reasons": [["exact_duplicate_after_repair"]] * int(later.sum())},
                schema={"row_id": pl.Int64, "verdict": pl.Utf8, "reasons": pl.List(pl.Utf8)},
            )
            dropped = pl.concat([dropped, new_drop])
            out = out.filter(pl.Series(~later))
            ids = ids[~later]
            drop_extra = int(later.sum())
        else:
            drop_extra = 0
    else:
        drop_extra = 0

    # ---- 6. summary and verification ---------------------------------------
    summary: dict[str, Any] = {
        "rows_in": int(df.height),
        "rows_out": int(out.height),
        "rows_dropped": int(drop.sum()) + drop_extra,
        "dropped_by_reason": {},
        "cells_changed": int(ledger.total),
        "cells_changed_by_code": dict(sorted(ledger.counts.items(), key=lambda kv: -kv[1])),
        "columns_retyped": coerced,
        "ledger_truncated": ledger.total > rc.max_log_rows,
    }
    for r_list in dropped["reasons"].to_list():
        for lab in r_list:
            code = lab.split(":")[0]
            summary["dropped_by_reason"][code] = summary["dropped_by_reason"].get(code, 0) + 1
    if rc.verify_after and out.height:
        quick = replace(tcfg, learned_enabled=False, text_enabled=False)
        after = triage_rows(out, quick).summary()
        summary["after"] = {"verdicts": after["verdicts"], "by_reason": after["by_reason"], "mean_quality": after["mean_quality"]}
    return RepairResult(out, ledger.frame(), dropped, summary, skipped)


def triage_and_repair(
    df: pl.DataFrame,
    triage_config: RowTriageConfig | None = None,
    repair_config: RepairConfig | None = None,
) -> tuple[RowTriageResult, RepairResult]:
    """One call: judge every row, then apply the repairs the verdicts justify."""
    result = triage_rows(df, triage_config)
    return result, repair_rows(df, result, repair_config, triage_config)