""" Semantic column classifier """

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

try:
    from app.core.config import settings

    _DEFAULT_ID_UNIQ = float(settings.ID_UNIQUENESS_THRESHOLD)
except Exception:  # pragma: no cover
    _DEFAULT_ID_UNIQ = 0.98

logger = logging.getLogger(__name__)

# Kept for backward compatibility with code importing these names.
ID_UNIQUENESS_THRESHOLD: float = _DEFAULT_ID_UNIQ
EXCEL_DATE_RANGE = (25569, 73050)  # Excel serial dates: 1970-2100
ID_COLUMN_PATTERNS = ["_id", "uuid", "guid", "pk", "primary_key", "row_id", "record_id"]
DATE_COLUMN_PATTERNS = ["date", "time", "timestamp", "created_at", "updated_at", "modified_at"]

_EXCLUDING_TYPES = ("empty", "constant", "near_constant", "date", "identifier", "phone", "geo_code")
_PRIORITY = {t: i for i, t in enumerate(_EXCLUDING_TYPES)}


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassifierConfig:
    id_uniqueness_threshold: float = _DEFAULT_ID_UNIQ
    exclude_threshold: float = 0.5
    review_threshold: float = 0.3

    min_rows_sequential: int = 10
    min_rows_uniqueness: int = 20
    near_constant_share: float = 0.99
    low_cv_threshold: float | None = None   # legacy rule (std/|mean| < x); off by default
    excel_date_range: tuple[float, float] = EXCEL_DATE_RANGE
    excel_int_ratio: float = 0.9
    year_range: tuple[int, int] = (1900, 2100)
    categorical_max_levels: int = 10

    # Types that are removed from `analytical`. Add "binary", "categorical_coded",
    # "ordinal" or "count" here to exclude them as well.
    exclude_types: frozenset[str] = frozenset(_EXCLUDING_TYPES)

    # Legacy behaviour: a name hint alone is enough to exclude.
    strict_name_rules: bool = False

    extra_id_tokens: tuple[str, ...] = ()
    extra_date_tokens: tuple[str, ...] = ()
    force_include: tuple[str, ...] = ()
    force_exclude: tuple[str, ...] = ()

    text_sample_size: int = 5000
    seed: int = 42
    date_string_min_match: float = 0.9


# ----------------------------------------------------------------------------
# Name analysis
# ----------------------------------------------------------------------------
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ENTITIES = {
    "user", "customer", "cust", "order", "account", "acct", "product", "prod", "item", "invoice",
    "vendor", "client", "retailer", "store", "row", "record", "employee", "emp", "student",
    "patient", "session", "transaction", "txn", "trans", "ticket", "device", "member", "merchant",
    "supplier", "sku", "shipment", "booking", "policy", "claim", "loan", "branch", "department",
    "dept", "region", "campaign", "lead", "contact", "person", "owner", "parent", "category",
    "company", "org", "tenant", "visitor", "event", "batch", "case", "game", "team", "player",
}
_ID_STRONG = {"id", "ids", "uuid", "guid", "pk", "pkey", "identifier"}
_ID_WEAK = {"code", "number", "num", "no", "nbr", "ref", "reference", "key", "serial"}
_INDEX_STRONG = {"unnamed", "sno", "srno", "slno", "rownum", "rowid", "rowno", "idx"}
_INDEX_WEAK = {"index", "row"}
_DATE_STRONG = {"date", "datetime", "timestamp", "dt", "dob", "created", "updated", "modified",
                "epoch", "ts", "datestamp", "birthdate"}
_DATE_WEAK = {"time", "period"}
_YEAR = {"year", "yr", "fy", "fiscal"}
_DURATION = {"duration", "elapsed", "delay", "latency", "runtime", "response", "wait", "waiting",
             "age", "lifetime", "interval", "uptime", "downtime", "ttl", "timeout", "spent",
             "minutes", "seconds", "hours", "days", "ms", "millis", "tenure"}
_PHONE = {"phone", "mobile", "tel", "telephone", "fax", "cell", "msisdn"}
_GEO = {"zip", "zipcode", "postal", "postcode", "pincode", "fips", "zcta"}
_LAT = {"lat", "latitude"}
_LON = {"lon", "lng", "long", "longitude"}
_PCT = {"pct", "percent", "percentage", "rate", "share", "ratio", "proportion", "prob", "probability"}
_MONEY = {"price", "amount", "cost", "revenue", "salary", "income", "sales", "profit", "fee", "tax",
          "balance", "budget", "spend", "wage", "payment", "total", "mrp"}
_CATEGORICAL_HINT = {"status", "type", "category", "cat", "class", "flag", "gender", "sex", "kind",
                     "mode", "label", "cluster", "segment", "group", "code", "region", "is", "has"}
_ORDINAL_HINT = {"rating", "rank", "grade", "level", "tier", "stars", "priority"}
_ANALYTICAL_HINT = {"feature", "feat", "metric", "measure", "value", "val", "target", "signal",
                    "input", "output", "pred", "prediction", "score", "weight", "x", "y", "z"}


def tokenize(name: str) -> list[str]:
    """'customerID' -> ['customer','id'], 'Order_Date' -> ['order','date']."""
    s = _CAMEL.sub(" ", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _name_features(name: str, cfg: ClassifierConfig) -> dict[str, Any]:
    toks = tokenize(name)
    tset = set(toks)
    compact = "".join(toks)

    id_strong = bool(tset & (_ID_STRONG | set(cfg.extra_id_tokens)))
    if not id_strong:
        for t in toks:
            if re.fullmatch(r"id\d+", t) or (t.endswith("id") and t[:-2] in _ENTITIES):
                id_strong = True
                break
    if not id_strong:  # "primary key", "foreign key", "surrogate key"
        for a, b in zip(toks, toks[1:]):
            if b == "key" and a in {"primary", "foreign", "surrogate"}:
                id_strong = True
    if compact in {"pk", "rowid", "recordid"}:
        id_strong = True

    duration = bool(tset & _DURATION)
    date_strong = bool(tset & (_DATE_STRONG | set(cfg.extra_date_tokens)))
    return {
        "tokens": toks,
        "id_strong": id_strong,
        "id_weak": bool(tset & _ID_WEAK),
        "index_strong": bool(tset & _INDEX_STRONG),
        "index_weak": bool(tset & _INDEX_WEAK),
        "date_strong": date_strong,
        "date_weak": bool(tset & _DATE_WEAK) and not duration,
        "year": bool(tset & _YEAR),
        "duration": duration,
        "phone": bool(tset & _PHONE),
        "geo": bool(tset & _GEO),
        "lat": bool(tset & _LAT),
        "lon": bool(tset & _LON),
        "pct": bool(tset & _PCT),
        "money": bool(tset & _MONEY),
        "cat_hint": bool(tset & _CATEGORICAL_HINT),
        "ordinal_hint": bool(tset & _ORDINAL_HINT),
        "analytical_hint": bool(tset & _ANALYTICAL_HINT),
    }


# ----------------------------------------------------------------------------
# Numeric column profiling
# ----------------------------------------------------------------------------
def _is_int_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_integer())
    except AttributeError:
        return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def _profile_numeric(s: pl.Series, n_rows: int) -> dict[str, Any] | None:
    """Single-pass profile of the finite values of a column. None if empty."""
    raw_non_null = s.drop_nulls().len()
    f = s.cast(pl.Float64, strict=False).drop_nulls()
    f = f.filter(f.is_finite())
    n = f.len()
    base = {
        "dtype": str(s.dtype),
        "null_pct": round(100.0 * (n_rows - raw_non_null) / n_rows, 2) if n_rows else 0.0,
        "n_valid": n,
    }
    if n == 0:
        return {**base, "empty": True}

    arr = f.to_numpy()
    vc = f.value_counts(sort=True)
    counts = vc[vc.columns[1]].to_numpy()
    n_unique = int(vc.height)
    mn, mx = float(arr.min()), float(arr.max())
    mean, std = float(arr.mean()), float(arr.std(ddof=1)) if n > 1 else 0.0
    rounded = np.round(arr)
    int_ratio = float((arr == rounded).mean())
    int_valued = int_ratio == 1.0

    prof: dict[str, Any] = {
        **base,
        "empty": False,
        "arr": arr,
        "n_unique": n_unique,
        "unique_ratio": n_unique / n,
        "min": mn, "max": mx, "mean": mean, "std": std,
        "median": float(np.median(arr)),
        "top_share": float(counts[0] / n),
        "int_ratio": int_ratio,
        "int_valued": int_valued,
        "int_dtype": _is_int_dtype(s.dtype),
        "step1": False, "const_step": None, "contiguous": False, "density": None,
        "digit_mode": None, "digit_share": 0.0,
    }

    if int_valued and n >= 2:
        diffs = np.diff(arr)
        if diffs.size:
            prof["step1"] = bool(np.all(diffs == 1))
            ud = np.unique(diffs)
            if ud.size == 1 and ud[0] > 0:
                prof["const_step"] = float(ud[0])
        span = mx - mn + 1
        prof["density"] = n_unique / span if span > 0 else None
        prof["contiguous"] = bool(n_unique == span and n_unique / n >= 0.98)
        pos = np.abs(arr[arr != 0])
        if pos.size:
            digits = (np.floor(np.log10(pos)) + 1).astype(int)
            bc = np.bincount(digits)
            prof["digit_mode"] = int(bc.argmax())
            prof["digit_share"] = float(bc.max() / digits.size)
    return prof


def _valid_yyyymmdd(v: np.ndarray) -> float:
    iv = v.astype(np.int64)
    m, d = (iv // 100) % 100, iv % 100
    ok = (iv >= 19000101) & (iv <= 21001231) & (m >= 1) & (m <= 12) & (d >= 1) & (d <= 31)
    return float(ok.mean())


def _valid_yyyymm(v: np.ndarray) -> float:
    iv = v.astype(np.int64)
    y, m = iv // 100, iv % 100
    return float(((y >= 1900) & (y <= 2100) & (m >= 1) & (m <= 12)).mean())


# ----------------------------------------------------------------------------
# Evidence collection
# ----------------------------------------------------------------------------
Evidence = dict[str, list[tuple[str, float, str | None]]]  # type -> [(reason, weight, subtype)]


def _collect_evidence(col: str, prof: dict[str, Any], cfg: ClassifierConfig, n_rows: int) -> tuple[Evidence, list[str], dict[str, Any]]:
    ev: Evidence = {t: [] for t in _EXCLUDING_TYPES}
    nf = _name_features(col, cfg)
    tags: list[str] = []
    strict = cfg.strict_name_rules
    name_w = lambda w: 1.0 if strict else w  # noqa: E731

    def add(t: str, reason: str, w: float, sub: str | None = None) -> None:
        ev[t].append((reason, w, sub))

    if prof["empty"]:
        add("empty", "all_null_or_non_numeric", 1.0)
        return ev, tags, nf

    n, nu = prof["n_valid"], prof["n_unique"]
    mn, mx = prof["min"], prof["max"]
    int_valued = prof["int_valued"]

    # Constant and near-constant ------------------------------------------------
    if nu <= 1 or prof["std"] <= 1e-12 * max(1.0, abs(prof["mean"])):
        add("constant", "low_variance", 1.0)
        add("constant", "constant_value", 0.0)
    elif prof["top_share"] >= cfg.near_constant_share:
        add("near_constant", "low_variance", 0.6)
        add("near_constant", f"dominant_value_{round(prof['top_share'] * 100)}%", 0.0)
    if cfg.low_cv_threshold is not None and abs(prof["mean"]) > 1e-9 and prof["std"] > 0:
        if abs(prof["std"] / prof["mean"]) < cfg.low_cv_threshold:
            add("near_constant", "low_variance", 0.6)

    # Identifier ----------------------------------------------------------------
    if nf["id_strong"]:
        add("identifier", "name_suggests_id", name_w(0.55))
    elif nf["index_strong"]:
        add("identifier", "name_suggests_index", name_w(0.55))
    elif nf["id_weak"] or nf["index_weak"]:
        add("identifier", "name_weakly_suggests_id", 0.2)

    has_id_name = nf["id_strong"] or nf["index_strong"] or nf["id_weak"] or nf["index_weak"]
    is_analytical = nf.get("analytical_hint", False) and not (nf["id_strong"] or nf["index_strong"])

    if int_valued and mn >= 0 and n >= cfg.min_rows_uniqueness and prof["unique_ratio"] >= cfg.id_uniqueness_threshold:
        if has_id_name:
            add("identifier", f"high_uniqueness_{round(prof['unique_ratio'] * 100)}%", 0.4)
            if prof["density"] is not None and prof["density"] >= 0.5:
                add("identifier", "dense_integer_range", 0.2)
        elif not is_analytical and prof["int_dtype"]:
            add("identifier", f"high_uniqueness_{round(prof['unique_ratio'] * 100)}%", 0.2)

    if int_valued and mn >= 0 and n >= cfg.min_rows_sequential and not is_analytical:
        if prof["step1"] and (prof["int_dtype"] or has_id_name):
            add("identifier", "strictly_sequential_auto_increment", 0.6, "row_index")
        elif prof["const_step"] is not None and (prof["int_dtype"] or has_id_name):
            add("identifier", "constant_step_sequence", 0.5, "row_index")
        elif prof["contiguous"] and has_id_name:
            add("identifier", "contiguous_integer_range", 0.5, "row_index")
    if (int_valued and prof["digit_mode"] and prof["digit_mode"] >= 7
            and prof["digit_share"] >= 0.95 and prof["unique_ratio"] >= 0.9):
        add("identifier", "fixed_length_digit_codes", 0.2)

    # Date stored as a number ------------------------------------------------------
    if nf["date_strong"]:
        add("date", "name_suggests_date", name_w(0.5))
    elif nf["date_weak"]:
        add("date", "name_weakly_suggests_date", 0.2)

    arr = prof["arr"]
    lo, hi = cfg.excel_date_range
    in_excel = float(((arr >= lo) & (arr <= hi)).mean())
    if in_excel >= 0.98 and prof["int_ratio"] > cfg.excel_int_ratio:
        add("date", "excel_serial_date_range", 0.35, "excel_serial")
    for unit, scale in (("s", 1.0), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        lo_e, hi_e = 3.15e8 * scale, 4.1e9 * scale
        if float(((arr >= lo_e) & (arr <= hi_e)).mean()) >= 0.98 and (unit == "s" or prof["int_ratio"] > 0.9):
            add("date", f"epoch_{unit}_range", 0.4, f"epoch_{unit}")
            med = abs(prof["median"])
            if med > 0 and (mx - mn) / med < 0.25:
                add("date", "values_clustered_like_timestamps", 0.2, f"epoch_{unit}")
            break
    if int_valued and n >= 5:
        if mn >= 19000101 and mx <= 21001231 and _valid_yyyymmdd(arr) >= 0.95:
            add("date", "yyyymmdd_pattern", 0.6, "yyyymmdd")
        elif 190001 <= mn and mx <= 210012 and _valid_yyyymm(arr) >= 0.95:
            add("date", "yyyymm_pattern", 0.4, "yyyymm")
        y_lo, y_hi = cfg.year_range
        if y_lo <= mn and mx <= y_hi and prof["int_ratio"] >= 0.95 and nu <= 150:
            if nf["year"]:
                add("date", "year_values", 0.6, "year")
            elif prof["std"] <= 30:
                add("date", "year_like_values", 0.35, "year")

    # Phone and geographic codes ----------------------------------------------------
    if nf["phone"]:
        add("phone", "name_suggests_phone", name_w(0.5))
    if int_valued and prof["digit_mode"] and 10 <= prof["digit_mode"] <= 15 and prof["digit_share"] >= 0.9:
        add("phone", "phone_digit_pattern", 0.4)
    if nf["geo"]:
        add("geo_code", "name_suggests_geo_code", name_w(0.5))

    # Non-excluding tags ----------------------------------------------------------
    if nu == 2:
        tags.append("binary")
    if int_valued and 2 < nu <= cfg.categorical_max_levels:
        if nf["ordinal_hint"]:
            tags.append("ordinal")
        elif nf["cat_hint"]:
            tags.append("categorical_coded")
        else:
            tags.append("discrete_low_cardinality")
    if int_valued and mn >= 0 and nu > cfg.categorical_max_levels:
        tags.append("count")
    if not int_valued and 0.0 <= mn and mx <= 1.0:
        tags.append("ratio_0_1")
    if 0.0 <= mn and mx <= 100.0 and nf["pct"]:
        tags.append("percentage")
    if nf["lat"] and -90 <= mn and mx <= 90:
        tags.append("coordinate")
    if nf["lon"] and -180 <= mn and mx <= 180:
        tags.append("coordinate")
    if nf["money"]:
        tags.append("monetary")
    if nf["duration"]:
        tags.append("duration")
    return ev, tags, nf


def _score(items: list[tuple[str, float, str | None]]) -> float:
    return min(1.0, sum(w for _, w, _ in items))


_TAG_PRIORITY = ("binary", "categorical_coded", "ordinal", "coordinate", "percentage", "ratio_0_1",
                 "monetary", "duration", "count", "discrete_low_cardinality")


def _classify_one(col: str, s: pl.Series, n_rows: int, cfg: ClassifierConfig) -> dict[str, Any]:
    prof = _profile_numeric(s, n_rows)
    assert prof is not None
    ev, tags, nf = _collect_evidence(col, prof, cfg, n_rows)
    scores = {t: round(_score(items), 3) for t, items in ev.items()}
    best_type = max(_EXCLUDING_TYPES, key=lambda t: (scores[t], -_PRIORITY[t]))
    best = scores[best_type]

    excluded = False
    semantic = None
    subtype = None
    review = None
    if col in cfg.force_exclude:
        excluded, semantic = True, best_type if best > 0 else "identifier"
    elif col in cfg.force_include:
        excluded = False
    elif best >= cfg.exclude_threshold:
        semantic = best_type
        excluded = semantic in cfg.exclude_types
    elif best >= cfg.review_threshold:
        review = {"column": col, "candidate_type": best_type, "score": best,
                  "reasons": [r for r, w, _ in ev[best_type] if w > 0]}

    if semantic is None:
        semantic = next((t for t in _TAG_PRIORITY if t in tags), "continuous")
        excluded = excluded or (semantic in cfg.exclude_types and col not in cfg.force_include)
    elif semantic == "date" or semantic == "identifier":
        subs = [(w, sub) for _, w, sub in ev[semantic] if sub]
        subtype = max(subs)[1] if subs else None

    reasons = [r for t in _EXCLUDING_TYPES for r, w, _ in ev[t] if w > 0 or t in ("constant", "near_constant")]
    reasons = list(dict.fromkeys(reasons))
    if col in cfg.force_exclude:
        reasons.append("force_exclude")
    if semantic in ("binary", "categorical_coded", "ordinal") and excluded:
        reasons.append(f"type_{semantic}_excluded_by_config")

    if excluded:
        confidence = 1.0 if col in cfg.force_exclude else best if best >= cfg.exclude_threshold else 0.6
    else:
        confidence = round(max(0.0, 1.0 - best), 2)

    details = {k: v for k, v in prof.items() if k not in ("arr",)}
    details["scores"] = {t: v for t, v in scores.items() if v > 0}
    details["evidence"] = {t: [r for r, w, _ in items if w > 0] for t, items in ev.items() if any(w > 0 for _, w, _ in items)}
    return {
        "excluded": excluded, "semantic_type": semantic, "subtype": subtype,
        "confidence": round(float(confidence), 2), "tags": tags, "reasons": reasons,
        "review": review, "details": details,
    }


def classify_numeric_columns(
    df: pl.DataFrame,
    numeric_cols: list[str],
    config: ClassifierConfig | None = None,
) -> dict[str, Any]:
    """
    Classify numeric columns. Keeps the original keys 'analytical', 'excluded'
    and 'exclusion_reasons', and adds semantic_types, subtypes, confidence,
    tags, groups, needs_review, details, recommended_casts and warnings.
    """
    cfg = config or ClassifierConfig()
    n_rows = df.height
    analytical: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, list[str]] = {}
    sem: dict[str, str] = {}
    subs: dict[str, str] = {}
    conf: dict[str, float] = {}
    tags: dict[str, list[str]] = {}
    details: dict[str, Any] = {}
    review: list[dict[str, Any]] = []
    casts: dict[str, str] = {}
    warnings: list[str] = []

    for col in numeric_cols:
        if col not in df.columns:
            warnings.append(f"{col}: column not found")
            continue
        try:
            r = _classify_one(col, df[col], n_rows, cfg)
        except Exception as ex:
            logger.warning("Classification failed for %s: %s", col, ex)
            analytical.append(col)
            sem[col], conf[col] = "unknown", 0.0
            warnings.append(f"{col}: classification failed ({ex}), kept as analytical")
            continue
        sem[col], conf[col], tags[col], details[col] = r["semantic_type"], r["confidence"], r["tags"], r["details"]
        if r["subtype"]:
            subs[col] = r["subtype"]
        if r["excluded"]:
            excluded.append(col)
            reasons[col] = r["reasons"]
        else:
            analytical.append(col)
        if r["review"]:
            review.append(r["review"])
            warnings.append(f"{col}: possibly {r['review']['candidate_type']} (score {r['review']['score']}), please review")
        if r["semantic_type"] == "date" and r["subtype"]:
            hint = _CAST_HINTS.get(r["subtype"])
            if hint:
                casts[col] = hint
        if "binary" in r["tags"] and not r["excluded"]:
            pass

    groups: dict[str, list[str]] = {}
    for c, t in sem.items():
        groups.setdefault(t, []).append(c)

    return {
        "analytical": analytical,
        "excluded": excluded,
        "exclusion_reasons": reasons,
        "semantic_types": sem,
        "subtypes": subs,
        "confidence": conf,
        "tags": tags,
        "groups": groups,
        "needs_review": review,
        "recommended_casts": casts,
        "warnings": warnings,
        "details": details,
    }


# ----------------------------------------------------------------------------
# All dtypes
# ----------------------------------------------------------------------------
_RE_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_RE_URL = re.compile(r"^(https?|ftp)://\S+$", re.I)
_RE_IPV4 = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_BOOL_WORDS = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"}
_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
                 "%d.%m.%Y", "%Y%m%d", "%d-%b-%Y", "%b %d, %Y"]
_DATETIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
                     "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S%.f"]

_CAST_HINTS = {
    "excel_serial": "excel_serial_to_datetime",
    "epoch_s": "epoch_s_to_datetime",
    "epoch_ms": "epoch_ms_to_datetime",
    "epoch_us": "epoch_us_to_datetime",
    "epoch_ns": "epoch_ns_to_datetime",
    "yyyymmdd": "yyyymmdd_to_date",
}


def _share(sample: list[str], rx: re.Pattern[str]) -> float:
    return sum(1 for v in sample if rx.match(v)) / len(sample) if sample else 0.0


def _detect_date_format(sample: pl.Series, min_match: float) -> tuple[str, str] | None:
    n = sample.len()
    if n == 0:
        return None
    best: tuple[float, str, str] | None = None
    for kind, formats in (("datetime", _DATETIME_FORMATS), ("date", _DATE_FORMATS)):
        for fmt in formats:
            try:
                parsed = (sample.str.to_datetime(fmt, strict=False) if kind == "datetime"
                          else sample.str.to_date(fmt, strict=False))
            except Exception:
                continue
            rate = 1.0 - parsed.null_count() / n
            if rate >= min_match and (best is None or rate > best[0]):
                best = (rate, kind, fmt)
        if best:
            break  # prefer datetime formats when one matches well
    return (best[1], best[2]) if best else None


def _classify_text(col: str, s: pl.Series, n_rows: int, cfg: ClassifierConfig) -> dict[str, Any]:
    nn = s.drop_nulls().str.strip_chars()
    nn = nn.filter(nn != "")
    n = nn.len()
    out: dict[str, Any] = {"semantic_type": "empty", "subtype": None, "cast": None, "excluded_from_categorical": True}
    if n == 0:
        return out
    n_unique = nn.n_unique()
    ratio = n_unique / n
    sample = nn.sample(n=min(n, cfg.text_sample_size), seed=cfg.seed) if n > cfg.text_sample_size else nn
    vals = sample.to_list()
    nf = _name_features(col, cfg)

    if _share(vals, _RE_UUID) >= 0.9:
        return {**out, "semantic_type": "identifier", "subtype": "uuid"}
    if _share(vals, _RE_EMAIL) >= 0.9:
        return {**out, "semantic_type": "email"}
    if _share(vals, _RE_URL) >= 0.9:
        return {**out, "semantic_type": "url"}
    if _share(vals, _RE_IPV4) >= 0.9:
        return {**out, "semantic_type": "ip_address"}

    lowered = {v.lower() for v in vals}
    if lowered <= _BOOL_WORDS and len(lowered) <= 2:
        return {**out, "semantic_type": "boolean_string", "cast": "boolean", "excluded_from_categorical": False}

    cleaned = sample.str.replace_all(r"[,$€£₹%\s]", "")
    if 1.0 - cleaned.cast(pl.Float64, strict=False).null_count() / sample.len() >= 0.95:
        hint = "float"
        if any(ch in v for v in vals[:200] for ch in "$€£₹"):
            hint = "float_strip_currency"
        elif any("%" in v for v in vals[:200]):
            hint = "float_strip_percent"
        elif any("," in v for v in vals[:200]):
            hint = "float_strip_thousands"
        return {**out, "semantic_type": "numeric_string", "cast": hint, "excluded_from_categorical": False}

    fmt = _detect_date_format(sample, cfg.date_string_min_match)
    if fmt:
        return {**out, "semantic_type": "date_string", "subtype": fmt[0],
                "cast": f"{fmt[0]}:{fmt[1]}", "excluded_from_categorical": False}

    avg_len = float(nn.str.len_chars().mean() or 0.0)
    avg_words = float(np.mean([len(v.split()) for v in vals]))
    if avg_len > 50 or avg_words >= 5:
        return {**out, "semantic_type": "free_text"}
    if n >= cfg.min_rows_uniqueness and ratio >= cfg.id_uniqueness_threshold and avg_words < 2:
        return {**out, "semantic_type": "identifier", "subtype": "string_id"}
    if nf["id_strong"] and ratio >= 0.5:
        return {**out, "semantic_type": "identifier", "subtype": "string_id"}
    if n_unique > 50 and ratio > 0.5:
        return {**out, "semantic_type": "categorical_high_cardinality", "excluded_from_categorical": False}
    return {**out, "semantic_type": "categorical", "excluded_from_categorical": False}


def classify_columns(
    df: pl.DataFrame,
    cols: list[str] | None = None,
    config: ClassifierConfig | None = None,
) -> dict[str, Any]:
    """
    Classify every column of any dtype. The numeric part reuses
    classify_numeric_columns(); the result adds categorical, text, boolean,
    datetime and recommended_casts lists.
    """
    cfg = config or ClassifierConfig()
    cols = [c for c in (cols or df.columns) if c in df.columns]
    numeric = [c for c in cols if df.schema[c].is_numeric()]
    res = classify_numeric_columns(df, numeric, cfg)

    res.update({"categorical": [], "text": [], "boolean": [], "datetime": [], "other": [],
                "text_identifiers": [], "numeric_like_text": []})
    n_rows = df.height
    for c in cols:
        if c in numeric:
            if res["semantic_types"].get(c) == "date" and c in res["excluded"]:
                res["datetime"].append(c)
            continue
        dt = df.schema[c]
        try:
            if dt == pl.Boolean:
                res["semantic_types"][c] = "boolean"
                res["boolean"].append(c)
            elif dt == pl.Date or dt == pl.Datetime:
                res["semantic_types"][c] = "datetime"
                res["datetime"].append(c)
            elif dt == pl.Utf8:
                r = _classify_text(c, df[c], n_rows, cfg)
                t = r["semantic_type"]
                res["semantic_types"][c] = t
                if r["subtype"]:
                    res["subtypes"][c] = r["subtype"]
                if t == "boolean_string":
                    res["boolean"].append(c)
                elif t == "numeric_string":
                    res["numeric_like_text"].append(c)
                elif t == "date_string":
                    res["datetime"].append(c)
                elif t in ("free_text",):
                    res["text"].append(c)
                elif t in ("identifier", "email", "url", "ip_address"):
                    res["text_identifiers"].append(c)
                    res["excluded"].append(c)
                    res["exclusion_reasons"][c] = [f"text_{t}"]
                elif t == "empty":
                    res["excluded"].append(c)
                    res["exclusion_reasons"][c] = ["all_null"]
                else:
                    res["categorical"].append(c)
                if r["cast"]:
                    res["recommended_casts"][c] = r["cast"]
            elif dt == pl.Categorical or str(dt).startswith("Enum"):
                res["semantic_types"][c] = "categorical"
                res["categorical"].append(c)
            else:
                res["semantic_types"][c] = "other"
                res["other"].append(c)
        except Exception as ex:
            logger.warning("Text classification failed for %s: %s", c, ex)
            res["other"].append(c)
            res["warnings"].append(f"{c}: classification failed ({ex})")
    res["groups"] = {}
    for c, t in res["semantic_types"].items():
        res["groups"].setdefault(t, []).append(c)
    return res


# ----------------------------------------------------------------------------
# Applying the recommended casts
# ----------------------------------------------------------------------------
def apply_casts(df: pl.DataFrame, casts: dict[str, str], inplace: bool = True) -> pl.DataFrame:
    """
    Apply recommended_casts. New columns get the suffix "__cast" unless
    inplace=True. Values that cannot be converted become null.
    """
    exprs: list[pl.Expr] = []
    for c, hint in casts.items():
        if c not in df.columns:
            continue
        x = pl.col(c)
        if hint == "excel_serial_to_datetime":
            e = pl.datetime(1899, 12, 30) + pl.duration(
                microseconds=(x.cast(pl.Float64, strict=False) * 86_400_000_000).cast(pl.Int64, strict=False))
        elif hint.startswith("epoch_") and hint.endswith("_to_datetime"):
            unit = hint.split("_")[1]
            e = pl.from_epoch(x.cast(pl.Int64, strict=False), time_unit=unit)
        elif hint == "yyyymmdd_to_date":
            e = x.cast(pl.Int64, strict=False).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        elif hint == "boolean":
            e = x.str.to_lowercase().str.strip_chars().is_in(["true", "yes", "y", "t", "1"])
            e = pl.when(x.is_null()).then(None).otherwise(e)
        elif hint.startswith("float"):
            e = x.str.replace_all(r"[,$€£₹%\s]", "").cast(pl.Float64, strict=False)
        elif hint.startswith("date:"):
            e = x.str.strip_chars().str.to_date(hint.split(":", 1)[1], strict=False)
        elif hint.startswith("datetime:"):
            e = x.str.strip_chars().str.to_datetime(hint.split(":", 1)[1], strict=False)
        else:
            logger.warning("Unknown cast hint %r for %s", hint, c)
            continue
        exprs.append(e.alias(c if inplace else f"{c}__cast"))
    return df.with_columns(exprs) if exprs else df


__all__ = ["ClassifierConfig", "classify_numeric_columns", "classify_columns", "apply_casts", "tokenize"]