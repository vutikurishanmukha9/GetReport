"""Outlier & Anomaly Detection Engine"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import polars as pl

try:  # keep working outside the app (tests, notebooks)
    from app.core.config import settings
except Exception:  # pragma: no cover
    settings = None

logger = logging.getLogger(__name__)

__all__ = [
    "OutlierConfig",
    "OutlierReport",
    "analyze_outliers",
    "detect_outliers",
    "detect_time_series_stl_outliers",
    "detect_time_series_outliers_multi",
]


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default) if settings is not None else default


# Back-compat module constants
IQR_LOWER_MULTIPLIER: float = _setting("IQR_LOWER_MULTIPLIER", 1.5)
IQR_UPPER_MULTIPLIER: float = _setting("IQR_UPPER_MULTIPLIER", 1.5)

NON_NEGATIVE_KEYWORDS = {
    "price", "cost", "age", "count", "quantity", "qty", "revenue", "amount", "salary",
    "fee", "rate", "income", "sales", "weight", "height", "length", "distance",
    "duration", "population", "volume", "size",
}
_PCT_KEYWORDS = {"pct", "percent", "percentage", "perc"}
_UNIT_KEYWORDS = {"proportion", "probability", "prob"}
_LAT_KEYWORDS = {"lat", "latitude"}
_LON_KEYWORDS = {"lon", "lng", "long", "longitude"}
_ID_KEYWORDS = {"id", "uuid", "guid", "index", "idx", "key"}
_SOFT_UPPER = {"age": 130.0}
_SENTINELS = (-1, -9, -99, -999, -9999, -99999, 99, 999, 9999, 99999, 999999, 9999999)

Treatment = Literal["none", "clip", "null", "review", "log_transform"]
_AUTO_STRATEGY = {"none": "none", "clip": "clip", "null": "null",
                  "review": "none", "log_transform": "clip"}


# ───────────────────────────── configuration ─────────────────────────────
@dataclass
class OutlierConfig:
    # univariate thresholds
    iqr_lower_multiplier: float = field(default_factory=lambda: IQR_LOWER_MULTIPLIER)
    iqr_upper_multiplier: float = field(default_factory=lambda: IQR_UPPER_MULTIPLIER)
    severe_multiplier: float = 3.0
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5
    heavy_skew_threshold: float = 2.0
    min_votes: int = 2                       # method families that must agree
    min_rows: int = 4
    discrete_unique_max: int = 10            # <= this many distinct values -> not treated as continuous
    gesd_alpha: float = 0.05
    gesd_max_rows: int = 5000
    gesd_max_fraction: float = 0.10
    medcouple_sample: int = 3000
    # domain
    infer_domain_rules: bool = True
    domain_rules: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    detect_sentinels: bool = True
    # ingestion
    coerce_numeric_strings: bool = True
    skip_id_like: bool = True
    max_columns: int | None = None
    # contextual
    group_by: list[str] | None = None
    min_group_size: int = 30
    merge_group_flags: bool = False          # True: contextual outliers also enter the final mask
    # multivariate
    multivariate: bool = True
    multivariate_contamination: float = 0.02
    multivariate_max_cols: int = 12
    multivariate_max_fit_rows: int = 10_000
    multivariate_min_rows: int = 30
    multivariate_min_votes: int = 2
    n_jobs: int = -1
    # output
    max_indices_returned: int = 200
    sample_size: int = 20
    include_clean_columns: bool = False
    include_meta: bool = True
    random_state: int = 42


# ───────────────────────────── low-level stats ─────────────────────────────
def _f(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _medcouple(v: np.ndarray, max_n: int, rng: np.random.Generator) -> float:
    """Medcouple (robust skewness, Brys et al.). Subsampled to bound the O(n^2) kernel."""
    if v.size < 5:
        return 0.0
    if v.size > max_n:
        v = rng.choice(v, max_n, replace=False)
    med = float(np.median(v))
    xl, xr = v[v <= med], v[v >= med]
    if xl.size == 0 or xr.size == 0:
        return 0.0
    num = (xr[None, :] - med) - (med - xl[:, None])
    den = xr[None, :] - xl[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(den != 0, num / den, 0.0)
    return float(np.clip(np.median(h), -0.99, 0.99))


def _iqr_fences(q1: float, q3: float, iqr: float, lo_m: float, hi_m: float) -> tuple[float, float]:
    return q1 - lo_m * iqr, q3 + hi_m * iqr


def _adjusted_fences(q1: float, q3: float, iqr: float, mc: float) -> tuple[float, float]:
    """Hubert & Vandervieren (2008) adjusted boxplot."""
    if mc >= 0:
        return q1 - 1.5 * math.exp(-4.0 * mc) * iqr, q3 + 1.5 * math.exp(3.0 * mc) * iqr
    return q1 - 1.5 * math.exp(-3.0 * mc) * iqr, q3 + 1.5 * math.exp(4.0 * mc) * iqr


def _skew_transform(v: np.ndarray, skew: float) -> tuple[np.ndarray, str]:
    """Variance-stabilising transform appropriate to the skew direction."""
    if skew >= 0:
        shift = min(0.0, float(v.min()))
        return np.log1p(v - shift), f"log1p(x{-shift:+g})" if shift else "log1p(x)"
    vmax = float(v.max())
    return np.log1p(vmax - v), "reflected_log1p"


def _gesd_mask(v: np.ndarray, alpha: float, max_frac: float) -> np.ndarray:
    """Rosner's Generalized ESD test. Returns boolean mask over ``v``."""
    n = v.size
    out = np.zeros(n, dtype=bool)
    if n < 15:
        return out
    try:
        from scipy import stats
    except Exception:
        return out
    r = max(1, min(int(n * max_frac), 100))
    alive = np.ones(n, dtype=bool)
    removed: list[int] = []
    stat: list[float] = []
    crit: list[float] = []
    for i in range(1, r + 1):
        w = v[alive]
        s = w.std(ddof=1)
        if s == 0:
            break
        dev = np.abs(v - w.mean())
        dev[~alive] = -1.0
        j = int(dev.argmax())
        stat.append(float(dev[j] / s))
        removed.append(j)
        alive[j] = False
        m = n - i + 1
        t = stats.t.ppf(1 - alpha / (2 * m), m - 2)
        crit.append(float((m - 1) * t / math.sqrt((m - 2 + t * t) * m)))
    k = 0
    for i, (a, b) in enumerate(zip(stat, crit), start=1):
        if a > b:
            k = i
    if k:
        out[removed[:k]] = True
    return out


@dataclass
class _Core:
    stats: dict[str, float]
    fences: dict[str, float]
    methods: dict[str, np.ndarray]          # method -> bool mask over valid values
    ensemble: np.ndarray
    votes: np.ndarray
    modz: np.ndarray
    n_active: int
    heavy_skew: bool
    transform: str | None
    iqr_degenerate: bool


def _core(v: np.ndarray, cfg: OutlierConfig, rng: np.random.Generator) -> _Core | None:
    """Run every univariate method on a clean, finite vector."""
    n = v.size
    if n < cfg.min_rows:
        return None
    q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
    mean, std = float(v.mean()), float(v.std(ddof=1))
    if std == 0:
        return None
    iqr = q3 - q1
    absdev = np.abs(v - med)
    mad = float(np.median(absdev))
    meanad = float(absdev.mean())
    pop_std = float(v.std())
    z = (v - mean) / pop_std
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean() - 3.0)

    # IQR can collapse to 0 on zero-inflated / heavily tied data -> robust fallback scale
    degenerate = iqr <= 0
    if not degenerate:
        iqr_eff = iqr
    elif mad > 0:
        iqr_eff = 1.349 * 1.4826 * mad
    else:
        iqr_eff = 1.349 * 1.2533 * meanad

    with np.errstate(divide="ignore", invalid="ignore"):
        if mad > 0:
            modz = 0.6745 * absdev / mad
        elif meanad > 0:
            modz = absdev / (1.253314 * meanad)
        else:
            modz = np.zeros(n)

    lo, hi = _iqr_fences(q1, q3, iqr_eff, cfg.iqr_lower_multiplier, cfg.iqr_upper_multiplier)
    lo_s, hi_s = _iqr_fences(q1, q3, iqr_eff, cfg.severe_multiplier, cfg.severe_multiplier)
    mc = _medcouple(v, cfg.medcouple_sample, rng)
    adj_lo, adj_hi = _adjusted_fences(q1, q3, iqr_eff, mc)

    heavy = abs(skew) > cfg.heavy_skew_threshold
    methods: dict[str, np.ndarray] = {
        "tukey": (v < lo) | (v > hi),                   # reported, not a voter
        "tukey_severe": (v < lo_s) | (v > hi_s),
        "adjusted_boxplot": (v < adj_lo) | (v > adj_hi),
        "mad": modz > cfg.mad_threshold,
        "zscore": np.abs(z) > cfg.zscore_threshold,
    }
    voters = ["adjusted_boxplot", "mad", "zscore"]
    fences = {"lower": lo, "upper": hi, "lower_severe": lo_s, "upper_severe": hi_s,
              "adj_lower": adj_lo, "adj_upper": adj_hi}
    transform_name: str | None = None
    t = v
    if heavy:
        t, transform_name = _skew_transform(v, skew)
        tq1, tq3 = np.percentile(t, [25, 75])
        tiqr = float(tq3 - tq1)
        if tiqr > 0:
            tlo, thi = tq1 - 1.5 * tiqr, tq3 + 1.5 * tiqr
            methods["log_iqr"] = (t < tlo) | (t > thi)
            voters.append("log_iqr")
            # back-transform the fences to the original scale (for treatment bounds)
            if skew >= 0:
                shift = min(0.0, float(v.min()))
                fences["log_upper"] = float(np.expm1(thi) + shift)
                fences["log_lower"] = float(np.expm1(tlo) + shift)
            else:
                vmax = float(v.max())
                fences["log_lower"] = float(vmax - np.expm1(thi))
                fences["log_upper"] = float(vmax - np.expm1(tlo))
    if 15 <= n <= cfg.gesd_max_rows:
        methods["gesd"] = _gesd_mask(t if heavy else v, cfg.gesd_alpha, cfg.gesd_max_fraction)
        voters.append("gesd")

    votes = np.sum([methods[m] for m in voters], axis=0)
    need = max(1, min(cfg.min_votes, len(voters)))
    ensemble = votes >= need
    return _Core(
        stats=dict(q1=q1, q3=q3, median=med, mean=mean, std=std, iqr=iqr, iqr_eff=iqr_eff,
                   mad=mad, skew=skew, kurtosis=kurt, medcouple=mc),
        fences=fences, methods=methods, ensemble=ensemble, votes=votes, modz=modz,
        n_active=len(voters), heavy_skew=heavy, transform=transform_name,
        iqr_degenerate=degenerate,
    )


def _expand_core(c: _Core, sel: np.ndarray) -> _Core:
    """Lift a _Core computed on v[sel] back to full length (unselected rows are never flagged)."""
    def lift(a: np.ndarray, fill: float | bool) -> np.ndarray:
        out = np.full(sel.size, fill, dtype=a.dtype)
        out[sel] = a
        return out
    return replace(
        c,
        methods={k: lift(m, False) for k, m in c.methods.items()},
        ensemble=lift(c.ensemble, False), votes=lift(c.votes, 0), modz=lift(c.modz, 0.0),
    )


# ───────────────────────────── ingestion helpers ─────────────────────────────
def _coerce_frame(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, dict):
        return pl.DataFrame(df)
    try:  # pandas
        return pl.from_pandas(df)
    except Exception as e:
        raise TypeError(f"Unsupported input type {type(df)!r}") from e


def _is_numeric_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_numeric())
    except AttributeError:  # very old polars
        return dt in getattr(pl, "NUMERIC_DTYPES", set())


def _is_int_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_integer())
    except AttributeError:
        return dt in getattr(pl, "INTEGER_DTYPES", set())


def _numeric_series(s: pl.Series, coerce_strings: bool) -> pl.Series | None:
    """Return a Float64 Series, or None if the column is not numeric-like."""
    dt = s.dtype
    if dt == pl.Boolean:
        return None
    if _is_numeric_dtype(dt):
        return s.cast(pl.Float64, strict=False)
    if coerce_strings and dt == pl.Utf8:
        cleaned = s.str.replace_all(r"[,$€£₹%\s]", "").cast(pl.Float64, strict=False)
        nn = s.drop_nulls().len()
        if nn > 0 and cleaned.drop_nulls().len() / nn >= 0.9:
            return cleaned
    return None


def _to_array(s: pl.Series, cfg: OutlierConfig) -> np.ndarray | None:
    fs = _numeric_series(s, cfg.coerce_numeric_strings)
    if fs is None:
        return None
    return np.asarray(fs.fill_null(float("nan")).to_numpy(), dtype=np.float64)


def _name_tokens(col: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col).lower()
    toks = {t for t in re.split(r"[^a-z0-9]+", spaced) if t}
    return toks | {t[:-1] for t in toks if t.endswith("s") and len(t) > 3}


def _looks_like_id(col: str, arr: np.ndarray, is_int: bool) -> bool:
    v = arr[np.isfinite(arr)]
    if v.size < 20:
        return False
    uniq_ratio = np.unique(v).size / v.size
    if _name_tokens(col) & _ID_KEYWORDS and uniq_ratio > 0.9:
        return True
    if is_int and uniq_ratio >= 0.98:
        d = np.diff(arr[np.isfinite(arr)])
        return bool((d >= 0).all() or (d <= 0).all())
    return False


def _domain_bounds(col: str, v: np.ndarray, cfg: OutlierConfig) -> tuple[float | None, float | None, str]:
    if col in cfg.domain_rules:
        lo, hi = cfg.domain_rules[col]
        return lo, hi, "user_rule"
    if not cfg.infer_domain_rules or v.size == 0:
        return None, None, ""
    toks = _name_tokens(col)
    if toks & _PCT_KEYWORDS:
        return 0.0, (1.0 if float(v.max()) <= 1.0 else 100.0), "percent"
    if toks & _UNIT_KEYWORDS:
        return 0.0, 1.0, "unit_interval"
    if toks & _LAT_KEYWORDS:
        return -90.0, 90.0, "latitude"
    if toks & _LON_KEYWORDS:
        return -180.0, 180.0, "longitude"
    if toks & NON_NEGATIVE_KEYWORDS:
        hi = next((h for k, h in _SOFT_UPPER.items() if k in toks), None)
        return 0.0, hi, "non_negative"
    return None, None, ""


def _find_sentinels(v: np.ndarray, outlier_like: np.ndarray, cfg: OutlierConfig) -> list[dict]:
    if v.size < 50 or np.unique(v).size < 20:
        return []
    found = []
    floor = max(3, int(0.005 * v.size))
    for s in _SENTINELS:
        m = v == s
        c = int(m.sum())
        if c >= floor and outlier_like[m].mean() > 0.5:
            found.append({"value": s, "count": c})
    return found


# ───────────────────────────── column analysis ─────────────────────────────
@dataclass
class _ColResult:
    info: dict
    mask: np.ndarray
    bounds: tuple[float | None, float | None] | None
    skipped: str | None = None
    skewed: bool = False
    stat_enabled: bool = False
    null_frac: float = 0.0


def _classify(c: _Core | None, v: np.ndarray, n_unique: int) -> str:
    if c is None:
        return "insufficient"
    if (v == 0).mean() > 0.3:
        return "zero_inflated"
    if c.stats["skew"] > 1.0 and c.heavy_skew:
        return "heavy_right_skew"
    if c.stats["skew"] < -1.0 and c.heavy_skew:
        return "heavy_left_skew"
    if abs(c.stats["skew"]) < 0.5 and abs(c.stats["kurtosis"]) < 1.0:
        return "approx_normal"
    if c.stats["kurtosis"] > 3.0:
        return "heavy_tailed"
    return "continuous"


def _recommend(pct: float, domain_n: int, sentinels: list, c: _Core | None, severe: int) -> tuple[Treatment, str]:
    if sentinels:
        return "null", "Suspected sentinel codes (e.g. 999 / -1): treat as missing, not as extreme values."
    if domain_n:
        return "null", "Values violate the column's valid range: treat as invalid/missing and investigate the source."
    if pct > 10:
        return "review", "More than 10% flagged: this is a distribution-shape issue, not stray outliers. Consider segmenting."
    if c is not None and c.heavy_skew:
        return "log_transform", "Heavy skew: model on a log scale or winsorise instead of deleting rows."
    if severe:
        return "clip", "Severe outliers present: verify against source, then clip to treatment bounds."
    return "clip", "Mild outliers: clipping to robust bounds is low-risk."


def _analyze_column(col: str, arr: np.ndarray, is_int: bool, cfg: OutlierConfig,
                    rng: np.random.Generator, n_rows: int) -> _ColResult:
    inf_mask = np.isinf(arr)
    finite = np.isfinite(arr)
    v = arr[finite]
    null_frac = float(np.isnan(arr).mean())
    empty = np.zeros(n_rows, dtype=bool)
    if v.size == 0:
        return _ColResult({}, empty, None, "no valid numeric values", null_frac=null_frac)

    n_unique = int(np.unique(v).size)
    stat_ok = True
    skip_reason = None
    if n_unique <= 1:
        stat_ok, skip_reason = False, "constant"
    elif n_unique == 2:
        stat_ok, skip_reason = False, "binary / flag column"
    elif n_unique <= cfg.discrete_unique_max and (n_unique / v.size) < 0.05:
        stat_ok, skip_reason = False, f"low-cardinality discrete ({n_unique} distinct values)"
    elif cfg.skip_id_like and _looks_like_id(col, arr, is_int):
        stat_ok, skip_reason = False, "identifier-like column"

    core = _core(v, cfg, rng) if stat_ok else None
    zero_inflated = bool(stat_ok and (v == 0).mean() > 0.3)
    if zero_inflated:  # judge the non-zero magnitudes; structural zeros are not outliers
        nz = v != 0
        if nz.sum() >= cfg.min_rows:
            sub = _core(v[nz], cfg, rng)
            core = _expand_core(sub, nz) if sub is not None else core
    if stat_ok and core is None:
        skip_reason = "too few rows or zero variance"

    # ── domain / validity ──
    lo_d, hi_d, rule = _domain_bounds(col, v, cfg)
    dom_valid = np.zeros(v.size, dtype=bool)
    domain_msgs: list[str] = []
    if lo_d is not None:
        neg = v < lo_d
        if neg.any():
            dom_valid |= neg
            if rule == "non_negative":
                domain_msgs.append(f"{int(neg.sum())} negative values in non-negative domain '{col}'")
            else:
                domain_msgs.append(f"{int(neg.sum())} values below {lo_d:g} ({rule})")
    if hi_d is not None:
        over = v > hi_d
        if over.any():
            dom_valid |= over
            domain_msgs.append(f"{int(over.sum())} values above {hi_d:g} ({rule})")
    n_inf = int(inf_mask.sum())
    if n_inf:
        domain_msgs.append(f"{n_inf} infinite values")

    # ── statistical ensemble mapped back to full length ──
    stat_valid = core.ensemble if core is not None else np.zeros(v.size, dtype=bool)
    sentinels: list[dict] = []
    if cfg.detect_sentinels and core is not None:
        sentinels = _find_sentinels(v, core.methods["tukey"] | dom_valid, cfg)
        for s in sentinels:
            stat_valid = stat_valid | (v == s["value"])
    final_valid = stat_valid | dom_valid

    full = empty.copy()
    full[np.flatnonzero(finite)] = final_valid
    full |= inf_mask
    dom_full = inf_mask.copy()
    dom_full[np.flatnonzero(finite)] |= dom_valid
    stat_full = empty.copy()
    stat_full[np.flatnonzero(finite)] = stat_valid

    count = int(full.sum())
    pct = round(count / n_rows * 100, 2)

    # ── treatment bounds (deliberately the *widest* robust fence; clipping should be conservative) ──
    bounds: tuple[float | None, float | None] | None = None
    if core is not None:
        f = core.fences
        lows = [f["lower"], f["adj_lower"], f.get("log_lower", np.inf)]
        highs = [f["upper"], f["adj_upper"], f.get("log_upper", -np.inf)]
        b_lo, b_hi = float(min(lows)), float(max(highs))
        if lo_d is not None:
            b_lo = max(b_lo, lo_d)
        if hi_d is not None:
            b_hi = min(b_hi, hi_d)
        bounds = (b_lo, b_hi)
    elif lo_d is not None or hi_d is not None:
        bounds = (lo_d, hi_d)

    info: dict[str, Any] = {}
    if count > 0 or cfg.include_clean_columns or domain_msgs:
        idx_all = np.flatnonzero(full)
        score = np.zeros(n_rows)
        if core is not None:
            score[np.flatnonzero(finite)] = core.modz
        score[idx_all[~np.isfinite(arr[idx_all])]] = np.inf
        order = idx_all[np.argsort(-score[idx_all], kind="stable")]
        flagged_vals = arr[order]
        fin_flagged = flagged_vals[np.isfinite(flagged_vals)]
        lower_n = int((fin_flagged < (core.fences["adj_lower"] if core else -np.inf)).sum()) if core else 0
        upper_n = int((fin_flagged > (core.fences["adj_upper"] if core else np.inf)).sum()) if core else 0
        direction = "both" if lower_n and upper_n else "upper" if upper_n else "lower" if lower_n else "n/a"

        s = core.stats if core else {}
        fe = core.fences if core else {}
        m = core.methods if core else {}
        severe_count = int(m["tukey_severe"][core.ensemble | m["tukey_severe"]].sum()) if core else 0
        any_flag = np.zeros(v.size, dtype=bool)
        if core:
            for k in ("adjusted_boxplot", "mad", "zscore", "log_iqr", "gesd"):
                if k in m:
                    any_flag |= m[k]
        if core and core.ensemble.any():
            ratio = float(core.votes[core.ensemble].mean() / core.n_active)
            consensus = "high" if ratio >= 0.8 else "moderate" if ratio >= 0.5 else "low"
        else:
            consensus = "low" if any_flag.any() else "none"

        rec, note = _recommend(pct, int(dom_full.sum()), sentinels, core, severe_count)
        info = {
            # legacy keys
            "count": count,
            "percentage": pct,
            "severe_count": severe_count,
            "mad_outlier_count": int(m["mad"].sum()) if core else 0,
            "log_outlier_count": int(m["log_iqr"].sum()) if core and "log_iqr" in m else 0,
            "is_heavy_skew": bool(core.heavy_skew) if core else False,
            "skewness": _f(s.get("skew"), 2) or 0.0,
            "min_outlier": _f(fin_flagged.min()) if fin_flagged.size else None,
            "max_outlier": _f(fin_flagged.max()) if fin_flagged.size else None,
            "lower_bound": _f(fe.get("lower")),
            "upper_bound": _f(fe.get("upper")),
            "lower_bound_severe": _f(fe.get("lower_severe")),
            "upper_bound_severe": _f(fe.get("upper_severe")),
            "skew_adjusted_lower": _f(fe.get("adj_lower")),
            "skew_adjusted_upper": _f(fe.get("adj_upper")),
            "consensus_rating": consensus,
            "sample_values": [_f(x) for x in flagged_vals[: cfg.sample_size]],
            "domain_anomalies": domain_msgs,
            # new keys
            "statistical_count": int(stat_full.sum()),
            "domain_violation_count": int(dom_full.sum()),
            "iqr_count": int(m["tukey"].sum()) if core else 0,
            "method_counts": {k: int(x.sum()) for k, x in m.items()} if core else {},
            "direction": direction,
            "lower_outlier_count": lower_n,
            "upper_outlier_count": upper_n,
            "kurtosis": _f(s.get("kurtosis"), 2),
            "medcouple": _f(s.get("medcouple"), 3),
            "iqr_degenerate": bool(core.iqr_degenerate) if core else False,
            "skew_transform": core.transform if core else None,
            "distribution": _classify(core, v, n_unique),
            "zero_inflated": zero_inflated,
            "n_valid": int(v.size),
            "null_pct": round(null_frac * 100, 2),
            "suspected_sentinels": sentinels,
            "skipped_statistical_reason": skip_reason if core is None else None,
            "outlier_indices": [int(i) for i in order[: cfg.max_indices_returned]],
            "indices_truncated": bool(order.size > cfg.max_indices_returned),
            "treatment_bounds": [_f(bounds[0]), _f(bounds[1])] if bounds else None,
            "recommended_treatment": rec,
            "treatment_note": note,
        }
    return _ColResult(info, full, bounds, skip_reason if core is None else None,
                      skewed=bool(core.heavy_skew) if core else False,
                      stat_enabled=core is not None, null_frac=null_frac)


# ───────────────────────────── multivariate ─────────────────────────────
def _multivariate(df_arrays: dict[str, np.ndarray], skewed: set[str], univariate_any: np.ndarray,
                  cfg: OutlierConfig, n_rows: int) -> tuple[dict | None, np.ndarray | None]:
    cols = list(df_arrays)
    if len(cols) < 2 or n_rows < cfg.multivariate_min_rows:
        return None, None
    try:
        from sklearn.covariance import EllipticEnvelope
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
    except Exception as e:  # sklearn optional
        logger.debug("Multivariate detection skipped (sklearn unavailable): %s", e)
        return None, None

    X = np.column_stack([df_arrays[c] for c in cols]).astype(np.float64)
    X[~np.isfinite(X)] = np.nan
    all_missing = np.isnan(X).all(axis=1)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    for j, c in enumerate(cols):
        if c in skewed:
            X[:, j] = np.sign(X[:, j]) * np.log1p(np.abs(X[:, j]))
    cmed = np.median(X, axis=0)
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    scale = (q3 - q1) / 1.349
    sd = X.std(axis=0)
    scale = np.where(scale > 0, scale, np.where(sd > 0, sd, 1.0))
    Z = (X - cmed) / scale

    rng = np.random.default_rng(cfg.random_state)
    n = Z.shape[0]
    sampled = n > cfg.multivariate_max_fit_rows
    fit_idx = np.sort(rng.choice(n, cfg.multivariate_max_fit_rows, replace=False)) if sampled else np.arange(n)
    Zf = Z[fit_idx]
    c = cfg.multivariate_contamination
    labels: dict[str, np.ndarray] = {}

    iso = IsolationForest(contamination=c, random_state=cfg.random_state, n_estimators=200, n_jobs=cfg.n_jobs)
    iso.fit(Zf)
    labels["isolation_forest"] = iso.predict(Z) == -1

    k = max(5, min(20, Zf.shape[0] - 1))
    lof = LocalOutlierFactor(n_neighbors=k, contamination=c, novelty=True, n_jobs=cfg.n_jobs)
    lof.fit(Zf)
    lof_lab = np.zeros(n, dtype=bool)
    lof_lab[fit_idx] = lof.negative_outlier_factor_ < lof.offset_      # training rows: self-excluded scores
    rest = np.setdiff1d(np.arange(n), fit_idx, assume_unique=True)
    if rest.size:
        lof_lab[rest] = lof.predict(Z[rest]) == -1
    labels["lof"] = lof_lab

    p = Z.shape[1]
    if p <= 15 and Zf.shape[0] >= 10 * p:
        try:
            ee = EllipticEnvelope(contamination=c, random_state=cfg.random_state).fit(Zf)
            labels["robust_mahalanobis"] = ee.predict(Z) == -1
        except Exception as e:
            logger.debug("Robust Mahalanobis skipped: %s", e)

    votes = np.sum(list(labels.values()), axis=0)
    need = max(1, min(cfg.multivariate_min_votes, len(labels)))
    flagged = (votes >= need) & ~all_missing
    count = int(flagged.sum())
    if count == 0:
        return None, flagged

    idx = np.flatnonzero(flagged)
    order = idx[np.argsort(-np.abs(Z[idx]).max(axis=1), kind="stable")]
    drivers = []
    for i in order[: cfg.sample_size]:
        top = np.argsort(-np.abs(Z[i]))[:3]
        drivers.append({"row": int(i), "drivers": [{"column": cols[j], "robust_z": _f(Z[i, j], 2)} for j in top]})

    summary = {
        "count": count,
        "percentage": round(count / n_rows * 100, 2),
        "columns_evaluated": cols,
        "isolation_forest_count": int(labels["isolation_forest"].sum()),
        "high_confidence_multivariate_count": count,
        "sampled": bool(sampled),
        "lof_count": int(labels["lof"].sum()),
        "robust_mahalanobis_count": int(labels["robust_mahalanobis"].sum()) if "robust_mahalanobis" in labels else None,
        "multivariate_only_count": int((flagged & ~univariate_any).sum()),
        "outlier_indices": [int(i) for i in order[: cfg.max_indices_returned]],
        "indices_truncated": bool(order.size > cfg.max_indices_returned),
        "top_drivers": drivers,
    }
    return summary, flagged


# ───────────────────────────── report & treatment ─────────────────────────────
@dataclass
class OutlierReport:
    infos: dict[str, dict]
    masks: dict[str, np.ndarray]
    bounds: dict[str, tuple[float | None, float | None]]
    n_rows: int
    skipped: dict[str, str]
    multivariate: dict | None
    multivariate_mask: np.ndarray | None
    group_masks: dict[str, np.ndarray]
    config: OutlierConfig

    # -- serialisation (legacy-compatible shape) --
    def to_dict(self) -> dict[str, dict]:
        out: dict[str, dict] = {c: i for c, i in self.infos.items() if i}
        if self.multivariate:
            out["_multivariate_summary"] = self.multivariate
        if self.config.include_meta and (self.masks or self.infos or self.multivariate):
            out["_meta"] = {
                "n_rows": self.n_rows,
                "columns_analyzed": list(self.masks),
                "columns_skipped": self.skipped,
                "rows_flagged_any": int(self.row_flags().sum()),
                "pct_rows_flagged_any": round(float(self.row_flags().mean()) * 100, 2) if self.n_rows else 0.0,
            }
        return out

    def row_flags(self) -> np.ndarray:
        m = np.zeros(self.n_rows, dtype=bool)
        for x in self.masks.values():
            m |= x
        if self.multivariate_mask is not None:
            m |= self.multivariate_mask
        return m

    def flag_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append row-level columns: __outlier_score, __outlier_columns, __is_outlier."""
        score = np.zeros(self.n_rows, dtype=np.int32)
        names: dict[int, list[str]] = {}
        for c, m in self.masks.items():
            score += m
            for i in np.flatnonzero(m):
                names.setdefault(int(i), []).append(c)
        if self.multivariate_mask is not None:
            score += self.multivariate_mask
            for i in np.flatnonzero(self.multivariate_mask):
                names.setdefault(int(i), []).append("_multivariate")
        col_str = [",".join(names.get(i, [])) for i in range(self.n_rows)]
        return df.with_columns(
            pl.Series("__outlier_score", score),
            pl.Series("__outlier_columns", col_str),
            pl.Series("__is_outlier", score > 0),
        )

    def apply(self, df: pl.DataFrame, strategy: str = "auto", columns: list[str] | None = None,
              per_column: dict[str, str] | None = None) -> tuple[pl.DataFrame, list[dict]]:
        """
        Treat flagged values.  strategy / per_column values:
          auto   -> each column's ``recommended_treatment``
          clip   -> clamp to conservative robust bounds (keeps rows)
          null   -> set flagged cells to null (keeps rows)
          median -> replace flagged cells with the median of unflagged values
          drop   -> remove rows flagged in any selected column
          log    -> log1p-transform the column (non-negative columns only)
          flag   -> add ``<col>__is_outlier`` boolean column
          none   -> leave untouched
        Returns (new_df, change_log).
        """
        per_column = per_column or {}
        log: list[dict] = []
        out = df
        drop = np.zeros(self.n_rows, dtype=bool)
        for col in columns or list(self.masks):
            mask = self.masks.get(col)
            if mask is None or col not in out.columns or not mask.any():
                continue
            strat = per_column.get(col, strategy)
            if strat == "auto":
                strat = _AUTO_STRATEGY.get(self.infos.get(col, {}).get("recommended_treatment", "none"), "none")
            if strat == "none":
                continue
            src = out[col]
            num = _numeric_series(src, self.config.coerce_numeric_strings)
            if num is None:
                log.append({"column": col, "strategy": strat, "rows_affected": 0, "detail": "column not numeric"})
                continue
            is_int = _is_int_dtype(src.dtype)
            n_aff = int(mask.sum())
            m_expr = pl.lit(pl.Series(mask))

            if strat == "flag":
                out = out.with_columns(pl.Series(f"{col}__is_outlier", mask))
                detail = f"added {col}__is_outlier"
            elif strat == "drop":
                drop |= mask
                detail = "rows queued for removal"
            elif strat == "null":
                out = out.with_columns(pl.when(m_expr).then(None).otherwise(num).alias(col))
                detail = "flagged cells set to null"
            elif strat == "median":
                arr = np.asarray(num.fill_null(float("nan")).to_numpy(), dtype=np.float64)
                good = arr[~mask & np.isfinite(arr)]
                if good.size == 0:
                    continue
                fill = float(np.median(good))
                fill = round(fill) if is_int else fill
                out = out.with_columns(pl.when(m_expr).then(fill).otherwise(num).alias(col))
                detail = f"flagged cells replaced with median {fill:g}"
            elif strat == "clip":
                lo, hi = self.bounds.get(col, (None, None))
                lo = lo if lo is not None and math.isfinite(lo) else None
                hi = hi if hi is not None and math.isfinite(hi) else None
                if lo is None and hi is None:
                    continue
                if is_int:
                    lo = math.ceil(lo) if lo is not None else None
                    hi = math.floor(hi) if hi is not None else None
                clipped = num.clip(lo, hi)
                out = out.with_columns((clipped.round(0).cast(src.dtype, strict=False) if is_int else clipped).alias(col))
                detail = f"clipped to [{lo}, {hi}]"
            elif strat == "log":
                if float(num.min() or 0.0) < 0:
                    log.append({"column": col, "strategy": strat, "rows_affected": 0,
                                "detail": "skipped: negative values present"})
                    continue
                out = out.with_columns(num.log1p().alias(col))
                n_aff = int(num.len())
                detail = "log1p applied to whole column"
            else:
                raise ValueError(f"Unknown strategy '{strat}'")
            log.append({"column": col, "strategy": strat, "rows_affected": n_aff, "detail": detail})

        if drop.any():
            out = out.filter(pl.Series(~drop))
            log.append({"column": "*", "strategy": "drop", "rows_affected": int(drop.sum()),
                        "detail": "rows removed"})
        return out, log


# ───────────────────────────── public API ─────────────────────────────
def analyze_outliers(df: Any, numeric_cols: list[str] | None = None,
                     config: OutlierConfig | None = None, **overrides: Any) -> OutlierReport:
    """Full-featured entry point. ``overrides`` are OutlierConfig fields, e.g. group_by=["region"]."""
    cfg = replace(config or OutlierConfig(), **overrides) if (config or overrides) else OutlierConfig()
    frame = _coerce_frame(df)
    n_rows = frame.height
    empty = OutlierReport({}, {}, {}, n_rows, {}, None, None, {}, cfg)
    if n_rows == 0:
        return empty

    cols = numeric_cols
    if cols is None:  # auto-detect numeric (and numeric-like string) columns
        cols = [c for c in frame.columns
                if c not in (cfg.group_by or []) and _numeric_series(frame[c], cfg.coerce_numeric_strings) is not None]
    cols = [c for c in cols if c in frame.columns]
    if cfg.max_columns:
        cols = cols[: cfg.max_columns]
    if not cols:
        return empty

    rng = np.random.default_rng(cfg.random_state)
    infos, masks, bounds, skipped = {}, {}, {}, {}
    arrays: dict[str, np.ndarray] = {}
    results: dict[str, _ColResult] = {}

    for col in cols:
        try:
            arr = _to_array(frame[col], cfg)
            if arr is None:
                skipped[col] = "not numeric"
                continue
            res = _analyze_column(col, arr, _is_int_dtype(frame[col].dtype), cfg, rng, n_rows)
            arrays[col] = arr
            results[col] = res
            if res.skipped and not res.info:
                skipped[col] = res.skipped
            infos[col] = res.info
            masks[col] = res.mask
            if res.bounds is not None:
                bounds[col] = res.bounds
        except Exception as e:
            logger.error("Outlier detection failed for column %s: %s", col, e)
            skipped[col] = f"error: {e}"

    # ── contextual (group-wise) outliers ──
    group_masks: dict[str, np.ndarray] = {}
    gk = [g for g in (cfg.group_by or []) if g in frame.columns]
    if gk and arrays:
        keyed = frame.select(gk).with_columns(pl.Series("__i", np.arange(n_rows)))
        groups = [np.asarray(ix, dtype=np.int64) for ix in keyed.group_by(gk).agg(pl.col("__i"))["__i"].to_list()]
        for col, arr in arrays.items():
            if not results[col].stat_enabled:
                continue
            gm = np.zeros(n_rows, dtype=bool)
            for ix in groups:
                if ix.size < cfg.min_group_size:
                    continue
                sub = arr[ix]
                fin = np.isfinite(sub)
                core = _core(sub[fin], cfg, rng)
                if core is not None:
                    gm[ix[fin]] = core.ensemble
            group_masks[col] = gm
            ctx_only = gm & ~masks[col]
            info = infos[col]
            if not info and gm.any():
                info = infos[col] = {"count": int(masks[col].sum()), "percentage": 0.0}
            if info is not None and (gm.any() or info):
                info["group_by"] = gk
                info["group_outlier_count"] = int(gm.sum())
                info["contextual_only_count"] = int(ctx_only.sum())
                info["contextual_indices"] = [int(i) for i in np.flatnonzero(ctx_only)[: cfg.max_indices_returned]]
            if cfg.merge_group_flags:
                masks[col] = masks[col] | gm

    # ── multivariate ──
    mv_summary, mv_mask = None, None
    if cfg.multivariate:
        usable = [c for c in arrays if results[c].stat_enabled]
        usable.sort(key=lambda c: results[c].null_frac)
        usable = usable[: cfg.multivariate_max_cols]
        if len(usable) >= 2:
            uni_any = np.zeros(n_rows, dtype=bool)
            for m in masks.values():
                uni_any |= m
            try:
                mv_summary, mv_mask = _multivariate(
                    {c: arrays[c] for c in usable},
                    {c for c in usable if results[c].skewed}, uni_any, cfg, n_rows)
            except Exception as e:
                logger.debug("Multivariate outlier detection skipped: %s", e)

    return OutlierReport(infos, masks, bounds, n_rows, skipped, mv_summary, mv_mask, group_masks, cfg)


def detect_outliers(df: Any, numeric_cols: list[str] | None = None, **kwargs: Any) -> dict[str, dict]:
    """
    Backward-compatible wrapper: same name/shape as the original, now ensemble-driven.

    ``count`` is now the *final ensemble verdict* (>= min_votes method families agree, or a
    domain violation); the old Tukey-only count is available as ``iqr_count``.
    Extra kwargs are OutlierConfig fields.  Use ``analyze_outliers`` to also get masks/treatment.
    """
    if numeric_cols is not None and len(numeric_cols) == 0:
        return {}
    try:
        report = analyze_outliers(df, numeric_cols, **kwargs)
        if not report.masks and not report.infos and not report.multivariate:
            return {}
        return report.to_dict()
    except Exception as e:
        logger.error("Outlier detection failed: %s", e)
        return {}


# ───────────────────────────── time series ─────────────────────────────
_S_PERIODS = ((3600, 24), (1800, 48), (900, 96), (60, 60), (86400, 7), (604800, 52))


def _acf_period(y: np.ndarray, max_lag: int) -> int | None:
    n = y.size
    if n < 16:
        return None
    x = y - y.mean()
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] <= 0:
        return None
    acf = acf / acf[0]
    top = min(max_lag, n // 2)
    for lag in range(3, top):
        if acf[lag] > 0.3 and acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]:
            return lag
    return None


def _infer_period(step_s: float | None, y: np.ndarray) -> int | None:
    n = y.size
    if step_s:
        if 27 * 86400 <= step_s <= 32 * 86400:
            cand = 12
        elif 89 * 86400 <= step_s <= 93 * 86400:
            cand = 4
        else:
            cand = next((p for s, p in _S_PERIODS if abs(step_s - s) / s < 0.05), None)
        if cand and n >= 2 * cand:
            return cand
    return _acf_period(y, max_lag=max(4, n // 3))


def _centered_ma(y: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average (2xMA for even w) with reflected edges."""
    n = y.size
    pad = min(w, n - 1)
    yp = np.pad(y, pad, mode="reflect")
    k = np.r_[0.5, np.ones(w - 1), 0.5] / w if w % 2 == 0 else np.ones(w) / w
    return np.convolve(yp, k, mode="same")[pad:pad + n]


def _numpy_decompose(y: np.ndarray, period: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust classical decomposition: MA trend, phase-median seasonality, iterative winsorising."""
    n = y.size
    if not (period and period >= 2 and n >= 2 * period):
        w = max(5, n // 10)
        w += (w % 2 == 0)
        pad = min(w // 2, n - 1)
        win = np.lib.stride_tricks.sliding_window_view(np.pad(y, pad, mode="reflect"), 2 * pad + 1)
        trend = np.median(win, axis=1)[:n]
        return trend, np.zeros(n), y - trend
    yw = y.copy()
    trend = seasonal = resid = np.zeros(n)
    for _ in range(3):
        trend = _centered_ma(yw, period)
        det = yw - trend
        pat = np.array([np.median(det[i::period]) for i in range(period)])
        pat -= pat.mean()
        seasonal = np.resize(pat, n)
        resid = y - trend - seasonal
        sig = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if sig == 0:
            break
        yw = np.where(np.abs(resid) > 3 * sig, trend + seasonal + np.sign(resid) * 3 * sig, y)
    return trend, seasonal, resid


def _decompose(y: np.ndarray, period: int | None, robust: bool = True):
    if period and period >= 2 and y.size >= 2 * period:
        try:
            from statsmodels.tsa.seasonal import STL
            r = STL(y, period=period, robust=robust).fit()
            return np.asarray(r.trend), np.asarray(r.seasonal), np.asarray(r.resid), "stl"
        except Exception as e:
            logger.debug("STL unavailable, using NumPy fallback: %s", e)
    t, s, r = _numpy_decompose(y, period)
    return t, s, r, "numpy_fallback"


def _time_to_us(s: pl.Series) -> np.ndarray | None:
    dt = s.dtype
    try:
        if dt == pl.Utf8:
            s = s.str.to_datetime(strict=False)
            dt = s.dtype
        if dt == pl.Date:
            s = s.cast(pl.Datetime("us"))
        if isinstance(s.dtype, pl.Datetime) or s.dtype == pl.Datetime:
            return np.asarray(s.dt.epoch("us").to_numpy(), dtype=np.float64)
        if _is_numeric_dtype(dt):
            return np.asarray(s.cast(pl.Float64).to_numpy(), dtype=np.float64)
    except Exception:
        pass
    return None


def detect_time_series_stl_outliers(df: pl.DataFrame, date_col: str, numeric_col: str,
                                    period: int | None = None, mad_threshold: float = 3.5,
                                    robust: bool = True, max_report: int = 50) -> dict:
    """
    Seasonal-trend decomposition outliers (observed = trend + seasonal + residual).

    Upgrades: period inferred from sampling frequency / autocorrelation (or pass ``period``),
    duplicate timestamps averaged, gaps interpolated for decomposition (flags only real points),
    robust STL, NumPy fallback with reflected edges, level-shift detection, strength metrics,
    and the flagged timestamps themselves are returned.
    """
    df = _coerce_frame(df)
    if date_col not in df.columns or numeric_col not in df.columns or df.height < 14:
        return {"has_stl_outliers": False, "reason": "Insufficient observations for STL decomposition"}
    try:
        t_raw = _time_to_us(df[date_col])
        y_raw = _numeric_series(df[numeric_col], True)
        if y_raw is None:
            return {"has_stl_outliers": False, "reason": f"'{numeric_col}' is not numeric"}
        y_arr = np.asarray(y_raw.fill_null(float("nan")).to_numpy(), dtype=np.float64)
        ts_is_time = t_raw is not None
        if t_raw is None:
            t_raw = np.arange(df.height, dtype=np.float64)
        ok = np.isfinite(y_arr) & np.isfinite(t_raw)
        t_raw, y_arr = t_raw[ok], y_arr[ok]
        if t_raw.size < 14:
            return {"has_stl_outliers": False, "reason": "Fewer than 14 non-null observations"}

        # sort + average duplicate timestamps
        order = np.argsort(t_raw, kind="stable")
        t_raw, y_arr = t_raw[order], y_arr[order]
        uniq, inv = np.unique(t_raw, return_inverse=True)
        if uniq.size < t_raw.size:
            y_arr = np.bincount(inv, weights=y_arr) / np.bincount(inv)
            t_raw = uniq
        n = t_raw.size
        if n < 14:
            return {"has_stl_outliers": False, "reason": "Fewer than 14 distinct timestamps"}

        steps = np.diff(t_raw)
        step = float(np.median(steps))
        step_s = step / 1e6 if ts_is_time else None
        notes: list[str] = []
        monthly = step_s is not None and 27 * 86400 <= step_s <= 32 * 86400
        y_dec, pos = y_arr, np.arange(n)
        if step > 0 and steps.max() > 1.5 * step and not monthly:
            m = int(round((t_raw[-1] - t_raw[0]) / step)) + 1
            if m <= 3 * n:
                grid = t_raw[0] + step * np.arange(m)
                y_dec = np.interp(grid, t_raw, y_arr)
                pos = np.clip(np.round((t_raw - t_raw[0]) / step).astype(int), 0, m - 1)
                notes.append(f"{m - n} gaps interpolated for decomposition")
            else:
                notes.append("highly irregular sampling; treated as index-ordered")

        per = period or _infer_period(step_s, y_dec)
        trend_g, seas_g, resid_g, method = _decompose(y_dec, per, robust)
        trend, seas, resid = trend_g[pos], seas_g[pos], resid_g[pos]

        med_r = float(np.median(resid))
        mad_r = float(np.median(np.abs(resid - med_r)))
        if mad_r > 0:
            score = 0.6745 * np.abs(resid - med_r) / mad_r
            flag = score > mad_threshold
        else:
            sd = float(resid.std())
            score = np.abs(resid - resid.mean()) / sd if sd > 0 else np.zeros(n)
            flag = score > 3.0
        cnt = int(flag.sum())

        # level shifts: abrupt jumps in the trend component
        dtr = np.diff(trend_g)
        mad_d = float(np.median(np.abs(dtr - np.median(dtr))))
        shifts: list[int] = []
        if mad_d > 0:
            shifts = [int(i + 1) for i in np.flatnonzero(np.abs(dtr - np.median(dtr)) > 8 * 1.4826 * mad_d)]
        merged: list[int] = []
        for i in shifts:  # a smoothed step spans several points: keep the sharpest of each run
            if merged and i - merged[-1][-1] <= 1:
                merged[-1].append(i)
            else:
                merged.append([i])
        shifts = [max(g, key=lambda j: abs(dtr[j - 1])) for g in merged][:max_report]

        var_r = float(resid.var())
        seas_strength = max(0.0, 1 - var_r / float((resid + seas).var())) if (resid + seas).var() > 0 else 0.0
        trend_strength = max(0.0, 1 - var_r / float((resid + trend).var())) if (resid + trend).var() > 0 else 0.0

        def label(i: int) -> Any:
            if not ts_is_time:
                return float(t_raw[i])
            return str(np.datetime64(int(t_raw[i]), "us"))

        top = np.flatnonzero(flag)
        top = top[np.argsort(-score[top], kind="stable")][:max_report]
        points = [{
            "timestamp": label(int(i)),
            "value": _f(y_arr[i]),
            "expected": _f(trend[i] + seas[i]),
            "residual": _f(resid[i]),
            "score": _f(score[i], 2),
            "type": "spike" if resid[i] > 0 else "dip",
        } for i in top]

        return {
            "has_stl_outliers": cnt > 0,
            "stl_outlier_count": cnt,
            "total_observations": n,
            "pct_stl_outliers": round(cnt / n * 100, 2),
            "date_col": date_col,
            "numeric_col": numeric_col,
            "trend_summary": {"min": float(np.min(trend)), "max": float(np.max(trend))},
            "seasonal_summary": {"amplitude": float(np.max(seas) - np.min(seas))},
            "residual_mad": mad_r,
            # new
            "method": method,
            "period": per,
            "period_inferred": period is None,
            "sampling_step_seconds": _f(step_s, 2) if step_s else None,
            "seasonal_strength": round(seas_strength, 3),
            "trend_strength": round(trend_strength, 3),
            "outliers": points,
            "level_shift_count": len(shifts),
            "level_shifts": [{"timestamp": label(i), "before": _f(trend[i - 1]), "after": _f(trend[i])}
                             for i in shifts if i < n],
            "notes": notes,
        }
    except Exception as err:
        logger.debug("STL time series decomposition skipped: %s", err)
        return {"has_stl_outliers": False, "reason": str(err)}


def detect_time_series_outliers_multi(df: Any, date_col: str, numeric_cols: list[str] | None = None,
                                      **kwargs: Any) -> dict[str, dict]:
    """Run the STL detector over several metrics at once."""
    frame = _coerce_frame(df)
    cols = numeric_cols or [c for c in frame.columns
                            if c != date_col and _numeric_series(frame[c], True) is not None]
    return {c: detect_time_series_stl_outliers(frame, date_col, c, **kwargs) for c in cols}