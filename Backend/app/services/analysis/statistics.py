""" Statistics engine """

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from statistics import NormalDist
from typing import Any, Iterable, Literal

import numpy as np
import polars as pl

try:  # project settings are optional so the module is usable standalone
    from app.core.config import settings

    _DEFAULT_STRONG = float(settings.CORRELATION_STRONG_THRESHOLD)
    _DEFAULT_SKEW = float(settings.SKEWNESS_THRESHOLD)
except Exception:  # pragma: no cover
    _DEFAULT_STRONG = 0.7
    _DEFAULT_SKEW = 1.0

try:  # SciPy is optional, everything degrades gracefully without it
    from scipy import stats as _sp_stats
except Exception:  # pragma: no cover
    _sp_stats = None

logger = logging.getLogger(__name__)

# Kept for backward compatibility with code that imports these names.
CORRELATION_STRONG_THRESHOLD: float = _DEFAULT_STRONG
SKEWNESS_THRESHOLD: float = _DEFAULT_SKEW

_MAD_SCALE = 1.4826
_EPS = 1e-12


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StatsConfig:
    # Correlation
    strong_corr: float = _DEFAULT_STRONG
    multicollinear_threshold: float = 0.90
    vif_threshold: float = 5.0
    nonlinear_gap: float = 0.15          # |rho| - |r| gap that signals nonlinearity
    correlation_method: Literal["pearson", "spearman", "both"] = "both"
    missing: Literal["pairwise", "listwise"] = "pairwise"
    min_periods: int = 5
    include_spearman_only_pairs: bool = True
    exact_spearman_pair_limit: int = 2000
    alpha: float = 0.05

    # Distribution
    skew_threshold: float = _DEFAULT_SKEW
    kurtosis_threshold: float = 1.0
    trim_fraction: float = 0.05
    iqr_multiplier: float = 1.5
    modified_z_threshold: float = 3.5
    percentiles: tuple[float, ...] = (0.01, 0.05, 0.10, 0.90, 0.95, 0.99)

    # Scale controls
    max_rows: int = 500_000
    seed: int = 42

    # Nonlinear association
    include_nonlinear: bool = True
    nonlinear_sample: int = 5000
    nmi_threshold: float = 0.30
    max_pairs_nonlinear: int = 2000

    # Categorical
    top_k_categories: int = 10
    cat_max_cardinality: int = 50
    rare_level_pct: float = 1.0
    association_threshold: float = 0.30
    max_groups: int = 20

    # Quality flags
    high_missing_pct: float = 30.0


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _f(x: Any, nd: int = 4, default: float = 0.0) -> float:
    """Safe float: None, NaN and inf become `default`; otherwise rounded."""
    try:
        if x is None:
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return round(v, nd)
    except (TypeError, ValueError):
        return default


def _is_numeric_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_numeric())
    except AttributeError:  # very old Polars
        return dt in pl.NUMERIC_DTYPES


def _is_float_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_float())
    except AttributeError:
        return dt in (pl.Float32, pl.Float64)


def _is_integer_dtype(dt: Any) -> bool:
    try:
        return bool(dt.is_integer())
    except AttributeError:
        return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


def _is_string_dtype(dt: Any) -> bool:
    return dt == pl.Utf8


def _is_datetime_dtype(dt: Any) -> bool:
    return dt == pl.Date or dt == pl.Datetime


def _clean_expr(col: str) -> pl.Expr:
    """Float64 expression where null, NaN and +/-inf all become null."""
    raw = pl.col(col).cast(pl.Float64, strict=False)
    return pl.when(raw.is_finite()).then(raw).otherwise(None)


def _sample(df: pl.DataFrame, cfg: StatsConfig) -> pl.DataFrame:
    if df.height > cfg.max_rows:
        return df.sample(n=cfg.max_rows, seed=cfg.seed)
    return df


def _t_crit(dof: float, alpha: float) -> float:
    if dof <= 0:
        return float("nan")
    if _sp_stats is not None:
        return float(_sp_stats.t.ppf(1 - alpha / 2, dof))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return z + (z**3 + z) / (4 * dof) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * dof**2)


def _rank_avg(a: np.ndarray) -> np.ndarray:
    """Average ranks (1 based) with correct tie handling."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    _, inv, counts = np.unique(sorted_a, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + 1 + cum) / 2.0
    ranks = np.empty(a.shape[0], dtype=float)
    ranks[order] = avg[inv]
    return ranks


def _bh_adjust(p: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(list(p), dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _corr_pvalue(r: float, n: int, spearman: bool = False) -> float:
    if n < 4:
        return 1.0
    if abs(r) >= 1.0:
        return 0.0
    if not spearman and _sp_stats is not None:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return float(2 * _sp_stats.t.sf(abs(t), n - 2))
    z = math.atanh(r) * math.sqrt((n - 3) / (1.06 if spearman else 1.0))
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _fisher_ci(r: float, n: int, alpha: float, spearman: bool = False) -> tuple[float, float]:
    if n < 4 or abs(r) >= 1.0:
        return (r, r)
    se = math.sqrt((1.06 if spearman else 1.0) / (n - 3))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    c = math.atanh(r)
    return (math.tanh(c - z * se), math.tanh(c + z * se))


def _pct_label(p: float) -> str:
    return f"{p * 100:g}%"


# ----------------------------------------------------------------------------
# Column type inference
# ----------------------------------------------------------------------------
def infer_column_types(df: pl.DataFrame, config: StatsConfig | None = None) -> dict[str, list[str]]:
    """
    Classify every column. Returns lists under the keys:
    numeric, boolean, categorical, datetime, text, id_like, constant, empty,
    numeric_like_text, other.
    """
    cfg = config or StatsConfig()
    out: dict[str, list[str]] = {
        k: [] for k in
        ("numeric", "boolean", "categorical", "datetime", "text", "id_like",
         "constant", "empty", "numeric_like_text", "other")
    }
    n = df.height
    for c, dt in df.schema.items():
        try:
            nn = df[c].drop_nulls()
            non_null = nn.len()
            if non_null == 0:
                out["empty"].append(c)
                continue
            n_unique = nn.n_unique()
            if n_unique <= 1:
                out["constant"].append(c)
                continue
            ratio = n_unique / non_null

            if dt == pl.Boolean:
                out["boolean"].append(c)
            elif _is_datetime_dtype(dt):
                out["datetime"].append(c)
            elif _is_numeric_dtype(dt):
                if _is_integer_dtype(dt) and n >= 50 and ratio >= 0.98:
                    out["id_like"].append(c)
                else:
                    out["numeric"].append(c)
            elif _is_string_dtype(dt) or dt == pl.Categorical or str(dt).startswith("Enum"):
                avg_len = 0.0
                if _is_string_dtype(dt):
                    avg_len = float(nn.str.len_chars().mean() or 0.0)
                    parsed = nn.cast(pl.Float64, strict=False)
                    if (non_null - parsed.null_count()) / non_null >= 0.95:
                        out["numeric_like_text"].append(c)
                if avg_len > 50:
                    out["text"].append(c)
                elif n >= 50 and ratio >= 0.98:
                    out["id_like"].append(c)
                else:
                    out["categorical"].append(c)
            else:
                out["other"].append(c)
        except Exception as ex:
            logger.warning("Type inference failed for %s: %s", c, ex)
            out["other"].append(c)
    return out


# ----------------------------------------------------------------------------
# Numeric summary
# ----------------------------------------------------------------------------
def _normality(arr: np.ndarray, skew: float, kurt: float, cfg: StatsConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    n = arr.size
    if n < 8:
        return out
    # Jarque-Bera with moment skewness and excess kurtosis; chi2 with 2 dof has
    # survival function exp(-x / 2), so no SciPy is required.
    jb = n / 6.0 * (skew**2 + (kurt**2) / 4.0)
    out["jb_stat"] = _f(jb)
    out["jb_pvalue"] = _f(math.exp(-jb / 2.0), 6)
    if _sp_stats is not None and n >= 3:
        try:
            sample = arr
            if n > 5000:
                sample = np.random.default_rng(cfg.seed).choice(arr, 5000, replace=False)
            if np.ptp(sample) > 0:
                out["shapiro_pvalue"] = _f(_sp_stats.shapiro(sample).pvalue, 6)
        except Exception as ex:  # pragma: no cover
            logger.debug("Shapiro failed: %s", ex)
    return out


def _robust_block(arr: np.ndarray, mean_val: float, median: float, q1: float, q3: float,
                  cfg: StatsConfig) -> dict[str, float]:
    n = arr.size
    dev = np.abs(arr - median)
    mad = float(np.median(dev))

    s = np.sort(arr)
    k = int(math.floor(n * cfg.trim_fraction))
    if k == 0 and n > 2:
        k = 1
    trimmed = float(s[k:n - k].mean()) if n - 2 * k > 0 else mean_val

    iqr = q3 - q1
    lo, hi = q1 - cfg.iqr_multiplier * iqr, q3 + cfg.iqr_multiplier * iqr
    elo, ehi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    low_ct = int((arr < lo).sum())
    high_ct = int((arr > hi).sum())
    extreme = int(((arr < elo) | (arr > ehi)).sum())

    if mad > 0:
        mz = 0.6745 * (arr - median) / mad
    else:
        mean_ad = float(dev.mean())
        mz = (arr - median) / (1.253314 * mean_ad) if mean_ad > 0 else np.zeros(n)
    modz = int((np.abs(mz) > cfg.modified_z_threshold).sum())

    out = {
        "mad": mad,
        "mad_scaled": mad * _MAD_SCALE,
        "trimmed_mean": trimmed,
        "lower_fence": lo,
        "upper_fence": hi,
        "outlier_low_count": float(low_ct),
        "outlier_high_count": float(high_ct),
        "outlier_iqr_count": float(low_ct + high_ct),
        "outlier_iqr_pct": 100.0 * (low_ct + high_ct) / n,
        "outlier_extreme_count": float(extreme),
        "outlier_modz_count": float(modz),
    }
    if abs(median) > 1e-9:
        out["robust_cv"] = mad * _MAD_SCALE / abs(median)
    if float(s[0]) > 0:
        out["geometric_mean"] = float(np.exp(np.log(arr).mean()))
    return out


def compute_summary(
    df: pl.DataFrame,
    numeric_cols: list[str],
    config: StatsConfig | None = None,
) -> dict[str, dict[str, float]]:
    """
    Comprehensive summary for numeric columns. All values are floats, so the
    result stays JSON friendly. Keys from the previous version are preserved.
    Labels and flags live in describe_distribution().
    """
    cfg = config or StatsConfig()
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return {}

    qs = sorted({0.25, 0.75, *cfg.percentiles})
    aggs: list[pl.Expr] = []
    for i, c in enumerate(cols):
        e = _clean_expr(c)
        raw = pl.col(c).cast(pl.Float64, strict=False)
        k = f"c{i}"
        aggs += [
            e.count().alias(f"{k}_count"),
            e.mean().alias(f"{k}_mean"),
            e.std().alias(f"{k}_std"),
            e.var().alias(f"{k}_var"),
            e.min().alias(f"{k}_min"),
            e.max().alias(f"{k}_max"),
            e.sum().alias(f"{k}_sum"),
            e.median().alias(f"{k}_median"),
            e.skew().alias(f"{k}_skew"),
            e.kurtosis().alias(f"{k}_kurt"),
            e.drop_nulls().n_unique().alias(f"{k}_nuniq"),
            (e == 0).sum().alias(f"{k}_zeros"),
            (e < 0).sum().alias(f"{k}_neg"),
            raw.is_infinite().sum().alias(f"{k}_inf"),
            raw.is_nan().sum().alias(f"{k}_nan"),
        ]
        for qi, q in enumerate(qs):
            aggs.append(e.quantile(q, interpolation="linear").alias(f"{k}_q{qi}"))

    try:
        row = df.lazy().select(aggs).collect().row(0, named=True)
    except Exception as ex:
        logger.error("Summary aggregation failed: %s", ex)
        return {}

    cleaned = df.select([_clean_expr(c).alias(c) for c in cols])
    n_rows = df.height
    result: dict[str, dict[str, float]] = {}

    for i, c in enumerate(cols):
        k = f"c{i}"
        try:
            count = int(row[f"{k}_count"] or 0)
            mean_val = _f(row[f"{k}_mean"])
            std_val = _f(row[f"{k}_std"])
            min_val = _f(row[f"{k}_min"])
            max_val = _f(row[f"{k}_max"])
            median = _f(row[f"{k}_median"])
            quant = {q: _f(row[f"{k}_q{qi}"]) for qi, q in enumerate(qs)}
            q25, q75 = quant[0.25], quant[0.75]
            skew_val = _f(row[f"{k}_skew"])
            kurt_val = _f(row[f"{k}_kurt"])
            iqr = q75 - q25

            entry: dict[str, float] = {
                "count": float(count),
                "null_count": float(n_rows - count),
                "null_pct": _f(100.0 * (n_rows - count) / n_rows if n_rows else 0.0, 2),
                "inf_count": float(row[f"{k}_inf"] or 0),
                "nan_count": float(row[f"{k}_nan"] or 0),
                "n_unique": float(row[f"{k}_nuniq"] or 0),
                "unique_ratio": _f((row[f"{k}_nuniq"] or 0) / count if count else 0.0),
                "zeros": float(row[f"{k}_zeros"] or 0),
                "zero_pct": _f(100.0 * (row[f"{k}_zeros"] or 0) / count if count else 0.0, 2),
                "negatives": float(row[f"{k}_neg"] or 0),
                "negative_pct": _f(100.0 * (row[f"{k}_neg"] or 0) / count if count else 0.0, 2),
                "mean": mean_val,
                "std": std_val,
                "variance": _f(row[f"{k}_var"]),
                "min": min_val,
                "max": max_val,
                "range": _f(max_val - min_val),
                "sum": _f(row[f"{k}_sum"]),
                "50%": median,
                "median": median,
                "25%": q25,
                "75%": q75,
                "iqr": _f(iqr, 6),
                "skewness": skew_val,
                "kurtosis": kurt_val,
                "bowley_skewness": _f((q75 + q25 - 2.0 * median) / iqr) if iqr > _EPS else 0.0,
                "pearson_median_skewness": _f(3.0 * (mean_val - median) / std_val) if std_val > _EPS else 0.0,
                "coefficient_of_variation": _f(std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else 0.0,
            }
            for q in cfg.percentiles:
                entry[_pct_label(q)] = quant[q]

            if count >= 2:
                se = std_val / math.sqrt(count)
                tc = _t_crit(count - 1, cfg.alpha)
                entry["standard_error"] = _f(se, 6)
                entry["ci_low"] = _f(mean_val - tc * se)
                entry["ci_high"] = _f(mean_val + tc * se)

            if count > 0:
                arr = cleaned[c].drop_nulls().to_numpy()
                rb = _robust_block(arr, mean_val, median, q25, q75, cfg)
                for key, val in rb.items():
                    entry[key] = _f(val, 6 if key == "mad" else 4)
                entry["trimmed_mean_5pct"] = entry["trimmed_mean"]
                entry.update(_normality(arr, skew_val, kurt_val, cfg))

            result[c] = entry
        except Exception as ex:
            logger.warning("Summary failed for column %s: %s", c, ex)
    return result


def describe_distribution(stats: dict[str, dict[str, float]], config: StatsConfig | None = None) -> dict[str, dict[str, Any]]:
    """Human readable labels and quality flags derived from compute_summary()."""
    cfg = config or StatsConfig()
    out: dict[str, dict[str, Any]] = {}
    for col, s in stats.items():
        skew, kurt = s.get("skewness", 0.0), s.get("kurtosis", 0.0)
        if skew > cfg.skew_threshold:
            shape = "right_skewed"
        elif skew < -cfg.skew_threshold:
            shape = "left_skewed"
        else:
            shape = "approximately_symmetric"
        tails = "heavy_tailed" if kurt > cfg.kurtosis_threshold else (
            "light_tailed" if kurt < -cfg.kurtosis_threshold else "normal_tailed")

        mn = s.get("min", 0.0)
        if shape == "right_skewed":
            transform = "log" if mn > 0 else ("log1p" if mn >= 0 else "yeo_johnson")
        elif shape == "left_skewed":
            transform = "reflect_then_log" if s.get("max", 0.0) > 0 else "yeo_johnson"
        else:
            transform = "none"

        count = s.get("count", 0.0)
        flags: list[str] = []
        if s.get("max", 0.0) == s.get("min", 0.0):
            flags.append("constant")
        if s.get("null_pct", 0.0) >= cfg.high_missing_pct:
            flags.append("high_missing")
        if s.get("inf_count", 0.0) > 0:
            flags.append("has_infinite")
        if s.get("nan_count", 0.0) > 0:
            flags.append("has_nan")
        if s.get("outlier_extreme_count", 0.0) > 0:
            flags.append("extreme_outliers")
        if s.get("n_unique", 0.0) <= 10 and count > 0 and s.get("n_unique", 0.0) > 1:
            flags.append("low_cardinality_possible_categorical")
        if s.get("zero_pct", 0.0) >= 50.0:
            flags.append("zero_inflated")
        if "jb_pvalue" in s and s["jb_pvalue"] < cfg.alpha:
            flags.append("non_normal_jb")
        if 0 < count < 30:
            flags.append("small_sample")

        out[col] = {"shape": shape, "tails": tails, "suggested_transform": transform, "flags": flags}
    return out


# ----------------------------------------------------------------------------
# Correlation engine
# ----------------------------------------------------------------------------
def _prepare_matrix(df: pl.DataFrame, cols: list[str], cfg: StatsConfig) -> tuple[np.ndarray, bool]:
    sub = df.select([_clean_expr(c).alias(c) for c in cols])
    sampled = False
    if sub.height > cfg.max_rows:
        sub = sub.sample(n=cfg.max_rows, seed=cfg.seed)
        sampled = True
    if cfg.missing == "listwise":
        sub = sub.drop_nulls()
    X = np.asarray(sub.to_numpy(), dtype=float)
    return X, sampled


def _pearson_pairwise(X: np.ndarray, min_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised pairwise-complete Pearson. Returns (R, N, constant_mask).
    Columns are centred first, which keeps the sums-of-squares form stable.
    """
    M = np.isfinite(X)
    Mf = M.astype(float)
    cnt = M.sum(axis=0)
    Xz = np.where(M, X, 0.0)
    mu = Xz.sum(axis=0) / np.maximum(cnt, 1)
    Xc = np.where(M, X - mu, 0.0)

    n = Mf.T @ Mf
    Sx = Xc.T @ Mf
    Sxx = (Xc ** 2).T @ Mf
    Sxy = Xc.T @ Xc

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Sxy - Sx * Sx.T / n
        vx = Sxx - Sx ** 2 / n
        vy = vx.T
        R = cov / np.sqrt(vx * vy)

    mx = np.where(M, X, -np.inf).max(axis=0)
    mn = np.where(M, X, np.inf).min(axis=0)
    scale = np.maximum(1.0, np.abs(mu))
    const = (cnt < 2) | ((mx - mn) <= 1e-12 * scale)

    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.clip(R, -1.0, 1.0)
    R[n < min_periods] = 0.0
    R[const, :] = 0.0
    R[:, const] = 0.0
    np.fill_diagonal(R, 1.0)
    return R, n, const


def _rank_columns(X: np.ndarray) -> np.ndarray:
    """Column-wise average ranks over available values, NaN preserved."""
    out = np.full_like(X, np.nan, dtype=float)
    for j in range(X.shape[1]):
        m = np.isfinite(X[:, j])
        if m.any():
            out[m, j] = _rank_avg(X[m, j])
    return out


def _spearman_matrix(X: np.ndarray, N: np.ndarray, cfg: StatsConfig) -> tuple[np.ndarray, str]:
    p = X.shape[1]
    M = np.isfinite(X)
    if M.all():
        R, _, _ = _pearson_pairwise(_rank_columns(X), cfg.min_periods)
        return R, "exact"

    n_pairs = p * (p - 1) // 2
    if n_pairs <= cfg.exact_spearman_pair_limit:
        R = np.eye(p)
        for i in range(p):
            for j in range(i + 1, p):
                m = M[:, i] & M[:, j]
                if m.sum() < cfg.min_periods:
                    continue
                a, b = _rank_avg(X[m, i]), _rank_avg(X[m, j])
                if np.ptp(a) < _EPS or np.ptp(b) < _EPS:
                    continue
                r = float(np.corrcoef(a, b)[0, 1])
                R[i, j] = R[j, i] = 0.0 if not math.isfinite(r) else r
        return R, "exact_pairwise"

    R, _, _ = _pearson_pairwise(_rank_columns(X), cfg.min_periods)
    return R, "approximate_column_ranks"


def _collinear_groups(cols: list[str], R: np.ndarray, threshold: float) -> list[list[str]]:
    parent = list(range(len(cols)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(R[i, j]) >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[str]] = {}
    for i, c in enumerate(cols):
        groups.setdefault(find(i), []).append(c)
    return [g for g in groups.values() if len(g) > 1]


def compute_correlation_report(
    df: pl.DataFrame,
    numeric_cols: list[str],
    config: StatsConfig | None = None,
) -> dict[str, Any]:
    """Full correlation report. compute_correlation() is a thin wrapper."""
    cfg = config or StatsConfig()
    cols = [c for c in numeric_cols if c in df.columns]
    empty = {
        "columns": cols, "pearson": {}, "spearman": {}, "n_obs": {}, "strong_pairs": [],
        "vif": {}, "condition_number": None, "multicollinear_groups": [],
        "constant_columns": [], "spearman_mode": None, "n_rows_used": 0, "sampled": False,
    }
    if len(cols) < 2:
        return empty

    try:
        X, sampled = _prepare_matrix(df, cols, cfg)
        if X.shape[0] < 2:
            return empty

        R_p, N, const = _pearson_pairwise(X, cfg.min_periods)
        R_s, s_mode = _spearman_matrix(X, N, cfg)
        R_s[const, :] = 0.0
        R_s[:, const] = 0.0
        np.fill_diagonal(R_s, 1.0)

        p = len(cols)
        pearson: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        spearman: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        n_obs: dict[str, dict[str, int]] = {c: {c: int(N[i, i])} for i, c in enumerate(cols)}

        pairs_meta: list[tuple[int, int]] = []
        p_pear: list[float] = []
        p_spear: list[float] = []
        for i in range(p):
            for j in range(i + 1, p):
                a, b = cols[i], cols[j]
                r, rho, nij = float(R_p[i, j]), float(R_s[i, j]), int(N[i, j])
                pearson[a][b] = pearson[b][a] = round(r, 4)
                spearman[a][b] = spearman[b][a] = round(rho, 4)
                n_obs[a][b] = n_obs[b][a] = nij
                pairs_meta.append((i, j))
                p_pear.append(_corr_pvalue(r, nij))
                p_spear.append(_corr_pvalue(rho, nij, spearman=True))

        adj_pear = _bh_adjust(p_pear)
        adj_spear = _bh_adjust(p_spear)

        method = cfg.correlation_method
        strong_pairs: list[dict[str, Any]] = []
        for idx, (i, j) in enumerate(pairs_meta):
            r, rho, nij = float(R_p[i, j]), float(R_s[i, j]), int(N[i, j])
            by_p = abs(r) >= cfg.strong_corr
            by_s = abs(rho) >= cfg.strong_corr
            if method == "pearson":
                selected = by_p
            elif method == "spearman":
                selected = by_s
            else:
                selected = by_p or (by_s and cfg.include_spearman_only_pairs)
            if not selected:
                continue

            if by_p and by_s:
                detected = "both"
            elif by_p:
                detected = "pearson"
            else:
                detected = "spearman"

            gap = abs(rho) - abs(r)
            if by_s and gap >= cfg.nonlinear_gap:
                relationship = "monotonic_nonlinear"
            elif by_p and -gap >= cfg.nonlinear_gap:
                relationship = "possibly_outlier_driven"
            else:
                relationship = "linear"

            lo, hi = _fisher_ci(r, nij, cfg.alpha)
            slo, shi = _fisher_ci(rho, nij, cfg.alpha, spearman=True)
            strength_val = max(abs(r), abs(rho)) if method != "pearson" else abs(r)
            strong_pairs.append({
                "column_a": cols[i],
                "column_b": cols[j],
                "r_value": round(r, 4),
                "spearman_rho": round(rho, 4),
                "direction": "positive" if (r if by_p or method == "pearson" else rho) > 0 else "negative",
                "strength": "very strong" if strength_val >= 0.9 else "strong",
                "is_multicollinear": abs(r) >= cfg.multicollinear_threshold,
                "n": nij,
                "p_value": _f(p_pear[idx], 8),
                "p_value_adj": _f(adj_pear[idx], 8),
                "spearman_p_value": _f(p_spear[idx], 8),
                "spearman_p_value_adj": _f(adj_spear[idx], 8),
                "significant": bool(adj_pear[idx] < cfg.alpha or adj_spear[idx] < cfg.alpha),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "spearman_ci_low": round(slo, 4),
                "spearman_ci_high": round(shi, 4),
                "relationship": relationship,
                "detected_by": detected,
            })
        strong_pairs.sort(key=lambda d: max(abs(d["r_value"]), abs(d["spearman_rho"])), reverse=True)

        # Multicollinearity diagnostics on non-constant columns
        vif: dict[str, float] = {}
        cond: float | None = None
        valid = [i for i in range(p) if not const[i]]
        if len(valid) >= 2:
            sub = R_p[np.ix_(valid, valid)].copy()
            np.fill_diagonal(sub, 1.0)
            try:
                # Pairwise-complete matrices can be indefinite. Clip eigenvalues
                # and rescale to a unit diagonal before inverting.
                w, V = np.linalg.eigh(sub)
                fixed = (V * np.clip(w, 1e-4, None)) @ V.T
                d = np.sqrt(np.diag(fixed))
                fixed = fixed / np.outer(d, d)
                inv = np.linalg.inv(fixed)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(sub)
            diag = np.clip(np.diag(inv), 1.0, 1e4)
            vif = {cols[i]: _f(v, 3) for i, v in zip(valid, diag)}
            try:
                cond = _f(np.linalg.cond(sub), 3)
            except Exception:
                cond = None

        return {
            "columns": cols,
            "pearson": pearson,
            "spearman": spearman,
            "n_obs": n_obs,
            "strong_pairs": strong_pairs,
            "vif": vif,
            "condition_number": cond,
            "multicollinear_groups": _collinear_groups(cols, R_p, cfg.multicollinear_threshold),
            "constant_columns": [cols[i] for i in range(p) if const[i]],
            "spearman_mode": s_mode,
            "n_rows_used": int(X.shape[0]),
            "sampled": sampled,
        }
    except Exception as e:
        logger.error("Correlation report failed: %s", e)
        return empty


def compute_correlation(df: pl.DataFrame, numeric_cols: list[str], config: StatsConfig | None = None):
    """
    Backward compatible entry point: returns (corr_dict, strong_pairs).
    corr_dict is Pearson unless config.correlation_method == "spearman".
    """
    cfg = config or StatsConfig()
    rep = compute_correlation_report(df, numeric_cols, cfg)
    matrix = rep["spearman"] if cfg.correlation_method == "spearman" else rep["pearson"]
    return matrix, rep["strong_pairs"]


# ----------------------------------------------------------------------------
# Nonlinear association (normalised mutual information)
# ----------------------------------------------------------------------------
def _binned_nmi(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    bins = int(min(20, max(3, math.sqrt(n / 5.0))))

    def digitize(v: np.ndarray) -> np.ndarray:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        if edges.size < 3:
            return np.zeros(n, dtype=int)
        return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, edges.size - 2)

    bx, by = digitize(x), digitize(y)
    kx, ky = int(bx.max()) + 1, int(by.max()) + 1
    if kx < 2 or ky < 2:
        return 0.0
    table = np.bincount(bx * ky + by, minlength=kx * ky).reshape(kx, ky).astype(float)
    pxy = table / n
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    nz = pxy > 0
    outer = np.outer(px, py)
    mi = float((pxy[nz] * np.log(pxy[nz] / outer[nz])).sum())
    mi -= (kx - 1) * (ky - 1) / (2.0 * n)  # remove expected bias under independence
    hx = -float((px[px > 0] * np.log(px[px > 0])).sum())
    hy = -float((py[py > 0] * np.log(py[py > 0])).sum())
    denom = math.sqrt(hx * hy)
    return max(0.0, mi / denom) if denom > _EPS else 0.0


def compute_nonlinear_associations(
    df: pl.DataFrame,
    numeric_cols: list[str],
    config: StatsConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Pairs with high normalised mutual information. Pairs where NMI is high but
    Pearson and Spearman are both weak point to non-monotonic structure
    (U shapes, cycles, interactions) that correlation misses.
    """
    cfg = config or StatsConfig()
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return []
    sub = df.select([_clean_expr(c).alias(c) for c in cols])
    if sub.height > cfg.nonlinear_sample:
        sub = sub.sample(n=cfg.nonlinear_sample, seed=cfg.seed)
    X = np.asarray(sub.to_numpy(), dtype=float)

    out: list[dict[str, Any]] = []
    checked = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if checked >= cfg.max_pairs_nonlinear:
                break
            checked += 1
            m = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            n = int(m.sum())
            if n < 30:
                continue
            a, b = X[m, i], X[m, j]
            if np.ptp(a) < _EPS or np.ptp(b) < _EPS:
                continue
            nmi = _binned_nmi(a, b)
            if nmi < cfg.nmi_threshold:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            rho = float(np.corrcoef(_rank_avg(a), _rank_avg(b))[0, 1])
            hidden = max(abs(r), abs(rho)) < cfg.strong_corr
            out.append({
                "column_a": cols[i], "column_b": cols[j], "nmi": _f(nmi),
                "pearson": _f(r), "spearman": _f(rho), "n": n,
                "hidden_from_correlation": bool(hidden),
            })
    out.sort(key=lambda d: d["nmi"], reverse=True)
    return out


# ----------------------------------------------------------------------------
# Categorical, datetime and missingness profiling
# ----------------------------------------------------------------------------
def compute_categorical_summary(
    df: pl.DataFrame, cat_cols: list[str], config: StatsConfig | None = None
) -> dict[str, dict[str, Any]]:
    cfg = config or StatsConfig()
    n_rows = df.height
    out: dict[str, dict[str, Any]] = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        try:
            s = df[c]
            nn = s.drop_nulls()
            cnt = nn.len()
            info: dict[str, Any] = {
                "count": cnt,
                "null_count": n_rows - cnt,
                "null_pct": _f(100.0 * (n_rows - cnt) / n_rows if n_rows else 0.0, 2),
            }
            if cnt == 0:
                out[c] = info
                continue

            vc = nn.value_counts(sort=True)
            vals = vc[vc.columns[0]].to_list()
            counts = np.asarray(vc[vc.columns[1]].to_list(), dtype=float)
            probs = counts / cnt
            k = len(vals)
            entropy = float(-(probs * np.log2(probs)).sum())
            rare = int((probs * 100.0 < cfg.rare_level_pct).sum())

            info.update({
                "n_unique": k,
                "unique_ratio": _f(k / cnt),
                "mode": str(vals[0]),
                "mode_pct": _f(100.0 * probs[0], 2),
                "entropy_bits": _f(entropy),
                "normalized_entropy": _f(entropy / math.log2(k)) if k > 1 else 0.0,
                "gini_impurity": _f(1.0 - float((probs ** 2).sum())),
                "rare_level_count": rare,
                "is_high_cardinality": bool(k > cfg.cat_max_cardinality or k / cnt > 0.5),
                "top_values": [
                    {"value": str(v), "count": int(n_), "pct": _f(100.0 * n_ / cnt, 2)}
                    for v, n_ in zip(vals[: cfg.top_k_categories], counts[: cfg.top_k_categories])
                ],
            })

            if _is_string_dtype(s.dtype):
                lens = nn.str.len_chars()
                info["length_min"] = int(lens.min() or 0)
                info["length_mean"] = _f(lens.mean())
                info["length_max"] = int(lens.max() or 0)
                info["empty_string_count"] = int((nn == "").sum())
                info["whitespace_padded_count"] = int((nn != nn.str.strip_chars()).sum())
                normalized_unique = nn.str.to_lowercase().str.strip_chars().n_unique()
                info["case_or_space_duplicate_levels"] = int(k - normalized_unique)
                parsed = nn.cast(pl.Float64, strict=False)
                info["numeric_like_pct"] = _f(100.0 * (cnt - parsed.null_count()) / cnt, 2)
            out[c] = info
        except Exception as ex:
            logger.warning("Categorical summary failed for %s: %s", c, ex)
    return out


def compute_datetime_summary(df: pl.DataFrame, dt_cols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    n_rows = df.height
    for c in dt_cols:
        if c not in df.columns:
            continue
        try:
            s = df[c].drop_nulls()
            cnt = s.len()
            info: dict[str, Any] = {
                "count": cnt, "null_count": n_rows - cnt,
                "null_pct": _f(100.0 * (n_rows - cnt) / n_rows if n_rows else 0.0, 2),
            }
            if cnt == 0:
                out[c] = info
                continue
            sd = s.cast(pl.Datetime("us"))
            raw_us = sd.dt.epoch("us").to_numpy()
            info["min"], info["max"] = str(s.min()), str(s.max())
            info["span_days"] = _f((raw_us.max() - raw_us.min()) / 86_400_000_000.0, 3)
            info["n_unique"] = int(np.unique(raw_us).size)
            info["is_monotonic_increasing"] = bool(np.all(np.diff(raw_us) >= 0))

            uniq = np.unique(raw_us)
            if uniq.size >= 3:
                steps = np.diff(uniq) / 1e6  # seconds
                med = float(np.median(steps))
                info["median_step_seconds"] = _f(med, 3)
                info["regularity"] = _f(float((np.isclose(steps, med, rtol=0.05)).mean()))
                info["gap_count"] = int((steps > 3.0 * med).sum()) if med > 0 else 0
                labels = [(1, "secondly"), (60, "minutely"), (3600, "hourly"), (86400, "daily"),
                          (604800, "weekly")]
                freq = "irregular"
                for secs, name in labels:
                    if math.isclose(med, secs, rel_tol=0.05):
                        freq = name
                if 28 * 86400 <= med <= 31 * 86400:
                    freq = "monthly"
                elif 89 * 86400 <= med <= 92 * 86400:
                    freq = "quarterly"
                elif 364 * 86400 <= med <= 366 * 86400:
                    freq = "yearly"
                info["inferred_frequency"] = freq
            out[c] = info
        except Exception as ex:
            logger.warning("Datetime summary failed for %s: %s", c, ex)
    return out


def compute_missingness(
    df: pl.DataFrame, cols: list[str] | None = None, config: StatsConfig | None = None
) -> dict[str, Any]:
    """Per-column missingness, row completeness and co-missing column pairs."""
    cfg = config or StatsConfig()
    cols = [c for c in (cols or df.columns) if c in df.columns][:300]
    n = df.height
    if n == 0 or not cols:
        return {}

    exprs = []
    for c in cols:
        e = pl.col(c)
        miss = e.is_null()
        if _is_float_dtype(df.schema[c]):
            miss = miss | e.is_nan()
        exprs.append(miss.fill_null(True).alias(c))
    ind = df.select(exprs).to_numpy().astype(bool)

    per_col = ind.sum(axis=0)
    row_missing = ind.sum(axis=1)
    out: dict[str, Any] = {
        "columns": {c: {"missing": int(m), "missing_pct": _f(100.0 * m / n, 2)}
                    for c, m in zip(cols, per_col)},
        "complete_rows": int((row_missing == 0).sum()),
        "complete_row_pct": _f(100.0 * (row_missing == 0).mean(), 2),
        "rows_over_half_missing": int((row_missing > len(cols) / 2).sum()),
    }

    var_idx = [i for i in range(len(cols)) if 0 < per_col[i] < n]
    pairs: list[dict[str, Any]] = []
    if len(var_idx) >= 2:
        sub = ind[:, var_idx].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.nan_to_num(np.corrcoef(sub.T), nan=0.0)
        for a in range(len(var_idx)):
            for b in range(a + 1, len(var_idx)):
                if phi[a, b] >= 0.5:
                    pairs.append({
                        "column_a": cols[var_idx[a]], "column_b": cols[var_idx[b]],
                        "phi": _f(phi[a, b]),
                    })
        pairs.sort(key=lambda d: d["phi"], reverse=True)
    out["co_missing_pairs"] = pairs[:50]
    return out


# ----------------------------------------------------------------------------
# Mixed type associations
# ----------------------------------------------------------------------------
def _cat_codes(s: pl.Series) -> np.ndarray:
    """Integer codes with -1 for null."""
    phys = s.cast(pl.Utf8).cast(pl.Categorical).to_physical().fill_null(-1)
    return np.asarray(phys.to_numpy(), dtype=np.int64)


def _chi2_pvalue(chi2: float, dof: int) -> float:
    if dof <= 0:
        return 1.0
    if _sp_stats is not None:
        return float(_sp_stats.chi2.sf(chi2, dof))
    # Wilson-Hilferty approximation
    z = ((chi2 / dof) ** (1.0 / 3.0) - (1 - 2.0 / (9 * dof))) / math.sqrt(2.0 / (9 * dof))
    return float(1.0 - NormalDist().cdf(z))


def compute_mixed_associations(
    df: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    config: StatsConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Cramer's V (cat vs cat, bias corrected) and eta (cat vs numeric)."""
    cfg = config or StatsConfig()
    d = _sample(df, cfg)
    usable: list[str] = []
    for c in categorical_cols:
        if c in d.columns:
            nu = d[c].drop_nulls().n_unique()
            if 2 <= nu <= cfg.cat_max_cardinality:
                usable.append(c)

    codes = {c: _cat_codes(d[c]) for c in usable}
    cat_cat: list[dict[str, Any]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            m = (codes[a] >= 0) & (codes[b] >= 0)
            n = int(m.sum())
            if n < cfg.min_periods:
                continue
            _, ia = np.unique(codes[a][m], return_inverse=True)
            _, ib = np.unique(codes[b][m], return_inverse=True)
            ia, ib = ia.reshape(-1), ib.reshape(-1)
            r_, k_ = int(ia.max()) + 1, int(ib.max()) + 1
            if r_ < 2 or k_ < 2:
                continue
            table = np.bincount(ia * k_ + ib, minlength=r_ * k_).reshape(r_, k_).astype(float)
            expected = np.outer(table.sum(1), table.sum(0)) / n
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.nansum((table - expected) ** 2 / expected))
            phi2 = chi2 / n
            phi2c = max(0.0, phi2 - (k_ - 1) * (r_ - 1) / (n - 1))
            rc = r_ - (r_ - 1) ** 2 / (n - 1)
            kc = k_ - (k_ - 1) ** 2 / (n - 1)
            denom = min(kc - 1, rc - 1)
            v = math.sqrt(phi2c / denom) if denom > 0 else 0.0
            if v >= cfg.association_threshold:
                cat_cat.append({
                    "column_a": a, "column_b": b, "cramers_v": _f(v), "n": n,
                    "chi2_p_value": _f(_chi2_pvalue(chi2, (r_ - 1) * (k_ - 1)), 8),
                })

    cat_num: list[dict[str, Any]] = []
    nums = [c for c in numeric_cols if c in d.columns]
    num_arr = {c: d.select(_clean_expr(c)).to_series().to_numpy() for c in nums}
    for c in usable:
        for x in nums:
            v = num_arr[x]
            m = (codes[c] >= 0) & np.isfinite(v)
            n = int(m.sum())
            if n < cfg.min_periods:
                continue
            _, g = np.unique(codes[c][m], return_inverse=True)
            g = g.reshape(-1)
            ng = int(g.max()) + 1
            xv = v[m]
            grand = xv.mean()
            sst = float(((xv - grand) ** 2).sum())
            if ng < 2 or sst <= _EPS:
                continue
            counts = np.bincount(g, minlength=ng).astype(float)
            sums = np.bincount(g, weights=xv, minlength=ng)
            ssb = float((counts * (sums / counts - grand) ** 2).sum())
            eta2 = ssb / sst
            eta2_adj = 1.0 - (1.0 - eta2) * (n - 1) / (n - ng) if n > ng else eta2
            eta = math.sqrt(max(eta2, 0.0))
            if eta >= cfg.association_threshold:
                cat_num.append({
                    "categorical": c, "numeric": x, "eta": _f(eta),
                    "eta_squared": _f(eta2), "eta_squared_adj": _f(max(eta2_adj, 0.0)),
                    "n_groups": ng, "n": n,
                })

    cat_cat.sort(key=lambda r: r["cramers_v"], reverse=True)
    cat_num.sort(key=lambda r: r["eta"], reverse=True)
    return {"cat_cat": cat_cat[:50], "cat_num": cat_num[:50]}


# ----------------------------------------------------------------------------
# Group-wise summary
# ----------------------------------------------------------------------------
def compute_group_summary(
    df: pl.DataFrame,
    group_col: str,
    numeric_cols: list[str],
    config: StatsConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-group count, mean, median, std, min, max for the largest groups."""
    cfg = config or StatsConfig()
    cols = [c for c in numeric_cols if c in df.columns and c != group_col]
    if group_col not in df.columns or not cols:
        return {}
    aggs: list[pl.Expr] = [pl.len().alias("__n")]
    for i, c in enumerate(cols):
        e = _clean_expr(c)
        aggs += [
            e.mean().alias(f"{i}_mean"), e.median().alias(f"{i}_median"),
            e.std().alias(f"{i}_std"), e.min().alias(f"{i}_min"), e.max().alias(f"{i}_max"),
        ]
    g = (df.filter(pl.col(group_col).is_not_null())
           .group_by(group_col).agg(aggs).sort("__n", descending=True).head(cfg.max_groups))
    out: dict[str, dict[str, Any]] = {}
    for r in g.iter_rows(named=True):
        out[str(r[group_col])] = {"count": int(r["__n"]), **{
            c: {s: _f(r[f"{i}_{s}"]) for s in ("mean", "median", "std", "min", "max")}
            for i, c in enumerate(cols)
        }}
    return out


# ----------------------------------------------------------------------------
# Optional transforms (the only part of this module that produces new data)
# ----------------------------------------------------------------------------
_TRANSFORMS = ("none", "log", "log1p", "sqrt", "signed_log1p", "zscore", "robust",
               "minmax", "winsorize_iqr", "winsorize_pct")


def apply_transforms(
    df: pl.DataFrame,
    plan: dict[str, str],
    config: StatsConfig | None = None,
    inplace: bool = False,
) -> pl.DataFrame:
    """
    Apply a {column: transform_name} plan. New columns are named
    "<col>__<transform>" unless inplace=True. Negative values are shifted for
    log and sqrt so the transform never produces NaN, and a warning is logged.
    """
    cfg = config or StatsConfig()
    exprs: list[pl.Expr] = []
    for c, name in plan.items():
        if c not in df.columns or name == "none":
            continue
        if name not in _TRANSFORMS:
            raise ValueError(f"Unknown transform {name!r}. Choose from {_TRANSFORMS}")
        x = _clean_expr(c)
        stats = df.select(
            x.min().alias("mn"), x.mean().alias("mu"), x.std().alias("sd"),
            x.median().alias("md"), x.quantile(0.25, interpolation="linear").alias("q1"),
            x.quantile(0.75, interpolation="linear").alias("q3"), x.max().alias("mx"),
            x.quantile(0.01, interpolation="linear").alias("p1"),
            x.quantile(0.99, interpolation="linear").alias("p99"),
        ).row(0, named=True)
        mn = stats["mn"] if stats["mn"] is not None else 0.0

        if name == "log":
            shift = 0.0 if mn > 0 else (1e-9 - mn)
            e = (x + shift).log()
            if shift:
                logger.warning("log transform on %s shifted by %g", c, shift)
        elif name == "log1p":
            shift = 0.0 if mn >= 0 else -mn
            e = (x + shift).log1p()
            if shift:
                logger.warning("log1p transform on %s shifted by %g", c, shift)
        elif name == "sqrt":
            shift = 0.0 if mn >= 0 else -mn
            e = (x + shift).sqrt()
        elif name == "signed_log1p":
            e = x.sign() * x.abs().log1p()
        elif name == "zscore":
            sd = stats["sd"] or 0.0
            e = (x - (stats["mu"] or 0.0)) / sd if sd > _EPS else x * 0.0
        elif name == "robust":
            iqr = (stats["q3"] or 0.0) - (stats["q1"] or 0.0)
            e = (x - (stats["md"] or 0.0)) / iqr if iqr > _EPS else x * 0.0
        elif name == "minmax":
            rng = (stats["mx"] or 0.0) - mn
            e = (x - mn) / rng if rng > _EPS else x * 0.0
        elif name == "winsorize_iqr":
            iqr = (stats["q3"] or 0.0) - (stats["q1"] or 0.0)
            e = x.clip((stats["q1"] or 0.0) - cfg.iqr_multiplier * iqr,
                       (stats["q3"] or 0.0) + cfg.iqr_multiplier * iqr)
        else:  # winsorize_pct
            e = x.clip(stats["p1"], stats["p99"])

        exprs.append(e.alias(c if inplace else f"{c}__{name}"))
    return df.with_columns(exprs) if exprs else df


# ----------------------------------------------------------------------------
# One call orchestrator
# ----------------------------------------------------------------------------
ALL_SECTIONS = ("types", "summary", "distribution", "correlation", "nonlinear",
                "categorical", "datetime", "missingness", "associations", "target", "transforms")


def _build_warnings(report: dict[str, Any], cfg: StatsConfig) -> list[str]:
    w: list[str] = []
    types = report.get("column_types", {})
    for key, msg in (("constant", "constant column"), ("empty", "entirely empty column"),
                     ("id_like", "looks like an identifier, exclude from modelling"),
                     ("numeric_like_text", "text column that parses as numbers, consider casting")):
        for c in types.get(key, []):
            w.append(f"{c}: {msg}")
    for c, d in report.get("distribution", {}).items():
        for f in d["flags"]:
            if f in ("high_missing", "has_infinite", "extreme_outliers", "small_sample", "zero_inflated"):
                w.append(f"{c}: {f.replace('_', ' ')}")
        if d["shape"] != "approximately_symmetric":
            w.append(f"{c}: {d['shape'].replace('_', ' ')}, try {d['suggested_transform']}")
    corr = report.get("correlation", {})
    for g in corr.get("multicollinear_groups", []):
        w.append(f"multicollinear group: {', '.join(g)}")
    for c, v in corr.get("vif", {}).items():
        if v >= cfg.vif_threshold:
            w.append(f"{c}: VIF {v} exceeds {cfg.vif_threshold}")
    for p in report.get("nonlinear", []):
        if p["hidden_from_correlation"]:
            w.append(f"{p['column_a']} and {p['column_b']}: nonlinear dependence (NMI {p['nmi']}) hidden from correlation")
    for c, m in report.get("missingness", {}).get("columns", {}).items():
        if m["missing_pct"] >= cfg.high_missing_pct:
            w.append(f"{c}: {m['missing_pct']}% missing")
    return list(dict.fromkeys(w))


def analyze(
    df: pl.DataFrame,
    config: StatsConfig | None = None,
    *,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    datetime_cols: list[str] | None = None,
    target: str | None = None,
    group_by: str | None = None,
    sections: Iterable[str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Run any combination of sections. Column lists are inferred when omitted.
    Config fields can be overridden inline: analyze(df, strong_corr=0.6).
    """
    cfg = replace(config or StatsConfig(), **overrides) if overrides else (config or StatsConfig())
    wanted = set(sections) if sections else set(ALL_SECTIONS)
    if not cfg.include_nonlinear:
        wanted.discard("nonlinear")
    report: dict[str, Any] = {"shape": {"rows": df.height, "columns": df.width},
                              "errors": {}, "timings_s": {}}

    def run(name: str, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as ex:
            logger.exception("Section %s failed", name)
            report["errors"][name] = str(ex)
            return None
        finally:
            report["timings_s"][name] = round(time.perf_counter() - t0, 4)

    types = run("types", lambda: infer_column_types(df, cfg)) or {}
    report["column_types"] = types
    num = numeric_cols if numeric_cols is not None else types.get("numeric", [])
    cat = categorical_cols if categorical_cols is not None else (
        types.get("categorical", []) + types.get("boolean", []))
    dts = datetime_cols if datetime_cols is not None else types.get("datetime", [])

    if "summary" in wanted:
        report["summary"] = run("summary", lambda: compute_summary(df, num, cfg)) or {}
        if "distribution" in wanted:
            report["distribution"] = describe_distribution(report["summary"], cfg)
            report["recommended_transforms"] = {
                c: d["suggested_transform"] for c, d in report["distribution"].items()
                if d["suggested_transform"] != "none"}
    if "correlation" in wanted:
        report["correlation"] = run("correlation", lambda: compute_correlation_report(df, num, cfg)) or {}
    if "nonlinear" in wanted:
        report["nonlinear"] = run("nonlinear", lambda: compute_nonlinear_associations(df, num, cfg)) or []
    if "categorical" in wanted and cat:
        report["categorical"] = run("categorical", lambda: compute_categorical_summary(df, cat, cfg)) or {}
    if "datetime" in wanted and dts:
        report["datetime"] = run("datetime", lambda: compute_datetime_summary(df, dts)) or {}
    if "missingness" in wanted:
        report["missingness"] = run("missingness", lambda: compute_missingness(df, None, cfg)) or {}
    if "associations" in wanted and cat:
        report["associations"] = run("associations", lambda: compute_mixed_associations(df, num, cat, cfg)) or {}
    if group_by:
        report["group_summary"] = run("group_summary", lambda: compute_group_summary(df, group_by, num, cfg)) or {}

    if "target" in wanted and target and target in report.get("correlation", {}).get("columns", []):
        corr = report["correlation"]
        rows = [{"feature": c, "pearson": corr["pearson"][target].get(c, 0.0),
                 "spearman": corr["spearman"][target].get(c, 0.0),
                 "n": corr["n_obs"][target].get(c, 0)}
                for c in corr["columns"] if c != target]
        rows.sort(key=lambda r: max(abs(r["pearson"]), abs(r["spearman"])), reverse=True)
        report["target_relationships"] = {"target": target, "features": rows}

    report["warnings"] = _build_warnings(report, cfg)
    return report


__all__ = [
    "StatsConfig", "analyze", "infer_column_types", "compute_summary", "describe_distribution",
    "compute_correlation", "compute_correlation_report", "compute_nonlinear_associations",
    "compute_categorical_summary", "compute_datetime_summary", "compute_missingness",
    "compute_mixed_associations", "compute_group_summary", "apply_transforms",
]