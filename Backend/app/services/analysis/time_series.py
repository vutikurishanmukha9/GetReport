""" Time Series Intelligence Engine """
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

__all__ = [
    "TimeSeriesConfig", "TimeSeriesData", "detect_time_columns", "prepare_series",
    "mann_kendall_trend_test", "detect_trend", "detect_seasonality", "analyze_time_series",
]


# ───────────────────────────── configuration ─────────────────────────────
@dataclass
class TimeSeriesConfig:
    max_columns: int = 10
    min_points: int = 10
    alpha: float = 0.05
    agg: str = "auto"                     # auto | sum | mean | median | min | max | first | last
    max_missing_pct: float = 40.0         # beyond this the series is flagged "sparse"
    parse_string_dates: bool = True
    min_seasonal_strength: float = 0.25
    seasonal_p_threshold: float = 0.01
    max_changepoints: int = 5
    changepoints: bool = True
    anomalies: bool = True
    forecast: bool = True
    forecast_horizon: int | None = None
    relationships: bool = True
    group_by: list[str] | None = None
    max_groups: int = 10
    min_group_points: int = 10
    kats: bool = True
    mk_max_points: int = 5000
    seed: int = 42


# ───────────────────────────── numeric helpers ─────────────────────────────
def _f(x: Any, nd: int = 4) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_p(t: float, df: float) -> float:
    """Two-sided p-value of a t statistic (exact, no SciPy needed)."""
    if not math.isfinite(t):
        return 0.0
    if df <= 0:
        return 1.0
    return float(_betai(df / 2.0, 0.5, df / (df + t * t)))


def _f_sf(F: float, d1: float, d2: float) -> float:
    if F <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    return float(_betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * F)))


def _chi2_sf(x: float, k: int) -> float:
    try:
        from scipy import stats
        return float(stats.chi2.sf(x, k))
    except Exception:  # Wilson-Hilferty approximation
        if k <= 0:
            return 1.0
        z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return 0.5 * math.erfc(z / math.sqrt(2))


def _normal_two_sided_p_value(z_score: float) -> float:
    return math.erfc(abs(z_score) / math.sqrt(2))


def _acf(y: np.ndarray, nlags: int) -> np.ndarray:
    n = y.size
    x = y - y.mean()
    f = np.fft.rfft(x, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    return a / a[0] if a[0] > 0 else np.zeros(nlags + 1)


def _mad_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def _centered_ma(y: np.ndarray, w: int) -> np.ndarray:
    n = y.size
    pad = min(w, n - 1)
    yp = np.pad(y, pad, mode="reflect")
    k = np.r_[0.5, np.ones(w - 1), 0.5] / w if w % 2 == 0 else np.ones(w) / w
    return np.convolve(yp, k, mode="same")[pad:pad + n]


def _decompose(y: np.ndarray, m: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust classical decomposition (MA trend + phase-median seasonality, winsorised iterations)."""
    n = y.size
    if not (m and m >= 2 and n >= 2 * m):
        w = max(5, n // 10)
        w += (w % 2 == 0)
        pad = min(w // 2, n - 1)
        win = np.lib.stride_tricks.sliding_window_view(np.pad(y, pad, mode="reflect"), 2 * pad + 1)
        trend = np.median(win, axis=1)[:n]
        return trend, np.zeros(n), y - trend
    yw = y.copy()
    trend = seas = resid = np.zeros(n)
    for _ in range(3):
        trend = _centered_ma(yw, m)
        det = yw - trend
        pat = np.array([np.median(det[i::m]) for i in range(m)])
        pat -= pat.mean()
        seas = np.resize(pat, n)
        resid = y - trend - seas
        sig = _mad_sigma(resid)
        if sig == 0:
            break
        yw = np.where(np.abs(resid - np.median(resid)) > 3 * sig,
                      trend + seas + np.sign(resid) * 3 * sig, y)
    return trend, seas, resid


# ───────────────────────────── time axis & series preparation ─────────────────────────────
def _is_numeric(dt: Any) -> bool:
    try:
        return bool(dt.is_numeric())
    except AttributeError:
        return dt in getattr(pl, "NUMERIC_DTYPES", set())


def _is_temporal(dt: Any) -> bool:
    return dt == pl.Date or isinstance(dt, pl.Datetime) or dt == pl.Datetime


def detect_time_columns(df: pl.DataFrame, include_parsable_strings: bool = False) -> list[str]:
    """Identify datetime columns (and optionally date-like string columns)."""
    cols = [c for c, t in df.schema.items() if _is_temporal(t)]
    if include_parsable_strings:
        for c, t in df.schema.items():
            if t == pl.Utf8 and c not in cols and _parse_ok(df[c]):
                cols.append(c)
    return cols


def _parse_ok(s: pl.Series) -> bool:
    nn = s.drop_nulls()
    if nn.len() < 5:
        return False
    try:
        sample = nn.head(200)
        parsed = sample.str.to_datetime(strict=False)
        return parsed.drop_nulls().len() / sample.len() >= 0.9
    except Exception:
        return False


def _time_to_us(s: pl.Series) -> np.ndarray | None:
    """Series -> float64 microseconds since epoch (NaN where null/unparsable)."""
    try:
        if s.dtype == pl.Utf8:
            s = s.str.to_datetime(strict=False)
        if s.dtype == pl.Date:
            s = s.cast(pl.Datetime("us"))
        if isinstance(s.dtype, pl.Datetime) or s.dtype == pl.Datetime:
            s = s.cast(pl.Datetime("us"))
            return np.asarray(s.dt.epoch("us").cast(pl.Float64).fill_null(float("nan")).to_numpy(), dtype=np.float64)
    except Exception as e:
        logger.debug("time conversion failed: %s", e)
    return None


_DAY = 86_400_000_000
_NAME_SUM = {"revenue", "sales", "count", "quantity", "qty", "amount", "orders", "clicks", "visits",
             "views", "units", "total", "transactions", "volume"}


def _name_tokens(col: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col).lower()
    toks = {t for t in re.split(r"[^a-z0-9]+", spaced) if t}
    return toks | {t[:-1] for t in toks if t.endswith("s") and len(t) > 3}


@dataclass
class _Grid:
    freq: str
    cal: str | None            # "M", "Q", "Y" or None
    step_us: float
    origin: int
    ord0: int
    m: int
    ts: np.ndarray             # datetime64[us], length m
    irregular_sorted: np.ndarray | None = None


def _label_freq(s: float) -> tuple[str, str | None]:
    d = 86400.0
    if 27 * d <= s <= 32 * d:
        return "monthly", "M"
    if 88 * d <= s <= 93 * d:
        return "quarterly", "Q"
    if 360 * d <= s <= 370 * d:
        return "yearly", "Y"
    if 6.5 * d <= s <= 7.5 * d:
        return "weekly", None
    if 0.95 * d <= s <= 1.05 * d:
        return "daily", None
    if 3420 <= s <= 3780:
        return "hourly", None
    if 55 <= s <= 65:
        return "minutely", None
    if 0.9 <= s <= 1.1:
        return "secondly", None
    return "custom", None


def _ordinals(t: np.ndarray, g: _Grid) -> np.ndarray:
    ti = t.astype(np.int64)
    if g.freq == "irregular":
        return np.searchsorted(g.irregular_sorted, ti)
    if g.cal:
        mo = ti.astype("datetime64[us]").astype("datetime64[M]").astype(np.int64)
        if g.cal == "M":
            return mo
        if g.cal == "Q":
            return mo // 3
        return ti.astype("datetime64[us]").astype("datetime64[Y]").astype(np.int64)
    return np.floor((ti - g.origin) / g.step_us).astype(np.int64)


def _grid_ts(g: _Grid, ords: np.ndarray) -> np.ndarray:
    if g.freq == "irregular":
        return g.irregular_sorted[ords].astype("datetime64[us]")
    if g.cal == "M":
        return ords.astype("datetime64[M]").astype("datetime64[us]")
    if g.cal == "Q":
        return (ords * 3).astype("datetime64[M]").astype("datetime64[us]")
    if g.cal == "Y":
        return ords.astype("datetime64[Y]").astype("datetime64[us]")
    return (g.origin + ords * g.step_us).astype(np.int64).astype("datetime64[us]")


def _build_grid(t_us: np.ndarray) -> _Grid | None:
    ok = np.isfinite(t_us)
    if ok.sum() < 3:
        return None
    u = np.unique(t_us[ok].astype(np.int64))
    if u.size < 3:
        return None
    med = float(np.median(np.diff(u)))
    freq, cal = _label_freq(med / 1e6)
    step_us = med if freq in ("custom",) else {
        "weekly": 7 * _DAY, "daily": _DAY, "hourly": 3_600_000_000,
        "minutely": 60_000_000, "secondly": 1_000_000}.get(freq, med)
    origin = int(u[0]) if freq in ("weekly", "custom") else 0
    g = _Grid(freq, cal, float(step_us), origin, 0, 0, np.array([], dtype="datetime64[us]"))
    ords = _ordinals(u.astype(np.float64), g)
    m = int(ords.max() - ords.min() + 1)
    if freq == "custom" and step_us <= 0:
        m = 10 ** 9
    close = float(np.mean(np.abs(np.diff(u) - med) <= 0.1 * med))
    if (np.unique(ords).size / m < 0.5 or m > max(20 * u.size, 1000) or m > 500_000
            or (freq == "custom" and close < 0.6)):
        g = _Grid("irregular", None, med, 0, 0, u.size, u.astype("datetime64[us]"), irregular_sorted=u)
        g.ord0, g.m = 0, u.size
        g.ts = _grid_ts(g, np.arange(u.size))
        return g
    g.ord0, g.m = int(ords.min()), m
    g.ts = _grid_ts(g, np.arange(g.ord0, g.ord0 + m))
    return g


def _period_hints(g: _Grid) -> list[int]:
    f = g.freq
    s = g.step_us / 1e6
    if f == "monthly":
        return [12]
    if f == "quarterly":
        return [4]
    if f == "weekly":
        return [52]
    if f == "daily":
        return [7, 365]
    if f == "hourly":
        return [24, 168]
    if f == "minutely":
        return [60]
    if f in ("custom", "secondly") and 0 < s < 86400:
        day = 86400 / s
        if abs(day - round(day)) / day < 0.015 and 2 <= round(day) <= 1440:
            hints = [int(round(day))]
            if hints[0] * 7 <= 2500:
                hints.append(hints[0] * 7)
            return hints
    return []


@dataclass
class TimeSeriesData:
    name: str
    ts: np.ndarray               # datetime64[us] (trimmed grid)
    y: np.ndarray                # gap-filled
    y_obs: np.ndarray            # NaN where unobserved
    obs: np.ndarray
    freq: str
    cal: str | None
    step_us: float
    step_days: float
    period_hints: list[int]
    agg: str
    quality: dict
    y_full: np.ndarray = field(default_factory=lambda: np.array([]))   # untrimmed, NaN-gapped (for relationships)


def _auto_agg(col: str, cfg: TimeSeriesConfig) -> str:
    if cfg.agg != "auto":
        return cfg.agg
    return "sum" if _name_tokens(col) & _NAME_SUM else "mean"


def _aggregate(idx: np.ndarray, y: np.ndarray, m: int, how: str) -> np.ndarray:
    out = np.full(m, np.nan)
    if idx.size == 0:
        return out
    cnt = np.bincount(idx, minlength=m)
    has = cnt > 0
    if how in ("sum", "mean"):
        s = np.bincount(idx, weights=y, minlength=m)
        out[has] = s[has] / (cnt[has] if how == "mean" else 1)
    elif how in ("first", "last"):
        src = idx if how == "first" else idx[::-1]
        vals = y if how == "first" else y[::-1]
        u, first = np.unique(src, return_index=True)
        out[u] = vals[first]
    elif how in ("min", "max"):
        order = np.argsort(idx, kind="stable")
        si, sv = idx[order], y[order]
        starts = np.r_[0, np.flatnonzero(np.diff(si)) + 1]
        red = (np.minimum if how == "min" else np.maximum).reduceat(sv, starts)
        out[si[starts]] = red
    else:  # median
        order = np.argsort(idx, kind="stable")
        si, sv = idx[order], y[order]
        starts = np.r_[0, np.flatnonzero(np.diff(si)) + 1, si.size]
        for a, b in zip(starts[:-1], starts[1:]):
            out[si[a]] = np.median(sv[a:b])
    return out


def _ts_str(t: Any, freq: str) -> str:
    unit = "D" if freq in ("daily", "weekly", "monthly", "quarterly", "yearly") else "s"
    return str(np.datetime_as_string(np.datetime64(t, "us"), unit=unit))


def _build_series(name: str, g: _Grid, t_us: np.ndarray, y_rows: np.ndarray, cfg: TimeSeriesConfig,
                  rows: np.ndarray | None = None) -> tuple[TimeSeriesData | None, str | None]:
    if rows is not None:
        t_us, y_rows = t_us[rows], y_rows[rows]
    valid = np.isfinite(t_us) & np.isfinite(y_rows)
    n_valid = int(valid.sum())
    if n_valid < cfg.min_points:
        return None, "Insufficient data points"
    idx = _ordinals(t_us[valid], g) - g.ord0
    keep = (idx >= 0) & (idx < g.m)
    idx, yv = idx[keep], y_rows[valid][keep]
    how = _auto_agg(name, cfg)
    y_obs = _aggregate(idx, yv, g.m, how)
    obs = np.isfinite(y_obs)
    if obs.sum() < cfg.min_points:
        return None, "Insufficient data points"
    first = int(np.argmax(obs))
    last = int(g.m - 1 - np.argmax(obs[::-1]))
    sl = slice(first, last + 1)
    y_obs_t, obs_t = y_obs[sl], obs[sl]
    pos = np.arange(y_obs_t.size)
    y_fill = np.interp(pos, pos[obs_t], y_obs_t[obs_t])
    dup_buckets = int((np.bincount(idx, minlength=g.m) > 1).sum())
    ts = g.ts[sl]
    step_days = g.step_us / _DAY if g.freq != "monthly" else 30.4375
    if g.freq == "quarterly":
        step_days = 91.31
    elif g.freq == "yearly":
        step_days = 365.25
    q = {
        "rows_total": int(t_us.size), "rows_used": int(idx.size),
        "duplicate_timestamp_buckets": dup_buckets, "aggregation": how if dup_buckets else None,
        "gaps_filled": int((~obs_t).sum()),
        "missing_pct": round(float((~obs_t).mean()) * 100, 2),
        "regular": g.freq != "irregular", "frequency": g.freq,
        "start": _ts_str(ts[0], g.freq), "end": _ts_str(ts[-1], g.freq),
        "span_days": _f((ts[-1] - ts[0]) / np.timedelta64(1, "D"), 2),
        "sparse": bool((~obs_t).mean() * 100 > cfg.max_missing_pct),
    }
    sr = TimeSeriesData(name, ts, y_fill, y_obs_t, obs_t, g.freq, g.cal, g.step_us, step_days,
                        _period_hints(g), how, q, y_full=y_obs)
    return sr, None


def _numeric_array(s: pl.Series) -> np.ndarray | None:
    if s.dtype == pl.Boolean or not _is_numeric(s.dtype):
        return None
    a = np.array(s.cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy(), dtype=np.float64, copy=True)
    a[~np.isfinite(a)] = np.nan
    return a


def prepare_series(df: pl.DataFrame, time_col: str, value_col: str,
                   config: TimeSeriesConfig | None = None) -> tuple[TimeSeriesData | None, str | None]:
    """Public helper: DataFrame columns -> regularised ``TimeSeriesData`` (or (None, reason))."""
    cfg = config or TimeSeriesConfig()
    t = _time_to_us(df[time_col])
    y = _numeric_array(df[value_col])
    if t is None:
        return None, f"'{time_col}' is not a time column"
    if y is None:
        return None, f"'{value_col}' is not numeric"
    g = _build_grid(t)
    if g is None:
        return None, "Not enough distinct timestamps"
    return _build_series(value_col, g, t, y, cfg)


# ───────────────────────────── Mann-Kendall & trend ─────────────────────────────
def _mk_s(y: np.ndarray) -> int:
    n = y.size
    s = 0
    cols = np.arange(n)
    for a in range(0, n, 400):
        b = min(n, a + 400)
        d = np.sign(y[None, :] - y[a:b, None])
        s += int((d * (cols[None, :] > np.arange(a, b)[:, None])).sum())
    return s


def _sen_slope(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, float]:
    n = y.size
    if n < 2:
        return 0.0, float(y.mean()) if n else 0.0
    if n <= 1500:
        i, j = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, 1_500_000)
        j = rng.integers(0, n, 1_500_000)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        i, j = lo, hi
    dx = x[j] - x[i]
    ok = dx > 0
    slope = float(np.median((y[j][ok] - y[i][ok]) / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


def mann_kendall_trend_test(series: np.ndarray, x: np.ndarray | None = None, alpha: float = 0.05) -> dict[str, Any]:
    """
    Mann-Kendall trend test with tie-corrected variance, Hamed-Rao autocorrelation correction
    (only ever makes the test more conservative) and Theil-Sen slope.
    Legacy keys preserved: s_stat, kendall_tau, z_score, p_value, statistically_significant, direction.
    """
    y = np.asarray(series, dtype=np.float64)
    y = y[np.isfinite(y)] if x is None else y
    n0 = y.size
    if n0 < 4:
        return {"trend": "insufficient_data", "tau": 0.0, "p_value": 1.0, "direction": "flat_or_no_trend",
                "statistically_significant": False}
    xs = np.arange(n0, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    sampled = False
    if n0 > 5000:  # evenly thinned to bound the quadratic work
        keep = np.linspace(0, n0 - 1, 5000).astype(int)
        y, xs, sampled = y[keep], xs[keep], True
    n = y.size

    s = _mk_s(y)
    _, counts = np.unique(y, return_counts=True)
    tie = float((counts * (counts - 1) * (2 * counts + 5)).sum())
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    pairs = n * (n - 1) / 2.0
    tau = s / pairs

    def zval(v: float) -> float:
        if v <= 0:
            return 0.0
        return (s - np.sign(s)) / math.sqrt(v) if s != 0 else 0.0

    z = zval(var_s)
    p = _normal_two_sided_p_value(z)

    slope, intercept = _sen_slope(xs, y)
    # Hamed-Rao modified variance
    factor = 1.0
    try:
        ranks = np.argsort(np.argsort(y - slope * xs)).astype(float)
        lags = min(n - 1, max(1, n // 3))
        rho = _acf(ranks, lags)[1:]
        k = np.arange(1, lags + 1)
        zc = 1.959964
        lo = (-1 - zc * np.sqrt(np.maximum(n - k - 1, 0))) / (n - k)
        hi = (-1 + zc * np.sqrt(np.maximum(n - k - 1, 0))) / (n - k)
        sig = (rho < lo) | (rho > hi)
        if sig.any():
            w = (n - k) * (n - k - 1) * (n - k - 2)
            nns = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * float((w[sig] * rho[sig]).sum())
            factor = max(1.0, nns)
    except Exception:
        pass
    z_mod = zval(var_s * factor)
    p_mod = _normal_two_sided_p_value(z_mod)
    sig_final = p_mod < alpha
    direction = ("upward" if tau > 0 else "downward") if sig_final else "flat_or_no_trend"
    return {
        "s_stat": int(s),
        "kendall_tau": round(float(tau), 4),
        "z_score": round(float(z), 4),
        "p_value": round(float(p), 4),                   # classical (tie-corrected)
        "p_value_modified": round(float(p_mod), 4),      # autocorrelation-corrected
        "statistically_significant": bool(sig_final),
        "direction": direction,
        "sen_slope": float(slope),
        "autocorrelation_inflation": round(float(factor), 3),
        "sampled": sampled,
    }


def _linear_trend_p_value(slope: float, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Two-sided p-value for the slope (exact t-distribution; legacy signature)."""
    n = len(x)
    if n < 3:
        return None
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0:
        return None
    rv = float(((y - y_pred) ** 2).sum()) / (n - 2)
    if rv <= 0:
        return 0.0
    return _t_p(slope / math.sqrt(rv / sxx), n - 2)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = x.size
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    pred = slope * x + intercept
    res = y - pred
    sse = float((res ** 2).sum())
    sst = float(((y - ym) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    dof = max(n - 2, 1)
    se = math.sqrt(sse / dof / sxx) if sse > 0 else 0.0
    t = slope / se if se > 0 else math.inf
    p = _t_p(t, dof)
    rho = float((res[:-1] * res[1:]).sum() / sse) if sse > 0 else 0.0
    dw = float((np.diff(res) ** 2).sum() / sse) if sse > 0 else 2.0
    n_eff = n * (1 - rho) / (1 + rho) if rho > 0 else n
    n_eff = max(n_eff, 3.0)
    p_adj = _t_p(t * math.sqrt(n_eff / n), n_eff - 2) if se > 0 else 0.0
    return dict(slope=slope, intercept=intercept, r2=r2, p=p, p_adj=p_adj, dw=dw, rho=rho,
                resid_std=float(res.std()))


def _curvature(x: np.ndarray, y: np.ndarray, slope: float) -> dict[str, Any]:
    n = x.size
    if n < 12:
        return {"shape": "unknown"}
    u = (x - x.mean()) / (x.std() or 1.0)
    X = np.column_stack([np.ones(n), u, u ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float((res ** 2).sum()) / (n - 3)
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {"shape": "unknown"}
    se = math.sqrt(max(cov[2, 2], 0.0))
    p = _t_p(beta[2] / se, n - 3) if se > 0 else 0.0
    if p >= 0.01:
        shape = "linear"
    elif slope >= 0:
        shape = "accelerating" if beta[2] > 0 else "decelerating"
    else:
        shape = "accelerating_decline" if beta[2] < 0 else "decelerating_decline"
    return {"shape": shape, "quadratic_p_value": round(p, 4)}


def _trend_from_series(sr: TimeSeriesData, cfg: TimeSeriesConfig) -> dict[str, Any]:
    x = np.flatnonzero(sr.obs).astype(np.float64)
    y = sr.y_obs[sr.obs]
    n = y.size
    if n < 10:
        return {"detected": False, "reason": "Insufficient data points"}
    if float(np.ptp(y)) == 0:
        return {"detected": False, "reason": "Constant values"}
    lin = _linear_fit(x, y)
    mk = mann_kendall_trend_test(y, x=x, alpha=cfg.alpha)
    slope = lin["slope"]
    span = float(x[-1] - x[0]) or 1.0
    scale = float(np.abs(y).mean()) or 1.0
    if abs(slope) * span < 1e-9 * scale:
        direction = "flat"
    else:
        direction = "upward" if slope > 0 else "downward"
    r2 = lin["r2"]
    strength = "strong" if r2 >= 0.7 else "moderate" if r2 >= 0.3 else "weak"
    sig_lin = lin["p_adj"] < cfg.alpha
    sig_mk = mk["statistically_significant"]
    confirmation = "confirmed" if (sig_lin and sig_mk) else "tentative" if (sig_lin or sig_mk) else "none"
    curv = _curvature(x, y, slope)
    if confirmation == "none":
        ttype = "none"
    elif mk["statistically_significant"] and not sig_lin:
        ttype = "monotonic_nonlinear"
    else:
        ttype = curv["shape"] if curv["shape"] not in ("unknown",) else "linear"
    fitted_start, fitted_end = lin["intercept"] + slope * x[0], lin["intercept"] + slope * x[-1]
    return {
        # legacy keys
        "detected": True, "direction": direction, "slope": round(slope, 6),
        "r_squared": round(r2, 4), "p_value": _f(lin["p"]),
        "statistically_significant": bool(lin["p"] < cfg.alpha),
        "strength": strength, "data_points": int(n), "mann_kendall": mk,
        # new keys
        "p_value_autocorr_adjusted": _f(lin["p_adj"]),
        "significant_after_autocorr_adjustment": bool(sig_lin),
        "confirmation": confirmation,
        "consensus_direction": direction if confirmation != "none" else "no_trend",
        "trend_type": ttype,
        "curvature": curv,
        "sen_slope": _f(mk["sen_slope"], 6),
        "slope_per_period": round(slope, 6),
        "slope_per_day": _f(slope / sr.step_days, 6) if sr.step_days else None,
        "slope_pct_per_period": _f(slope / scale * 100, 4),
        "fitted_change_pct": _f((fitted_end - fitted_start) / abs(fitted_start) * 100, 2) if fitted_start else None,
        "durbin_watson": _f(lin["dw"], 3),
        "residual_autocorr_lag1": _f(lin["rho"], 3),
        "frequency": sr.freq,
    }


def detect_trend(df: pl.DataFrame, time_col: str, value_col: str,
                 config: TimeSeriesConfig | None = None) -> dict[str, Any]:
    """Trend via OLS + Mann-Kendall + Theil-Sen on a regularised series (legacy signature)."""
    try:
        cfg = config or TimeSeriesConfig()
        sr, why = prepare_series(df, time_col, value_col, cfg)
        if sr is None:
            return {"detected": False, "reason": why}
        return _trend_from_series(sr, cfg)
    except Exception as e:
        logger.warning("Trend detection failed: %s", e)
        return {"detected": False, "reason": str(e)}


# ───────────────────────────── seasonality ─────────────────────────────
def _period_name(L: int, step_days: float) -> str:
    d = L * step_days
    for lo, hi, nm in ((0.95, 1.05, "daily"), (6.5, 7.5, "weekly"), (13, 15.5, "biweekly"),
                       (28, 31.5, "monthly"), (88, 93, "quarterly"), (175, 186, "semiannual"),
                       (360, 371, "yearly")):
        if lo <= d <= hi:
            return nm
    return f"period_{L}"


def _seasonal_strength(y: np.ndarray, m: int) -> tuple[float, float, np.ndarray]:
    """
    Degrees-of-freedom-adjusted seasonal strength in [0,1] (classical decomposition), ANOVA p-value
    across phases, and the seasonal pattern.  The adjustment stops long candidate periods from
    "winning" simply because they have more free parameters (which also absorb trend / level shifts).
    """
    n = y.size
    trend = _centered_ma(y, m)
    det = y - trend
    phases = np.arange(n) % m
    means = np.array([det[phases == k].mean() for k in range(m)])
    seas = means[phases] - means.mean()
    resid = det - seas
    ssd = float(((det - det.mean()) ** 2).sum())
    ssw = float((resid ** 2).sum())
    strength = max(0.0, 1.0 - (ssw / max(n - m, 1)) / (ssd / (n - 1))) if ssd > 0 else 0.0
    ssb = float(((means - det.mean()) ** 2 * np.bincount(phases, minlength=m)).sum())
    if m > 1 and n > m and ssw > 0:
        p = _f_sf((ssb / (m - 1)) / (ssw / (n - m)), m - 1, n - m)
    else:
        p = 1.0
    return strength, p, means - means.mean()


def _periodogram_periods(y: np.ndarray, top: int = 3) -> tuple[list[int], float]:
    n = y.size
    x = np.arange(n)
    slope, icpt = np.polyfit(x, y, 1)
    d = (y - (slope * x + icpt)) * np.hanning(n)
    pw = np.abs(np.fft.rfft(d)) ** 2
    pw[0] = 0
    freqs = np.arange(pw.size)
    valid = (freqs >= 3) & (freqs <= n // 2)         # period between 2 and n/3 (>= 3 cycles)
    if valid.sum() < 3:
        return [], 1.0
    p = pw * valid
    tot = p.sum()
    g = p.max() / tot if tot > 0 else 0.0
    mcount = int(valid.sum())
    fisher_p = min(1.0, mcount * (1 - g) ** (mcount - 1)) if g > 0 else 1.0
    med = float(np.median(pw[valid]))
    peaks = [k for k in np.argsort(-p)[: top * 4]
             if p[k] > 8 * med and p[k] >= p[k - 1] and (k + 1 >= p.size or p[k] >= p[k + 1])]
    periods: list[int] = []
    for k in peaks:
        L = int(round(n / k))
        if 2 <= L <= n // 2 and all(abs(L - q) > max(1, 0.05 * L) for q in periods):
            periods.append(L)
        if len(periods) >= top:
            break
    return periods, float(fisher_p)


def _seasonality_from_series(sr: TimeSeriesData, cfg: TimeSeriesConfig) -> dict[str, Any]:
    y = sr.y
    n = y.size
    if n < 14:
        return {"detected": False, "reason": "Insufficient data for seasonality analysis"}
    if float(np.ptp(y)) == 0:
        return {"detected": False, "reason": "No residual variance after detrending"}
    cands: dict[int, str] = {L: "calendar" for L in sr.period_hints}
    per_found, fisher_p = _periodogram_periods(y)
    acf = _acf(y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n)), min(n // 2, 800))
    for L in per_found:  # refine with the ACF peak next to the periodogram estimate
        lo, hi = max(2, int(L * 0.9)), min(len(acf) - 2, int(math.ceil(L * 1.1)))
        if hi > lo:
            L = lo + int(np.argmax(acf[lo:hi + 1]))
        cands.setdefault(L, "periodogram")
    bound = 1.96 / math.sqrt(n)
    results = []
    for L, how in sorted(cands.items()):
        need = 2 if how == "calendar" else 3          # data-driven periods must repeat >= 3 times
        if L < 2 or n < need * L:
            continue
        strength, p_f, pattern = _seasonal_strength(y, L)
        a = float(acf[L]) if L < len(acf) else float("nan")
        passed = strength >= cfg.min_seasonal_strength and p_f < cfg.seasonal_p_threshold
        results.append(dict(L=L, how=how, strength=strength, p=p_f, acf=a, pattern=pattern, passed=passed))
    accepted: list[dict] = []
    for r in results:  # drop harmonics / multiples of an already accepted shorter period
        if not r["passed"]:
            continue
        def multiple(L: int, a: int) -> bool:
            k = round(L / a)
            return k >= 2 and abs(L - k * a) <= max(1, 0.03 * L)
        if not any(multiple(r["L"], a["L"]) and r["strength"] < a["strength"] + 0.10 for a in accepted):
            accepted.append(r)
    if not accepted:
        best = max(results, key=lambda r: r["strength"], default=None)
        return {"detected": False, "reason": "No significant seasonal structure at tested periods",
                "best_candidate": None if best is None else {
                    "period_length": best["L"], "seasonal_strength": _f(best["strength"], 3),
                    "anova_p_value": _f(best["p"])},
                "periodogram_fisher_p": _f(fisher_p)}
    accepted.sort(key=lambda r: -r["strength"])
    patterns = []
    for r in accepted:
        pat = r["pattern"]
        patterns.append({
            "period": _period_name(r["L"], sr.step_days),
            "lag_days": _f(r["L"] * sr.step_days, 2),                # legacy key, now frequency-aware
            "period_length": int(r["L"]),
            "autocorrelation": _f(r["acf"]),
            "acf_significant": bool(r["acf"] > bound) if math.isfinite(r["acf"]) else False,
            "seasonal_strength": _f(r["strength"], 3),
            "anova_p_value": _f(r["p"]),
            "strength": "strong" if r["strength"] >= 0.6 else "moderate",
            "amplitude": _f(float(pat.max() - pat.min())),
            "peak_phase": int(np.argmax(pat)), "trough_phase": int(np.argmin(pat)),
            "discovered_by": r["how"],
        })
    return {"detected": True, "patterns": patterns, "primary_period": patterns[0]["period"],
            "primary_period_length": patterns[0]["period_length"],
            "multiple_seasonalities": len(patterns) > 1, "periodogram_fisher_p": _f(fisher_p),
            "frequency": sr.freq}


def detect_seasonality(df: pl.DataFrame, time_col: str, value_col: str,
                       config: TimeSeriesConfig | None = None) -> dict[str, Any]:
    """Frequency-aware seasonality detection (legacy signature and keys)."""
    try:
        cfg = config or TimeSeriesConfig()
        sr, why = prepare_series(df, time_col, value_col, cfg)
        if sr is None:
            return {"detected": False, "reason": why}
        return _seasonality_from_series(sr, cfg)
    except Exception as e:
        logger.warning("Seasonality detection failed: %s", e)
        return {"detected": False, "reason": str(e)}


# ───────────────────────────── change points ─────────────────────────────
def _hampel(x: np.ndarray, w: int = 7, k: float = 5.0) -> np.ndarray:
    """Replace isolated spikes by the local median so a single outlier is never a 'change point'."""
    n = x.size
    pad = min(w, n - 1)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="reflect"), 2 * pad + 1)
    med = np.median(win, axis=1)[:n]
    sig = _mad_sigma(x - med) or float(x.std()) or 1.0
    return np.where(np.abs(x - med) > k * sig, med, x)


def _seg_linear(y: np.ndarray, penalty: float, min_size: int, max_cp: int) -> list[int]:
    """
    Best-first binary segmentation where every segment has its OWN line (level + slope).
    Trend therefore never masquerades as a change, and level jumps / slope breaks are both found.
    """
    n = y.size
    yc = y - y.mean()
    x = np.arange(n, dtype=float)
    cx, cy = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(yc)]
    cxx, cxy, cyy = np.r_[0, np.cumsum(x * x)], np.r_[0, np.cumsum(x * yc)], np.r_[0, np.cumsum(yc * yc)]

    def sse(a, b):
        m = b - a
        sx, sy = cx[b] - cx[a], cy[b] - cy[a]
        vxx = (cxx[b] - cxx[a]) - sx ** 2 / m
        vxy = (cxy[b] - cxy[a]) - sx * sy / m
        vyy = (cyy[b] - cyy[a]) - sy ** 2 / m
        return vyy - np.where(vxx > 0, vxy ** 2 / np.where(vxx > 0, vxx, 1), 0.0)

    def best(a: int, b: int):
        if b - a < 2 * min_size:
            return None
        k = np.arange(a + min_size, b - min_size + 1)
        gain = sse(a, b) - (sse(a, k) + sse(k, b))
        j = int(np.argmax(gain))
        return float(gain[j]), a, b, int(k[j])

    cand = [r for r in [best(0, n)] if r]
    cuts: list[int] = []
    while cand and len(cuts) < max_cp:
        cand.sort(key=lambda c: -c[0])
        g, a, b, k = cand.pop(0)
        if g <= penalty:
            break
        cuts.append(k)
        cand += [r for r in (best(a, k), best(k, b)) if r]
    return sorted(cuts)


def _pettitt(x: np.ndarray) -> dict[str, Any]:
    n = x.size
    ranks = np.argsort(np.argsort(x)).astype(float) + 1
    k = np.arange(1, n)
    U = 2 * np.cumsum(ranks)[:-1] - k * (n + 1)
    j = int(np.argmax(np.abs(U)))
    K = float(abs(U[j]))
    p = min(1.0, 2 * math.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))
    return {"index": int(j + 1), "p_value": round(p, 4)}


def _changepoints(sr: TimeSeriesData, period: int | None, trend: dict, cfg: TimeSeriesConfig):
    y = sr.y
    n = y.size
    if n < 24:
        return [], {}
    adj = y.copy()
    if period and n >= 2 * period:
        _, seas, _ = _decompose(y, period)
        adj = y - seas
    work = _hampel(adj)
    sig = _mad_sigma(np.diff(work)) / math.sqrt(2) or float(work.std()) or 1.0
    min_size = max(8, n // 20)
    pen = 4.0 * sig ** 2 * math.log(n)            # two extra parameters per cut (level + slope)
    cuts = _seg_linear(work, pen, min_size, cfg.max_changepoints)

    def residuals(cs: list[int]) -> np.ndarray:
        r = np.empty(n)
        edges = [0, *cs, n]
        for a, b in zip(edges[:-1], edges[1:]):
            xs = np.arange(a, b, dtype=float)
            co = np.polyfit(xs, work[a:b], 1)
            r[a:b] = work[a:b] - np.polyval(co, xs)
        return r

    r = residuals(cuts)
    rho = float(np.clip(_acf(r, 1)[1], 0.0, 0.9)) if r.std() > 0 else 0.0
    if rho > 0.1:                                  # correlated noise -> inflate penalty, redo once
        cuts = _seg_linear(work, pen * (1 + rho) / (1 - rho), min_size, cfg.max_changepoints)
        r = residuals(cuts)
    sig_res = float(r.std()) or sig

    out: list[dict] = []
    edges = [0, *cuts, n]
    w = min(30, max(5, n // 10))
    for i, k in enumerate(cuts):
        a, b = edges[i], edges[i + 2]
        cl = np.polyfit(np.arange(a, k, dtype=float), work[a:k], 1)
        cr = np.polyfit(np.arange(k, b, dtype=float), work[k:b], 1)
        jump = float(np.polyval(cr, k) - np.polyval(cl, k))
        dslope = float(cr[0] - cl[0])
        kind = "level_shift" if abs(jump) >= 3 * sig_res else "trend_break"
        before, after = float(adj[max(0, k - w):k].mean()), float(adj[k:k + w].mean())
        item = {"index": int(k), "timestamp": _ts_str(sr.ts[k], sr.freq), "type": kind,
                "direction": ("up" if jump > 0 else "down") if kind == "level_shift" else ("up" if dslope > 0 else "down"),
                "mean_before": _f(before), "mean_after": _f(after),
                "jump": _f(jump), "jump_sigma": _f(abs(jump) / sig_res, 2),
                "slope_before": _f(cl[0], 6), "slope_after": _f(cr[0], 6),
                "shift_pct": _f((after - before) / abs(before) * 100, 2) if before else None,
                "source": "native"}
        out.append(item)

    if n >= 60:                                    # variance regimes on log squared residuals
        lv = np.log(r ** 2 + 1e-9 * (r.var() + 1e-12))
        mean_cuts = _seg_mean(lv, 3.0 * 4.93 * math.log(n), max(15, n // 12), 2)
        for k in mean_cuts:
            s1, s2 = float(r[max(0, k - 2 * w):k].std()), float(r[k:k + 2 * w].std())
            if s1 > 0 and s2 > 0 and max(s1, s2) / min(s1, s2) >= 1.8:
                out.append({"index": int(k), "timestamp": _ts_str(sr.ts[k], sr.freq), "type": "variance_shift",
                            "direction": "up" if s2 > s1 else "down", "std_before": _f(s1), "std_after": _f(s2),
                            "ratio": _f(max(s1, s2) / min(s1, s2), 2), "source": "native"})
    out.sort(key=lambda c: c["index"])
    summary: dict[str, Any] = {"noise_sigma": _f(sig_res), "n_changepoints": len(out),
                               "method": "piecewise-linear binary segmentation"}
    if trend.get("confirmation") != "confirmed":   # Pettitt assumes no trend
        pet = _pettitt(work)
        summary["pettitt"] = {**pet, "timestamp": _ts_str(sr.ts[pet["index"]], sr.freq),
                              "significant": pet["p_value"] < cfg.alpha}
    else:
        summary["pettitt"] = None
        summary["note"] = "Pettitt test skipped: a confirmed trend violates its no-trend assumption."
    return out, summary


def _seg_mean(x: np.ndarray, penalty: float, min_size: int, max_cp: int) -> list[int]:
    """Binary segmentation for pure mean shifts (used for variance regimes)."""
    cand: list[tuple[float, int, int, int]] = []
    cuts: list[int] = []

    def best(a: int, b: int):
        n = b - a
        if n < 2 * min_size:
            return None
        cs = np.cumsum(x[a:b])
        k = np.arange(min_size, n - min_size + 1)
        S, tot = cs[k - 1], cs[-1]
        gain = S ** 2 / k + (tot - S) ** 2 / (n - k) - tot ** 2 / n
        j = int(np.argmax(gain))
        return float(gain[j]), a, b, a + int(k[j])

    r0 = best(0, x.size)
    if r0:
        cand.append(r0)
    while cand and len(cuts) < max_cp:
        cand.sort(key=lambda c: -c[0])
        g, a, b, k = cand.pop(0)
        if g <= penalty:
            break
        cuts.append(k)
        cand += [r for r in (best(a, k), best(k, b)) if r]
    return sorted(cuts)


# ───────────────────────────── forecasting ─────────────────────────────
def _ets(y: np.ndarray, m: int | None, a: float, b: float, g: float, phi: float,
         trend: bool, season: bool):
    n = y.size
    if season:
        lvl = float(np.mean(y[:m]))
        s = list(y[:m] - lvl)
        bb = float((np.mean(y[m:2 * m]) - np.mean(y[:m])) / m) if trend and n >= 2 * m else 0.0
    else:
        lvl = float(y[0])
        k = min(4, n - 1)
        bb = float((y[k] - y[0]) / k) if trend and k > 0 else 0.0
        s = None
    sse, resid = 0.0, []
    for t in range(n):
        st = s[t % m] if season else 0.0
        e = float(y[t]) - (lvl + phi * bb + st)
        sse += e * e
        resid.append(e)
        lvl = lvl + phi * bb + a * e
        bb = phi * bb + b * e if trend else 0.0
        if season:
            s[t % m] = st + g * e
    return sse, lvl, bb, s, np.array(resid)


def _fit_ets(y: np.ndarray, m: int | None, trend: bool, season: bool, damped: bool):
    y = y[-2000:]
    alphas, betas = (0.1, 0.3, 0.6), (0.01, 0.1) if trend else (0.0,)
    gammas = (0.05, 0.2, 0.5) if season else (0.0,)
    phis = (0.9, 0.98) if damped else (1.0,)
    best = None
    for a in alphas:
        for b in betas:
            for g in gammas:
                for phi in phis:
                    sse, *_ = _ets(y, m, a, b, g, phi, trend, season)
                    if best is None or sse < best[0]:
                        best = (sse, a, b, g, phi)
    _, a, b, g, phi = best
    sse, lvl, bb, s, resid = _ets(y, m, a, b, g, phi, trend, season)
    skip = m if season else 1
    sig1 = float(np.sqrt(np.mean(resid[skip:] ** 2))) if resid.size > skip else float(resid.std())
    return dict(a=a, b=b, g=g, phi=phi, lvl=lvl, bb=bb, s=s, n=y.size, sig1=sig1, trend=trend, season=season, m=m)


def _model_forecast(name: str, y: np.ndarray, m: int | None, h: int):
    """Returns (point forecast, sigma per horizon, params)."""
    n = y.size
    hh = np.arange(1, h + 1)
    if name == "naive":
        sig1 = float(np.std(np.diff(y))) if n > 2 else float(y.std())
        return np.full(h, y[-1]), sig1 * np.sqrt(hh), {}
    if name == "drift":
        d = (y[-1] - y[0]) / (n - 1)
        sig1 = float(np.std(np.diff(y) - d)) if n > 2 else float(y.std())
        return y[-1] + d * hh, sig1 * np.sqrt(hh * (1 + hh / (n - 1))), {"drift_per_period": _f(d, 6)}
    if name == "seasonal_naive":
        fc = np.array([y[n - m + ((k - 1) % m)] for k in hh])
        sig1 = float(np.std(y[m:] - y[:-m]))
        return fc, sig1 * np.sqrt(np.floor((hh - 1) / m) + 1), {"period": m}
    trend = True
    season = name == "holt_winters"
    damped = name == "holt_damped" or season
    fit = _fit_ets(y, m, trend, season, damped)
    phi, a, b, g = fit["phi"], fit["a"], fit["b"], fit["g"]
    cum = np.cumsum(phi ** hh) if phi != 1 else hh.astype(float)
    fc = fit["lvl"] + fit["bb"] * cum
    if season:
        fc = fc + np.array([fit["s"][(fit["n"] + k - 1) % m] for k in hh])
    c = np.array([a + b * (np.cumsum(phi ** np.arange(1, j + 1))[-1] if j > 0 else 0)
                  + (g if (m and j % m == 0 and j > 0) else 0.0) for j in range(1, h)])
    var = fit["sig1"] ** 2 * (1 + np.r_[0.0, np.cumsum(c ** 2)])
    return fc, np.sqrt(var), {"alpha": a, "beta": b, "gamma": g if season else None, "phi": phi}


def _metrics(actual: np.ndarray, pred: np.ndarray, train: np.ndarray, m: int | None) -> dict[str, float | None]:
    err = actual - pred
    lag = m if (m and train.size > m) else 1
    scale = float(np.mean(np.abs(train[lag:] - train[:-lag]))) or None
    denom = (np.abs(actual) + np.abs(pred)) / 2
    return {"mae": _f(np.mean(np.abs(err))), "rmse": _f(math.sqrt(np.mean(err ** 2))),
            "smape_pct": _f(float(np.mean(np.where(denom > 0, np.abs(err) / np.where(denom > 0, denom, 1), 0))) * 100, 2),
            "mase": _f(float(np.mean(np.abs(err))) / scale) if scale else None}


def _future_ts(sr: TimeSeriesData, h: int) -> list[str]:
    last = sr.ts[-1]
    if sr.cal in ("M", "Q", "Y"):
        unit = {"M": "M", "Q": "M", "Y": "Y"}[sr.cal]
        mult = 3 if sr.cal == "Q" else 1
        base = np.datetime64(last, unit)
        return [_ts_str((base + np.timedelta64(mult * k, unit)).astype("datetime64[us]"), sr.freq) for k in range(1, h + 1)]
    return [_ts_str(last + np.timedelta64(int(sr.step_us * k), "us"), sr.freq) for k in range(1, h + 1)]


def _forecast(sr: TimeSeriesData, period: int | None, cfg: TimeSeriesConfig) -> dict[str, Any]:
    y = sr.y
    n = y.size
    if n < 15:
        return {"success": False, "reason": "Need at least 15 observations to forecast"}
    m = period if (period and period >= 2 and n >= 2 * period + 2) else None
    h = cfg.forecast_horizon or int(min(max(m or 0, 12), max(3, n // 2)))
    names = ["naive", "drift"] + (["seasonal_naive"] if m else []) + ["holt_damped"] + (["holt_winters"] if m else [])
    h_bt = int(min(h, max(3, n // 4)))
    train, test = y[:-h_bt], y[-h_bt:]
    board: dict[str, dict] = {}
    if train.size >= max(10, 2 * (m or 1) + 2):
        for nm in names:
            try:
                fc, _, _ = _model_forecast(nm, train, m, h_bt)
                board[nm] = _metrics(test, fc, train, m)
            except Exception as e:
                logger.debug("backtest %s failed: %s", nm, e)
    simple = [k for k in ("naive", "drift", "seasonal_naive") if k in board]
    if board:
        ranked = sorted(board, key=lambda k: board[k]["mae"] if board[k]["mae"] is not None else math.inf)
        chosen = ranked[0]
        if chosen not in simple and simple:  # prefer simplicity unless clearly better
            best_simple = min(simple, key=lambda k: board[k]["mae"])
            if board[chosen]["mae"] > 0.95 * board[best_simple]["mae"]:
                chosen = best_simple
    else:
        chosen = "holt_winters" if m else "drift"
    fc, sig, params = _model_forecast(chosen, y, m, h)
    nonneg = bool(np.all(y >= 0))
    rows = []
    for k, ts in enumerate(_future_ts(sr, h)):
        row = {"timestamp": ts, "value": fc[k]}
        for lvl, z in (("80", 1.2816), ("95", 1.96)):
            row[f"lower_{lvl}"], row[f"upper_{lvl}"] = fc[k] - z * sig[k], fc[k] + z * sig[k]
        if nonneg:
            row = {kk: (max(0.0, v) if isinstance(v, float) else v) for kk, v in row.items()}
        rows.append({kk: (_f(v) if not isinstance(v, str) else v) for kk, v in row.items()})
    return {"success": True, "source": "native", "model": chosen, "horizon": h, "period": m,
            "frequency": sr.freq, "params": params, "backtest": board.get(chosen),
            "candidates": board, "forecast": rows}


# ───────────────────────────── diagnostics ─────────────────────────────
def _anomalies(sr: TimeSeriesData, period: int | None, cuts: list[int] | None = None) -> dict[str, Any]:
    """Residual-MAD anomalies. With change points, residuals come from per-regime lines so a level
    shift or trend break is reported as a change point, not as a burst of anomalies."""
    y = sr.y
    n = y.size
    if n < 14:
        return {"count": 0}
    trend, seas, resid = _decompose(y, period)
    expected = trend + seas
    if cuts:
        adj = y - seas
        edges = [0, *sorted(cuts), n]
        fit = np.empty(n)
        for a, b in zip(edges[:-1], edges[1:]):
            xs = np.arange(a, b, dtype=float)
            fit[a:b] = np.polyval(np.polyfit(xs, adj[a:b], 1), xs)
        resid, expected = adj - fit, fit + seas
    med, sig = float(np.median(resid)), _mad_sigma(resid)
    if sig == 0:
        return {"count": 0}
    z = np.abs(resid - med) / sig
    flag = (z > 3.5) & sr.obs
    idx = np.flatnonzero(flag)
    idx = idx[np.argsort(-z[idx])][:10]
    return {"count": int(flag.sum()), "pct": round(float(flag.sum()) / n * 100, 2),
            "top": [{"timestamp": _ts_str(sr.ts[i], sr.freq), "value": _f(y[i]),
                     "expected": _f(expected[i]), "robust_z": _f(z[i], 2),
                     "type": "spike" if resid[i] > med else "dip"} for i in idx]}


def _autocorr_info(y: np.ndarray, alpha: float) -> dict[str, Any]:
    n = y.size
    L = min(40, n // 3)
    if L < 3:
        return {}
    a = _acf(y - np.polyval(np.polyfit(np.arange(n), y, 1), np.arange(n)), L)
    bound = 1.96 / math.sqrt(n)
    lags = [int(k) for k in range(1, L + 1) if abs(a[k]) > bound][:6]
    lb = min(10, n // 5)
    q = n * (n + 2) * float(sum(a[k] ** 2 / (n - k) for k in range(1, lb + 1)))
    p = _chi2_sf(q, lb)
    return {"lag1": _f(a[1], 3), "significant_lags": lags, "ljung_box_p_value": _f(p),
            "residual_autocorrelation": bool(p < alpha)}


def _stationarity(y: np.ndarray, alpha: float) -> dict[str, Any]:
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, p, *_ = adfuller(y, autolag="AIC")
        return {"method": "adf", "statistic": _f(stat, 3), "p_value": _f(p),
                "stationary": bool(p < alpha), "suggest_differencing": bool(p >= alpha)}
    except Exception:
        thirds = np.array_split(y, 3)
        sd = float(y.std()) or 1.0
        mean_drift = abs(thirds[-1].mean() - thirds[0].mean()) / sd
        v = [float(t.var()) for t in thirds]
        ok = mean_drift < 0.5 and (max(v) / (min(v) or 1e-12)) < 3
        return {"method": "heuristic", "mean_drift_in_sd": _f(mean_drift, 2), "stationary": bool(ok),
                "suggest_differencing": not ok}


def _volatility(y: np.ndarray, x: np.ndarray, period: int | None) -> dict[str, Any]:
    n = y.size
    _, _, r = _decompose(y, period)
    h = n // 2
    a, b = r[:h], r[h:]
    va, vb = _mad_sigma(a) ** 2, _mad_sigma(b) ** 2          # robust: one spike must not look like volatility
    ratio = (vb / va) if va > 0 else None
    p = _f_sf(max(vb, va) / min(vb, va), h - 1, n - h - 1) * 2 if va > 0 and vb > 0 else None
    return {"coefficient_of_variation": _f(y.std() / abs(y.mean())) if y.mean() else None,
            "variance_ratio_second_vs_first_half": _f(ratio, 3),
            "heteroscedastic": bool(p is not None and min(p, 1.0) < 0.01),
            "volatility_trend": None if ratio is None else ("increasing" if ratio > 1.5 else "decreasing" if ratio < 0.67 else "stable")}


def _summary(sr: TimeSeriesData) -> dict[str, Any]:
    y = sr.y_obs[sr.obs]
    n = y.size
    first, last = float(y[0]), float(y[-1])
    span_years = (sr.ts[-1] - sr.ts[0]) / np.timedelta64(1, "D") / 365.25
    k = max(3, n // 10)
    d = np.diff(y)
    streak_up = streak_dn = cu = cd = 0
    for v in d:
        cu = cu + 1 if v > 0 else 0
        cd = cd + 1 if v < 0 else 0
        streak_up, streak_dn = max(streak_up, cu), max(streak_dn, cd)
    sig = _mad_sigma(y)
    out = {
        "first": _f(first), "last": _f(last), "min": _f(y.min()), "max": _f(y.max()),
        "mean": _f(y.mean()), "median": _f(np.median(y)),
        "pct_change_total": _f((last - first) / abs(first) * 100, 2) if first else None,
        "cagr_pct": _f(((last / first) ** (1 / span_years) - 1) * 100, 2) if first > 0 and last > 0 and span_years >= 1 else None,
        "momentum_pct": _f((y[-k:].mean() - y[-2 * k:-k].mean()) / abs(y[-2 * k:-k].mean()) * 100, 2)
        if n >= 2 * k and y[-2 * k:-k].mean() else None,
        "longest_up_streak": streak_up, "longest_down_streak": streak_dn,
        "last_value_robust_z": _f((last - np.median(y)) / sig, 2) if sig > 0 else None,
    }
    if np.all(y > 0):
        out["max_drawdown_pct"] = _f(float((y / np.maximum.accumulate(y) - 1).min()) * 100, 2)
    return out


def _relationships(series: dict[str, TimeSeriesData], cfg: TimeSeriesConfig) -> list[dict]:
    names = list(series)[:8]
    diffs = {}
    for nm in names:
        f = series[nm].y_full
        pos = np.arange(f.size)
        ok = np.isfinite(f)
        if ok.sum() < 20:
            continue
        diffs[nm] = np.diff(np.interp(pos, pos[ok], f[ok]))
    out = []
    keys = list(diffs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = diffs[keys[i]], diffs[keys[j]]
            L = min(a.size, b.size)
            if L < 20:
                continue
            a, b = a[-L:], b[-L:]
            if a.std() == 0 or b.std() == 0:
                continue
            max_lag = min(12, L // 5)
            best = (0, float(np.corrcoef(a, b)[0, 1]))
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    r = np.corrcoef(a[:-lag], b[lag:])[0, 1]
                elif lag < 0:
                    r = np.corrcoef(a[-lag:], b[:lag])[0, 1]
                else:
                    r = np.corrcoef(a, b)[0, 1]
                if abs(r) > abs(best[1]) + 0.02:
                    best = (lag, float(r))
            r0 = float(np.corrcoef(a, b)[0, 1])
            if abs(best[1]) >= 0.3:
                out.append({"a": keys[i], "b": keys[j], "correlation_of_changes": _f(r0, 3),
                            "best_lag": best[0], "correlation_at_best_lag": _f(best[1], 3),
                            "interpretation": ("same-time co-movement" if best[0] == 0 else
                                               f"{keys[i]} leads {keys[j]} by {best[0]} periods" if best[0] > 0 else
                                               f"{keys[j]} leads {keys[i]} by {-best[0]} periods")})
    out.sort(key=lambda d: -abs(d["correlation_at_best_lag"]))
    return out[:10]


# ───────────────────────────── orchestration ─────────────────────────────
def _time_score(name: str, s: pl.Series) -> float:
    t = _time_to_us(s)
    if t is None:
        return -1.0
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return -1.0
    v = t[ok]
    uniq = np.unique(v).size / v.size
    mono = float(np.mean(np.diff(v) >= 0)) if v.size > 1 else 0.0
    name_bonus = 0.5 if _name_tokens(name) & {"date", "time", "timestamp", "datetime", "day", "month", "period"} else 0.0
    return uniq + mono + name_bonus + ok.mean()


def _resolve_numeric(df: pl.DataFrame, exclude: set[str]) -> list[str]:
    out = []
    for c, t in df.schema.items():
        if c in exclude or t == pl.Boolean or not _is_numeric(t):
            continue
        v = _numeric_array(df[c])
        if v is None:
            continue
        v = v[np.isfinite(v)]
        if v.size < 5 or np.ptp(v) == 0:
            continue
        if np.unique(v).size / v.size >= 0.98 and np.all(np.diff(v) >= 0) and np.all(v == np.round(v)):
            continue  # monotonic integer id / counter
        out.append(c)
    return out


def _coerce_frame(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if isinstance(df, dict):
        return pl.DataFrame(df)
    return pl.from_pandas(df)


def _group_trends(df: pl.DataFrame, gk: list[str], t_us: np.ndarray, g: _Grid, col: str,
                  y_rows: np.ndarray, cfg: TimeSeriesConfig, global_dir: str | None) -> dict[str, Any]:
    keyed = df.select(gk).with_columns(pl.Series("__i", np.arange(df.height)))
    agg = keyed.group_by(gk).agg(pl.col("__i"))
    entries = []
    for row in agg.iter_rows():
        ix = np.asarray(row[-1], dtype=np.int64)
        if ix.size >= cfg.min_group_points:
            entries.append((" | ".join(str(v) for v in row[:-1]), ix))
    entries.sort(key=lambda e: -e[1].size)
    out: dict[str, Any] = {}
    for label, ix in entries[: cfg.max_groups]:
        sr, _ = _build_series(col, g, t_us, y_rows, cfg, rows=ix)
        if sr is None:
            continue
        tr = _trend_from_series(sr, cfg)
        if not tr.get("detected"):
            continue
        out[label] = {"rows": int(ix.size), "direction": tr["consensus_direction"],
                      "confirmation": tr["confirmation"], "slope_pct_per_period": tr["slope_pct_per_period"],
                      "diverges_from_overall": bool(global_dir in ("upward", "downward") and
                                                    tr["consensus_direction"] in ("upward", "downward") and
                                                    tr["consensus_direction"] != global_dir)}
    return out


def analyze_time_series(df: Any, numeric_cols: list[str] | None = None, time_col: str | None = None,
                        config: TimeSeriesConfig | None = None, **overrides: Any) -> dict[str, Any]:
    """
    Complete time-series analysis. Legacy keys: has_time_series, time_column, trends, seasonality,
    kats_intelligence, changepoints, forecasts.  New keys: frequency, data_quality, summary, anomalies,
    change_summary, volatility, autocorrelation, stationarity, relationships, group_trends,
    time_columns_found, skipped, notes.
    """
    cfg = replace(config or TimeSeriesConfig(), **overrides) if (config or overrides) else TimeSeriesConfig()
    df = _coerce_frame(df)
    time_cols = detect_time_columns(df, include_parsable_strings=cfg.parse_string_dates)
    if time_col and time_col in df.columns:
        primary = time_col
    elif time_cols:
        primary = max(time_cols, key=lambda c: _time_score(c, df[c]))
    else:
        return {"has_time_series": False, "message": "No datetime or numeric columns found"}
    gk = [c for c in (cfg.group_by or []) if c in df.columns]
    cols = numeric_cols if numeric_cols is not None else _resolve_numeric(df, set(time_cols) | set(gk) | {primary})
    cols = [c for c in cols if c in df.columns and c != primary]
    if not cols:
        return {"has_time_series": False, "message": "No datetime or numeric columns found"}

    results: dict[str, Any] = {
        "has_time_series": True, "time_column": primary, "time_columns_found": time_cols,
        "trends": {}, "seasonality": {}, "kats_intelligence": {}, "changepoints": {}, "forecasts": {},
        "change_summary": {}, "anomalies": {}, "volatility": {}, "autocorrelation": {},
        "stationarity": {}, "summary": {}, "data_quality": {}, "group_trends": {},
        "skipped": {}, "notes": [],
    }
    t_us = _time_to_us(df[primary])
    g = _build_grid(t_us) if t_us is not None else None
    if g is None:
        results.update(has_time_series=False, message=f"Could not build a time axis from '{primary}'")
        return results
    results["frequency"] = g.freq
    if g.freq == "irregular":
        results["notes"].append("Irregular sampling: analysed in observation order; seasonality limited to data-driven periods.")

    series: dict[str, TimeSeriesData] = {}
    for col in cols[: cfg.max_columns]:
        try:
            y_rows = _numeric_array(df[col])
            if y_rows is None:
                results["skipped"][col] = "not numeric"
                continue
            sr, why = _build_series(col, g, t_us, y_rows, cfg)
            if sr is None:
                results["skipped"][col] = why
                results["trends"][col] = {"detected": False, "reason": why}
                results["seasonality"][col] = {"detected": False, "reason": why}
                continue
            series[col] = sr
            results["data_quality"][col] = sr.quality
            trend = _trend_from_series(sr, cfg)
            seas = _seasonality_from_series(sr, cfg)
            results["trends"][col], results["seasonality"][col] = trend, seas
            period = seas.get("primary_period_length") if seas.get("detected") else None
            results["summary"][col] = _summary(sr)
            y_obs = sr.y_obs[sr.obs]
            results["stationarity"][col] = _stationarity(sr.y, cfg.alpha)
            results["autocorrelation"][col] = _autocorr_info(sr.y, cfg.alpha)
            results["volatility"][col] = _volatility(sr.y, np.arange(sr.y.size), period)
            native_cp: list[dict] = []
            if cfg.changepoints:
                native_cp, cps = _changepoints(sr, period, trend, cfg)
                results["changepoints"][col], results["change_summary"][col] = native_cp, cps
            if cfg.anomalies:
                results["anomalies"][col] = _anomalies(
                    sr, period, [c["index"] for c in native_cp if c["type"] in ("level_shift", "trend_break")])
            native_fc: dict = {}
            if cfg.forecast:
                try:
                    native_fc = _forecast(sr, period, cfg)
                except Exception as e:
                    native_fc = {"success": False, "reason": str(e)}
                results["forecasts"][col] = native_fc
            if cfg.kats:  # optional Meta Kats intelligence; native results above are the fallback
                try:
                    from app.services.timeseries_kats import run_kats_time_series_intelligence
                    kats = run_kats_time_series_intelligence(df, primary, col)
                    if kats.get("success"):
                        results["kats_intelligence"][col] = kats
                        if not native_cp and kats.get("changepoints"):
                            results["changepoints"][col] = kats["changepoints"]
                        if not native_fc.get("success") and kats.get("forecast"):
                            results["forecasts"][col] = kats["forecast"]
                except Exception as kerr:
                    logger.debug("Kats intelligence unavailable for %s: %s", col, kerr)
            if gk:
                results["group_trends"][col] = _group_trends(
                    df, gk, t_us, g, col, y_rows, cfg, trend.get("consensus_direction"))
        except Exception as e:
            logger.warning("Time series analysis failed for %s: %s", col, e)
            results["skipped"][col] = f"error: {e}"
    if cfg.relationships and len(series) >= 2:
        try:
            results["relationships"] = _relationships(series, cfg)
        except Exception as e:
            logger.debug("relationship analysis skipped: %s", e)

    # Legacy & ranking compatibility
    results["analyses"] = {
        col: {
            "trend": results["trends"].get(col, {}),
            "seasonality": results["seasonality"].get(col, {}),
        }
        for col in series
    }
    drift_detected = []
    for col, cps in results.get("changepoints", {}).items():
        if isinstance(cps, list):
            for cp in cps:
                if isinstance(cp, dict) and cp.get("type") in ("level_shift", "trend_break"):
                    drift_detected.append({
                        "column": col,
                        "shift_pct": cp.get("shift_pct", 0.0) or 0.0,
                        "timestamp": cp.get("timestamp"),
                        "p_value": cp.get("p_value", 0.01),
                    })
    results["drift_detected"] = drift_detected

    return results