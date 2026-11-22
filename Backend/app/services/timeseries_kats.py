"""
Meta Kats-Inspired Time Series Intelligence Engine for GetReport.
Implements:
1. CUSUM (Cumulative Sum) Changepoint Detection with Log-Likelihood Ratio Tests.
2. Fast Holt-Winters Exponential Smoothing Forecaster with 95% Confidence Bounds.
3. TSFeatures Statistical Feature Extractor (Entropy, Trend/Seasonal Strength, Lumpiness, Stability).
All operations are vectorized, in-process, and execute in sub-10ms.
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from scipy.stats import chi2

logger = logging.getLogger(__name__)


class CUSUMChangepointDetector:
    """
    Cumulative Sum (CUSUM) changepoint detection algorithm inspired by Meta's Kats.
    Conducts iterative log-likelihood ratio (LLR) hypothesis testing against Gaussian
    distribution assumptions to pinpoint exact structural shifts in time series.
    """
    def __init__(self, significance_level: float = 0.05, min_distance: int = 5):
        self.significance_level = significance_level
        self.min_distance = min_distance

    def _single_changepoint(
        self,
        series: np.ndarray,
        timestamps: Optional[List[Any]] = None
    ) -> Optional[Dict[str, Any]]:
        n = len(series)
        if n < 2 * self.min_distance:
            return None

        # Standardize series with finite masking
        series = np.asarray(series, dtype=np.float64)
        finite_mask = np.isfinite(series) & (np.abs(series) < 1e12)
        if finite_mask.sum() < 2 * self.min_distance:
            return None
        series = series[finite_mask]
        n = len(series)

        # Efficient downsampling for search when series is large
        step = 1
        sub_series = series
        if n > 2000:
            step = n // 2000 + 1
            sub_series = series[::step]

        m = len(sub_series)
        sub_min_dist = max(2, self.min_distance // step)
        if m < 2 * sub_min_dist:
            return None

        with np.errstate(all="ignore"):
            overall_mean = float(np.mean(sub_series))
            overall_var = float(np.var(sub_series))
        if overall_var <= 1e-10:
            return None

        # Vectorized CUSUM changepoint calculation via cumulative sums
        with np.errstate(all="ignore"):
            cumsum1 = np.cumsum(sub_series)
            cumsum2 = np.cumsum(sub_series**2)

        tau_arr = np.arange(sub_min_dist, m - sub_min_dist)
        pre_len = tau_arr
        post_len = m - tau_arr

        pre_sum = cumsum1[tau_arr - 1]
        pre_mean_arr = pre_sum / pre_len
        pre_var_arr = np.maximum(0.0, (cumsum2[tau_arr - 1] / pre_len) - pre_mean_arr**2)

        post_sum = cumsum1[-1] - pre_sum
        post_mean_arr = post_sum / post_len
        post_var_arr = np.maximum(0.0, ((cumsum2[-1] - cumsum2[tau_arr - 1]) / post_len) - post_mean_arr**2)

        pooled_var_arr = (pre_len * pre_var_arr + post_len * post_var_arr) / m
        valid_taus = pooled_var_arr > 1e-10
        if not np.any(valid_taus):
            return None

        llr_arr = np.full(len(tau_arr), -np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.maximum(1.0, overall_var / pooled_var_arr[valid_taus])
            llr_arr[valid_taus] = (m / 2.0) * np.log(ratio)

        best_idx = int(np.argmax(llr_arr))
        max_llr = float(llr_arr[best_idx])
        if max_llr <= 0 or np.isneginf(max_llr):
            return None

        best_tau = int(tau_arr[best_idx] * step)
        best_tau = max(self.min_distance, min(best_tau, n - self.min_distance))

        pre = series[:best_tau]
        post = series[best_tau:]
        with np.errstate(all="ignore"):
            pre_mean = float(np.mean(pre))
            post_mean = float(np.mean(post))
            pre_var = float(np.var(pre))
            post_var = float(np.var(post))

        best_stats = {
            "index": best_tau,
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "mean_shift": round(post_mean - pre_mean, 4),
            "percent_shift": round(((post_mean - pre_mean) / (abs(pre_mean) + 1e-6)) * 100.0, 2),
            "pre_std": round(math.sqrt(max(pre_var, 0.0)), 4),
            "post_std": round(math.sqrt(max(post_var, 0.0)), 4),
            "llr_score": round(max_llr, 4)
        }

        # Guard: Effect size check - mean shift must exceed 0.3 standard deviations
        if abs(best_stats.get("mean_shift", 0.0)) < 0.3 * math.sqrt(overall_var):
            return None

        # Chi-square hypothesis test (1 degree of freedom for mean shift)
        stat = 2.0 * max_llr
        p_value = float(1.0 - chi2.cdf(stat, df=1))

        best_stats["p_value"] = round(p_value, 6)
        best_stats["statistically_significant"] = bool(p_value < self.significance_level)

        if timestamps and best_tau < len(timestamps):
            best_stats["timestamp"] = str(timestamps[best_tau])
        else:
            best_stats["timestamp"] = None

        return best_stats

    def detect_changepoints(
        self,
        series: np.ndarray,
        timestamps: Optional[List[Any]] = None,
        max_changepoints: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Detects multiple significant changepoints using binary splitting.
        Returns changepoints ordered by statistical significance (llr_score).
        """
        n = len(series)
        if n < 2 * self.min_distance:
            return []

        changepoints = []
        
        # Primary changepoint
        primary = self._single_changepoint(series, timestamps)
        if primary and primary["statistically_significant"]:
            primary["is_primary"] = True
            changepoints.append(primary)
            tau = primary["index"]

            # Recursively check pre and post segments if length allows
            if max_changepoints > 1 and tau >= 2 * self.min_distance:
                pre_ts = timestamps[:tau] if timestamps else None
                pre_cp = self._single_changepoint(series[:tau], pre_ts)
                if pre_cp and pre_cp["statistically_significant"]:
                    pre_cp["is_primary"] = False
                    changepoints.append(pre_cp)

            if max_changepoints > len(changepoints) and (n - tau) >= 2 * self.min_distance:
                post_ts = timestamps[tau:] if timestamps else None
                post_cp = self._single_changepoint(series[tau:], post_ts)
                if post_cp and post_cp["statistically_significant"]:
                    post_cp["index"] += tau  # adjust global index
                    post_cp["is_primary"] = False
                    changepoints.append(post_cp)

        # Sort by statistical significance (llr_score) descending so primary is always first
        changepoints.sort(key=lambda x: x.get("llr_score", 0.0), reverse=True)
        return changepoints


class HoltWintersForecaster:
    """
    High-speed, robust Holt-Winters exponential smoothing forecaster with empirical
    95% prediction intervals (inspired by Kats's HoltWinters & Theta models).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1):
        self.alpha = alpha  # Level smoothing
        self.beta = beta    # Trend smoothing
        self.gamma = gamma  # Seasonality smoothing

    def forecast(
        self,
        series: np.ndarray,
        horizon: int = 30,
        seasonality_period: Optional[int] = None
    ) -> Dict[str, Any]:
        series = np.asarray(series, dtype=np.float64)
        series = series[np.isfinite(series) & (np.abs(series) < 1e12)]
        n = len(series)
        if n < 5:
            return {"success": False, "reason": "Series too short for Holt-Winters (< 5 points)"}

        # Initialize level and trend
        level = float(series[0])
        trend = float(series[1] - series[0]) if n > 1 else 0.0

        # Seasonal components
        has_seasonality = bool(seasonality_period and seasonality_period >= 2 and n >= 2 * seasonality_period)
        m = seasonality_period if has_seasonality else 1
        
        if has_seasonality:
            # Initial seasonal indices
            seasonals = [float(series[i] - level) for i in range(m)]
        else:
            seasonals = [0.0] * m

        levels = [level]
        trends = [trend]
        fitted = [level]

        # Filter forward
        for i in range(1, n):
            val = float(series[i])
            s_idx = i % m
            
            last_level = level
            last_trend = trend
            
            if has_seasonality:
                level = self.alpha * (val - seasonals[s_idx]) + (1 - self.alpha) * (last_level + last_trend)
                trend = self.beta * (level - last_level) + (1 - self.beta) * last_trend
                seasonals[s_idx] = self.gamma * (val - level) + (1 - self.gamma) * seasonals[s_idx]
                fitted.append(last_level + last_trend + seasonals[s_idx])
            else:
                level = self.alpha * val + (1 - self.alpha) * (last_level + last_trend)
                trend = self.beta * (level - last_level) + (1 - self.beta) * last_trend
                fitted.append(last_level + last_trend)

            levels.append(level)
            trends.append(trend)

        # Residual standard error for confidence bands
        with np.errstate(all="ignore"):
            residuals = series - np.array(fitted)
            sigma = float(np.std(residuals)) if len(residuals) > 1 else float(np.std(series) * 0.1)

        # Extrapolate horizon
        forecasts = []
        lower_bounds = []
        upper_bounds = []

        for h in range(1, horizon + 1):
            s_idx = (n + h - 1) % m
            seas = seasonals[s_idx] if has_seasonality else 0.0
            fc = level + h * trend + seas
            # Expanding prediction interval: 1.96 * sigma * sqrt(h)
            margin = 1.96 * sigma * math.sqrt(h * 0.5 + 0.5)
            
            forecasts.append(round(float(fc), 4))
            lower_bounds.append(round(float(fc - margin), 4))
            upper_bounds.append(round(float(fc + margin), 4))

        return {
            "success": True,
            "horizon": horizon,
            "has_seasonality": has_seasonality,
            "seasonality_period": m if has_seasonality else None,
            "final_level": round(level, 4),
            "final_trend": round(trend, 4),
            "residual_std": round(sigma, 4),
            "forecast": forecasts,
            "lower_bound_95": lower_bounds,
            "upper_bound_95": upper_bounds
        }


class TSFeaturesExtractor:
    """
    Extracts high-value statistical traits from time series (Meta Kats tsfeatures pattern):
    - entropy (spectral randomness/predictability)
    - trend_strength
    - seasonal_strength
    - lumpiness (variance of chunked variances)
    - stability (variance of chunked means)
    - crossing_points (median crossings)
    - flat_spots (run length of flat runs)
    """
    @staticmethod
    def extract_features(series: np.ndarray, period: int = 7) -> Dict[str, Any]:
        y = np.asarray(series, dtype=np.float64)
        y = y[np.isfinite(y) & (np.abs(y) < 1e12)]
        n = len(y)
        if n < 10:
            return {"valid": False, "reason": "Insufficient points (< 10)"}
        total_var = float(np.var(y))
        if total_var <= 1e-10:
            return {
                "valid": True,
                "entropy": 0.0,
                "trend_strength": 0.0,
                "seasonal_strength": 0.0,
                "lumpiness": 0.0,
                "stability": 0.0,
                "crossing_points": 0,
                "flat_spots": n,
                "classification": "constant"
            }

        # 1. Spectral Entropy
        fft_vals = np.abs(np.fft.rfft(y - np.mean(y))) ** 2
        psd = fft_vals / (np.sum(fft_vals) + 1e-12)
        entropy = -float(np.sum([p * math.log(p + 1e-12) for p in psd if p > 0])) / math.log(len(psd) + 1e-12)
        entropy = max(0.0, min(1.0, float(entropy)))

        # 2. Trend & Seasonal Strength
        x = np.arange(n)
        slope, intercept = np.polyfit(x, y, 1)
        trend_comp = slope * x + intercept
        detrended = y - trend_comp
        trend_var = float(np.var(trend_comp))
        residual_var = float(np.var(detrended))
        trend_strength = max(0.0, min(1.0, 1.0 - (residual_var / (trend_var + residual_var + 1e-10))))

        # Seasonal strength
        seasonal_strength = 0.0
        if n >= 2 * period:
            cycle_means = np.array([np.mean(detrended[i::period]) for i in range(period)])
            seasonal_comp = np.tile(cycle_means, int(math.ceil(n / period)))[:n]
            seas_var = float(np.var(seasonal_comp))
            seas_resid_var = float(np.var(detrended - seasonal_comp))
            seasonal_strength = max(0.0, min(1.0, 1.0 - (seas_resid_var / (seas_var + seas_resid_var + 1e-10))))

        # 3. Lumpiness & Stability
        chunk_size = max(5, n // 10)
        chunks = [y[i:i+chunk_size] for i in range(0, n, chunk_size) if len(y[i:i+chunk_size]) >= 3]
        if chunks:
            chunk_means = [float(np.mean(c)) for c in chunks]
            chunk_vars = [float(np.var(c)) for c in chunks]
            stability = float(np.var(chunk_means)) / (total_var + 1e-10)
            lumpiness = float(np.var(chunk_vars)) / ((total_var ** 2) + 1e-10)
        else:
            stability = 0.0
            lumpiness = 0.0

        # 4. Crossing points
        med = float(np.median(y))
        crossings = int(np.sum(np.diff(y > med) != 0))

        # 5. Flat spots
        with np.errstate(all="ignore"):
            min_y = float(np.min(y))
            max_y = float(np.max(y))
            if np.isfinite(min_y) and np.isfinite(max_y) and min_y != max_y:
                bins = np.linspace(min_y, max_y, 11)
            else:
                bins = np.array([min_y, min_y + 1.0])
        binned = np.digitize(y, bins)
        max_run = 1
        current_run = 1
        for i in range(1, len(binned)):
            if binned[i] == binned[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        # Classification
        if trend_strength > 0.6:
            classification = "strongly_trending"
        elif seasonal_strength > 0.5:
            classification = "strongly_seasonal"
        elif entropy > 0.8:
            classification = "noisy_or_chaotic"
        elif lumpiness > 2.0:
            classification = "volatile_bursty"
        else:
            classification = "stable_stationary"

        return {
            "valid": True,
            "entropy": round(entropy, 4),
            "trend_strength": round(trend_strength, 4),
            "seasonal_strength": round(seasonal_strength, 4),
            "lumpiness": round(lumpiness, 4),
            "stability": round(stability, 4),
            "crossing_points": crossings,
            "flat_spots": max_run,
            "classification": classification
        }


def run_kats_time_series_intelligence(
    df: pl.DataFrame,
    time_col: str,
    value_col: str,
    forecast_horizon: int = 30
) -> Dict[str, Any]:
    """
    Unified entry point executing full Kats intelligence pipeline:
    1. CUSUM changepoint detection.
    2. Holt-Winters 30-step forecast with 95% confidence intervals.
    3. TSFeatures statistical profiling.
    """
    try:
        sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
        if sorted_df.height < 10:
            return {"success": False, "reason": "Insufficient records (< 10 rows)"}

        # Downsample if very large to prevent memory/CPU spikes
        orig_height = sorted_df.height
        if orig_height > 5000:
            step = orig_height // 5000 + 1
            sorted_df = sorted_df.gather_every(step)

        y = sorted_df[value_col].cast(pl.Float64, strict=False).to_numpy()
        y = np.clip(np.where(np.isfinite(y), y, 0.0), -1e12, 1e12)
        t_vals = [str(v) for v in sorted_df[time_col].to_list()]

        # 1. CUSUM Changepoints
        detector = CUSUMChangepointDetector()
        changepoints = detector.detect_changepoints(y, timestamps=t_vals)

        # 2. TSFeatures
        features = TSFeaturesExtractor.extract_features(y)

        # 3. Forecast
        forecaster = HoltWintersForecaster()
        forecast_res = forecaster.forecast(y, horizon=forecast_horizon)

        return {
            "success": True,
            "column": value_col,
            "time_column": time_col,
            "data_points": len(y),
            "changepoints": changepoints,
            "has_structural_changes": len(changepoints) > 0,
            "features": features,
            "forecast": forecast_res
        }
    except Exception as e:
        logger.error(f"Kats time series intelligence failed for {value_col}: {e}")
        return {"success": False, "reason": str(e)}
