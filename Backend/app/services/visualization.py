from __future__ import annotations

import io
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")                              # original: non-interactive backend for server
import matplotlib.pyplot as plt                    # noqa: E402
import seaborn as sns                              # noqa: E402
import pandas as pd                                # noqa: E402
import numpy as np                                 # noqa: E402

# ─── Logger ──────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
CHART_DPI: int                    = 100     # original DPI preserved
TOP_DISTRIBUTION_FEATURES: int    = 5       # original: top 5 by variance
TOP_CATEGORICAL_COLUMNS: int      = 3       # original: first 3 categorical columns
BAR_CHART_MAX_CARDINALITY: int    = 20      # original: nunique < 20
PIE_CHART_MAX_CARDINALITY: int    = 5       # Reduced from 10 to avoid "Pie Overload" critique
TREND_CHART_MAX_COLUMNS: int      = 5       # max numeric columns to plot as trends

# ─── Global Seaborn Theme ───────────────────────────────────────────────────
# Set once at module load — applies consistently to every chart in the report.
# Original code had no global theme; each chart inherited default styling.
sns.set_theme(style="whitegrid", palette="muted")


# ─── Custom Exceptions ───────────────────────────────────────────────────────
class InvalidDataFrameError(TypeError):
    """Raised when generate_charts receives something that is not a DataFrame."""


class EmptyDataFrameError(ValueError):
    """Raised when the DataFrame has zero rows or zero columns."""


# ─── Chart Report Metadata ───────────────────────────────────────────────────
@dataclass
class ChartReport:
    """
    Tracks exactly what happened during chart generation.
    Returned alongside the charts dict so pdf_report.py and the caller
    know which charts made it and which ones failed.

    Attributes:
        charts_generated: List of chart types that were successfully created.
        charts_failed:    List of {chart_type, reason} for every failure.
        total_images:     Total number of individual chart images produced.
        timing_ms:        How long the entire generation took (milliseconds).
    """
    charts_generated: list[str]              = field(default_factory=list)
    charts_failed:    list[dict[str, str]]   = field(default_factory=list)
    total_images:     int                    = 0
    timing_ms:        float                  = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "charts_generated": self.charts_generated,
            "charts_failed":    self.charts_failed,
            "total_images":     self.total_images,
            "timing_ms":        round(self.timing_ms, 2),
        }


# ─── Figure → Base64 ────────────────────────────────────────────────────────
def fig_to_base64(fig) -> str:
    """
    Convert a Matplotlib figure to a Base64-encoded PNG string.
    """
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)   # original
        buf.seek(0)                                                            # original
        img_str = base64.b64encode(buf.read()).decode("utf-8")                 # original
        return img_str
    except Exception as e:
        logger.error("fig_to_base64 failed: %s", str(e))
        raise
    finally:
        plt.close(fig)     # original: critical memory free
        buf.close()        # enhanced: close the BytesIO buffer too


# ─── Input Validation ────────────────────────────────────────────────────────
def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Verify the input is a non-empty DataFrame before any chart work begins.
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"generate_charts expects a pandas DataFrame, got {type(df).__name__}."
        )
    if df.empty or len(df.columns) == 0:
        raise EmptyDataFrameError(
            "The DataFrame is empty — nothing to chart."
        )
    logger.info("Chart input validated — %d rows × %d columns.", len(df), len(df.columns))


# ─── Chart 1: Correlation Heatmap ────────────────────────────────────────────
def _generate_correlation_heatmap(
    numeric_df: pd.DataFrame,
    report: ChartReport,
) -> str | None:
    """
    Generate a Pearson correlation heatmap for all numeric columns.
    """
    # Original guard condition preserved exactly
    if numeric_df.empty or numeric_df.shape[1] <= 1:
        logger.debug("Correlation heatmap skipped — need 2+ numeric columns, have %d.", numeric_df.shape[1])
        return None

    try:
        fig, ax = plt.subplots(figsize=(10, 8))                                  # original figsize
        sns.heatmap(
            numeric_df.corr(),                                                   # original
            annot=True,                                                          # original
            cmap="coolwarm",                                                     # original
            fmt=".2f",                                                           # original
            ax=ax,
        )
        ax.set_title("Correlation Matrix")                                       # original title
        result = fig_to_base64(fig)                                              # original conversion

        report.charts_generated.append("correlation_heatmap")
        report.total_images += 1
        logger.info("Correlation heatmap generated — %d columns.", numeric_df.shape[1])
        return result

    except Exception as e:
        report.charts_failed.append({"chart": "correlation_heatmap", "reason": str(e)})
        logger.warning("Correlation heatmap failed: %s", str(e))
        return None


# ─── Chart 2: Distribution Histograms ────────────────────────────────────────
def _generate_distribution_plots(
    numeric_df: pd.DataFrame,
    report: ChartReport,
) -> list[dict[str, str]]:
    """
    Generate KDE-overlaid histograms for the top numeric columns by variance.
    """
    if numeric_df.empty:
        logger.debug("Distribution plots skipped — no numeric columns.")
        return []

    # Original logic: sort by variance, take top N
    variances    = numeric_df.var().sort_values(ascending=False)                 # original
    top_features = variances.head(TOP_DISTRIBUTION_FEATURES).index.tolist()      # original

    logger.info(
        "Distribution plots — top %d columns by variance: %s",
        len(top_features), top_features
    )

    distributions: list[dict[str, str]] = []

    for col in top_features:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))                               # original figsize
            sns.histplot(numeric_df[col], kde=True, color="skyblue", ax=ax)      # original
            ax.set_title(f"Distribution of {col}")                               # original title

            distributions.append({                                               # original structure
                "column": col,
                "image":  fig_to_base64(fig),                                    # original conversion
            })

            report.total_images += 1
            logger.info("Distribution plot generated — '%s'.", col)

        except Exception as e:
            report.charts_failed.append({"chart": f"distribution:{col}", "reason": str(e)})
            logger.warning("Distribution plot failed for '%s': %s", col, str(e))

    if distributions:
        report.charts_generated.append("distributions")

    return distributions


# ─── Chart 3: Categorical Bar Charts ─────────────────────────────────────────
def _generate_bar_charts(
    df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    report: ChartReport,
) -> list[dict[str, str]]:
    """
    Generate horizontal bar charts for the top categorical columns.
    """
    if categorical_df.empty:
        logger.debug("Bar charts skipped — no categorical columns.")
        return []

    # Original logic: take first N categorical columns
    columns_to_check = categorical_df.columns[:TOP_CATEGORICAL_COLUMNS]          # original

    bar_charts: list[dict[str, str]] = []

    for col in columns_to_check:
        # Original guard: skip if cardinality too high
        unique_count = categorical_df[col].nunique()
        if unique_count >= BAR_CHART_MAX_CARDINALITY:
            logger.info(
                "Bar chart skipped for '%s' — %d unique values (limit: %d).",
                col, unique_count, BAR_CHART_MAX_CARDINALITY
            )
            continue

        try:
            fig, ax = plt.subplots(figsize=(10, 6))                              # original figsize
            sns.countplot(                                                        # original
                y=df[col],                                                       # original
                order=df[col].value_counts().index,                              # original
                ax=ax,
            )
            ax.set_title(f"Counts: {col}")                                       # original title

            bar_charts.append({                                                  # original structure
                "column": col,
                "image":  fig_to_base64(fig),                                    # original conversion
            })

            report.total_images += 1
            logger.info("Bar chart generated — '%s' (%d categories).", col, unique_count)

        except Exception as e:
            report.charts_failed.append({"chart": f"bar_chart:{col}", "reason": str(e)})
            logger.warning("Bar chart failed for '%s': %s", col, str(e))

    if bar_charts:
        report.charts_generated.append("bar_charts")

    return bar_charts


# ─── Chart 4: Pie Charts (Enhanced — new chart type) ────────────────────────
def _generate_pie_charts(
    categorical_df: pd.DataFrame,
    report: ChartReport,
) -> list[dict[str, str]]:
    """
    Generate pie charts for categorical columns with very low cardinality.
    """
    if categorical_df.empty:
        logger.debug("Pie charts skipped — no categorical columns.")
        return []

    pie_charts: list[dict[str, str]] = []

    for col in categorical_df.columns:
        unique_count = categorical_df[col].nunique()

        # Only pie-chart very low cardinality columns
        if unique_count >= PIE_CHART_MAX_CARDINALITY or unique_count < 2:
            continue

        try:
            value_counts = categorical_df[col].value_counts()

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(
                value_counts.values,
                labels=value_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=sns.color_palette("muted", len(value_counts)),
            )
            ax.set_title(f"Composition: {col}")

            pie_charts.append({
                "column": col,
                "image":  fig_to_base64(fig),
            })

            report.total_images += 1
            logger.info("Pie chart generated — '%s' (%d categories).", col, unique_count)

        except Exception as e:
            report.charts_failed.append({"chart": f"pie_chart:{col}", "reason": str(e)})
            logger.warning("Pie chart failed for '%s': %s", col, str(e))

    if pie_charts:
        report.charts_generated.append("pie_charts")

    return pie_charts


# ─── Chart 5: Trend Line Charts (Enhanced — new chart type) ─────────────────
def _generate_trend_charts(
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    report: ChartReport,
) -> list[dict[str, str]]:
    """
    Generate line charts showing numeric values over time when a datetime
    column is detected in the DataFrame.
    """
    if numeric_df.empty:
        logger.debug("Trend charts skipped — no numeric columns.")
        return []

    # Detect datetime columns
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not datetime_cols:
        logger.debug("Trend charts skipped — no datetime column detected.")
        return []

    # Sort the DataFrame once? No, needs sorting per time column if we support multiple.
    
    trend_charts: list[dict[str, str]] = []
    
    # Pick top numeric columns by variance
    variances    = numeric_df.var().sort_values(ascending=False)
    top_features = variances.head(TREND_CHART_MAX_COLUMNS).index.tolist()

    # Iterate over ALL detected time columns (up to reasonable limit, e.g. 2)
    # This fixes the "blindly pick first column" critique.
    for time_col in datetime_cols[:2]:
        df_sorted = df.sort_values(by=time_col)
        
        for col in top_features:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(
                    df_sorted[time_col],
                    df_sorted[col],
                    color="steelblue",
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                )
                ax.set_title(f"Trend: {col} over {time_col}")
                ax.set_xlabel(time_col)
                ax.set_ylabel(col)
                fig.autofmt_xdate()

                trend_charts.append({
                    "column": col,
                    "image":  fig_to_base64(fig),
                })

                report.total_images += 1
                logger.info("Trend chart generated — '%s' over '%s'.", col, time_col)

            except Exception as e:
                report.charts_failed.append({"chart": f"trend:{col}_vs_{time_col}", "reason": str(e)})
                logger.warning("Trend chart failed for '%s' vs '%s': %s", col, time_col, str(e))

    if trend_charts:
        report.charts_generated.append("trend_charts")

    return trend_charts


# ─── Main Entry Point ────────────────────────────────────────────────────────
def generate_charts(df: pd.DataFrame) -> tuple[dict[str, Any], ChartReport]:
    """
    Generate the full suite of visualizations for the dataset.
    Returns:
        Tuple of (charts dict with base64 strings, ChartReport with metadata).
        Never raises — on any failure returns what it has plus the error log.
    """
    start_time = time.perf_counter()
    report     = ChartReport()
    logger.info("═══ Chart Generation Started ═══")

    # ── 1. Validate input ────────────────────────────────────────────────────
    try:
        _validate_dataframe(df)
    except (InvalidDataFrameError, EmptyDataFrameError) as e:
        logger.error("Chart generation aborted: %s", str(e))
        report.charts_failed.append({"chart": "all", "reason": str(e)})
        report.timing_ms = (time.perf_counter() - start_time) * 1000
        return {}, report                                                        # original behavior: no crash

    # ── 2. Split columns (original logic preserved) ─────────────────────────
    numeric_df     = df.select_dtypes(include=["number"])                        # original
    categorical_df = df.select_dtypes(exclude=["number"])

    logger.info(
        "Columns split — Numeric: %d, Categorical: %d.",
        len(numeric_df.columns), len(categorical_df.columns)
    )

    charts: dict[str, Any] = {}

    # ── 3. Generate each chart type independently ───────────────────────────
    
    # Chart 1: Correlation Heatmap (original)
    heatmap = _generate_correlation_heatmap(numeric_df, report)
    if heatmap:
        charts["correlation_heatmap"] = heatmap                                  # original key

    # Chart 2: Distribution Histograms (original)
    distributions = _generate_distribution_plots(numeric_df, report)
    if distributions:
        charts["distributions"] = distributions                                  # original key

    # Chart 3: Bar Charts (original logic, key aligned to pdf_report.py)
    bar_charts = _generate_bar_charts(df, categorical_df, report)
    if bar_charts:
        charts["bar_charts"] = bar_charts                                        # aligned key

    # Chart 4: Pie Charts (Stricter limits based on critique)
    pie_charts = _generate_pie_charts(categorical_df, report)
    if pie_charts:
        charts["pie_charts"] = pie_charts

    # Chart 5: Trend Charts (Enhanced: Check ALL datetime columns)
    trend_charts = _generate_trend_charts(df, numeric_df, report)
    if trend_charts:
        charts["trend_charts"] = trend_charts

    # ── 4. Finalize ──────────────────────────────────────────────────────────
    report.timing_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "═══ Chart Generation Complete — %d image(s) in %.2f ms | Types: %s | Failures: %d ═══",
        report.total_images,
        report.timing_ms,
        report.charts_generated,
        len(report.charts_failed),
    )
    return charts, report