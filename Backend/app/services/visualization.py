from __future__ import annotations
import logging
import io
import base64
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Constants
CHART_DPI = 100
MAX_SAMPLES = 5000  # for scatter plots
PIE_MAX = 5

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_charts(df: pl.DataFrame) -> tuple[dict[str, str], list[str]]:
    """
    Generate charts using Polars for data prep and Matplotlib/Seaborn for rendering.
    """
    charts = {}
    warnings = []
    
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    
    # 1. Correlation Heatmap
    if len(numeric_cols) > 1:
        try:
            # We can use the already computed correlation or recompute
            # For visualization, we need a Pandas Correlation Matrix
            # Convert numeric columns to pandas (careful with size)
            # If rows > 10000 -> Sample??
            # Correlation is sensitive to sampling. But 50MB file fits in RAM for pandas if we only take numeric cols.
            # Polars to_pandas() is zero-copy in some cases via Arrow.
            
            pdf = df.select(numeric_cols).head(50000).to_pandas()
            corr = pdf.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Matrix")
            charts["correlation_heatmap"] = _fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")
            
    # 2. Distributions (Histograms)
    dist_list = []
    for col in numeric_cols[:5]: # distinct top 5
        try:
            # Polars -> Pandas for plotting is fine for histogram bins
            # Sample if huge
            series = df[col].sample(n=min(len(df), 10000)).to_pandas()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(series, kde=True, ax=ax, color="#3b82f6")
            ax.set_title(f"Distribution: {col}")
            dist_list.append({"column": col, "image": _fig_to_base64(fig)})
        except Exception as e:
            pass
    if dist_list:
        charts["distributions"] = dist_list
        
    return charts, warnings