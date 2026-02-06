from __future__ import annotations
import logging
import io
import base64
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Constants
CHART_DPI = 100

def _fig_to_base64(fig) -> str:
    """Convert Matplotlib figure to Base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_charts(df: pl.DataFrame) -> tuple[dict[str, str], list[str]]:
    """
    Generate charts using Polars and pure Matplotlib (No Pandas/Seaborn).
    """
    charts = {}
    warnings = []
    
    # Identify columns
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    
    # ── 1. Correlation Heatmap ────────────────────────────────────────────────
    if len(numeric_cols) > 1:
        try:
            # Prepare data
            target_cols = numeric_cols[:20] # Limit to 20 for readability
            
            # Use Polars to get numpy array (drop nulls is safer)
            # We must drop rows where ANY of the target cols are null to be safe for corrcoef
            data_matrix = df.select(target_cols).drop_nulls().to_numpy().T
            
            if data_matrix.shape[1] > 1: # at least 2 rows
                corr_matrix = np.corrcoef(data_matrix)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                
                # Annotations
                # Only if dimension is small
                if len(target_cols) <= 10:
                    for i in range(len(target_cols)):
                        for j in range(len(target_cols)):
                            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                                           ha="center", va="center", color="black", fontsize=8)

                # Axis Labels
                ax.set_xticks(np.arange(len(target_cols)))
                ax.set_yticks(np.arange(len(target_cols)))
                ax.set_xticklabels(target_cols, rotation=45, ha="right")
                ax.set_yticklabels(target_cols)
                ax.set_title("Correlation Matrix (Numeric)")
                fig.colorbar(im, ax=ax)
                
                charts["correlation_heatmap"] = _fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")

    # ── 2. Distributions (Histograms) ────────────────────────────────────────
    dist_list = []
    for col in numeric_cols[:5]:
        try:
            # Get data
            data = df[col].drop_nulls().to_numpy()
            if len(data) == 0: continue
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(data, bins=20, color="#3b82f6", edgecolor="white", alpha=0.8)
            
            # Add KDE-like line (Standard Normal approximation or simple smoothing?)
            # Matplotlib doesn't have KDE built-in easily without scipy/seaborn.
            # We stick to Histogram for pure matplotlib speed/simplicity.
            
            ax.set_title(f"Distribution: {col}")
            ax.set_ylabel("Frequency")
            ax.grid(axis='y', alpha=0.3)
            
            dist_list.append({"column": col, "image": _fig_to_base64(fig)})
        except Exception as e:
            logger.warning(f"Histogram failed for {col}: {e}")
            
    if dist_list:
        charts["distributions"] = dist_list

    # ── 3. Bar Charts (Frequency) ────────────────────────────────────────────
    bar_list = []
    plot_cats = [c for c in cat_cols if df[c].n_unique() <= 20]
    for col in plot_cats[:3]:
        try:
            # Polars Value Counts
            vc = df[col].value_counts(sort=True).head(15)
            # vc has columns: col, count
            
            labels = vc[col].to_list()
            counts = vc["count"].to_list()
            
            # Handle None/Null labels
            labels = [str(l) if l is not None else "Unknown" for l in labels]
            
            fig, ax = plt.subplots(figsize=(7, 4))
            # Create colors using a colormap
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
            
            ax.bar(labels, counts, color=colors)
            ax.set_title(f"Frequency: {col}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Count")
            
            bar_list.append({"column": col, "image": _fig_to_base64(fig)})
        except Exception as e:
            logger.warning(f"Bar chart failed for {col}: {e}")

    if bar_list:
        charts["bar_charts"] = bar_list

    # ── 4. Boxplots (Bivariate) ──────────────────────────────────────────────
    box_list = []
    target_cats = [c for c in cat_cols if 2 <= df[c].n_unique() <= 10]
    
    if target_cats and numeric_cols:
        cat_col = target_cats[0]
        
        for num_col in numeric_cols[:3]:
            try:
                # Group data by category
                # Polars group_by
                groups = []
                labels = []
                
                # We need a list of arrays for plt.boxplot
                # Filter nulls
                valid_df = df.select([cat_col, num_col]).drop_nulls()
                
                # Get unique categories
                cats = valid_df[cat_col].unique().to_list()
                cats = sorted([str(c) for c in cats]) # Sort for consistency
                
                for c in cats:
                    # Filter for this category
                    vals = valid_df.filter(pl.col(cat_col) == c)[num_col].to_numpy()
                    if len(vals) > 0:
                        groups.append(vals)
                        labels.append(c)
                
                if groups:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.boxplot(groups, tick_labels=labels, patch_artist=True,
                               boxprops=dict(facecolor="#99d6ff"))
                    ax.set_title(f"{num_col} by {cat_col}")
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.grid(axis='y', alpha=0.3)
                    
                    box_list.append({"column": f"{num_col} vs {cat_col}", "image": _fig_to_base64(fig)})
            except Exception as e:
                logger.warning(f"Boxplot failed for {num_col} vs {cat_col}: {e}")

    if box_list:
        charts["boxplots"] = box_list

    return charts, warnings


# ─── Tier 1 Enhancement: Interactive Charts (Plotly.js JSON) ─────────────────
def generate_interactive_charts(df: pl.DataFrame) -> dict[str, Any]:
    """
    Generate interactive chart specifications as Plotly.js-compatible JSON.
    Frontend can render these directly with Plotly.newPlot().
    
    Returns:
        dict with chart specs: {chart_name: {data: [...], layout: {...}}}
    """
    from typing import Any
    
    charts = {}
    
    # Identify columns
    numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    
    # ── 1. Interactive Histograms ─────────────────────────────────────────────
    histograms = []
    for col in numeric_cols[:5]:
        try:
            data = df[col].drop_nulls().to_list()
            if not data:
                continue
            
            histograms.append({
                "column": col,
                "plotly_spec": {
                    "data": [{
                        "type": "histogram",
                        "x": data[:5000],  # Limit for performance
                        "marker": {"color": "#3b82f6"},
                        "opacity": 0.8,
                        "nbinsx": 30
                    }],
                    "layout": {
                        "title": f"Distribution: {col}",
                        "xaxis": {"title": col},
                        "yaxis": {"title": "Frequency"},
                        "bargap": 0.05,
                        "template": "plotly_white"
                    }
                }
            })
        except Exception as e:
            logger.warning(f"Interactive histogram failed for {col}: {e}")
    
    if histograms:
        charts["histograms"] = histograms
    
    # ── 2. Interactive Scatter Matrix (Top 4 numeric) ─────────────────────────
    if len(numeric_cols) >= 2:
        try:
            scatter_cols = numeric_cols[:4]
            sample_df = df.select(scatter_cols).drop_nulls()
            if sample_df.height > 1000:
                sample_df = sample_df.sample(1000)
            
            # Create scatter matrix spec
            dimensions = []
            for col in scatter_cols:
                dimensions.append({
                    "label": col,
                    "values": sample_df[col].to_list()
                })
            
            charts["scatter_matrix"] = {
                "plotly_spec": {
                    "data": [{
                        "type": "splom",
                        "dimensions": dimensions,
                        "marker": {
                            "color": "#6366f1",
                            "size": 5,
                            "opacity": 0.6
                        },
                        "diagonal": {"visible": True},
                        "showupperhalf": True,
                        "showlowerhalf": True
                    }],
                    "layout": {
                        "title": "Scatter Matrix (Top Numeric Columns)",
                        "template": "plotly_white",
                        "height": 600,
                        "width": 700
                    }
                }
            }
        except Exception as e:
            logger.warning(f"Scatter matrix failed: {e}")
    
    # ── 3. Interactive Bar Charts ─────────────────────────────────────────────
    bar_charts = []
    plot_cats = [c for c in cat_cols if df[c].n_unique() <= 20]
    for col in plot_cats[:3]:
        try:
            vc = df[col].value_counts(sort=True).head(15)
            labels = [str(v) if v is not None else "Unknown" for v in vc[col].to_list()]
            counts = vc["count"].to_list()
            
            bar_charts.append({
                "column": col,
                "plotly_spec": {
                    "data": [{
                        "type": "bar",
                        "x": labels,
                        "y": counts,
                        "marker": {
                            "color": counts,
                            "colorscale": "Viridis"
                        }
                    }],
                    "layout": {
                        "title": f"Frequency: {col}",
                        "xaxis": {"title": col, "tickangle": -45},
                        "yaxis": {"title": "Count"},
                        "template": "plotly_white"
                    }
                }
            })
        except Exception as e:
            logger.warning(f"Interactive bar chart failed for {col}: {e}")
    
    if bar_charts:
        charts["bar_charts"] = bar_charts
    
    # ── 4. Interactive Heatmap (Correlation) ──────────────────────────────────
    if len(numeric_cols) > 1:
        try:
            target_cols = numeric_cols[:15]
            data_matrix = df.select(target_cols).drop_nulls().to_numpy().T
            
            if data_matrix.shape[1] > 1:
                corr_matrix = np.corrcoef(data_matrix)
                # Replace NaN with 0
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                
                charts["correlation_heatmap"] = {
                    "plotly_spec": {
                        "data": [{
                            "type": "heatmap",
                            "z": corr_matrix.tolist(),
                            "x": target_cols,
                            "y": target_cols,
                            "colorscale": "RdBu",
                            "zmin": -1,
                            "zmax": 1,
                            "hoverongaps": False
                        }],
                        "layout": {
                            "title": "Correlation Matrix",
                            "xaxis": {"tickangle": -45},
                            "template": "plotly_white",
                            "height": 500,
                            "width": 600
                        }
                    }
                }
        except Exception as e:
            logger.warning(f"Interactive heatmap failed: {e}")
    
    # ── 5. Interactive Box Plots ──────────────────────────────────────────────
    box_plots = []
    target_cats = [c for c in cat_cols if 2 <= df[c].n_unique() <= 10]
    
    if target_cats and numeric_cols:
        cat_col = target_cats[0]
        
        for num_col in numeric_cols[:2]:
            try:
                valid_df = df.select([cat_col, num_col]).drop_nulls()
                categories = valid_df[cat_col].unique().to_list()
                
                traces = []
                for cat in sorted([str(c) for c in categories])[:8]:
                    vals = valid_df.filter(pl.col(cat_col) == cat)[num_col].to_list()
                    if vals:
                        traces.append({
                            "type": "box",
                            "name": str(cat),
                            "y": vals[:1000]  # Limit
                        })
                
                if traces:
                    box_plots.append({
                        "column": f"{num_col} by {cat_col}",
                        "plotly_spec": {
                            "data": traces,
                            "layout": {
                                "title": f"{num_col} by {cat_col}",
                                "yaxis": {"title": num_col},
                                "template": "plotly_white"
                            }
                        }
                    })
            except Exception as e:
                logger.warning(f"Interactive boxplot failed: {e}")
    
    if box_plots:
        charts["box_plots"] = box_plots
    
    return charts