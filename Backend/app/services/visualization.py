from __future__ import annotations
import logging
import io
import base64
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import semantic column classifier
from app.services.analysis import classify_numeric_columns

logger = logging.getLogger(__name__)

# ─── Brand-Consistent Chart Theme ────────────────────────────────────────────
CHART_DPI = 130
BRAND_PRIMARY   = "#6366f1"
BRAND_SECONDARY = "#4338ca"
BRAND_ACCENT    = "#a5b4fc"
BRAND_BG        = "#fafafe"
BRAND_TEXT       = "#1e1b4b"
BRAND_GRID      = "#e0e7ff"
BRAND_PALETTE   = ["#6366f1", "#8b5cf6", "#a78bfa", "#c084fc", "#e879f9",
                   "#f472b6", "#fb7185", "#f97316", "#facc15", "#34d399"]

# Apply global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.facecolor": BRAND_BG,
    "axes.edgecolor": "#c7d2fe",
    "axes.labelcolor": BRAND_TEXT,
    "axes.titlecolor": BRAND_TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": BRAND_GRID,
    "grid.alpha": 0.5,
    "xtick.color": BRAND_TEXT,
    "ytick.color": BRAND_TEXT,
    "figure.facecolor": "white",
    "figure.dpi": CHART_DPI,
})

def _fig_to_base64(fig) -> str:
    """Convert Matplotlib figure to Base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=CHART_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def generate_charts(df: pl.DataFrame) -> tuple[dict[str, str], list[str]]:
    """
    Generate charts using Polars and pure Matplotlib.
    Filters out ID/date columns for meaningful visualizations.
    Uses Brand color palette for a premium, cohesive look.
    """
    charts = {}
    warnings = []
    
    # Identify columns
    all_numeric_cols = [c for c, t in df.schema.items() if t in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)]
    cat_cols = [c for c in df.columns if c not in all_numeric_cols]
    
    logger.info(f"Visualization: Total columns={df.width}, numeric={len(all_numeric_cols)}, categorical={len(cat_cols)}")
    
    # Filter to only analytical numeric columns (exclude IDs, dates, low-variance)
    column_classification = classify_numeric_columns(df, all_numeric_cols)
    numeric_cols = column_classification["analytical"]
    
    logger.info(f"Visualization: After classification - analytical={len(numeric_cols)}, excluded={len(column_classification.get('excluded', []))}")
    
    if column_classification["excluded"]:
        warnings.append(f"Excluded from charts: {', '.join(column_classification['excluded'])} (ID/date/low-variance columns)")
        logger.info(f"Visualization: Excluded columns {column_classification['excluded']}")
    
    # ── 1. Correlation Heatmap ────────────────────────────────────────────────
    corr_matrix = None  # Keep for scatter plot logic below
    if len(numeric_cols) > 1:
        try:
            target_cols = numeric_cols[:20]
            data_matrix = df.select(target_cols).drop_nulls().to_numpy().T
            
            if data_matrix.shape[1] > 1:
                corr_matrix = np.corrcoef(data_matrix)
                
                # Custom purple-white-orange diverging colormap
                from matplotlib.colors import LinearSegmentedColormap
                brand_cmap = LinearSegmentedColormap.from_list(
                    "brand_corr", ["#6366f1", "#ffffff", "#f97316"]
                )
                
                fig, ax = plt.subplots(figsize=(8, 6.5))
                im = ax.imshow(corr_matrix, cmap=brand_cmap, vmin=-1, vmax=1, aspect="auto")
                
                # Annotations (only for ≤10 columns)
                if len(target_cols) <= 10:
                    for i in range(len(target_cols)):
                        for j in range(len(target_cols)):
                            val = corr_matrix[i, j]
                            color = "white" if abs(val) > 0.6 else BRAND_TEXT
                            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                    color=color, fontsize=9, fontweight="bold")

                ax.set_xticks(np.arange(len(target_cols)))
                ax.set_yticks(np.arange(len(target_cols)))
                ax.set_xticklabels(target_cols, rotation=45, ha="right", fontsize=9)
                ax.set_yticklabels(target_cols, fontsize=9)
                ax.set_title("Correlation Matrix")
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Pearson r", fontsize=9)
                fig.tight_layout()
                
                charts["correlation_heatmap"] = _fig_to_base64(fig)
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")

    # ── 2. Distributions (Histograms with mean/median lines) ──────────────────
    dist_list = []
    for idx, col in enumerate(numeric_cols[:5]):
        try:
            data = df[col].drop_nulls().to_numpy()
            if len(data) == 0: continue
            
            color = BRAND_PALETTE[idx % len(BRAND_PALETTE)]
            fig, ax = plt.subplots(figsize=(6.5, 4))
            
            n, bins, patches = ax.hist(data, bins=25, color=color, edgecolor="white",
                                        alpha=0.85, linewidth=0.5)
            
            # Mean & Median reference lines
            mean_val = float(np.mean(data))
            median_val = float(np.median(data))
            ax.axvline(mean_val, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.1f}")
            ax.axvline(median_val, color="#f97316", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:,.1f}")
            
            ax.set_title(f"Distribution: {col}")
            ax.set_ylabel("Frequency")
            ax.set_xlabel(col)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
            fig.tight_layout()
            
            dist_list.append({"column": col, "image": _fig_to_base64(fig)})
        except Exception as e:
            logger.warning(f"Histogram failed for {col}: {e}")
            
    if dist_list:
        charts["distributions"] = dist_list

    # ── 3. Bar Charts (Categorical Frequency) ────────────────────────────────
    bar_list = []
    plot_cats = [c for c in cat_cols if df[c].n_unique() <= 20]
    for col in plot_cats[:3]:
        try:
            vc = df[col].value_counts(sort=True).head(12)
            labels = [str(l) if l is not None else "Unknown" for l in vc[col].to_list()]
            counts = vc["count"].to_list()
            
            # Gradient using Brand palette
            n = len(labels)
            bar_colors = [BRAND_PALETTE[i % len(BRAND_PALETTE)] for i in range(n)]
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bars = ax.barh(labels[::-1], counts[::-1], color=bar_colors[::-1], edgecolor="white", linewidth=0.5)
            
            # Value labels on bars
            for bar, count in zip(bars, counts[::-1]):
                ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{count:,}", va="center", fontsize=8, color=BRAND_TEXT)
            
            ax.set_title(f"Frequency: {col}")
            ax.set_xlabel("Count")
            ax.invert_yaxis()
            fig.tight_layout()
            
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
                valid_df = df.select([cat_col, num_col]).drop_nulls()
                cats = sorted([str(c) for c in valid_df[cat_col].unique().to_list()])
                
                groups = []
                group_labels = []
                for c in cats:
                    vals = valid_df.filter(pl.col(cat_col) == c)[num_col].to_numpy()
                    if len(vals) > 0:
                        groups.append(vals)
                        group_labels.append(c)
                
                if groups:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    bp = ax.boxplot(groups, tick_labels=group_labels, patch_artist=True,
                                    medianprops=dict(color="#ef4444", linewidth=2),
                                    whiskerprops=dict(color=BRAND_SECONDARY),
                                    capprops=dict(color=BRAND_SECONDARY))
                    
                    # Color each box with brand palette
                    for i, patch in enumerate(bp["boxes"]):
                        patch.set_facecolor(BRAND_PALETTE[i % len(BRAND_PALETTE)])
                        patch.set_alpha(0.7)
                    
                    ax.set_title(f"{num_col} by {cat_col}")
                    ax.set_ylabel(num_col)
                    ax.set_xticklabels(group_labels, rotation=45, ha="right")
                    fig.tight_layout()
                    
                    box_list.append({"column": f"{num_col} vs {cat_col}", "image": _fig_to_base64(fig)})
            except Exception as e:
                logger.warning(f"Boxplot failed for {num_col} vs {cat_col}: {e}")

    if box_list:
        charts["boxplots"] = box_list

    # ── 5. Scatter Plot (Top Correlated Pair) ─────────────────────────────────
    if corr_matrix is not None and len(numeric_cols) >= 2:
        try:
            target_cols = numeric_cols[:20]
            # Find the strongest off-diagonal correlation
            abs_corr = np.abs(corr_matrix)
            np.fill_diagonal(abs_corr, 0)
            max_idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
            col_x, col_y = target_cols[max_idx[0]], target_cols[max_idx[1]]
            r_val = corr_matrix[max_idx[0], max_idx[1]]
            
            scatter_df = df.select([col_x, col_y]).drop_nulls()
            x_data = scatter_df[col_x].to_numpy()
            y_data = scatter_df[col_y].to_numpy()
            
            # Subsample if too many points
            if len(x_data) > 2000:
                idx = np.random.choice(len(x_data), 2000, replace=False)
                x_data, y_data = x_data[idx], y_data[idx]
            
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x_data, y_data, c=BRAND_PRIMARY, alpha=0.4, s=12, edgecolors="none")
            
            # Trend line
            z = np.polyfit(x_data.astype(float), y_data.astype(float), 1)
            p = np.poly1d(z)
            x_line = np.linspace(float(x_data.min()), float(x_data.max()), 100)
            ax.plot(x_line, p(x_line), color="#ef4444", linewidth=2, linestyle="--",
                    label=f"r = {r_val:.3f}")
            
            ax.set_title(f"{col_x} vs {col_y}")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.legend(fontsize=10, loc="upper left", framealpha=0.8)
            fig.tight_layout()
            
            charts["scatter_plot"] = {"columns": f"{col_x} vs {col_y}", "image": _fig_to_base64(fig)}
        except Exception as e:
            logger.warning(f"Scatter plot failed: {e}")

    # ── 6. Donut Chart (Top Categorical Column) ──────────────────────────────
    if cat_cols:
        try:
            # Pick the categorical column with the most information (2-12 unique values)
            donut_candidates = [c for c in cat_cols if 2 <= df[c].n_unique() <= 12]
            if donut_candidates:
                donut_col = donut_candidates[0]
                vc = df[donut_col].value_counts(sort=True).head(8)
                labels = [str(l) if l is not None else "Unknown" for l in vc[donut_col].to_list()]
                sizes = vc["count"].to_list()
                
                # Add "Other" if there are remaining
                total = df.height
                shown_total = sum(sizes)
                if shown_total < total:
                    labels.append("Other")
                    sizes.append(total - shown_total)
                
                n = len(labels)
                pie_colors = [BRAND_PALETTE[i % len(BRAND_PALETTE)] for i in range(n)]
                
                fig, ax = plt.subplots(figsize=(6.5, 5))
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=None, colors=pie_colors, autopct="%1.1f%%",
                    startangle=90, pctdistance=0.78,
                    wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
                )
                
                for t in autotexts:
                    t.set_fontsize(8)
                    t.set_fontweight("bold")
                    t.set_color("white")
                
                ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
                ax.set_title(f"Distribution: {donut_col}")
                fig.tight_layout()
                
                charts["donut_chart"] = {"column": donut_col, "image": _fig_to_base64(fig)}
        except Exception as e:
            logger.warning(f"Donut chart failed: {e}")

    # Log summary of generated charts
    chart_summary = {k: len(v) if isinstance(v, list) else 1 for k, v in charts.items()}
    logger.info(f"Visualization: Generated charts summary: {chart_summary}")
    if not charts:
        logger.warning("Visualization: No charts were generated!")
    
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