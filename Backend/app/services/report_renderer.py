import io
import datetime
import logging
from typing import Dict, Any, Tuple, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

from app.services.report_styles import get_custom_styles
from app.services.report_components import (
    build_stat_table, create_chart_section,
    build_correlations_table, build_outlier_table,
)

logger = logging.getLogger(__name__)

def generate_pdf_report(
    analysis_data: Dict[str, Any],
    charts_data: Dict[str, Any],
    filename: str
) -> Tuple[io.BytesIO, Dict[str, Any]]:
    """
    Generates a comprehensive PDF report using ReportLab Platypus.
    Includes all chart types: heatmap, distributions, bar charts, boxplots,
    scatter plot, donut chart.
    Includes data tables: correlations, outliers, ranked insights.
    Returns (pdf_buffer, metadata).
    """
    start_time = datetime.datetime.now()
    styles = get_custom_styles()
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    story = []
    
    # ─── 1. Header ──────────────────────────────────────────────────────────
    story.append(Paragraph("Analysis Report", styles['ModernTitle']))
    story.append(Paragraph(f"<b>Dataset:</b> {filename}", styles['ModernBody']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['ModernBody']))
    story.append(Spacer(1, 20))
    
    # ─── 2. Executive Summary (Grid) ────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles['ModernHeading']))
    
    # Extract stats considering multiple potential formats
    meta = analysis_data.get("metadata", {})
    rows = meta.get("rows", meta.get("total_rows", analysis_data.get("info", {}).get("rows", "N/A")))
    cols_count = meta.get("total_columns", 0)
    
    # Handle list vs int for counts
    def get_count(val):
        if isinstance(val, int): return val
        if isinstance(val, list): return len(val)
        return 0

    num_count = get_count(meta.get("numeric_columns", []))
    cat_count = get_count(meta.get("categorical_columns", []))
    
    stat_data = [
        [
            Paragraph(str(rows), styles['MetaValue']),
            Paragraph(str(cols_count), styles['MetaValue']),
            Paragraph(str(num_count), styles['MetaValue']),
            Paragraph(str(cat_count), styles['MetaValue'])
        ],
        [
            Paragraph("Rows", styles['MetaLabel']),
            Paragraph("Columns", styles['MetaLabel']),
            Paragraph("Numeric", styles['MetaLabel']),
            Paragraph("Categorical", styles['MetaLabel'])
        ]
    ]
    
    story.append(build_stat_table(stat_data))
    story.append(Spacer(1, 20))

    # ─── 2b. Cleaning Report ────────────────────────────────────────────────
    cleaning = analysis_data.get("cleaning_report", {})
    if cleaning and cleaning.get("total_changes", 0) > 0:
        story.append(Paragraph("Data Cleaning Actions", styles['ModernHeading']))
        
        clean_text = []
        if cleaning.get("empty_rows_dropped", 0) > 0:
            clean_text.append(f"• Dropped {cleaning['empty_rows_dropped']} empty rows.")
        if cleaning.get("duplicate_rows_removed", 0) > 0:
            clean_text.append(f"• Removed {cleaning['duplicate_rows_removed']} duplicate rows.")
        if cleaning.get("numeric_nans_filled", 0) > 0:
            clean_text.append(f"• Filled {cleaning['numeric_nans_filled']:,} missing numeric values.")
        if cleaning.get("categorical_nans_filled", 0) > 0:
            clean_text.append(f"• Filled {cleaning['categorical_nans_filled']:,} missing categorical values.")
        if cleaning.get("columns_renamed"):
            clean_text.append(f"• Standardized {len(cleaning['columns_renamed'])} column names.")
            
        if clean_text:
            story.append(Paragraph("<br/>".join(clean_text), styles['ModernBody']))
            story.append(Spacer(1, 20))

    # ─── 3. Key Correlations Table ──────────────────────────────────────────
    strong_corrs = analysis_data.get("strong_correlations", [])
    if strong_corrs:
        story.append(Paragraph("Key Correlations", styles['ModernHeading']))
        story.append(Paragraph(
            "Pairs of columns with strong linear relationships (|r| ≥ 0.7).",
            styles['ModernBody']
        ))
        story.append(Spacer(1, 8))
        
        corr_table = build_correlations_table(strong_corrs, styles)
        if corr_table:
            story.append(corr_table)
            story.append(Spacer(1, 20))

    # ─── 4. Outlier Summary Table ───────────────────────────────────────────
    outliers = analysis_data.get("outliers", {})
    if outliers:
        story.append(Paragraph("Outlier Summary (IQR Method)", styles['ModernHeading']))
        story.append(Paragraph(
            "Values beyond 1.5× the interquartile range.",
            styles['ModernBody']
        ))
        story.append(Spacer(1, 8))
        
        outlier_table = build_outlier_table(outliers, styles)
        if outlier_table:
            story.append(outlier_table)
            story.append(Spacer(1, 20))

    # ─── 5. Top Insights ────────────────────────────────────────────────────
    ranked_insights = analysis_data.get("ranked_insights", [])
    if ranked_insights:
        story.append(Paragraph("Top Data Insights", styles['ModernHeading']))
        
        insight_items = []
        for idx, insight in enumerate(ranked_insights[:5]):  # Top 5
            if isinstance(insight, dict):
                title = insight.get("title", insight.get("type", f"Insight {idx+1}"))
                desc = insight.get("description", insight.get("detail", ""))
                score = insight.get("score", insight.get("importance", ""))
                
                text = f"<b>{idx+1}. {title}</b>"
                if desc:
                    text += f"<br/>{desc}"
                if score:
                    text += f" <i>(Score: {score})</i>"
                insight_items.append(text)
            elif isinstance(insight, str):
                insight_items.append(f"<b>{idx+1}.</b> {insight}")
        
        if insight_items:
            story.append(Paragraph("<br/><br/>".join(insight_items), styles['InsightBox']))
            story.append(Spacer(1, 20))

    # ─── 6. AI-Powered Insights ─────────────────────────────────────────────
    insights = analysis_data.get("insights", {})
    insights_text = ""
    if isinstance(insights, dict):
         insights_text = insights.get('insights_text', insights.get('response', ''))
    elif isinstance(insights, str):
         insights_text = insights

    if insights_text and "unavailable" not in insights_text.lower():
        story.append(Paragraph("AI-Powered Insights", styles['ModernHeading']))
        formatted_text = insights_text.replace('\n', '<br/>')
        story.append(Paragraph(formatted_text, styles['InsightBox']))
        story.append(Spacer(1, 20))

    # ─── 7. Visualizations ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Deep Dive: Key Visualizations", styles['ModernTitle']))
    story.append(Paragraph(
        "Each chart below tells a specific story about your data. "
        "Read the narrative caption to understand what the visualization reveals.",
        styles['ModernBody']
    ))
    story.append(Spacer(1, 15))

    charts = charts_data or {}

    def _add_chart_with_narrative(chart_data, title, styles, story):
        """Render a chart with its narrative caption."""
        if not chart_data:
            return
        # Handle both plain string (legacy) and dict with narrative
        if isinstance(chart_data, str):
            image_b64 = chart_data
            narrative = None
        elif isinstance(chart_data, dict):
            image_b64 = chart_data.get("image")
            narrative = chart_data.get("narrative")
        else:
            return
        
        if not image_b64:
            return
        
        create_chart_section(image_b64, title, styles, story)
        if narrative:
            story.append(Paragraph(f"<i>{narrative}</i>", styles['InsightBox']))
            story.append(Spacer(1, 15))
    
    # 7a. Correlation Heatmap
    _add_chart_with_narrative(
        charts.get('correlation_heatmap'),
        "Correlation Heatmap", styles, story
    )
    
    # 7b. Scatter Plot (top correlated pair)
    scatter = charts.get("scatter_plot")
    if scatter and isinstance(scatter, dict):
        _add_chart_with_narrative(
            scatter,
            f"Relationship: {scatter.get('columns', '')}",
            styles, story
        )
    
    # 7c. Distribution Histograms
    for dist in charts.get('distributions', []):
        _add_chart_with_narrative(
            dist,
            f"Distribution of {dist.get('column', '')}",
            styles, story
        )
    
    # 7d. Bar Charts (Categorical)
    bar_charts = charts.get("bar_charts", [])
    if bar_charts:
        story.append(PageBreak())
        story.append(Paragraph("Categorical Analysis", styles['ModernTitle']))
        story.append(Spacer(1, 10))
        for bar in bar_charts:
            _add_chart_with_narrative(
                bar,
                f"Category Breakdown: {bar.get('column', '')}",
                styles, story
            )
    
    # 7e. Donut Chart
    donut = charts.get("donut_chart")
    if donut and isinstance(donut, dict):
        _add_chart_with_narrative(
            donut,
            f"Composition: {donut.get('column', '')}",
            styles, story
        )
    
    # 7f. Boxplots (Bivariate)
    boxplots = charts.get("boxplots", [])
    if boxplots:
        story.append(PageBreak())
        story.append(Paragraph("Bivariate Deep Dive", styles['ModernTitle']))
        story.append(Spacer(1, 10))
        for box in boxplots:
            _add_chart_with_narrative(
                box,
                f"Comparison: {box.get('column', '')}",
                styles, story
            )

    # ─── Build ──────────────────────────────────────────────────────────────
    try:
        doc.build(story)
        buffer.seek(0)
        
        file_size = buffer.getbuffer().nbytes
        metadata = {
            "filename": f"Report_{filename}.pdf",
            "size_bytes": file_size,
            "generated_at": start_time.isoformat(),
            "engine": "ReportLab Platypus",
            "sections": _count_sections(charts, analysis_data),
        }
        return buffer, metadata
        
    except Exception as e:
        logger.error(f"Platypus build failed: {str(e)}")
        raise


def _count_sections(charts: dict, analysis_data: dict) -> dict:
    """Count what sections were included in the report."""
    counts = {}
    for key, val in charts.items():
        if isinstance(val, list):
            counts[key] = len(val)
        elif val:
            counts[key] = 1
    counts["correlations_table"] = 1 if analysis_data.get("strong_correlations") else 0
    counts["outlier_table"] = 1 if analysis_data.get("outliers") else 0
    counts["ranked_insights"] = len(analysis_data.get("ranked_insights", []))
    return counts
