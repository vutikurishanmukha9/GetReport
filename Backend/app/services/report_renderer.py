import io
import datetime
import logging
from typing import Dict, Any, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

from app.services.report_styles import get_custom_styles
from app.services.report_components import build_stat_table, create_chart_section

logger = logging.getLogger(__name__)

def generate_pdf_report(
    analysis_data: Dict[str, Any],
    charts_data: Dict[str, Any],
    filename: str
) -> Tuple[io.BytesIO, Dict[str, Any]]:
    """
    Generates a PDF file using ReportLab Platypus.
    Orchestrates the 'Story' using components and styles.
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
            clean_text.append(f"• Filled {cleaning['numeric_nans_filled']} missing numeric values.")
        if cleaning.get("categorical_nans_filled", 0) > 0:
            clean_text.append(f"• Filled {cleaning['categorical_nans_filled']} missing categorical values.")
            
        if clean_text:
            story.append(Paragraph("<br/>".join(clean_text), styles['ModernBody']))
            story.append(Spacer(1, 20))

    # ─── 3. AI Insights ─────────────────────────────────────────────────────
    insights = analysis_data.get("insights", {})
    # Support both old string format and new InsightResult dict format
    insights_text = ""
    if isinstance(insights, dict):
         insights_text = insights.get('insights_text', insights.get('response', ''))
    elif isinstance(insights, str):
         insights_text = insights

    if insights_text:
        story.append(Paragraph("AI-Powered Insights", styles['ModernHeading']))
        formatted_text = insights_text.replace('\n', '<br/>')
        story.append(Paragraph(formatted_text, styles['InsightBox']))
        story.append(Spacer(1, 20))

    # ─── 4. Visualizations ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Key Visualizations", styles['ModernTitle']))

    charts = charts_data or {}
    
    create_chart_section(charts.get('correlation_heatmap'), "Correlation Heatmap", styles, story)
    
    for dist in charts.get('distributions', []):
        create_chart_section(dist.get('image'), f"Distribution: {dist.get('column')}", styles, story)

    # ─── Build ──────────────────────────────────────────────────────────────
    try:
        doc.build(story)
        buffer.seek(0)
        
        file_size = buffer.getbuffer().nbytes
        metadata = {
            "filename": f"Report_{filename}.pdf",
            "size_bytes": file_size,
            "generated_at": start_time.isoformat(),
            "engine": "ReportLab Platypus"
        }
        return buffer, metadata
        
    except Exception as e:
        logger.error(f"Platypus build failed: {str(e)}")
        raise
