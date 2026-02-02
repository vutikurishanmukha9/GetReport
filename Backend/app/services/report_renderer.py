import io
import base64
import datetime
import logging
from typing import Dict, Any, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

# ─── Style Definitions (Tailwind-ish Look) ───────────────────────────────────
def get_custom_styles():
    styles = getSampleStyleSheet()
    
    # Title - H1 equivalent
    styles.add(ParagraphStyle(
        name='ModernTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        leading=30,
        spaceAfter=20,
        borderPadding=10,
        borderColor=colors.HexColor('#3b82f6'),
        borderWidth=0,
        borderBottomWidth=2,
    ))
    
    # Subtitle - H2 equivalent
    styles.add(ParagraphStyle(
        name='ModernHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=20,
        spaceAfter=10,
        leading=22
    ))
    
    # Text
    styles.add(ParagraphStyle(
        name='ModernBody',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        leading=16,
        spaceAfter=10
    ))

    # Metadata Label
    styles.add(ParagraphStyle(
        name='MetaLabel',
        fontSize=9,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER
    ))
    
    # Metadata Value
    styles.add(ParagraphStyle(
        name='MetaValue',
        fontSize=14,
        textColor=colors.HexColor('#1e3a8a'),
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    ))
    
    # Insight Box
    styles.add(ParagraphStyle(
        name='InsightBox',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1e40af'),
        backColor=colors.HexColor('#eff6ff'),
        borderColor=colors.HexColor('#3b82f6'),
        borderPadding=15,
        borderWidth=0,
        borderLeftWidth=4,
        leading=16,
    ))
    
    return styles

def generate_pdf_report(
    analysis_data: Dict[str, Any],
    charts_data: Dict[str, Any],
    filename: str
) -> Tuple[io.BytesIO, Dict[str, Any]]:
    """
    Generates a PDF file using ReportLab Platypus.
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
    
    # Safely extract stats
    meta = analysis_data.get("metadata", {})
    rows = meta.get("rows", analysis_data.get("info", {}).get("rows", "N/A"))
    cols_count = meta.get("total_columns", 0)
    num_count = len(meta.get("numeric_columns", []))
    cat_count = len(meta.get("categorical_columns", []))
    
    # Create a 4-column table for stats
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
    
    t = Table(stat_data, colWidths=[1.5*inch]*4)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f3f4f6')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [8, 8, 8, 8]), # ReportLab 4.x feature
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # ─── 3. AI Insights ─────────────────────────────────────────────────────
    insights = analysis_data.get("insights", {})
    if insights and "response" in insights:
        story.append(Paragraph("AI-Powered Insights", styles['ModernHeading']))
        # Clean specific characters if needed
        text = insights['response'].replace('\n', '<br/>')
        story.append(Paragraph(text, styles['InsightBox']))
        story.append(Spacer(1, 20))

    # ─── 4. Visualizations ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Key Visualizations", styles['ModernTitle']))

    # Helper to add base64 image
    def add_chart(b64_str, title):
        if not b64_str:
            return
        
        story.append(Paragraph(title, styles['ModernHeading']))
        
        img_buffer = io.BytesIO(base64.b64decode(b64_str))
        img = Image(img_buffer)
        
        # Resize to fit A4 width (approx 6 inches usable)
        max_width = 6 * inch
        aspect = img.imageHeight / float(img.imageWidth)
        img.drawWidth = max_width
        img.drawHeight = max_width * aspect
        
        story.append(img)
        story.append(Spacer(1, 20))

    charts = charts_data or {}
    add_chart(charts.get('correlation_heatmap'), "Correlation Heatmap")
    
    for dist in charts.get('distributions', []):
        add_chart(dist.get('image'), f"Distribution: {dist.get('column')}")

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
