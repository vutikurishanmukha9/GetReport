import io
import base64
import logging
from typing import Optional, List, Any, Dict

from reportlab.lib import colors
from reportlab.lib.units import inch, mm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import ParagraphStyle
from app.services.theme import Brand

logger = logging.getLogger(__name__)


def build_stat_table(stat_data: List[List[Any]]) -> Table:
    """
    Constructs the 4-column executive summary grid.
    """
    t = Table(stat_data, colWidths=[1.5*inch]*4)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eef2ff')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [8, 8, 8, 8]), 
    ]))
    return t


def create_chart_section(
    b64_str: Optional[str], 
    title: str, 
    styles: dict, 
    story: List[Any]
) -> None:
    """
    Decodes a base64 image and appends it to the story with a title.
    Handles errors gracefully (skips if invalid).
    """
    if not b64_str:
        return
    
    story.append(Paragraph(title, styles['ModernHeading']))
    
    try:
        img_buffer = io.BytesIO(base64.b64decode(b64_str))
        img = Image(img_buffer)
        
        # Resize to fit A4 width (approx 6 inches usable)
        max_width = 6 * inch
        aspect = img.imageHeight / float(img.imageWidth)
        img.drawWidth = max_width
        img.drawHeight = max_width * aspect
        
        # Cap height to avoid page overflow
        max_height = 4.5 * inch
        if img.drawHeight > max_height:
            img.drawHeight = max_height
            img.drawWidth = max_height / aspect
        
        story.append(img)
        story.append(Spacer(1, 15))
    except Exception as e:
        logger.warning(f"Error decoding chart '{title}': {e}")
        story.append(Paragraph(f"[Chart unavailable: {title}]", styles['ModernBody']))


def build_correlations_table(strong_correlations: List[Dict], styles: dict) -> Optional[Table]:
    """
    Build a styled table showing strong correlation pairs.
    Returns None if no data.
    """
    if not strong_correlations:
        return None
    
    # Header row
    header = [
        Paragraph("<b>Column A</b>", styles['ModernBody']),
        Paragraph("<b>Column B</b>", styles['ModernBody']),
        Paragraph("<b>Correlation (r)</b>", styles['ModernBody']),
        Paragraph("<b>Strength</b>", styles['ModernBody']),
    ]
    
    rows = [header]
    for pair in strong_correlations[:10]:  # Limit to top 10
        col1 = pair.get("col1", pair.get("column_a", ""))
        col2 = pair.get("col2", pair.get("column_b", ""))
        r = pair.get("correlation", pair.get("r", 0))
        
        abs_r = abs(r) if isinstance(r, (int, float)) else 0
        if abs_r >= 0.9:
            strength = "Very Strong"
        elif abs_r >= 0.7:
            strength = "Strong"
        else:
            strength = "Moderate"
        
        rows.append([
            Paragraph(str(col1), styles['ModernBody']),
            Paragraph(str(col2), styles['ModernBody']),
            Paragraph(f"{r:.3f}" if isinstance(r, (int, float)) else str(r), styles['ModernBody']),
            Paragraph(strength, styles['ModernBody']),
        ])
    
    if len(rows) <= 1:
        return None
    
    t = Table(rows, colWidths=[1.5*inch, 1.5*inch, 1.2*inch, 1.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), Brand.TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,0), Brand.TEXT_LIGHT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,-1), Brand.TABLE_ROW),
        ('GRID', (0,0), (-1,-1), 0.5, Brand.DIVIDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    # Alternating rows
    for i in range(2, len(rows), 2):
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, i), (-1, i), Brand.TABLE_ROW_ALT),
        ]))
    
    return t


def build_outlier_table(outliers: Dict[str, Any], styles: dict) -> Optional[Table]:
    """
    Build a styled table showing outlier summary per column.
    Returns None if no data.
    """
    if not outliers:
        return None
    
    header = [
        Paragraph("<b>Column</b>", styles['ModernBody']),
        Paragraph("<b>Outlier Count</b>", styles['ModernBody']),
        Paragraph("<b>Percentage</b>", styles['ModernBody']),
        Paragraph("<b>Lower Bound</b>", styles['ModernBody']),
        Paragraph("<b>Upper Bound</b>", styles['ModernBody']),
    ]
    
    rows = [header]
    for col_name, data in outliers.items():
        if isinstance(data, dict):
            count = data.get("count", data.get("outlier_count", 0))
            pct = data.get("percentage", data.get("outlier_pct", 0))
            lower = data.get("lower_bound", data.get("iqr_lower", "N/A"))
            upper = data.get("upper_bound", data.get("iqr_upper", "N/A"))
            
            if count and count > 0:
                rows.append([
                    Paragraph(str(col_name), styles['ModernBody']),
                    Paragraph(f"{count:,}" if isinstance(count, int) else str(count), styles['ModernBody']),
                    Paragraph(f"{pct:.1f}%" if isinstance(pct, (int, float)) else str(pct), styles['ModernBody']),
                    Paragraph(f"{lower:,.1f}" if isinstance(lower, (int, float)) else str(lower), styles['ModernBody']),
                    Paragraph(f"{upper:,.1f}" if isinstance(upper, (int, float)) else str(upper), styles['ModernBody']),
                ])
    
    if len(rows) <= 1:
        return None
    
    t = Table(rows, colWidths=[1.3*inch, 1.0*inch, 0.9*inch, 1.1*inch, 1.1*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), Brand.TABLE_HEADER),
        ('TEXTCOLOR', (0,0), (-1,0), Brand.TEXT_LIGHT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('BACKGROUND', (0,1), (-1,-1), Brand.TABLE_ROW),
        ('GRID', (0,0), (-1,-1), 0.5, Brand.DIVIDER),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    for i in range(2, len(rows), 2):
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, i), (-1, i), Brand.TABLE_ROW_ALT),
        ]))
    
    return t
