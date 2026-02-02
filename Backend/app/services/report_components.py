import io
import base64
from typing import Optional, List, Any, Dict

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import StyleSheet1

def build_stat_table(stat_data: List[List[Any]]) -> Table:
    """
    Constructs the 4-column executive summary grid.
    """
    t = Table(stat_data, colWidths=[1.5*inch]*4)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f3f4f6')),
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
    styles: StyleSheet1, 
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
        
        story.append(img)
        story.append(Spacer(1, 20))
    except Exception as e:
        # Graceful fallback - just log/print and skip image
        print(f"Error decoding chart '{title}': {e}")
        story.append(Paragraph(f"[Error displaying chart: {title}]", styles['ModernBody']))
