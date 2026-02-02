from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def get_custom_styles():
    """
    Define and return the custom stylesheet for the PDF report.
    Follows a strict 'Modern/Tailwind-ish' design system.
    """
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
