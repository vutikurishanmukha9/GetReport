from reportlab.lib import colors

# ─── Brand Color Palette ─────────────────────────────────────────────────────
# Consistent across every table, header, and accent in GetReport
class Brand:
    DARK_BG        = colors.HexColor("#722F37")   # Deep Burgundy title page background
    ACCENT         = colors.HexColor("#722F37")   # Headings, header bar, primary accent
    ACCENT_LIGHT   = colors.HexColor("#A84351")   # Sub-accents & secondary highlights
    TABLE_HEADER   = colors.HexColor("#722F37")   # Table header row background
    TABLE_ROW_ALT  = colors.HexColor("#FAF6F0")   # Warm Parchment alternating row fill
    TABLE_ROW      = colors.white                 # Default row background
    TEXT_DARK      = colors.HexColor("#1A1A1A")   # High-contrast charcoal body text
    TEXT_MUTED     = colors.HexColor("#555555")   # Muted labels & captions
    TEXT_LIGHT     = colors.white                 # Text on dark backgrounds
    DIVIDER        = colors.HexColor("#E5E0D8")   # Subtle warm divider line
    INSIGHT_BG     = colors.HexColor("#FFFDF9")   # Warm cream insight box background
    INSIGHT_BORDER = colors.HexColor("#722F37")   # Burgundy left accent border
    WARNING_BG     = colors.HexColor("#FFFBEB")   # Quality flag warning background
    WARNING_BORDER = colors.HexColor("#F59E0B")   # Quality flag warning border
