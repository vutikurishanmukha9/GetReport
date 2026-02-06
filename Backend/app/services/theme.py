from reportlab.lib import colors

# ─── Brand Color Palette ─────────────────────────────────────────────────────
# Consistent across every table, header, and accent in the report
class Brand:
    DARK_BG       = colors.HexColor("#1e1b4b")   # title page background
    ACCENT        = colors.HexColor("#6366f1")   # headings, header bar
    ACCENT_LIGHT  = colors.HexColor("#a5b4fc")   # sub-accents
    TABLE_HEADER  = colors.HexColor("#4338ca")   # table header row
    TABLE_ROW_ALT = colors.HexColor("#eef2ff")   # alternating row fill
    TABLE_ROW     = colors.white                 # default row
    TEXT_DARK     = colors.HexColor("#1e1b4b")   # body text
    TEXT_LIGHT    = colors.white                 # text on dark backgrounds
    DIVIDER       = colors.HexColor("#c7d2fe")   # horizontal rule color
    INSIGHT_BG    = colors.HexColor("#f0f4ff")   # insight box background
    WARNING_BG    = colors.HexColor("#fef3c7")   # quality flag warning background
    WARNING_BORDER= colors.HexColor("#f59e0b")   # quality flag border
