from reportlab.lib import colors

# ═══════════════════════════════════════════════════════════════════════════════
# GetReport — Unified Light Mode Design System Theme Specification
# Synthesized from Vercel Geist, Linear, and Stripe Developer Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

class Brand:
    # ── Canvas & Surfaces ───────────────────────────────────────────────────────
    CANVAS            = colors.HexColor("#FFFFFF")   # Pure crisp white canvas
    SURFACE           = colors.HexColor("#FAFAFA")   # Elevated card surface (zinc-50)
    SURFACE_WELL      = colors.HexColor("#F4F4F6")   # Recessed well / subdued container (zinc-100)
    DARK_BG           = colors.HexColor("#09090B")   # Obsidian 950 executive cover / dark surface

    # ── Typography & Hierarchy ──────────────────────────────────────────────────
    PRIMARY           = colors.HexColor("#09090B")   # Obsidian primary headings (zinc-950)
    PRIMARY_MUTED     = colors.HexColor("#27272A")   # Secondary text (zinc-800)
    TEXT_DARK         = colors.HexColor("#09090B")   # Body copy (high-contrast obsidian)
    TEXT_MUTED        = colors.HexColor("#71717A")   # Muted labels (zinc-500, accessible WCAG 4.8:1)
    TEXT_LIGHT        = colors.white                 # Inverted text on dark chips
    ACCENT            = colors.HexColor("#09090B")   # Obsidian headings & structural accents
    ACCENT_LIGHT      = colors.HexColor("#4F46E5")   # Technical indigo accent
    ACCENT_INDIGO     = colors.HexColor("#4F46E5")   # Linear/Stripe signature indigo

    # ── Structural Hairline Boundaries ──────────────────────────────────────────
    BORDER            = colors.HexColor("#E4E4E7")   # 1px hairline boundary (zinc-200)
    BORDER_SUBTLE     = colors.HexColor("#F4F4F6")   # Interior grid divider (zinc-100)
    DIVIDER           = colors.HexColor("#E4E4E7")   # Section separator hairline

    # ── Tables (Tabular Filings Style) ──────────────────────────────────────────
    TABLE_HEADER      = colors.HexColor("#F4F4F6")   # Recessed light zinc header background
    TABLE_HEADER_TEXT = colors.HexColor("#27272A")   # Charcoal/Obsidian header text
    TABLE_ROW_ALT     = colors.HexColor("#FAFAFA")   # Ultra-light alternating row fill
    TABLE_ROW         = colors.white                 # Pure white default row

    # ── Insights & Callouts ─────────────────────────────────────────────────────
    INSIGHT_BG        = colors.HexColor("#FAFAFA")   # Clean card background
    INSIGHT_BORDER    = colors.HexColor("#4F46E5")   # Indigo technical accent bar

    # ── Semantic Tints (10% Alpha Containers + AA Contrast Text) ────────────────
    SUCCESS_BG        = colors.HexColor("#ECFDF5")   # Emerald 50
    SUCCESS_BORDER    = colors.HexColor("#A7F3D0")   # Emerald 200
    SUCCESS_TEXT      = colors.HexColor("#047857")   # Emerald 700
    
    WARNING_BG        = colors.HexColor("#FFFBEB")   # Amber 50
    WARNING_BORDER    = colors.HexColor("#FDE68A")   # Amber 200
    WARNING_TEXT      = colors.HexColor("#B45309")   # Amber 700

    DANGER_BG         = colors.HexColor("#FFF1F2")   # Rose 50
    DANGER_BORDER     = colors.HexColor("#FECDD3")   # Rose 200
    DANGER_TEXT       = colors.HexColor("#BE123C")   # Rose 700

    INFO_BG           = colors.HexColor("#F0F9FF")   # Sky 50
    INFO_BORDER       = colors.HexColor("#BAE6FD")   # Sky 200
    INFO_TEXT         = colors.HexColor("#0369A1")   # Sky 700
