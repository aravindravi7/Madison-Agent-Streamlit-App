"""Brand theme injection.

Loads SignalScout's type system from Google Fonts and applies brand CSS:
- Space Grotesk (500, 700) for headings + tab labels
- Inter (400, 500, 600) for body
- JetBrains Mono (400) for numbers, timestamps, "code-feeling" text

Called once from ``app.main()``. Dark base and color tokens are locked in
``.streamlit/config.toml`` — this file handles typography + layout polish.
Colors here MUST match the tokens used elsewhere
(signalscout/ui/components.py, learning_view.py).

Token single-sourcing:
- Aquamarine  = #00F5D4  (primary accent / active tab / primary button)
- Ink Black   = #0B0F1A  (canvas)
- Alabaster   = #EAEAEA  (body text on dark)
- Panel BG    = #12172A  (cards)
- Panel border= rgba(234,234,234,0.1) used for every outer card stroke
- Radius      = 12px on every card/wrapper
"""

from __future__ import annotations

import streamlit as st

__all__ = ["inject_theme", "CARD_RADIUS", "CARD_BORDER", "CARD_BG"]


_AQUAMARINE = "#00F5D4"
_INK_BLACK = "#0B0F1A"
_ALABASTER = "#EAEAEA"
_PANEL_BG = "#12172A"
_MUTED = "#9CA3AF"

# Single-source card tokens — view modules import these to stay in lockstep.
CARD_RADIUS = "12px"
CARD_BORDER = "1px solid rgba(234, 234, 234, 0.1)"
CARD_BG = _PANEL_BG

_PAGE_MAX_WIDTH = "1400px"
_SIDEBAR_WIDTH = "280px"

_EVALUATOR_MIN_HEIGHT = "240px"     # Arena: all three columns equal-height
_LEADERBOARD_MIN_HEIGHT = "180px"   # Learning: all three leaderboard cards match

_FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400&"
    "family=Space+Grotesk:wght@500;600;700&display=swap"
)


def _css() -> str:
    # Single CSS payload. Deliberate scope:
    # - Font families
    # - Sidebar width + tighter padding + lockup area
    # - Page max-width cap
    # - Tab styling (Space Grotesk, Aquamarine active underline)
    # - Arena + leaderboard card min-heights for equal-row layout
    # - Card radius/border/bg tokens applied consistently where needed
    # No color/layout restructuring. Changes to color tokens go in one place.
    return f"""
<style>
    /* Google Fonts */
    @import url('{_FONT_IMPORT}');

    /* Body text — override Streamlit's default stack.
       Deliberately NOT !important so inline font-family declarations in
       view modules (e.g. JetBrains Mono on scores) still win. */
    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}

    /* Headings: Space Grotesk 700 */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {{
        font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }}

    /* Monospace classes + <code>/<pre> */
    .mono, .jb-mono, code, pre, kbd, samp {{
        font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace !important;
    }}

    /* Page max-width: cap main content on ultrawide screens */
    .main .block-container {{
        max-width: {_PAGE_MAX_WIDTH};
        padding-top: 2rem;
    }}

    /* Sidebar: wider, tighter top padding */
    [data-testid="stSidebar"] {{
        min-width: {_SIDEBAR_WIDTH} !important;
        max-width: {_SIDEBAR_WIDTH} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}

    /* Tab styling — Space Grotesk 600, 18px; Aquamarine 3px active underline */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        border-bottom: 1px solid rgba(234, 234, 234, 0.1);
        margin-bottom: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        color: {_ALABASTER} !important;
        opacity: 0.6;
        padding: 10px 6px 12px 6px !important;
        border-bottom: 3px solid transparent !important;
        transition: opacity 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        opacity: 0.85;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {_AQUAMARINE} !important;
        opacity: 1;
        font-weight: 700 !important;
        border-bottom: 3px solid {_AQUAMARINE} !important;
    }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 12px !important;
    }}
    /* Hide the default Streamlit underline/highlight bar below the tabs */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Primary button hover → Aquamarine accent */
    .stButton > button[kind="primary"]:hover,
    .stButton > button:hover {{
        border-color: {_AQUAMARINE} !important;
        color: {_AQUAMARINE} !important;
        transition: border-color 0.15s ease, color 0.15s ease;
    }}
    .stButton > button[kind="primary"] {{
        background: {_AQUAMARINE};
        color: {_INK_BLACK};
        font-weight: 600;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: {_AQUAMARINE};
        color: {_INK_BLACK} !important;
        filter: brightness(1.1);
        border-color: {_AQUAMARINE} !important;
    }}

    /* Captions - subtle muted text */
    [data-testid="stCaptionContainer"], .stCaption {{
        color: {_MUTED} !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }}

    /* Metric numbers in JetBrains Mono */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Inter', system-ui, sans-serif !important;
        color: {_MUTED};
    }}

    /* Arena evaluator panel: equal-height columns via min-height.
       Views apply .arena-panel; theme owns the dimensions. */
    .arena-panel {{
        min-height: {_EVALUATOR_MIN_HEIGHT};
        display: flex;
        flex-direction: column;
    }}

    /* Leaderboard card: equal heights across the three arms */
    .leaderboard-card {{
        min-height: {_LEADERBOARD_MIN_HEIGHT};
        display: flex;
        flex-direction: column;
    }}

    /* Arena item wrapper — groups title + panels + winner row + feedback */
    .arena-item {{
        border: {CARD_BORDER};
        border-radius: {CARD_RADIUS};
        padding: 20px;
        margin-bottom: 24px;
        background: rgba(255, 255, 255, 0.02);
    }}
</style>
    """


def inject_theme() -> None:
    """Inject SignalScout brand CSS into the current Streamlit page.

    Idempotent-ish: calling twice re-injects the block, which browsers
    deduplicate cleanly. Streamlit re-runs the script on each interaction,
    so this is called on every render.
    """
    st.markdown(_css(), unsafe_allow_html=True)
