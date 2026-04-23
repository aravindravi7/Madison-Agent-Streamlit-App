"""Brand theme injection.

Loads SignalScout's type system from Google Fonts and applies brand CSS:
- Space Grotesk (500, 700) for headings
- Inter (400, 500, 600) for body
- JetBrains Mono (400) for numbers, timestamps, "code-feeling" text

Called once from ``app.main()``. Dark base and color tokens are locked in
``.streamlit/config.toml`` — this file handles typography + small layout
polish only. Colors here MUST match the tokens used elsewhere
(signalscout/ui/components.py, learning_view.py).
"""

from __future__ import annotations

import streamlit as st

__all__ = ["inject_theme"]


_AQUAMARINE = "#00F5D4"
_INK_BLACK = "#0B0F1A"
_ALABASTER = "#EAEAEA"
_PANEL_BG = "#12172A"

_FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400&"
    "family=Space+Grotesk:wght@500;700&display=swap"
)


def _css() -> str:
    # Single CSS payload. Deliberate scope:
    # - Override font stack for the three families
    # - Tighten sidebar top padding
    # - Aquamarine hover on primary buttons
    # - .mono / .jb-mono class + <code> → JetBrains Mono
    # - h1/h2/h3 → Space Grotesk 700
    # Nothing here changes colors (those come from config.toml) or layout
    # structure. Anything that looks wrong on the deployed app should be
    # corrected here, not in individual view modules.
    return f"""
<style>
    /* Google Fonts */
    @import url('{_FONT_IMPORT}');

    /* Body text — override Streamlit's default font stack.
       Deliberately NOT !important so inline font-family declarations in
       view modules (e.g. JetBrains Mono on scores) still win. */
    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }}

    /* Subheaders + titles — Streamlit wraps these in specific data-test ids */
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {{
        font-family: 'Space Grotesk', 'Inter', system-ui, sans-serif !important;
    }}

    /* Monospace classes */
    .mono, .jb-mono, code, pre, kbd, samp {{
        font-family: 'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace !important;
    }}

    /* Sidebar: tighten the top gap above Settings */
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1rem;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
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
        color: #9CA3AF !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }}

    /* Metric numbers should be JetBrains Mono so they feel instrument-like */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'Inter', system-ui, sans-serif !important;
        color: #9CA3AF;
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
