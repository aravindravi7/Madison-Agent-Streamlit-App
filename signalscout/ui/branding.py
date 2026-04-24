"""Brand lockup helpers.

Renders the SignalScout logo where it belongs:
- Sidebar top (above Settings)
- Page header (st.title replacement)

Email uses a text+emoji fallback inside a dark container because SVG
doesn't render in Gmail / Outlook — see ``email_brief.py``.

If the SVG asset is missing at import time (packaged install without
assets), views fall back to a text+📡 lockup so the app never breaks.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

__all__ = [
    "render_sidebar_lockup",
    "render_page_header",
    "LOCKUP_SVG_PATH",
    "MARK_SVG_PATH",
    "have_svg_assets",
]

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LOCKUP_SVG_PATH = _ASSETS_DIR / "signalscout_lockup.svg"
MARK_SVG_PATH = _ASSETS_DIR / "signalscout_mark.svg"


def have_svg_assets() -> bool:
    """True when both SVG files are present on disk."""
    return LOCKUP_SVG_PATH.exists() and MARK_SVG_PATH.exists()


def _svg_data_uri(path: Path) -> str:
    """Base64-encode an SVG file as a data: URI for use in an <img src>."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def render_sidebar_lockup() -> None:
    """SVG lockup at the top of the sidebar. Falls back to text+📡 if missing.

    The packaged SVG is the cropped lockup (~2.5:1 aspect) — 56px tall
    renders legibly in a 280px sidebar without taking over the column.
    """
    if LOCKUP_SVG_PATH.exists():
        uri = _svg_data_uri(LOCKUP_SVG_PATH)
        st.sidebar.markdown(
            f'<div style="margin:0 0 14px 0;">'
            f'<img src="{uri}" alt="SignalScout" style="height:56px;display:block;" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _text_lockup(st.sidebar, font_size=28)


def render_page_header() -> None:
    """Page title lockup using the SVG. Falls back to text+emoji if missing.

    96px tall against the cropped wide aspect reads at a glance from
    demo-distance without crowding the tagline line beneath.
    """
    if LOCKUP_SVG_PATH.exists():
        uri = _svg_data_uri(LOCKUP_SVG_PATH)
        st.markdown(
            f'<div style="margin:0 0 4px 0;">'
            f'<img src="{uri}" alt="SignalScout" style="height:96px;display:block;" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
            'font-size:56px;line-height:1.1;margin:0 0 6px 0;color:#EAEAEA;">'
            '<span style="color:#00F5D4;margin-right:10px;">📡</span>SignalScout</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:500;'
        'font-size:16px;color:#9CA3AF;margin:-4px 0 4px 0;">'
        'Your taste, on autopilot.</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-weight:400;'
        'font-size:14px;color:rgba(234,234,234,0.6);margin:4px 0 12px 0;'
        'max-width:720px;line-height:1.5;">'
        'SignalScout learns what matters to you, then decides what\'s worth your attention.'
        '</div>',
        unsafe_allow_html=True,
    )


def _text_lockup(container, *, font_size: int) -> None:
    container.markdown(
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;'
        f'font-size:{font_size}px;line-height:1.1;margin-bottom:10px;color:#EAEAEA;">'
        f'<span style="color:#00F5D4;margin-right:6px;">📡</span>SignalScout</div>',
        unsafe_allow_html=True,
    )
