"""Brand theme injection.

Phase 1: a no-op stub. Dark mode is enabled via ``.streamlit/config.toml``
so we build against a dark canvas from day one.

Phase 4 will:
- Import Space Grotesk / Inter / JetBrains Mono from Google Fonts
- Inject brand CSS (Ink Black, Aquamarine, Azure Blue, Alabaster)
- Style score badges, evaluator cards, winner-glow effect
"""

from __future__ import annotations

__all__ = ["inject_theme"]


def inject_theme() -> None:
    """Phase 1 no-op. Dark mode comes from .streamlit/config.toml."""
    return None
