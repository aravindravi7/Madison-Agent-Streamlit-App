"""Reusable UI components for the Arena and future views.

Pure functions, no hidden state. Inline CSS is kept deliberately minimal —
Phase 4 will replace it with a real theme.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import streamlit as st

from ..evaluators import Evaluator
from ..models import EvaluatorVerdict, Feedback, Item
from ..storage import Storage

__all__ = [
    "score_badge",
    "signal_card_header",
    "agent_verdict_panel",
    "feedback_row",
    "disagreement_badge",
]


_MONO = "'JetBrains Mono','SFMono-Regular',Menlo,Consolas,monospace"


def score_badge(score: int, color: str) -> str:
    """Return an inline-HTML score pill in the given color."""
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{color};color:#0B0F1A;font-family:{_MONO};font-weight:600;'
        f'font-size:13px;">{int(score)}</span>'
    )


def disagreement_badge(disagreement: float) -> str:
    """HIGH >15 / MEDIUM 8–15 / LOW <8. Returns inline HTML badge."""
    if disagreement > 15:
        label, color = "HIGH", "#FF6B6B"
    elif disagreement >= 8:
        label, color = "MEDIUM", "#FFB84D"
    else:
        label, color = "LOW", "#6EE7B7"
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:transparent;border:1px solid {color};color:{color};'
        f'font-family:{_MONO};font-size:11px;letter-spacing:0.05em;">'
        f'DISAGREEMENT · {label}</span>'
    )


def signal_card_header(item: Item) -> None:
    """Render the item title + source + published-date line."""
    title = (item.title or "(untitled)")[:120]
    published = item.published_at.astimezone(UTC).strftime("%Y-%m-%d")
    source = item.source
    url = item.url
    st.markdown(
        f'<div style="margin:6px 0 2px 0;font-size:16px;font-weight:600;line-height:1.3;">'
        f'{_escape(title)}</div>'
        f'<div style="font-family:{_MONO};font-size:11px;color:#9CA3AF;margin-bottom:10px;">'
        f'{_escape(source)} · {published}'
        + (f' · <a href="{_escape(url)}" target="_blank" style="color:#00F5D4;">open ↗</a>' if url else '')
        + "</div>",
        unsafe_allow_html=True,
    )


def agent_verdict_panel(
    verdict: EvaluatorVerdict,
    evaluator: Evaluator,
    is_winner: bool,
) -> None:
    """Render one evaluator's column: name, score, confidence, reasoning."""
    border = f"2px solid {evaluator.color}" if is_winner else "1px solid #2A2F45"
    glow = (
        f"0 0 16px {evaluator.color}55, 0 0 4px {evaluator.color}33"
        if is_winner
        else "none"
    )
    crown = " 🏆" if is_winner else ""
    st.markdown(
        f'<div style="border:{border};border-radius:10px;padding:12px;box-shadow:{glow};'
        f'background:#12172A;min-height:180px;">'
        f'<div style="font-size:13px;color:#9CA3AF;margin-bottom:2px;">'
        f'{evaluator.avatar} {_escape(evaluator.name)}{crown}</div>'
        f'<div style="font-family:{_MONO};font-size:30px;font-weight:700;color:{evaluator.color};'
        f'line-height:1.1;margin-bottom:4px;">{int(verdict.score)}</div>'
        f'<div style="font-family:{_MONO};font-size:11px;color:#9CA3AF;margin-bottom:10px;">'
        f'conf: {verdict.confidence:.2f} · {verdict.latency_ms}ms</div>'
        f'<div style="font-size:13px;line-height:1.4;color:#EAEAEA;">'
        f'{_escape(verdict.reasoning or "")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def feedback_row(
    item_id: str,
    winner_id: str,
    user_id: str,
    storage: Storage,
    url: str | None = None,
) -> None:
    """Render the four feedback buttons. Writes to storage; shows a toast."""
    cols = st.columns([1, 1, 1, 1])
    buttons: list[tuple[str, str, Any]] = [
        ("👍 Useful", "thumbs_up", cols[0]),
        ("👎 Not relevant", "thumbs_down", cols[1]),
        ("🔖 Save", "saved", cols[2]),
    ]
    for label, signal, col in buttons:
        with col:
            if st.button(label, key=f"fb-{signal}-{item_id}"):
                inserted = storage.save_feedback(
                    Feedback(
                        item_id=item_id,
                        user_id=user_id,
                        signal=signal,  # type: ignore[arg-type]
                        evaluator_id=winner_id,
                        timestamp=datetime.now(UTC),
                    )
                )
                if inserted:
                    st.toast(f"Feedback logged: {signal}.")
                else:
                    st.toast(f"Already logged: {signal}.")
    with cols[3]:
        if url:
            st.link_button("Read full ↗", url)


def _escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
