"""Reusable UI components for the Arena and future views.

Pure functions, no hidden state. Inline CSS is kept deliberately minimal —
Phase 4 will replace it with a real theme.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import streamlit as st

from ..bandit import REWARD_MAP, TasteBandit
from ..evaluators import EVALUATORS, Evaluator
from ..models import EvaluatorVerdict, Feedback, Item
from ..storage import Storage

__all__ = [
    "score_badge",
    "signal_card_header",
    "agent_verdict_panel",
    "feedback_row",
    "disagreement_badge",
    "bandit_phase_badge",
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


_CALIBRATING_RE = re.compile(r"round\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)


def bandit_phase_badge(winner_reason: str) -> str:
    """Inline badge showing whether a pick was cold-start or learned.

    Returns:
    - CALIBRATING · N/10 in amber when ``winner_reason`` starts with
      ``"Calibrating"``. N/10 is parsed from "round N of 10" in the reason,
      falling back to just ``CALIBRATING`` if parsing fails.
    - LEARNED in Aquamarine (muted) when ``winner_reason`` starts with
      ``"Learned preference"``.
    - Empty string for anything else — no badge rendered.

    Matches the ``disagreement_badge`` visual style so the Arena row reads
    as one pill cluster.
    """
    reason = (winner_reason or "").strip()
    if reason.startswith("Calibrating"):
        label = "CALIBRATING"
        m = _CALIBRATING_RE.search(reason)
        if m:
            label = f"CALIBRATING · {m.group(1)}/{m.group(2)}"
        color = "#FFB547"  # amber
        return (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f'background:transparent;border:1px solid {color};color:{color};'
            f'font-family:{_MONO};font-size:11px;letter-spacing:0.05em;">'
            f'{label}</span>'
        )
    if reason.startswith("Learned preference"):
        color = "#00F5D4"  # Aquamarine
        # Intentionally quieter than the calibrating badge: 40% opacity border/text
        # so it reads as "the normal case" not a flag.
        return (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f'background:transparent;border:1px solid {color}66;color:{color}99;'
            f'font-family:{_MONO};font-size:11px;letter-spacing:0.05em;">'
            f'LEARNED</span>'
        )
    return ""


def signal_card_header(item: Item) -> None:
    """Render the item title + source + published-date line."""
    title = (item.title or "(untitled)")[:120]
    published = item.published_at.astimezone(UTC).strftime("%Y-%m-%d")
    source = item.source
    url = item.url
    # Non-breaking space + white-space:nowrap keeps "open↗" as one token so
    # the arrow never wraps to its own line.
    open_link = (
        f' · <a href="{_escape(url)}" target="_blank" '
        f'style="color:#00F5D4;white-space:nowrap;">open&nbsp;↗</a>'
        if url else ""
    )
    st.markdown(
        f'<div style="margin:6px 0 2px 0;font-size:16px;font-weight:600;line-height:1.3;">'
        f'{_escape(title)}</div>'
        f'<div class="mono" style="font-size:11px;color:#9CA3AF;margin-bottom:10px;">'
        f'{_escape(source)} · {published}{open_link}'
        f'</div>',
        unsafe_allow_html=True,
    )


def agent_verdict_panel(
    verdict: EvaluatorVerdict,
    evaluator: Evaluator,
    is_winner: bool,
) -> None:
    """Render one evaluator's column: name, score, confidence, reasoning.

    Uses the ``.arena-panel`` class from the theme so all three columns in
    a row have the same min-height; evaluators whose reasoning is shorter
    don't leave their card shorter than the others.
    """
    border = (
        f"2px solid {evaluator.color}"
        if is_winner
        else "1px solid rgba(234, 234, 234, 0.1)"
    )
    glow = (
        f"0 0 16px {evaluator.color}55, 0 0 4px {evaluator.color}33"
        if is_winner
        else "none"
    )
    crown = " 🏆" if is_winner else ""
    st.markdown(
        f'<div class="arena-panel" style="border:{border};border-radius:12px;padding:16px;'
        f'box-shadow:{glow};background:#12172A;">'
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-weight:500;font-size:18px;'
        f'color:#EAEAEA;margin-bottom:6px;">'
        f'{evaluator.avatar} {_escape(evaluator.name)}{crown}</div>'
        f'<div class="mono" style="font-size:32px;font-weight:700;color:{evaluator.color};'
        f'line-height:1.1;margin-bottom:4px;">{int(verdict.score)}</div>'
        f'<div class="mono" style="font-size:11px;color:#9CA3AF;margin-bottom:12px;">'
        f'conf: {verdict.confidence:.2f} · {verdict.latency_ms}ms</div>'
        f'<div style="font-size:13px;line-height:1.5;color:#EAEAEA;">'
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
    """Render the four feedback buttons.

    On a NEW feedback event, this also applies the bandit update against
    the context vector that was frozen at decision time (loaded from
    ``arena_results.context_vector_json``). Duplicate feedback — a click
    on a signal we've already recorded — is a no-op for the bandit,
    enforced by the unique ``(item_id, user_id, signal)`` index on the
    feedback table: ``save_feedback`` returns ``False`` and we skip the
    bandit update.
    """
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
                    _apply_bandit_update(
                        item_id=item_id,
                        winner_id=winner_id,
                        signal=signal,
                        user_id=user_id,
                        storage=storage,
                    )
                    st.toast(f"Feedback logged: {signal}.")
                else:
                    st.toast(f"Already logged: {signal}.")
    with cols[3]:
        if url:
            st.link_button("Read full ↗", url)


def _apply_bandit_update(
    *,
    item_id: str,
    winner_id: str,
    signal: str,
    user_id: str,
    storage: Storage,
) -> None:
    """Pull the context vector frozen at decision time and apply the reward.

    If the context vector isn't available (Phase-2 rows predate this column),
    the update is silently skipped — the feedback is still persisted, we
    just can't learn from it.
    """
    vec = storage.get_context_vector_for_item(item_id)
    if vec is None:
        return
    reward = REWARD_MAP.get(signal)
    if reward is None:
        return
    bandit = TasteBandit(
        user_id=user_id,
        evaluator_ids=[e.id for e in EVALUATORS],
        storage=storage,
    )
    bandit.update_from_feedback(
        evaluator_id=winner_id,
        context=np.asarray(vec, dtype=np.float64),
        reward=reward,
    )


def _escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
