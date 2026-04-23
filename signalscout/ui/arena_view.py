"""Arena tab: the three evaluators' verdicts per item, side by side.

Sort order is disagreement-desc — contested items first, because those are
where the signal is. Winner column glows in that evaluator's brand color.
Feedback buttons write to storage; the bandit consumes those signals in
Phase 3.

Above the per-item cards, a single mood header shows each evaluator's
current disposition, computed once per render from the last N verdicts
stored for this user.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from ..evaluators import EVALUATORS, evaluator_by_id
from ..models import ArenaResult, EvaluatorVerdict, Item
from ..moods import compute_mood
from ..storage import Storage
from . import components as C

__all__ = ["render"]

_MONO = "'JetBrains Mono','SFMono-Regular',Menlo,Consolas,monospace"


def render(*, storage: Storage | None = None, user_id: str | None = None) -> None:
    """Render the Arena tab against the most recent brief in session state."""
    storage = storage or st.session_state.get("storage")
    user_id = user_id or st.session_state.get("user_id") or ""

    container: dict[str, Any] | None = st.session_state.get("last_container")
    if not container or not container.get("arena_results"):
        st.info(
            "No arena yet. Run a brief from the **Run** tab — "
            "every item gets scored by all three evaluators in parallel."
        )
        return

    arena_results: list[ArenaResult] = list(container["arena_results"])
    items_by_id: dict[str, Item] = container.get("items_by_id", {})

    if not user_id:
        st.warning("Enter your email in the sidebar to enable feedback + learning.")

    st.subheader("Arena")
    st.caption(
        f"{len(arena_results)} items · sorted by disagreement (contested first). "
        "The glowing column is the item's winner."
    )

    # Moods — computed once, not per item.
    if storage is not None and user_id:
        _render_mood_header(storage=storage, user_id=user_id)

    # Most-contested first.
    arena_results.sort(key=lambda r: r.disagreement, reverse=True)

    for result in arena_results:
        item = items_by_id.get(result.item_id)
        if item is None:
            continue
        _render_card(result, item, storage=storage, user_id=user_id)
        st.markdown("---")


def _render_mood_header(*, storage: Storage, user_id: str) -> None:
    """One row, three evaluators' current moods with a short factual detail."""
    entries: list[tuple[Any, dict[str, str]]] = []
    for ev in EVALUATORS:
        recent: list[EvaluatorVerdict] = storage.get_recent_verdicts_for_evaluator(
            ev.id, user_id, limit=20
        )
        entries.append((ev, compute_mood(ev.id, recent)))

    cols = st.columns(len(entries))
    for col, (ev, mood) in zip(cols, entries):
        with col:
            st.markdown(
                f'<div style="padding:8px 12px;border:1px solid #2A2F45;border-radius:8px;'
                f'background:#12172A;">'
                f'<div style="color:#9CA3AF;font-size:11px;margin-bottom:2px;font-family:{_MONO};">'
                f'{ev.avatar} {_escape(ev.name)} · MOOD</div>'
                f'<div style="color:{ev.color};font-size:15px;font-weight:600;">{_escape(mood["label"])}</div>'
                f'<div style="color:#9CA3AF;font-size:11px;font-family:{_MONO};">{_escape(mood["detail"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


def _render_card(
    result: ArenaResult,
    item: Item,
    *,
    storage: Storage | None,
    user_id: str,
) -> None:
    C.signal_card_header(item)

    cols = st.columns(3)
    by_id = {v.evaluator_id: v for v in result.verdicts}
    for col, ev_id in zip(cols, ("skeptic", "scout", "operator")):
        verdict = by_id.get(ev_id)
        evaluator = evaluator_by_id(ev_id)
        if verdict is None or evaluator is None:
            continue
        with col:
            C.agent_verdict_panel(
                verdict=verdict,
                evaluator=evaluator,
                is_winner=(verdict.evaluator_id == result.winner_id),
            )

    winner_ev = evaluator_by_id(result.winner_id)
    winner_label = winner_ev.name if winner_ev else result.winner_id
    st.markdown(
        f'<div style="margin-top:10px;display:flex;gap:12px;align-items:center;">'
        f'<span style="color:#EAEAEA;">Winner: <b>{winner_label}</b>. '
        f"Reason: {result.winner_reason or 'Highest score among evaluators'}.</span>"
        f"{C.disagreement_badge(result.disagreement)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if storage is not None and user_id:
        C.feedback_row(
            item_id=result.item_id,
            winner_id=result.winner_id,
            user_id=user_id,
            storage=storage,
            url=item.url,
        )
    elif item.url:
        st.link_button("Read full ↗", item.url)


def _escape(s: str) -> str:
    return (
        str(s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
