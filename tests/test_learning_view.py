"""Tests for signalscout.ui.learning_view.

Pure-logic tests on the computation helpers (no Streamlit runtime), plus a
taste-summary cache test that exercises the full ``_render_taste_summary``
path with a mocked OpenAI client under a patched ``st.session_state``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from signalscout.bandit import TasteBandit
from signalscout.models import ArenaResult, EvaluatorVerdict, Feedback, Item
from signalscout.storage import Storage
from signalscout.ui import learning_view as LV


def _fresh_storage(tmp_path: Path) -> Storage:
    s = Storage(str(tmp_path / "lv.db"))
    s.init_db()
    return s


def _verdict(evaluator_id: str, score: int = 75, tags: list[str] | None = None) -> EvaluatorVerdict:
    return EvaluatorVerdict(
        evaluator_id=evaluator_id,
        score=score,
        reasoning="r",
        confidence=0.8,
        topic_tags=tags or ["T"],
        action="include" if score >= 75 else "skip",
        latency_ms=10,
    )


def _arena(item_id: str, winner: str, verdicts: list[EvaluatorVerdict] | None = None) -> ArenaResult:
    return ArenaResult(
        item_id=item_id,
        verdicts=verdicts or [_verdict("skeptic", 60), _verdict(winner, 80), _verdict("operator", 55)],
        winner_id=winner,
        winner_reason="Learned preference",
        bandit_sampled_values={},
        final_score=80,
        final_action="include",
        disagreement=0.0,
    )


def _item(item_id: str) -> Item:
    return Item(
        id=item_id,
        source="arxiv_cs_ai",
        title=f"T-{item_id}",
        url=f"https://e/{item_id}",
        published_at=datetime(2024, 6, 1, tzinfo=UTC),
        summary="x" * 80,
        fetched_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------- trend

def test_trend_up_when_last10_exceeds_prior10() -> None:
    outcomes = [False] * 10 + [True] * 10
    assert LV._compute_trend(outcomes) == "↑"


def test_trend_down_when_last10_below_prior10() -> None:
    outcomes = [True] * 10 + [False] * 10
    assert LV._compute_trend(outcomes) == "↓"


def test_trend_stable_within_threshold() -> None:
    outcomes = ([True] * 7 + [False] * 3) + ([True] * 7 + [False] * 3)
    assert LV._compute_trend(outcomes) == "→"


def test_trend_calibrating_when_under_min_pulls() -> None:
    assert LV._compute_trend([True] * 4) == "calibrating"
    assert LV._compute_trend([]) == "calibrating"


def test_trend_stable_when_no_prior_window() -> None:
    # Exactly one window of data → no prior to compare; should return "→".
    assert LV._compute_trend([True] * 10) == "→"


# ---------------------------------------------------------------- feedback/cumulative

def _history(rows: list[tuple[str, str, float | None]]) -> list[dict]:
    """Convenience: rows = (item_id, winner_id, feedback_reward)."""
    return [
        {
            "item_id": item_id,
            "winner_id": winner,
            "final_score": 80,
            "timestamp": f"2024-06-01T00:00:{i:02d}",
            "feedback_signal": ("thumbs_up" if (r is not None and r > 0) else "thumbs_down") if r is not None else None,
            "feedback_reward": r,
        }
        for i, (item_id, winner, r) in enumerate(rows)
    ]


def test_feedback_count_ignores_unrated() -> None:
    h = _history([("a", "scout", 1.0), ("b", "skeptic", -1.0), ("c", "scout", None)])
    assert LV._feedback_count(h) == 2


def test_cumulative_reward_sums_observed_only() -> None:
    h = _history([("a", "scout", 1.0), ("b", "skeptic", -1.0), ("c", "scout", None), ("d", "scout", 1.0)])
    assert LV._cumulative_reward(h) == [1.0, 0.0, 0.0, 1.0]


# ---------------------------------------------------------------- baseline seed

def test_random_baseline_is_deterministic() -> None:
    h = _history([(f"i{i}", "scout", 1.0 if i % 2 == 0 else -1.0) for i in range(20)])
    run1 = LV._random_baseline_cumulative(h)
    run2 = LV._random_baseline_cumulative(h)
    assert run1 == run2
    # Final value must be the same across two runs (seed fixed).
    assert len(run1) == 20


def test_random_baseline_skips_unrated_items() -> None:
    h = _history([("a", "scout", 1.0), ("b", "scout", None), ("c", "scout", 1.0)])
    cum = LV._random_baseline_cumulative(h)
    # Row b carries no feedback → baseline must not advance at that index.
    assert cum[0] == cum[0]  # starting point
    assert cum[1] == cum[0]  # unrated row preserves running total


# ---------------------------------------------------------------- outcomes_by_arm

def test_outcomes_by_arm_excludes_unrated() -> None:
    h = _history([
        ("a", "scout", 1.0),
        ("b", "skeptic", -1.0),
        ("c", "scout", None),
        ("d", "scout", 1.0),
        ("e", "operator", -0.5),
    ])
    by_arm = LV._outcomes_by_arm(h)
    assert by_arm["scout"] == [True, True]
    assert by_arm["skeptic"] == [False]
    assert by_arm["operator"] == [False]


# ---------------------------------------------------------------- top tags

def test_top_tags_split_groups_by_reward_sign(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    # Seed arena results with distinct tag sets.
    for item_id, winner, tags in [
        ("a", "scout", ["agents", "evaluation"]),
        ("b", "scout", ["agents"]),
        ("c", "skeptic", ["methodology", "benchmarks"]),
    ]:
        s.upsert_item(_item(item_id))
        verdict = _verdict(winner, score=80, tags=tags)
        s.save_arena_result(_arena(item_id, winner, [verdict]), user_id="u1")

    h = _history([("a", "scout", 1.0), ("b", "scout", 1.0), ("c", "skeptic", -1.0)])
    pos, neg = LV._top_tags_split(h, s)
    assert "agents" in pos  # appears twice on positive feedback
    assert "methodology" in neg


# ---------------------------------------------------------------- taste cache

class _FakeOpenAI:
    """Minimal client.chat.completions.create stand-in that tracks calls."""
    def __init__(self, response: str = "You reward Scout 2x more on agent-tagged items.") -> None:
        self.calls: list[dict] = []

        class _Completions:
            def create(_self, **kwargs):  # noqa: N805
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=response))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def test_taste_summary_caches_by_user_and_feedback_count(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    bandit = TasteBandit(
        user_id="u1",
        evaluator_ids=["skeptic", "scout", "operator"],
        storage=s,
    )
    client = _FakeOpenAI()
    # Build a history that clears the TASTE_THRESHOLD (≥5 feedback events).
    history = _history([
        ("a", "scout", 1.0), ("b", "scout", 1.0), ("c", "skeptic", -1.0),
        ("d", "scout", 1.0), ("e", "operator", 1.0),
    ])

    fake_state: dict = {}
    with patch("signalscout.ui.learning_view.st") as mock_st:
        mock_st.session_state = fake_state
        # First render → LLM call happens, result cached.
        LV._render_taste_summary(
            bandit=bandit, history=history, storage=s, user_id="u1", client=client,
        )
        # Second render with identical history → cache hit, no new LLM call.
        LV._render_taste_summary(
            bandit=bandit, history=history, storage=s, user_id="u1", client=client,
        )

    assert len(client.calls) == 1
    # Third render: same user, different feedback count → should call again.
    with patch("signalscout.ui.learning_view.st") as mock_st:
        mock_st.session_state = fake_state  # reuse the cache across renders
        bigger_history = history + _history([("f", "scout", 1.0)])
        LV._render_taste_summary(
            bandit=bandit, history=bigger_history, storage=s, user_id="u1", client=client,
        )
    assert len(client.calls) == 2


def test_taste_summary_placeholder_below_threshold(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    bandit = TasteBandit(
        user_id="u1",
        evaluator_ids=["skeptic", "scout", "operator"],
        storage=s,
    )
    client = _FakeOpenAI()
    # Below 5 feedback events → no LLM call.
    history = _history([("a", "scout", 1.0), ("b", "scout", -1.0)])

    with patch("signalscout.ui.learning_view.st") as mock_st:
        mock_st.session_state = {}
        LV._render_taste_summary(
            bandit=bandit, history=history, storage=s, user_id="u1", client=client,
        )
    assert client.calls == []


# ---------------------------------------------------------------- placeholders

def test_regret_curve_placeholder_below_threshold(tmp_path: Path) -> None:
    # Fewer than _REGRET_THRESHOLD (10) feedback events → no Plotly call.
    history = _history([(f"i{i}", "scout", 1.0) for i in range(5)])
    with patch("signalscout.ui.learning_view.st") as mock_st:
        LV._render_regret_curve(history)
    # st.plotly_chart must not have been called.
    assert not mock_st.plotly_chart.called


def test_regret_curve_renders_when_threshold_met() -> None:
    history = _history([(f"i{i}", "scout", 1.0 if i % 2 == 0 else -1.0) for i in range(12)])
    with patch("signalscout.ui.learning_view.st") as mock_st:
        LV._render_regret_curve(history)
    assert mock_st.plotly_chart.called


# ---------------------------------------------------------------- public render

def test_render_short_circuits_without_user_id(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    with patch("signalscout.ui.learning_view.st") as mock_st:
        LV.render(storage=s, bandit=None, user_id="", client=None)
    mock_st.info.assert_called()


def _seed_arena_and_feedback(s: Storage, bandit: TasteBandit, n: int) -> None:
    """Write ``n`` arena results + positive feedback for user u1."""
    from datetime import timedelta
    import numpy as np
    from signalscout.evaluators import EVALUATORS as _EVALS
    now = datetime.now(UTC)
    for i in range(n):
        s.upsert_item(_item(f"x{i}"))
        ctx = np.zeros(15)
        ctx[-1] = 1.0
        arm_id, _, reason = bandit.select_trusted_evaluator(ctx)
        bandit.record_pull(arm_id)
        verdicts = [_verdict(ev.id, score=80 if ev.id == arm_id else 50)
                    for ev in _EVALS]
        winner = next(v for v in verdicts if v.evaluator_id == arm_id)
        s.save_arena_result(
            ArenaResult(
                item_id=f"x{i}",
                verdicts=verdicts,
                winner_id=arm_id,
                winner_reason=reason,
                bandit_sampled_values={},
                final_score=winner.score,
                final_action=winner.action,
                disagreement=0.0,
            ),
            user_id="u1",
            context_vector=ctx.tolist(),
        )
        s.save_feedback(
            Feedback(
                item_id=f"x{i}",
                user_id="u1",
                signal="thumbs_up",
                evaluator_id=arm_id,
                timestamp=now - timedelta(minutes=n - i),
            )
        )
        bandit.update_from_feedback(arm_id, ctx, reward=1.0)


def test_render_tab_still_renders_when_taste_summary_raises(tmp_path: Path) -> None:
    """Leaderboard and regret curve must complete even if the taste summary
    section raises — failure is isolated to the taste card, which falls back
    to 'retrying shortly'."""
    from signalscout.evaluators import EVALUATORS as _EVALS
    s = _fresh_storage(tmp_path)
    bandit = TasteBandit(
        user_id="u1",
        evaluator_ids=[ev.id for ev in _EVALS],
        storage=s,
    )
    _seed_arena_and_feedback(s, bandit, n=15)

    class _MockCol:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("signalscout.ui.learning_view.st") as mock_st, \
         patch.object(LV, "_render_taste_summary", side_effect=RuntimeError("boom")):
        mock_st.session_state = {}
        mock_st.columns.return_value = [_MockCol() for _ in range(3)]
        LV.render(storage=s, bandit=bandit, user_id="u1", client=None)

    # Leaderboard + regret should have completed (3 subheaders: leaderboard + regret title + fallback taste title).
    assert mock_st.subheader.call_count == 3
    assert mock_st.plotly_chart.call_count == 1
    fallback_seen = any(
        "retrying shortly" in str(call) for call in mock_st.markdown.call_args_list
    )
    assert fallback_seen, "fallback message 'retrying shortly' should render when taste summary raises"
