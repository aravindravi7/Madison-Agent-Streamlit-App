"""Round-trip tests for the SQLite DAL.

One temp DB per test via the ``tmp_path`` fixture. Every model gets a
round-trip. ``get_evaluator_win_rates`` is exercised against a small
seeded dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from signalscout.models import (
    ArenaResult,
    BanditArmState,
    Brief,
    EvaluatorVerdict,
    Feedback,
    Item,
)
from signalscout.storage import Storage


def _fresh_storage(tmp_path: Path) -> Storage:
    s = Storage(str(tmp_path / "scout.db"))
    s.init_db()
    return s


def _make_item(suffix: str = "1") -> Item:
    return Item(
        id=f"item-{suffix}",
        source="arxiv",
        title=f"Title {suffix}",
        url=f"https://example.com/{suffix}",
        published_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        summary=f"Summary {suffix} " + ("x" * 60),
        fetched_at=datetime(2024, 6, 2, 9, 30, 0, tzinfo=timezone.utc),
    )


def _make_verdict(evaluator_id: str, score: int, action: str = "include") -> EvaluatorVerdict:
    return EvaluatorVerdict(
        evaluator_id=evaluator_id,
        score=score,
        reasoning=f"{evaluator_id} thinks score={score}",
        confidence=0.8,
        topic_tags=["LLM", "Evaluation"],
        action=action,  # type: ignore[arg-type]
        latency_ms=420,
    )


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    s = Storage(str(tmp_path / "s.db"))
    s.init_db()
    s.init_db()
    s.init_db()
    # Should not raise and should accept normal operations.
    s.upsert_item(_make_item())


def test_item_round_trip_via_arena(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    item = _make_item("42")
    s.upsert_item(item)
    # Upsert again with a changed title — should overwrite, not duplicate.
    updated = item.model_copy(update={"title": "Updated title"})
    s.upsert_item(updated)
    # We don't have a get_item yet; confirm via the arena FK path.
    arena = ArenaResult(
        item_id=item.id,
        verdicts=[_make_verdict("skeptic", 70), _make_verdict("scout", 85)],
        winner_id="scout",
        winner_reason="ignored on read",
        bandit_sampled_values={"skeptic": 0.5, "scout": 0.7},
        final_score=85,
        final_action="include",
        disagreement=7.5,
    )
    s.save_arena_result(arena)
    loaded = s.get_arena_result(item.id)
    assert loaded is not None
    assert loaded.item_id == item.id
    assert loaded.winner_id == "scout"
    assert loaded.final_score == 85
    assert loaded.final_action == "include"
    assert pytest.approx(loaded.disagreement, rel=1e-6) == 7.5
    assert {v.evaluator_id for v in loaded.verdicts} == {"skeptic", "scout"}
    assert loaded.bandit_sampled_values["scout"] == pytest.approx(0.7)


def test_arena_result_missing_returns_none(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    assert s.get_arena_result("nonexistent") is None


def test_feedback_round_trip(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    fb_in = Feedback(
        item_id="item-1",
        user_id="user-abc",
        signal="thumbs_up",
        evaluator_id="scout",
        timestamp=datetime(2024, 6, 3, 10, 0, 0, tzinfo=timezone.utc),
    )
    s.save_feedback(fb_in)
    rows = s.get_feedback_for_user("user-abc")
    assert len(rows) == 1
    assert rows[0].item_id == "item-1"
    assert rows[0].signal == "thumbs_up"
    assert rows[0].evaluator_id == "scout"
    assert rows[0].timestamp == fb_in.timestamp


def test_brief_round_trip(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    brief = Brief(
        id="brief-1",
        generated_at=datetime(2024, 6, 5, 8, 0, 0, tzinfo=timezone.utc),
        user_id="user-abc",
        included_items=["item-1", "item-2", "item-3"],
        theme_synthesis={"themes": [{"name": "Alignment", "confidence": 0.9}]},
        config={"arxiv_limit": 50, "max_evaluate": 25},
    )
    s.save_brief(brief)
    got = s.list_briefs("user-abc")
    assert len(got) == 1
    assert got[0].id == "brief-1"
    assert got[0].included_items == ["item-1", "item-2", "item-3"]
    assert got[0].theme_synthesis["themes"][0]["name"] == "Alignment"
    assert got[0].config["arxiv_limit"] == 50


def test_list_briefs_respects_user_and_limit(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    base = datetime(2024, 6, 5, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        s.save_brief(
            Brief(
                id=f"brief-{i}",
                generated_at=base.replace(hour=i),
                user_id="user-abc",
                included_items=[],
                theme_synthesis={},
                config={},
            )
        )
    s.save_brief(
        Brief(
            id="other-users-brief",
            generated_at=base,
            user_id="user-xyz",
            included_items=[],
            theme_synthesis={},
            config={},
        )
    )
    got = s.list_briefs("user-abc", limit=3)
    assert len(got) == 3
    # Newest first.
    assert [b.id for b in got] == ["brief-4", "brief-3", "brief-2"]


def test_bandit_state_round_trip(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    state = {
        "skeptic": BanditArmState(
            evaluator_id="skeptic",
            A=[[1.0, 0.0], [0.0, 1.0]],
            b=[0.2, -0.1],
            n_pulls=3,
            n_rewards_positive=2,
        ),
        "scout": BanditArmState(
            evaluator_id="scout",
            A=[[2.0, 0.1], [0.1, 2.0]],
            b=[0.5, 0.5],
            n_pulls=5,
            n_rewards_positive=4,
        ),
    }
    s.save_bandit_state("user-abc", state)
    loaded = s.get_bandit_state("user-abc")
    assert set(loaded.keys()) == {"skeptic", "scout"}
    assert loaded["scout"].A == [[2.0, 0.1], [0.1, 2.0]]
    assert loaded["scout"].b == [0.5, 0.5]
    assert loaded["scout"].n_pulls == 5
    assert loaded["skeptic"].n_rewards_positive == 2

    # Update one arm, persist again — should overwrite, not duplicate.
    state["scout"] = state["scout"].model_copy(update={"n_pulls": 11})
    s.save_bandit_state("user-abc", state)
    reloaded = s.get_bandit_state("user-abc")
    assert reloaded["scout"].n_pulls == 11
    assert len(reloaded) == 2


def test_get_bandit_state_empty_for_new_user(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    assert s.get_bandit_state("nobody") == {}


def test_evaluator_win_rates_on_seeded_dataset(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    ts = datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc)

    def fb(sig: str, eid: str, idx: int) -> Feedback:
        return Feedback(
            item_id=f"item-{idx}",
            user_id="user-abc",
            signal=sig,  # type: ignore[arg-type]
            evaluator_id=eid,
            timestamp=ts,
        )

    # scout: 3 positive, 1 negative -> 0.75
    s.save_feedback(fb("thumbs_up", "scout", 1))
    s.save_feedback(fb("saved", "scout", 2))
    s.save_feedback(fb("clicked", "scout", 3))
    s.save_feedback(fb("thumbs_down", "scout", 4))
    # skeptic: 1 positive, 3 negative -> 0.25
    s.save_feedback(fb("thumbs_up", "skeptic", 5))
    s.save_feedback(fb("dismissed", "skeptic", 6))
    s.save_feedback(fb("thumbs_down", "skeptic", 7))
    s.save_feedback(fb("dismissed", "skeptic", 8))
    # operator: no feedback -> omitted
    # other user: must not leak
    s.save_feedback(
        Feedback(
            item_id="item-9",
            user_id="user-xyz",
            signal="thumbs_up",
            evaluator_id="scout",
            timestamp=ts,
        )
    )

    rates = s.get_evaluator_win_rates("user-abc")
    assert set(rates.keys()) == {"scout", "skeptic"}
    assert rates["scout"] == pytest.approx(0.75)
    assert rates["skeptic"] == pytest.approx(0.25)

    other = s.get_evaluator_win_rates("user-xyz")
    assert other == {"scout": 1.0}


def test_returns_are_pydantic_models(tmp_path: Path) -> None:
    """No raw sqlite rows or dicts should escape — everything is a model."""
    s = _fresh_storage(tmp_path)
    s.upsert_item(_make_item("a"))
    s.save_arena_result(
        ArenaResult(
            item_id="item-a",
            verdicts=[_make_verdict("skeptic", 60), _make_verdict("scout", 80, "skip")],
            winner_id="scout",
            winner_reason="",
            bandit_sampled_values={"skeptic": 0.4, "scout": 0.6},
            final_score=80,
            final_action="skip",
            disagreement=10.0,
        )
    )
    arena = s.get_arena_result("item-a")
    assert isinstance(arena, ArenaResult)
    assert all(isinstance(v, EvaluatorVerdict) for v in arena.verdicts)


def test_winner_reason_round_trip(tmp_path: Path) -> None:
    """Phase 2 added the winner_reason column — it must persist + restore."""
    s = _fresh_storage(tmp_path)
    s.upsert_item(_make_item("wr"))
    s.save_arena_result(
        ArenaResult(
            item_id="item-wr",
            verdicts=[_make_verdict("skeptic", 65), _make_verdict("scout", 85)],
            winner_id="scout",
            winner_reason="Highest score among evaluators",
            bandit_sampled_values={},
            final_score=85,
            final_action="include",
            disagreement=10.0,
        )
    )
    loaded = s.get_arena_result("item-wr")
    assert loaded is not None
    assert loaded.winner_reason == "Highest score among evaluators"


def test_decision_history_joins_arena_with_feedback(tmp_path: Path) -> None:
    """LEFT JOIN contract: all arena rows surface, feedback fields are None
    when absent, summed when multiple signals land on the same item."""
    s = _fresh_storage(tmp_path)
    s.upsert_item(_make_item("1"))
    s.upsert_item(_make_item("2"))
    s.upsert_item(_make_item("3"))

    # Item 1: 👍 + 🔖 → reward 2.0, latest signal = saved
    s.save_arena_result(
        ArenaResult(
            item_id="item-1",
            verdicts=[_make_verdict("scout", 85), _make_verdict("skeptic", 60)],
            winner_id="scout", winner_reason="Learned preference",
            bandit_sampled_values={}, final_score=85, final_action="include",
            disagreement=12.5,
        ),
        user_id="u1",
    )
    s.save_feedback(Feedback(item_id="item-1", user_id="u1", signal="thumbs_up",
                             evaluator_id="scout", timestamp=datetime(2024, 6, 1, 10, tzinfo=timezone.utc)))
    s.save_feedback(Feedback(item_id="item-1", user_id="u1", signal="saved",
                             evaluator_id="scout", timestamp=datetime(2024, 6, 1, 11, tzinfo=timezone.utc)))

    # Item 2: no feedback
    s.save_arena_result(
        ArenaResult(
            item_id="item-2",
            verdicts=[_make_verdict("skeptic", 55)],
            winner_id="skeptic", winner_reason="Calibrating — round 2 of 10",
            bandit_sampled_values={}, final_score=55, final_action="skip",
            disagreement=0.0,
        ),
        user_id="u1",
    )

    # Item 3: 👎 only → reward -1.0
    s.save_arena_result(
        ArenaResult(
            item_id="item-3",
            verdicts=[_make_verdict("operator", 70)],
            winner_id="operator", winner_reason="Learned preference",
            bandit_sampled_values={}, final_score=70, final_action="skip",
            disagreement=0.0,
        ),
        user_id="u1",
    )
    s.save_feedback(Feedback(item_id="item-3", user_id="u1", signal="thumbs_down",
                             evaluator_id="operator", timestamp=datetime(2024, 6, 1, 12, tzinfo=timezone.utc)))

    history = s.get_decision_history("u1")
    assert [r["item_id"] for r in history] == ["item-1", "item-2", "item-3"]

    row_1 = history[0]
    assert row_1["winner_id"] == "scout"
    assert row_1["feedback_signal"] == "saved"  # latest of the two
    assert row_1["feedback_reward"] == 2.0      # 1.0 + 1.0

    row_2 = history[1]
    assert row_2["feedback_signal"] is None
    assert row_2["feedback_reward"] is None

    row_3 = history[2]
    assert row_3["feedback_signal"] == "thumbs_down"
    assert row_3["feedback_reward"] == -1.0


def test_decision_history_is_user_scoped(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    s.upsert_item(_make_item("a"))
    s.save_arena_result(
        ArenaResult(
            item_id="item-a",
            verdicts=[_make_verdict("scout", 80)],
            winner_id="scout", winner_reason="Learned preference",
            bandit_sampled_values={}, final_score=80, final_action="include",
            disagreement=0.0,
        ),
        user_id="u1",
    )
    s.save_feedback(Feedback(item_id="item-a", user_id="u2", signal="thumbs_up",
                             evaluator_id="scout",
                             timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc)))
    # u1's feedback stays empty even though u2 gave one — scope respected.
    u1 = s.get_decision_history("u1")
    assert len(u1) == 1
    assert u1[0]["feedback_signal"] is None
    u2 = s.get_decision_history("u2")
    # u2 has no arena_results scoped to them, so the history is empty.
    assert u2 == []


def test_decision_history_respects_limit(tmp_path: Path) -> None:
    s = _fresh_storage(tmp_path)
    for i in range(5):
        s.upsert_item(_make_item(str(i)))
        s.save_arena_result(
            ArenaResult(
                item_id=f"item-{i}",
                verdicts=[_make_verdict("scout", 80)],
                winner_id="scout", winner_reason="Learned preference",
                bandit_sampled_values={}, final_score=80, final_action="include",
                disagreement=0.0,
            ),
            user_id="u1",
        )
    assert len(s.get_decision_history("u1", limit=3)) == 3


def test_feedback_idempotency_on_item_user_signal(tmp_path: Path) -> None:
    """Re-clicking 👍 on the same item must be a no-op (unique index)."""
    s = _fresh_storage(tmp_path)
    fb = Feedback(
        item_id="item-dup",
        user_id="user-dup",
        signal="thumbs_up",
        evaluator_id="scout",
        timestamp=datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
    )
    inserted_first = s.save_feedback(fb)
    inserted_second = s.save_feedback(fb)
    rows = s.get_feedback_for_user("user-dup")
    assert inserted_first is True
    assert inserted_second is False
    assert len(rows) == 1
    # A different signal on the same item still goes through.
    other = fb.model_copy(update={"signal": "saved"})
    assert s.save_feedback(other) is True
    assert len(s.get_feedback_for_user("user-dup")) == 2
