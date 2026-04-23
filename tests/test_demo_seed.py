"""Tests for signalscout.demo_seed.

Locks the demo-day contract:
- First call writes exactly ``DEMO_DECISION_COUNT`` decisions + feedback.
- Second call is a no-op (returns 0) so judges clicking the button twice
  don't corrupt the learned state.
- All three evaluators appear as winners across the run so the Arena and
  Leaderboard aren't single-arm.
- The bandit's cold-start transition is tripped — ``total_pulls`` exceeds
  the cold-start threshold so the Arena shows "Learned preference" on
  most items, not just "Calibrating".
"""

from __future__ import annotations

from pathlib import Path

from signalscout.demo_seed import (
    DEMO_DECISION_COUNT,
    DEMO_USER_ID,
    seed_demo_user,
)
from signalscout.storage import Storage


def _storage(tmp_path: Path) -> Storage:
    s = Storage(str(tmp_path / "seed.db"))
    s.init_db()
    return s


def test_seed_demo_user_writes_expected_count(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    assert seed_demo_user(s) == DEMO_DECISION_COUNT
    history = s.get_decision_history(DEMO_USER_ID)
    assert len(history) == DEMO_DECISION_COUNT
    feedback = s.get_feedback_for_user(DEMO_USER_ID)
    assert len(feedback) == DEMO_DECISION_COUNT


def test_seed_demo_user_is_idempotent(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    first = seed_demo_user(s)
    second = seed_demo_user(s)
    third = seed_demo_user(s)
    assert first == DEMO_DECISION_COUNT
    assert second == 0
    assert third == 0
    # Count did not double.
    assert len(s.get_decision_history(DEMO_USER_ID)) == DEMO_DECISION_COUNT


def test_seed_produces_mixed_winners(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    seed_demo_user(s)
    winners = {r["winner_id"] for r in s.get_decision_history(DEMO_USER_ID)}
    assert winners == {"skeptic", "scout", "operator"}, (
        f"expected all three arms to appear; got {winners}"
    )


def test_seed_trips_cold_start_threshold(tmp_path: Path) -> None:
    """After seeding, the bandit should be out of cold-start for this user.
    30 pulls spread across 3 arms > the 10-pull cold-start window."""
    s = _storage(tmp_path)
    seed_demo_user(s)
    assert s.get_total_pulls(DEMO_USER_ID) == DEMO_DECISION_COUNT


def test_seed_is_deterministic(tmp_path: Path) -> None:
    """Same seeds → same decision sequence. Two clean seeds should produce
    identical winner sequences."""
    s1 = Storage(str(tmp_path / "a.db")); s1.init_db()
    s2 = Storage(str(tmp_path / "b.db")); s2.init_db()
    seed_demo_user(s1)
    seed_demo_user(s2)
    w1 = [r["winner_id"] for r in s1.get_decision_history(DEMO_USER_ID)]
    w2 = [r["winner_id"] for r in s2.get_decision_history(DEMO_USER_ID)]
    assert w1 == w2
