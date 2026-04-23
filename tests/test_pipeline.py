"""Pipeline tests: arena-on-item structure, winner selection, disagreement.

No network. Swap in fake evaluators that return canned verdicts so we
can assert on ``run_arena_on_item`` deterministically.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from signalscout.models import ArenaResult, EvaluatorVerdict, Item
from signalscout.pipeline import WINNER_REASON_HIGHEST_SCORE, run_arena_on_item


class _FakeEvaluator:
    """Duck-types ``Evaluator`` for the pipeline's purposes: just ``.evaluate()``."""

    def __init__(self, id: str, score: int, action: str = "include", latency_ms: int = 10) -> None:  # noqa: A002
        self.id = id
        self._score = score
        self._action = action
        self._latency = latency_ms

    def evaluate(self, client: Any, item: Item) -> EvaluatorVerdict:
        return EvaluatorVerdict(
            evaluator_id=self.id,
            score=self._score,
            reasoning=f"{self.id} says {self._score}",
            confidence=0.7,
            topic_tags=["T"],
            action=self._action,  # type: ignore[arg-type]
            latency_ms=self._latency,
        )


def _item() -> Item:
    return Item(
        id="item-1",
        source="arxiv",
        title="t",
        url="https://example.com/paper",
        published_at=datetime(2024, 6, 1, tzinfo=UTC),
        summary="x" * 80,
        fetched_at=datetime.now(UTC),
    )


def test_run_arena_returns_three_verdicts_and_correct_winner() -> None:
    evs = [
        _FakeEvaluator("skeptic", score=72),
        _FakeEvaluator("scout", score=88),
        _FakeEvaluator("operator", score=45, action="skip"),
    ]
    result = run_arena_on_item(client=None, item=_item(), evaluators=evs)  # type: ignore[arg-type]

    assert isinstance(result, ArenaResult)
    assert len(result.verdicts) == 3
    assert {v.evaluator_id for v in result.verdicts} == {"skeptic", "scout", "operator"}
    assert result.winner_id == "scout"
    assert result.final_score == 88
    assert result.final_action == "include"
    assert result.winner_reason == WINNER_REASON_HIGHEST_SCORE
    assert result.bandit_sampled_values == {}


def test_disagreement_matches_population_stdev() -> None:
    evs = [
        _FakeEvaluator("skeptic", score=90),
        _FakeEvaluator("scout", score=60),
        _FakeEvaluator("operator", score=30),
    ]
    result = run_arena_on_item(client=None, item=_item(), evaluators=evs)  # type: ignore[arg-type]
    # pstdev([90, 60, 30]) == sqrt(((30)^2 + 0 + (-30)^2) / 3) == sqrt(600) ≈ 24.495
    assert result.disagreement == pytest.approx(math.sqrt(600), rel=1e-6)


def test_verdicts_are_sorted_by_evaluator_id_for_determinism() -> None:
    evs = [
        _FakeEvaluator("scout", score=50),
        _FakeEvaluator("skeptic", score=40),
        _FakeEvaluator("operator", score=60),
    ]
    result = run_arena_on_item(client=None, item=_item(), evaluators=evs)  # type: ignore[arg-type]
    assert [v.evaluator_id for v in result.verdicts] == ["operator", "scout", "skeptic"]


def test_disagreement_is_zero_with_single_evaluator() -> None:
    result = run_arena_on_item(
        client=None,
        item=_item(),
        evaluators=[_FakeEvaluator("skeptic", score=55)],  # type: ignore[arg-type]
    )
    assert result.disagreement == 0.0
    assert result.winner_id == "skeptic"


def test_all_include_when_all_score_above_threshold() -> None:
    evs = [
        _FakeEvaluator("skeptic", score=76),
        _FakeEvaluator("scout", score=85),
        _FakeEvaluator("operator", score=78),
    ]
    result = run_arena_on_item(client=None, item=_item(), evaluators=evs)  # type: ignore[arg-type]
    assert result.final_action == "include"
    assert all(v.action == "include" for v in result.verdicts)
