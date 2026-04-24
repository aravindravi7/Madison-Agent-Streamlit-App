"""Evaluator tests with a mocked OpenAI client.

Covers:
- valid JSON → validated ``EvaluatorVerdict`` per evaluator
- malformed first response, valid retry → success on retry
- two consecutive malformed responses → failed verdict (no raise)
- score clamping + include/skip threshold enforcement (score 85 → include)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from signalscout.evaluators import EVALUATORS, OPERATOR, SCOUT, SKEPTIC
from signalscout.models import EvaluatorVerdict, Item


def _item() -> Item:
    return Item(
        id="item-x",
        source="arxiv",
        title="A paper about evaluation",
        url="https://example.com/paper",
        published_at=datetime(2024, 10, 1, tzinfo=UTC),
        summary="x" * 80,
        fetched_at=datetime.now(UTC),
    )


class _FakeClient:
    """Minimal OpenAI SDK lookalike: client.chat.completions.create(...)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

        class _Completions:
            def create(_self, **kwargs: Any) -> Any:  # noqa: N805
                self.calls.append(kwargs)
                body = self._responses.pop(0) if self._responses else ""
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
                )

        self.chat = SimpleNamespace(completions=_Completions())


def _ok_payload(score: int = 85, action: str = "include") -> str:
    return json.dumps(
        {
            "score": score,
            "reasoning": "Methodology is clear; ablations present.",
            "confidence": 0.82,
            "topic_tags": ["LLM Evaluation", "Benchmarks", "Safety"],
            "action": action,
        }
    )


@pytest.mark.parametrize("evaluator", EVALUATORS, ids=[e.id for e in EVALUATORS])
def test_each_evaluator_returns_valid_verdict(evaluator: Any) -> None:
    client = _FakeClient([_ok_payload(score=82)])
    verdict = evaluator.evaluate(client, _item())
    assert isinstance(verdict, EvaluatorVerdict)
    assert verdict.evaluator_id == evaluator.id
    assert verdict.score == 82
    assert verdict.action == "include"
    assert verdict.confidence == pytest.approx(0.82)
    assert verdict.reasoning.startswith("Methodology")
    assert verdict.latency_ms >= 0
    # Exactly one LLM call on the happy path.
    assert len(client.calls) == 1


def test_malformed_then_valid_retry_succeeds() -> None:
    client = _FakeClient(["this is not JSON at all", _ok_payload(score=78)])
    verdict = SCOUT.evaluate(client, _item())
    assert isinstance(verdict, EvaluatorVerdict)
    assert verdict.score == 78
    assert verdict.action == "include"
    assert verdict.reasoning != "parse_error"
    # Two calls: original + reminder retry.
    assert len(client.calls) == 2


def test_two_malformed_returns_failed_verdict() -> None:
    client = _FakeClient(["not json", "still not json"])
    verdict = SKEPTIC.evaluate(client, _item())
    assert verdict.reasoning == "parse_error"
    assert verdict.score == 0
    assert verdict.action == "skip"
    assert verdict.confidence == 0.0
    assert verdict.evaluator_id == "skeptic"
    assert len(client.calls) == 2


def test_action_is_forced_to_include_for_high_score() -> None:
    # LLM claims action="skip" despite score=85 — server must enforce include.
    client = _FakeClient([_ok_payload(score=85, action="skip")])
    verdict = OPERATOR.evaluate(client, _item())
    assert verdict.score == 85
    assert verdict.action == "include"


def test_action_is_forced_to_skip_for_low_score() -> None:
    client = _FakeClient([_ok_payload(score=30, action="include")])
    verdict = SKEPTIC.evaluate(client, _item())
    assert verdict.score == 30
    assert verdict.action == "skip"


def test_score_74_is_skipped_under_threshold_75() -> None:
    client = _FakeClient([_ok_payload(score=74, action="include")])
    verdict = SCOUT.evaluate(client, _item())
    assert verdict.score == 74
    assert verdict.action == "skip"  # threshold is 75


def test_score_75_is_included_on_threshold() -> None:
    client = _FakeClient([_ok_payload(score=75, action="skip")])
    verdict = SCOUT.evaluate(client, _item())
    assert verdict.score == 75
    assert verdict.action == "include"


def test_score_is_clamped_to_0_100() -> None:
    client = _FakeClient([_ok_payload(score=1500, action="include")])
    verdict = SCOUT.evaluate(client, _item())
    assert verdict.score == 100


def test_evaluator_has_brand_identity() -> None:
    # Baselines reflect the Phase 2 addendum's self-audit targets.
    assert SKEPTIC.avatar == "🔬" and SKEPTIC.color == "#3A86FF" and SKEPTIC.baseline_score == 50
    assert SCOUT.avatar == "📡" and SCOUT.color == "#00F5D4" and SCOUT.baseline_score == 45
    assert OPERATOR.avatar == "⚙️" and OPERATOR.color == "#EAEAEA" and OPERATOR.baseline_score == 40
