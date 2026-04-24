"""Tests for per-evaluator mood logic."""

from __future__ import annotations

from signalscout.models import EvaluatorVerdict
from signalscout.moods import MIN_VERDICTS, compute_mood


def _verdict(evaluator_id: str, score: int, action: str = "include") -> EvaluatorVerdict:
    return EvaluatorVerdict(
        evaluator_id=evaluator_id,
        score=score,
        reasoning="r",
        confidence=0.7,
        topic_tags=["T"],
        action=action,  # type: ignore[arg-type]
        latency_ms=10,
    )


def test_empty_verdicts_returns_calibrating() -> None:
    m = compute_mood("skeptic", [])
    assert m == {"label": "calibrating", "detail": "needs more data"}


def test_below_min_verdicts_returns_calibrating() -> None:
    m = compute_mood("scout", [_verdict("scout", 80)] * (MIN_VERDICTS - 1))
    assert m["label"] == "calibrating"


def test_skeptic_grumpy_when_include_rate_low() -> None:
    verdicts = (
        [_verdict("skeptic", 30, action="skip")] * 18
        + [_verdict("skeptic", 82, action="include")] * 2
    )
    # 2/20 = 10% include rate — below the 15% grumpy threshold.
    m = compute_mood("skeptic", verdicts)
    assert m["label"] == "grumpy"
    assert "10% include rate" in m["detail"]


def test_skeptic_credulous_when_include_rate_high() -> None:
    verdicts = (
        [_verdict("skeptic", 85, action="include")] * 12
        + [_verdict("skeptic", 30, action="skip")] * 8
    )
    # 12/20 = 60% include rate — above the 40% credulous threshold.
    m = compute_mood("skeptic", verdicts)
    assert m["label"] == "credulous"


def test_skeptic_going_through_motions_when_flat() -> None:
    # Include rate in the neutral zone (15%-40%) AND scores barely move (σ<8).
    # Per-spec priority is: include_rate first, then stdev — so we need a mix of
    # actions to dodge the grumpy/credulous branches.
    verdicts = (
        [_verdict("skeptic", 76, action="include")] * 2  # 20% include rate → neutral
        + [_verdict("skeptic", 74, action="skip")] * 8   # flat band of scores
    )
    m = compute_mood("skeptic", verdicts)
    assert m["label"] == "going through the motions"


def test_scout_unimpressed_when_mean_low() -> None:
    verdicts = [_verdict("scout", 25, action="skip")] * 10
    m = compute_mood("scout", verdicts)
    assert m["label"] == "unimpressed"
    assert "mean score 25" in m["detail"]


def test_scout_excited_when_mean_high() -> None:
    verdicts = [_verdict("scout", 80, action="include")] * 10
    m = compute_mood("scout", verdicts)
    assert m["label"] == "excited"
    assert "mean score 80" in m["detail"]


def test_scout_all_over_the_place_when_dispersion_high() -> None:
    # Mean in the middle, stdev > 25.
    scores = [10, 90, 10, 90, 10, 90, 50, 50]
    verdicts = [_verdict("scout", s, action="include" if s >= 75 else "skip") for s in scores]
    m = compute_mood("scout", verdicts)
    assert m["label"] == "all over the place"


def test_operator_unconvinced_when_nothing_ships() -> None:
    verdicts = [_verdict("operator", 30, action="skip")] * 10
    m = compute_mood("operator", verdicts)
    assert m["label"] == "unconvinced"


def test_operator_ready_to_ship_when_include_high() -> None:
    verdicts = (
        [_verdict("operator", 85, action="include")] * 12
        + [_verdict("operator", 30, action="skip")] * 8
    )
    m = compute_mood("operator", verdicts)
    assert m["label"] == "ready to ship"


def test_operator_pragmatic_is_the_default_middle() -> None:
    # Moderate include rate and normal dispersion.
    scores = [78, 30, 40, 76, 55, 35, 50, 82, 42, 48]
    verdicts = [
        _verdict("operator", s, action="include" if s >= 75 else "skip") for s in scores
    ]
    m = compute_mood("operator", verdicts)
    assert m["label"] == "pragmatic"


def test_unknown_evaluator_returns_unknown_label() -> None:
    verdicts = [_verdict("whoever", 50, action="skip")] * 10
    m = compute_mood("whoever", verdicts)
    assert m["label"] == "unknown"
