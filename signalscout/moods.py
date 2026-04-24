"""Evaluator 'moods' — a legible summary of how each persona has been scoring.

Each evaluator's mood is computed from the last N verdicts it produced for
this user. The thresholds below are per-evaluator on purpose: each persona
has its own axis (rigor / novelty / applicability), so "grumpy" for the
Skeptic means something different than "unimpressed" for the Scout.

Returned shape: ``{"label": str, "detail": str}``. ``label`` is the mood
word; ``detail`` is a short factual string showing the number that drove it.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any

from .models import EvaluatorVerdict

__all__ = ["compute_mood", "MIN_VERDICTS"]

MIN_VERDICTS = 5


def compute_mood(
    evaluator_id: str,
    recent_verdicts: list[EvaluatorVerdict],
) -> dict[str, Any]:
    """Return ``{"label", "detail"}`` for display above the Arena view.

    If fewer than ``MIN_VERDICTS`` verdicts are available, returns
    ``calibrating``. Otherwise dispatches to a per-evaluator mood function.
    """
    if len(recent_verdicts) < MIN_VERDICTS:
        return {"label": "calibrating", "detail": "needs more data"}

    include_rate = _include_rate(recent_verdicts)
    mean = _mean(recent_verdicts)
    stdev = _stdev(recent_verdicts)

    if evaluator_id == "skeptic":
        return _skeptic_mood(include_rate, mean, stdev, len(recent_verdicts))
    if evaluator_id == "scout":
        return _scout_mood(include_rate, mean, stdev, len(recent_verdicts))
    if evaluator_id == "operator":
        return _operator_mood(include_rate, mean, stdev, len(recent_verdicts))

    # Unknown evaluator: fall back to a neutral shape.
    return {"label": "unknown", "detail": f"no mood rules for {evaluator_id}"}


def _include_rate(verdicts: list[EvaluatorVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.action == "include") / len(verdicts)


def _mean(verdicts: list[EvaluatorVerdict]) -> float:
    if not verdicts:
        return 0.0
    return sum(v.score for v in verdicts) / len(verdicts)


def _stdev(verdicts: list[EvaluatorVerdict]) -> float:
    if len(verdicts) < 2:
        return 0.0
    return float(pstdev(v.score for v in verdicts))


def _skeptic_mood(include_rate: float, mean: float, stdev: float, n: int) -> dict[str, str]:
    """Skeptic = rigor axis. Mood tracks include rate (are they rejecting enough?)
    and dispersion (are they making distinctions?)."""
    pct = int(round(include_rate * 100))
    if include_rate < 0.15:
        return {"label": "grumpy", "detail": f"{pct}% include rate over last {n} items"}
    if include_rate > 0.40:
        return {"label": "credulous", "detail": f"{pct}% include rate over last {n} items"}
    if stdev < 8:
        return {"label": "going through the motions", "detail": f"scores barely move (σ={stdev:.1f})"}
    return {"label": "measured", "detail": f"{pct}% include, σ={stdev:.1f}"}


def _scout_mood(include_rate: float, mean: float, stdev: float, n: int) -> dict[str, str]:
    """Scout = novelty axis. Low mean = unimpressed; high mean = excited;
    very high dispersion = pattern hunting gone wild."""
    if mean < 35:
        return {"label": "unimpressed", "detail": f"mean score {mean:.0f} — nothing novel this week"}
    if mean > 65:
        return {"label": "excited", "detail": f"mean score {mean:.0f}"}
    if stdev > 25:
        return {"label": "all over the place", "detail": f"σ={stdev:.1f}, mean {mean:.0f}"}
    return {"label": "scanning", "detail": f"mean {mean:.0f}, σ={stdev:.1f}"}


def _operator_mood(include_rate: float, mean: float, stdev: float, n: int) -> dict[str, str]:
    """Operator = applicability axis. Low include = nothing ships; high include
    = pipeline is full; low dispersion with otherwise normal include = going through
    the motions without strong opinions."""
    pct = int(round(include_rate * 100))
    if include_rate < 0.15:
        return {"label": "unconvinced", "detail": f"{pct}% include rate over last {n} items"}
    if include_rate > 0.40:
        return {"label": "ready to ship", "detail": f"{pct}% include rate over last {n} items"}
    if stdev < 8:
        return {"label": "unimpressed", "detail": f"scores barely move (σ={stdev:.1f})"}
    return {"label": "pragmatic", "detail": f"{pct}% include, σ={stdev:.1f}"}
