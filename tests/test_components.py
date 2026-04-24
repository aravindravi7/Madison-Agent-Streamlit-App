"""Tests for reusable UI badge helpers.

Pure HTML-string assertions — no Streamlit runtime needed.
"""

from __future__ import annotations

from signalscout.ui.components import bandit_phase_badge, disagreement_badge


def test_calibrating_badge_includes_parsed_round_count() -> None:
    html = bandit_phase_badge("Calibrating — round 3 of 10")
    assert "CALIBRATING · 3/10" in html
    assert "#FFB547" in html  # amber accent


def test_calibrating_badge_falls_back_without_round_digits() -> None:
    html = bandit_phase_badge("Calibrating")
    assert "CALIBRATING" in html
    assert "·" not in html  # no fraction when parse fails
    assert "#FFB547" in html


def test_learned_preference_badge_is_quiet_aquamarine() -> None:
    html = bandit_phase_badge("Learned preference — sampled scout 0.82 vs skeptic 0.41")
    assert "LEARNED" in html
    assert "#00F5D4" in html  # Aquamarine
    # Quieter alpha on border + text (60% / 99% suffix).
    assert "#00F5D466" in html or "#00F5D499" in html


def test_unknown_reason_renders_no_badge() -> None:
    assert bandit_phase_badge("") == ""
    assert bandit_phase_badge("Highest score among evaluators") == ""
    assert bandit_phase_badge(None) == ""  # type: ignore[arg-type]


def test_disagreement_badge_tiers() -> None:
    assert "HIGH" in disagreement_badge(20)
    assert "MEDIUM" in disagreement_badge(10)
    assert "LOW" in disagreement_badge(5)
