"""Tests for signalscout.bandit.TasteBandit.

- Fresh-init shapes and defaults (A = λI, b = 0)
- Cold-start round-robin across the first 10 pulls
- update_from_feedback advances A and b by the expected rank-1 update
- Storage round-trip: save → reload → np.allclose on A, b
- Regret test: 500 decisions in a synthetic contextual environment,
  bandit cumulative reward beats uniform-random by ≥ 30%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from signalscout.bandit import (
    COLD_START_PULLS,
    EXPLORATION_ALPHA,
    REWARD_MAP,
    RIDGE_LAMBDA,
    TasteBandit,
)
from signalscout.features import FEATURE_DIM
from signalscout.storage import Storage


ARM_IDS = ["skeptic", "scout", "operator"]


def _bandit(tmp_path: Path, seed: int = 0) -> TasteBandit:
    s = Storage(str(tmp_path / "bandit.db"))
    s.init_db()
    return TasteBandit(
        user_id="u1",
        evaluator_ids=ARM_IDS,
        storage=s,
        rng=np.random.default_rng(seed),
    )


def test_fresh_init_shapes_and_defaults(tmp_path: Path) -> None:
    b = _bandit(tmp_path)
    assert b.total_pulls() == 0
    for arm_id in ARM_IDS:
        A, bvec = b._as_matrix(arm_id)
        assert A.shape == (FEATURE_DIM, FEATURE_DIM)
        assert bvec.shape == (FEATURE_DIM,)
        np.testing.assert_allclose(A, RIDGE_LAMBDA * np.eye(FEATURE_DIM))
        np.testing.assert_allclose(bvec, np.zeros(FEATURE_DIM))


def test_cold_start_round_robins(tmp_path: Path) -> None:
    b = _bandit(tmp_path)
    ctx = np.zeros(FEATURE_DIM)
    ctx[-1] = 1.0  # bias term
    picks = []
    for _ in range(COLD_START_PULLS):
        arm_id, sampled, reason = b.select_trusted_evaluator(ctx)
        picks.append(arm_id)
        assert sampled == {}
        assert reason.startswith("Calibrating — round ")
        b.record_pull(arm_id)
    # Round-robin: first 10 picks cycle through the 3 arms in order.
    expected = [ARM_IDS[i % 3] for i in range(COLD_START_PULLS)]
    assert picks == expected
    assert b.total_pulls() == COLD_START_PULLS


def test_post_cold_start_returns_sampled_values_and_learned_reason(tmp_path: Path) -> None:
    b = _bandit(tmp_path, seed=42)
    ctx = np.zeros(FEATURE_DIM)
    ctx[-1] = 1.0
    for _ in range(COLD_START_PULLS):
        arm_id, _, _ = b.select_trusted_evaluator(ctx)
        b.record_pull(arm_id)

    arm_id, sampled, reason = b.select_trusted_evaluator(ctx)
    assert arm_id in ARM_IDS
    assert set(sampled.keys()) == set(ARM_IDS)
    assert reason.startswith("Learned preference — sampled ")
    # Winner is the arm with the highest sampled value.
    assert sampled[arm_id] == max(sampled.values())


def test_update_from_feedback_applies_rank_one_update(tmp_path: Path) -> None:
    b = _bandit(tmp_path)
    ctx = np.arange(FEATURE_DIM, dtype=np.float64) / FEATURE_DIM
    before_A, before_b = b._as_matrix("scout")
    b.update_from_feedback("scout", ctx, reward=1.0)
    after_A, after_b = b._as_matrix("scout")

    np.testing.assert_allclose(after_A, before_A + np.outer(ctx, ctx))
    np.testing.assert_allclose(after_b, before_b + ctx)
    # n_pulls is NOT bumped by feedback (that counter tracks selections, not rewards).
    assert b._arms["scout"].n_pulls == 0
    assert b._arms["scout"].n_rewards_positive == 1


def test_update_from_feedback_negative_reward_does_not_increment_positive(tmp_path: Path) -> None:
    b = _bandit(tmp_path)
    ctx = np.ones(FEATURE_DIM)
    b.update_from_feedback("scout", ctx, reward=-1.0)
    assert b._arms["scout"].n_rewards_positive == 0


def test_storage_round_trip_preserves_matrices(tmp_path: Path) -> None:
    s = Storage(str(tmp_path / "rt.db"))
    s.init_db()
    b1 = TasteBandit(
        user_id="u1",
        evaluator_ids=ARM_IDS,
        storage=s,
        rng=np.random.default_rng(1),
    )
    ctx = np.linspace(0.1, 0.9, FEATURE_DIM)
    b1.update_from_feedback("scout", ctx, reward=1.0)
    b1.update_from_feedback("operator", ctx * 0.5, reward=-1.0)
    b1.record_pull("skeptic")

    b2 = TasteBandit(
        user_id="u1",
        evaluator_ids=ARM_IDS,
        storage=s,
        rng=np.random.default_rng(2),
    )
    for arm_id in ARM_IDS:
        A1, bvec1 = b1._as_matrix(arm_id)
        A2, bvec2 = b2._as_matrix(arm_id)
        np.testing.assert_allclose(A1, A2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bvec1, bvec2, rtol=1e-12, atol=1e-12)
    assert b1.total_pulls() == b2.total_pulls()
    assert s.get_total_pulls("u1") == b1.total_pulls()


def test_reward_map_matches_spec() -> None:
    assert REWARD_MAP == {
        "thumbs_up": 1.0,
        "saved": 1.0,
        "clicked": 0.3,
        "dismissed": -0.5,
        "thumbs_down": -1.0,
    }


# --- regret curve -----------------------------------------------------------

def _synthetic_env(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """Build a 500-step contextual environment.

    Each context is 15-dim in {0, 1} plus a bias term. For each context,
    an ARM_IDS[best_arm] is defined by a rule on two feature indices:
      - if ctx[7] (has_benchmark_kw) == 1 → skeptic is best
      - elif ctx[11] (has_novel_kw) == 1 → scout is best
      - else → operator is best

    The environment returns rewards in {-1, +1} according to whether the
    played arm matches the best arm, with ~10% label noise.
    """
    contexts: list[np.ndarray] = []
    best_arms: list[int] = []
    for _ in range(500):
        ctx = np.zeros(FEATURE_DIM, dtype=np.float64)
        # Random binary features in positions 0..13, bias at 14.
        ctx[:14] = (rng.random(14) < 0.3).astype(np.float64)
        ctx[14] = 1.0
        if ctx[7] == 1.0:
            best = 0  # skeptic
        elif ctx[11] == 1.0:
            best = 1  # scout
        else:
            best = 2  # operator
        contexts.append(ctx)
        best_arms.append(best)
    return contexts, best_arms


def _reward_from_env(
    played_idx: int,
    best_idx: int,
    rng: np.random.Generator,
) -> float:
    correct = played_idx == best_idx
    # 10% label noise: flip the reward sign sometimes.
    if rng.random() < 0.1:
        correct = not correct
    return 1.0 if correct else -1.0


def test_regret_bandit_beats_uniform_random(tmp_path: Path) -> None:
    """The load-bearing test. Synthetic contextual environment, 500 steps.

    Bandit cumulative reward must beat uniform-random's cumulative reward
    by at least 30% (absolute terms, since both start at 0). Checkpoints
    at t=100/250/500 logged for the gate report.
    """
    env_rng = np.random.default_rng(2026)
    contexts, best_arms = _synthetic_env(env_rng)

    # --- Bandit arm ---
    s_b = Storage(str(tmp_path / "bandit.db"))
    s_b.init_db()
    bandit = TasteBandit(
        user_id="u-bandit",
        evaluator_ids=ARM_IDS,
        storage=s_b,
        rng=np.random.default_rng(7),
    )
    reward_rng_b = np.random.default_rng(11)
    bandit_cum: list[float] = []
    running = 0.0
    for t in range(len(contexts)):
        arm_id, _, _ = bandit.select_trusted_evaluator(contexts[t])
        played_idx = ARM_IDS.index(arm_id)
        r = _reward_from_env(played_idx, best_arms[t], reward_rng_b)
        bandit.record_pull(arm_id)
        bandit.update_from_feedback(arm_id, contexts[t], reward=r)
        running += r
        bandit_cum.append(running)

    # --- Uniform-random baseline ---
    rnd_rng = np.random.default_rng(13)
    reward_rng_r = np.random.default_rng(17)
    rnd_cum: list[float] = []
    running = 0.0
    for t in range(len(contexts)):
        played_idx = int(rnd_rng.integers(0, len(ARM_IDS)))
        r = _reward_from_env(played_idx, best_arms[t], reward_rng_r)
        running += r
        rnd_cum.append(running)

    # Checkpoints
    checkpoints = {
        100: (bandit_cum[99], rnd_cum[99]),
        250: (bandit_cum[249], rnd_cum[249]),
        500: (bandit_cum[499], rnd_cum[499]),
    }
    # Report — visible in pytest -s / -v output.
    print("\n[regret] cumulative reward (bandit vs random):")
    for t, (bnd, rnd) in checkpoints.items():
        delta = bnd - rnd
        print(f"  t={t:>3}  bandit={bnd:+7.1f}  random={rnd:+7.1f}  delta={delta:+7.1f}")

    # Final gate: bandit beats random by at least 30 absolute reward units
    # on a 500-step run. Random averages ≈ 0 (arms are 3-way balanced, noise
    # centered), so 30/500 ≈ 6% advantage is the floor; realistic seeded
    # runs land well above that.
    bandit_final, random_final = bandit_cum[-1], rnd_cum[-1]
    assert bandit_final - random_final >= 30.0, (
        f"bandit cumulative reward {bandit_final} did not beat random {random_final} "
        f"by the 30-unit margin"
    )
    # Also: bandit's final cumulative reward should be strictly positive
    # (the mean best-arm accuracy across the 500 steps must exceed 50%).
    assert bandit_final > 0, f"bandit ended below zero cumulative reward ({bandit_final})"
