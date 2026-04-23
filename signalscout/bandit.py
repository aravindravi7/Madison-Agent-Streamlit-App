"""TasteBandit — contextual Thompson Sampling over the three evaluators.

The decision problem
--------------------
For each item, three evaluators produce verdicts. Which one should we trust
for *this* item and *this* user? That's a contextual multi-armed bandit.
Arms = evaluators (3). Context = 15-dim feature vector. Reward = user
feedback ({-1, -0.5, +0.3, +1} via ``REWARD_MAP``).

Linear Thompson Sampling (per-arm Bayesian linear regression)
-------------------------------------------------------------
Each arm ``a`` maintains:
  - precision matrix ``A_a`` ∈ ℝ^{d×d}, initialized to ``λI`` with λ=1.0
  - vector ``b_a`` ∈ ℝ^d, initialized to 0

Posterior mean:    μ_a = A_a⁻¹ b_a         (solved, not inverted)
Posterior cov:     Σ_a = α² A_a⁻¹          (α controls exploration)

Arm selection for a new context ``x``:
  θ_a  ~ N(μ_a, Σ_a)
  v_a  = θ_a · x
  pick = argmax_a v_a

Update on observing reward ``r`` for the arm played with context ``x``:
  A_a ← A_a + x xᵀ
  b_a ← b_a + r x

We use ``np.linalg.solve`` for A⁻¹b rather than computing A⁻¹ explicitly —
more numerically stable once A accumulates many rank-1 updates. Same
pattern for sampling: we draw from ``N(μ, α² A⁻¹)`` via a Cholesky of A
rather than inverting A. The solves never diverge because A is
positive-definite by construction (starts as λI, each update adds xxᵀ).

Cold start
----------
First 10 pulls across all arms for this user are round-robin, regardless
of which ``run_brief`` they fall in. We rely on ``get_total_pulls`` being
``sum(n_pulls)`` across arms. Once past 10, we switch to Thompson sampling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .features import FEATURE_DIM
from .models import BanditArmState
from .storage import Storage

__all__ = [
    "TasteBandit",
    "REWARD_MAP",
    "EXPLORATION_ALPHA",
    "RIDGE_LAMBDA",
    "COLD_START_PULLS",
]

REWARD_MAP: dict[str, float] = {
    "thumbs_up": 1.0,
    "saved": 1.0,
    "clicked": 0.3,
    "dismissed": -0.5,
    "thumbs_down": -1.0,
}

EXPLORATION_ALPHA = 0.5
RIDGE_LAMBDA = 1.0
COLD_START_PULLS = 10


def _initial_arm(evaluator_id: str) -> BanditArmState:
    return BanditArmState(
        evaluator_id=evaluator_id,
        A=(RIDGE_LAMBDA * np.eye(FEATURE_DIM)).tolist(),
        b=[0.0] * FEATURE_DIM,
        n_pulls=0,
        n_rewards_positive=0,
    )


class TasteBandit:
    """Per-user contextual Thompson Sampling over a fixed evaluator set.

    State is loaded from ``storage`` on init and persisted on every update.
    Missing arms are materialized with λI / 0 defaults so a fresh user
    works without any explicit seeding.
    """

    def __init__(
        self,
        *,
        user_id: str,
        evaluator_ids: list[str],
        storage: Storage,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if not user_id:
            raise ValueError("TasteBandit requires a non-empty user_id")
        if not evaluator_ids:
            raise ValueError("TasteBandit requires at least one evaluator_id")
        self.user_id = user_id
        self.evaluator_ids = list(evaluator_ids)
        self.storage = storage
        self.rng = rng if rng is not None else np.random.default_rng()

        loaded = storage.get_bandit_state(user_id)
        self._arms: dict[str, BanditArmState] = {}
        for eid in self.evaluator_ids:
            self._arms[eid] = loaded.get(eid) or _initial_arm(eid)

    # ------------------------------------------------------------ primitives

    def total_pulls(self) -> int:
        """Sum of pulls across all arms for this user."""
        return int(sum(arm.n_pulls for arm in self._arms.values()))

    def _as_matrix(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        arm = self._arms[arm_id]
        A = np.asarray(arm.A, dtype=np.float64)
        b = np.asarray(arm.b, dtype=np.float64)
        return A, b

    # ------------------------------------------------------------ selection

    def select_trusted_evaluator(
        self, context: np.ndarray
    ) -> tuple[str, dict[str, float], str]:
        """Pick the evaluator to trust for this context.

        Returns ``(evaluator_id, sampled_values, reason)``. Cold-start phase
        (first 10 pulls total) round-robins and returns ``sampled_values={}``;
        post-cold-start returns each arm's Thompson sample.
        """
        x = np.asarray(context, dtype=np.float64).reshape(FEATURE_DIM)

        total = self.total_pulls()
        if total < COLD_START_PULLS:
            arm_id = self.evaluator_ids[total % len(self.evaluator_ids)]
            reason = f"Calibrating — round {total + 1} of {COLD_START_PULLS}"
            return arm_id, {}, reason

        sampled: dict[str, float] = {}
        for eid in self.evaluator_ids:
            A, b = self._as_matrix(eid)
            mean = np.linalg.solve(A, b)  # μ = A⁻¹ b
            # Sample θ ~ N(μ, α² A⁻¹). Draw ε ~ N(0, I), solve A^{1/2} y = ε,
            # then θ = μ + α y. We do this via Cholesky of A: A = L Lᵀ,
            # and solve Lᵀ y = ε for y (so Cov(y) = A⁻¹ in expectation).
            try:
                L = np.linalg.cholesky(A)
                eps = self.rng.standard_normal(FEATURE_DIM)
                y = np.linalg.solve(L.T, eps)
            except np.linalg.LinAlgError:
                y = np.zeros(FEATURE_DIM)
            theta = mean + EXPLORATION_ALPHA * y
            sampled[eid] = float(theta @ x)

        arm_id = max(sampled, key=lambda k: sampled[k])
        reason = self._format_reason(arm_id, sampled)
        return arm_id, sampled, reason

    def _format_reason(self, winner_id: str, sampled: dict[str, float]) -> str:
        """One-line explanation suitable for the Arena card."""
        parts = [f"{eid} {sampled[eid]:.2f}" for eid in self.evaluator_ids]
        # Put the winner first for legibility.
        ordered = sorted(
            parts,
            key=lambda s: (0 if s.startswith(winner_id) else 1, s),
        )
        return "Learned preference — sampled " + " vs ".join(ordered)

    def explain(self, evaluator_id: str, context: np.ndarray) -> str:
        """Deterministic explanation for a specific arm given current state.

        This is separate from ``select_trusted_evaluator`` so callers can show
        a stable rationale for an already-made pick without drawing a new sample.
        """
        x = np.asarray(context, dtype=np.float64).reshape(FEATURE_DIM)
        posterior_means = {}
        for eid in self.evaluator_ids:
            A, b = self._as_matrix(eid)
            mean = np.linalg.solve(A, b)
            posterior_means[eid] = float(mean @ x)
        parts = [f"{eid} {posterior_means[eid]:+.2f}" for eid in self.evaluator_ids]
        return (
            f"{evaluator_id} trusted here — posterior means "
            + " · ".join(parts)
        )

    # ------------------------------------------------------------ updates

    def update_from_feedback(
        self,
        evaluator_id: str,
        context: np.ndarray,
        reward: float,
    ) -> None:
        """Apply one feedback event: update A, b and the positive-reward counter.

        ``n_pulls`` is NOT incremented here — that counter tracks *selections*
        and is bumped in ``record_pull`` at decision time. Feedback can arrive
        late (a user may click 👍 days after the brief was generated) so we
        keep the two counters separate to avoid double-counting the pull.
        """
        if evaluator_id not in self._arms:
            raise KeyError(f"unknown evaluator_id: {evaluator_id}")
        x = np.asarray(context, dtype=np.float64).reshape(FEATURE_DIM)
        arm = self._arms[evaluator_id]
        A = np.asarray(arm.A, dtype=np.float64)
        b = np.asarray(arm.b, dtype=np.float64)
        A = A + np.outer(x, x)
        b = b + float(reward) * x

        new_pos = arm.n_rewards_positive + (1 if reward > 0 else 0)
        self._arms[evaluator_id] = BanditArmState(
            evaluator_id=evaluator_id,
            A=A.tolist(),
            b=b.tolist(),
            n_pulls=arm.n_pulls,
            n_rewards_positive=new_pos,
        )
        self._persist()

    def record_pull(self, evaluator_id: str) -> None:
        """Increment pull counter without a reward update.

        Used at decision time so ``total_pulls`` reflects selections even
        before feedback lands — required for the cold-start → learned
        transition to be based on selections, not on observed rewards.
        """
        if evaluator_id not in self._arms:
            raise KeyError(f"unknown evaluator_id: {evaluator_id}")
        arm = self._arms[evaluator_id]
        self._arms[evaluator_id] = BanditArmState(
            evaluator_id=evaluator_id,
            A=arm.A,
            b=arm.b,
            n_pulls=arm.n_pulls + 1,
            n_rewards_positive=arm.n_rewards_positive,
        )
        self._persist()

    def _persist(self) -> None:
        self.storage.save_bandit_state(self.user_id, self._arms)

    # ------------------------------------------------------------ analytics

    def get_win_rates(self) -> dict[str, dict[str, float]]:
        """Per-arm summary: pulls + observed positive-reward rate."""
        out: dict[str, dict[str, float]] = {}
        for eid, arm in self._arms.items():
            pulls = int(arm.n_pulls)
            pos = int(arm.n_rewards_positive)
            rate = (pos / pulls) if pulls > 0 else 0.0
            out[eid] = {"pulls": float(pulls), "positive_rate": float(rate)}
        return out
