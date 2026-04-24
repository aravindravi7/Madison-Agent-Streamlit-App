"""Seed ``demo_user`` with synthetic decisions + feedback for a populated demo.

Extracted from the Phase 3C local seeding script so a Streamlit Cloud deploy
can populate its (ephemeral) database with one button click — critical for
demo day when a judge opens the app and must see a live Arena + Learning
state immediately.

Determinism: reward patterns are generated from a fixed seed so the same
"user taste" story appears every time — Scout dominates agent-tagged items,
Operator earns moderate credit on production-feeling items, Skeptic stays
conservative. The bandit learns to pick Scout for agent-tagged contexts.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np

from .bandit import TasteBandit
from .evaluators import EVALUATORS
from .models import ArenaResult, EvaluatorVerdict, Feedback, Item
from .storage import Storage

__all__ = ["seed_demo_user", "DEMO_USER_ID", "DEMO_DECISION_COUNT"]

DEMO_USER_ID = "demo_user"
DEMO_DECISION_COUNT = 30

_RANDOM_SEED = 2026
_NUMPY_SEED = 2026
_BANDIT_SEED = 11

_TAG_POOL: dict[str, list[list[str]]] = {
    "scout":    [["agents", "evaluation"], ["llm", "emergence"], ["agents", "tools"]],
    "skeptic":  [["methodology", "benchmarks"], ["replication", "stats"]],
    "operator": [["inference", "latency"], ["deployment", "cost"]],
}

_TITLE_POOL = [
    "Emergent Self-Correction in Multi-Agent Systems",
    "A Benchmark for Thermodynamic Reasoning in LLMs",
    "Inference Latency Reduction via Speculative Routing",
    "Replication Study: Chain-of-Thought at Scale",
    "Tool-Use Generalization Across Domain Shifts",
    "Cost-Aware Deployment Playbook for Open-Weight Models",
    "Agentic Workflows: When to Hand Off",
    "Stability of Attitude Change Models Under Fine-Tuning",
    "Evaluator Disagreement as a Novelty Signal",
    "Interpretability Probes for Tool-Augmented Agents",
    "Methodology Audit: Recent Safety Benchmarks",
    "Learning to Route Queries in Heterogeneous LLM Pools",
    "Signal-to-Noise in Research Aggregation Pipelines",
    "Scaling Laws for Multi-Agent Systems",
    "Zero-Shot Domain Transfer for Decision Support",
    "Red-Teaming Protocol for Production Agent Stacks",
    "Statistical Properties of LLM Output Distributions",
    "A Modular Framework for Agent Evaluation",
    "Latency Budgets in Agentic Pipelines",
    "Reproducibility Crisis in Agent Research",
    "Emergent Planning in Small Language Models",
    "Cost-Latency Pareto Frontier for Inference Serving",
    "Human-in-the-Loop Evaluation of Research Summaries",
    "Adversarial Prompts in Multi-Agent Debate",
    "Probing for Syntactic Understanding in LLMs",
    "Benchmarks for Long-Horizon Agent Tasks",
    "Memory Substrates for Persistent Agents",
    "A Framework for Evaluator Ensembles",
    "Production Traces of Agent Failure Modes",
    "Theoretical Limits of Context-Window Extension",
]


def _make_context(index: int) -> np.ndarray:
    """Deterministic context vector: alternate the has_agent_kw flag so half
    the items are agent-tagged (the cohort Scout tends to win on)."""
    ctx = np.zeros(15, dtype=np.float64)
    ctx[-1] = 1.0  # bias
    if index % 2 == 0:
        ctx[9] = 1.0  # has_agent_kw
    return ctx


def _decide_signal(arm_id: str, has_agent_kw: bool, py_rng: random.Random) -> tuple[str, float]:
    """Return (signal, reward) for a simulated user feedback event.

    Pattern the bandit should learn:
    - Scout on agent-tagged items → always 👍
    - Scout on non-agent items   → 70% 👍
    - Operator                   → 50/50
    - Skeptic                    → 70% 👎
    """
    if arm_id == "scout" and has_agent_kw:
        return "thumbs_up", 1.0
    if arm_id == "scout":
        return ("thumbs_up", 1.0) if py_rng.random() < 0.7 else ("thumbs_down", -1.0)
    if arm_id == "operator":
        return ("thumbs_up", 1.0) if py_rng.random() < 0.5 else ("thumbs_down", -1.0)
    # skeptic
    return ("thumbs_down", -1.0) if py_rng.random() < 0.7 else ("thumbs_up", 1.0)


def seed_demo_user(storage: Storage) -> int:
    """Populate storage with ``DEMO_DECISION_COUNT`` decisions + feedback
    events for ``DEMO_USER_ID``.

    Returns the number of decisions written. If ``demo_user`` already has
    bandit pulls recorded, this function is a no-op returning 0 so a second
    click of the seed button doesn't double-seed.
    """
    if storage.get_total_pulls(DEMO_USER_ID) > 0:
        return 0

    py_rng = random.Random(_RANDOM_SEED)
    np_rng = np.random.default_rng(_NUMPY_SEED)

    bandit = TasteBandit(
        user_id=DEMO_USER_ID,
        evaluator_ids=[e.id for e in EVALUATORS],
        storage=storage,
        rng=np.random.default_rng(_BANDIT_SEED),
    )

    now = datetime.now(UTC)
    evaluator_names = {e.id: e.name for e in EVALUATORS}

    for i in range(DEMO_DECISION_COUNT):
        item_id = f"demo-item-{i:02d}"
        title = _TITLE_POOL[i % len(_TITLE_POOL)]
        decision_time = now - timedelta(hours=DEMO_DECISION_COUNT - i)
        published = now - timedelta(days=max(1, i % 10))

        item = Item(
            id=item_id,
            source="arxiv_cs_ai",
            title=title,
            url=f"https://example.com/demo/{i:02d}",
            published_at=published,
            summary=(f"Summary for {title}. " * 5).strip(),
            fetched_at=now,
        )
        storage.upsert_item(item)

        ctx = _make_context(i)
        arm_id, _, reason = bandit.select_trusted_evaluator(ctx)
        bandit.record_pull(arm_id)

        verdicts: list[EvaluatorVerdict] = []
        for ev in EVALUATORS:
            score = 80 if ev.id == arm_id else int(np_rng.integers(40, 65))
            tag_set = py_rng.choice(_TAG_POOL[ev.id])
            verdicts.append(
                EvaluatorVerdict(
                    evaluator_id=ev.id,
                    score=score,
                    reasoning=f"{evaluator_names[ev.id]}'s assessment of {title[:40]}.",
                    confidence=0.75 + 0.2 * np_rng.random(),
                    topic_tags=list(tag_set),
                    action="include" if score >= 75 else "skip",
                    latency_ms=int(300 + 200 * np_rng.random()),
                )
            )
        winner_verdict = next(v for v in verdicts if v.evaluator_id == arm_id)
        result = ArenaResult(
            item_id=item_id,
            verdicts=sorted(verdicts, key=lambda v: v.evaluator_id),
            winner_id=arm_id,
            winner_reason=reason,
            bandit_sampled_values={},
            final_score=winner_verdict.score,
            final_action=winner_verdict.action,
            disagreement=float(
                np.std([v.score for v in verdicts])
            ),
        )
        storage.save_arena_result(result, user_id=DEMO_USER_ID, context_vector=ctx.tolist())

        signal, reward = _decide_signal(arm_id, bool(ctx[9] == 1.0), py_rng)
        storage.save_feedback(
            Feedback(
                item_id=item_id,
                user_id=DEMO_USER_ID,
                signal=signal,  # type: ignore[arg-type]
                evaluator_id=arm_id,
                timestamp=decision_time + timedelta(minutes=1),
            )
        )
        bandit.update_from_feedback(arm_id, ctx, reward=reward)

    return DEMO_DECISION_COUNT
