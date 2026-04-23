"""Pipeline orchestrator.

Phase 2: fetch → clean → arena (three evaluators in parallel per item) →
persist ArenaResults → theme synthesis over included items → Brief record.
The bandit lands in Phase 3; for now the winner is whoever scored highest,
and ``winner_reason`` is a placeholder.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from statistics import pstdev
from typing import Any, Callable, Optional

from openai import OpenAI

from .evaluators import EVALUATORS, Evaluator
from .models import ArenaResult, Brief, EvaluatorVerdict, Item
from .sources import BUILTIN_SOURCES, clean_and_validate
from .storage import Storage
from .synthesis import prepare_theme_input, run_theme_synthesis

__all__ = [
    "run_arena_on_item",
    "run_brief",
    "ProgressCallback",
    "WINNER_REASON_HIGHEST_SCORE",
]

ProgressCallback = Callable[[int, int, str], None]

WINNER_REASON_HIGHEST_SCORE = "Highest score among evaluators"


def run_arena_on_item(
    client: OpenAI,
    item: Item,
    evaluators: list[Evaluator] = EVALUATORS,
) -> ArenaResult:
    """Run all evaluators on one item in parallel; return an ArenaResult.

    Uses ``ThreadPoolExecutor(max_workers=3)`` — one thread per evaluator.
    Wall-clock is bounded by the slowest evaluator, not the sum. The
    highest-scoring verdict wins (bandit-driven selection lands in Phase 3).
    """
    verdicts: list[EvaluatorVerdict] = []
    with ThreadPoolExecutor(max_workers=max(1, len(evaluators))) as ex:
        futures = {ex.submit(ev.evaluate, client, item): ev for ev in evaluators}
        for future in as_completed(futures):
            verdicts.append(future.result())

    if not verdicts:
        raise RuntimeError("no verdicts produced for item")

    # Deterministic ordering in the persisted verdicts regardless of completion order.
    verdicts.sort(key=lambda v: v.evaluator_id)

    winner = max(verdicts, key=lambda v: v.score)
    scores = [v.score for v in verdicts]
    disagreement = float(pstdev(scores)) if len(scores) > 1 else 0.0

    return ArenaResult(
        item_id=item.id,
        verdicts=verdicts,
        winner_id=winner.evaluator_id,
        winner_reason=WINNER_REASON_HIGHEST_SCORE,
        bandit_sampled_values={},
        final_score=winner.score,
        final_action=winner.action,
        disagreement=disagreement,
    )


def _verdict_to_legacy_output(winner: EvaluatorVerdict, summary: str) -> dict[str, Any]:
    """Adapt the winning verdict into the legacy ``output`` dict the HTML
    renderer + email still consume. Temporary bridge until Phase 4 rebuilds
    those surfaces against the new model set."""
    return {
        "topic_tags": winner.topic_tags,
        "eval_relevance_score": winner.score,
        "why_it_matters": winner.reasoning,
        "clean_summary": summary[:300],
        "action": winner.action,
        "action_reason": WINNER_REASON_HIGHEST_SCORE,
    }


def _make_legacy_included_item(item: Item, result: ArenaResult) -> dict[str, Any]:
    winner = next(v for v in result.verdicts if v.evaluator_id == result.winner_id)
    return {
        "id": item.id,
        "source": item.source,
        "title": item.title,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "summary": item.summary,
        "output": _verdict_to_legacy_output(winner, item.summary),
    }


def _brief_id(user_id: str, batch_id: str) -> str:
    h = hashlib.sha256(f"{user_id}:{batch_id}".encode("utf-8")).hexdigest()
    return f"brief-{h[:16]}"


def run_brief(
    client: OpenAI,
    storage: Storage,
    user_id: str,
    arxiv_limit: int,
    smol_limit: int,
    max_evaluate: int,
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Full pipeline. Persists items + arena results + brief; returns a
    container dict the UI layer consumes."""
    sources_by_id = {s.id: s for s in BUILTIN_SOURCES}
    arxiv = sources_by_id["arxiv_cs_ai"].fetch(arxiv_limit)
    smol = sources_by_id["smol"].fetch(smol_limit)
    combined = arxiv + smol
    cleaned = clean_and_validate(combined)
    limited = cleaned[: max(0, int(max_evaluate))]

    if not limited:
        return {
            "error": "No items passed validation.",
            "arena_results": [],
            "items_by_id": {},
            "included_items": [],
        }

    arena_results: list[ArenaResult] = []
    items_by_id: dict[str, Item] = {}
    total = len(limited)
    for i, item in enumerate(limited):
        if on_progress:
            on_progress(i + 1, total, f"Evaluating item {i + 1}/{total}")
        storage.upsert_item(item)
        items_by_id[item.id] = item
        result = run_arena_on_item(client, item)
        storage.save_arena_result(result, user_id=user_id)
        arena_results.append(result)

    included_arena = [r for r in arena_results if r.final_action == "include"]
    included_items_legacy = [
        _make_legacy_included_item(items_by_id[r.item_id], r) for r in included_arena
    ]

    theme_input, tag_counts = prepare_theme_input(included_items_legacy)
    theme_synthesis = (
        run_theme_synthesis(client, theme_input, tag_counts) if included_items_legacy else {}
    )

    now = datetime.now(UTC)
    batch_id = now.strftime("%Y%m%dT%H%M%SZ")
    brief_id = _brief_id(user_id, batch_id)
    brief = Brief(
        id=brief_id,
        generated_at=now,
        user_id=user_id,
        included_items=[r.item_id for r in included_arena],
        theme_synthesis=theme_synthesis,
        config={
            "arxiv_limit": int(arxiv_limit),
            "smol_limit": int(smol_limit),
            "max_evaluate": int(max_evaluate),
        },
    )
    storage.save_brief(brief)

    return {
        "report_title": "SignalScout Research Brief",
        "brief_id": brief_id,
        "export_batch_id": batch_id,
        "generated_at": now.isoformat(),
        "user_id": user_id,
        "evaluated_count": len(arena_results),
        "included_count": len(included_arena),
        "arena_results": arena_results,
        "items_by_id": items_by_id,
        "included_items": included_items_legacy,
        "theme_synthesis": theme_synthesis,
    }
