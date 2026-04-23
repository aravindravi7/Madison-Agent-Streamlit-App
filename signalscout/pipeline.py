"""Pipeline orchestrator.

Phase 1 wiring: fetch → clean → legacy per-item eval → theme synthesis →
result container. The evaluator here is the *legacy* single-prompt scorer
ported from the original ``app.py`` so the Run tab keeps working end-to-end
while Phases 2–3 introduce the three-evaluator arena and the bandit.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from openai import OpenAI

from .models import Item
from .sources import BUILTIN_SOURCES, clean_and_validate
from .synthesis import prepare_theme_input, run_theme_synthesis

__all__ = ["run_workflow", "ProgressCallback"]

ProgressCallback = Callable[[int, int, str], None]


def _legacy_eval_one_item(client: OpenAI, item: Item) -> dict[str, Any]:
    """Single-prompt scorer ported from the pre-refactor ``app.py``.

    Returns a dict in the legacy shape the Phase 1 renderer expects:
    ``{ ...item fields..., "output": {...} }``. This is transitional —
    Phase 2 replaces it with the three-evaluator arena.
    """
    prompt = f"""You are an assistant helping build a model behavior evaluation research brief.

You will be given ONE content item (title + summary + source + url + published date). Your job is to:
- Tag the topic
- Score how relevant it is to model behavior evaluation
- Produce a clean, human-readable 1–2 sentence summary
- Decide whether to include it in the brief

Output rules (strict):
- Return a JSON object only. No markdown, no code fences, no extra text.
- Use exactly these keys: topic_tags, eval_relevance_score, why_it_matters, clean_summary, action, action_reason

Scoring (0–100 integer; spread scores):
- 90–100 (include): directly about evaluation methods, benchmarks, eval frameworks, red-teaming, alignment/safety evals.
- 70–89 (include): strong relevance to LLM behavior/capabilities/risks.
- 40–69 (usually skip): AI/ML only indirectly related.
- 0–39 (skip): no meaningful connection to LLM behavior/evaluation.

Action: if score >= 70 → action = "include"; if score < 70 → action = "skip".

topic_tags: 3–6 concise tags (e.g. "LLM Evaluation", "Benchmarking", "Safety").
why_it_matters: 1–2 sentences, specific.
clean_summary: 1–2 sentences for product/engineering audience.
action_reason: 1 sentence explaining include/skip.

Input content item:
Source: {item.source}
Title: {item.title}
Published: {item.published_at.isoformat()}
URL: {item.url}
Summary: {item.summary}

Return only the JSON object."""

    base = {
        "source": item.source,
        "title": item.title,
        "url": item.url,
        "published_at": item.published_at.isoformat(),
        "summary": item.summary,
    }
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if "```" in text:
            text = re.sub(r"^```\w*\n?", "", text).rstrip("`").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            data = json.loads(match.group(0)) if match else None
        if not data:
            raise ValueError("No JSON in response")
        score = int(data.get("eval_relevance_score") or 0)
        action = str(data.get("action") or "skip").lower().strip()
        if score >= 70 and action != "include":
            action = "include"
        return {
            **base,
            "output": {
                "topic_tags": data.get("topic_tags") or [],
                "eval_relevance_score": score,
                "why_it_matters": data.get("why_it_matters", ""),
                "clean_summary": data.get("clean_summary", ""),
                "action": action,
                "action_reason": data.get("action_reason", ""),
            },
        }
    except Exception as e:
        return {
            **base,
            "output": {
                "topic_tags": [],
                "eval_relevance_score": 0,
                "why_it_matters": "",
                "clean_summary": item.summary[:200],
                "action": "skip",
                "action_reason": str(e),
            },
        }


def run_workflow(
    client: OpenAI,
    arxiv_limit: int,
    smol_limit: int,
    max_evaluate: int,
    on_progress: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Full Phase 1 pipeline. Returns a container dict consumed by brief_view."""
    sources_by_id = {s.id: s for s in BUILTIN_SOURCES}
    arxiv = sources_by_id["arxiv_cs_ai"].fetch(arxiv_limit)
    smol = sources_by_id["smol"].fetch(smol_limit)
    combined = arxiv + smol
    cleaned = clean_and_validate(combined)
    limited = cleaned[: max(0, int(max_evaluate))]

    if not limited:
        return {"error": "No items passed validation.", "included_items": []}

    evaluated: list[dict[str, Any]] = []
    total = len(limited)
    for i, item in enumerate(limited):
        if on_progress:
            on_progress(i + 1, total, f"Evaluating item {i + 1}/{total}")
        evaluated.append(_legacy_eval_one_item(client, item))

    included = [e for e in evaluated if (e.get("output") or {}).get("action") == "include"]
    now = datetime.now(UTC)
    batch_id = now.strftime("%Y%m%dT%H%M%SZ")

    theme_input, tag_counts = prepare_theme_input(included)
    theme_synthesis = (
        run_theme_synthesis(client, theme_input, tag_counts) if included else {}
    )

    return {
        "report_title": "SignalScout Research Brief",
        "export_batch_id": batch_id,
        "generated_at": now.isoformat(),
        "included_count": len(included),
        "evaluated_count": len(evaluated),
        "included_items": included,
        "theme_synthesis": theme_synthesis,
    }
