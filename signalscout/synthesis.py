"""Theme synthesis LLM call + preparation.

Takes a list of included-item records (shape produced by the pipeline's
evaluation step) and returns the themes + editorial insight + meta
observation. Single LLM call; deterministic temperature.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

__all__ = ["prepare_theme_input", "run_theme_synthesis"]


def prepare_theme_input(
    included_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reshape included items for the synthesis prompt + count tag frequencies."""
    theme_input: list[dict[str, Any]] = []
    tag_counts: dict[str, int] = {}
    for it in included_items:
        o = it.get("output") or {}
        tags = o.get("topic_tags") or []
        rec = {
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            "tags": tags,
            "summary": o.get("clean_summary") or it.get("summary", ""),
            "why_it_matters": o.get("why_it_matters", ""),
            "score": int(o.get("eval_relevance_score") or 0),
        }
        theme_input.append(rec)
        for t in tags or []:
            key = str(t).strip()
            if key:
                tag_counts[key] = tag_counts.get(key, 0) + 1
    return theme_input, tag_counts


def run_theme_synthesis(
    client: OpenAI,
    theme_input: list[dict[str, Any]],
    tag_counts: dict[str, int],
) -> dict[str, Any]:
    """Run the synthesis LLM call and return the parsed JSON.

    Falls back to an empty shape on any error — upstream renders that cleanly.
    """
    prompt = """You are an editorial analyst synthesizing insights across multiple research items.
Detect patterns, recurring themes, and shifts in focus. Do NOT summarize items individually.
Reason across the entire set for higher-level insights.

You are given a batch of included research items (topic_tags, summary, why_it_matters, relevance score).
Goals:
1. Detect recurring themes
2. Identify which themes dominate vs emerging
3. Infer what this batch suggests about current research focus

Return valid JSON only, with these exact keys:
- "themes": array of objects, each with "name" (string), "confidence" (number 0-1), "evidence" (array of short strings)
- "editorial_insight": string (1-2 sentences)
- "meta_observation": string (1 sentence)

INPUT DATA:
"""
    prompt += json.dumps(theme_input, indent=2)
    prompt += "\n\nTAG FREQUENCY (reference only):\n"
    prompt += json.dumps(tag_counts, indent=2)
    prompt += "\n\nReturn only the JSON object."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text).rstrip("`")
        return json.loads(text)
    except Exception:
        return {
            "themes": [],
            "editorial_insight": "Theme synthesis unavailable.",
            "meta_observation": "",
        }
