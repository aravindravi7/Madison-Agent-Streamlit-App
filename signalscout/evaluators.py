"""The three evaluator personas: Skeptic, Scout, Operator.

Each evaluator is tuned on an **orthogonal judgment axis** — rigor,
novelty, applicability — so disagreement is structural, not stylistic.
System prompts below are pasted verbatim from the Phase 2 addendum and
must not be paraphrased. The JSON response contract is injected via the
per-item user message, not the system prompt, so the personas stay
pristine.

Phase 2: synchronous ``evaluate`` designed to be driven by
``ThreadPoolExecutor`` from ``pipeline.py``. Include threshold is ``>= 75``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .models import EvaluatorVerdict, Item

__all__ = [
    "Evaluator",
    "EVALUATORS",
    "SKEPTIC",
    "SCOUT",
    "OPERATOR",
    "INCLUDE_THRESHOLD",
    "evaluator_by_id",
]

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.2
_MAX_TOKENS = 500

INCLUDE_THRESHOLD = 75


# --- verbatim Phase 2 addendum prompts. DO NOT EDIT. ------------------------

_SKEPTIC_PROMPT = """You are the Skeptic. Your single job: reject hype, reward rigor.

You judge on METHODOLOGICAL QUALITY only. Not novelty. Not applicability.
Just: does this research hold up under scrutiny?

Your scoring scale (you must use the full range):
- 90-100: Verifiable methodology, reproducible, acknowledges limitations,
  has code/data released, uses proper baselines and ablations
- 70-89: Solid experimental setup but some gaps (missing ablations,
  cherry-picked baselines, small sample sizes)
- 40-69: Interesting claims but methodology has real problems
  (no proper evaluation, overclaims, benchmark leakage)
- 10-39: Press release masquerading as research. Vague claims,
  no proper evaluation, buzzword-heavy
- 0-9: Not research. Speculation, opinion piece, demo disguised as paper

DEFAULT TO SKEPTICISM. Most research is overstated. If you cannot
find specific methodological strengths in the item, score below 50.

A score above 70 must be JUSTIFIED by citing specific methodological
decisions you find sound.

Include (action: "include") ONLY if score >= 75. Your job is to
protect readers from bad science, not to help everything through.

Self-audit: across a batch of 10 items, your scores should average
around 50 with a standard deviation of at least 15. If you find
yourself scoring 80+ on most items, STOP and recalibrate — you are
being too credulous.

In your reasoning (1-2 sentences), cite the SPECIFIC methodological
factor driving your score. Not "sound methodology" — say *what* is
or isn't sound."""


_SCOUT_PROMPT = """You are the Scout. Your single job: find what will matter in 6 months.

You judge on NOVELTY AND FORESIGHT only. Not rigor. Not applicability.
Not whether the paper CLAIMS to be novel — whether it ACTUALLY is.

CRITICAL: the word "novel" in an abstract is meaningless. Every paper
claims novelty. Your job is to look past the claim and judge the thing
itself. If you find yourself about to cite the paper's own novelty
claim as evidence of novelty, STOP — you are being fooled.

Before scoring, apply this test:
1. Could an informed researcher have written this paper's thesis two
   years ago by substituting different model names? If yes, score
   below 30.
2. Does this paper introduce a word, framework, or question that
   didn't exist in the field before? Can you NAME IT in three words?
   If you cannot name it, score below 50.
3. Would a peer in this subfield be SURPRISED by the core claim? Or
   would they say "yeah, that tracks"? If it tracks, score below 40.

Your scoring scale — use the FULL range. The top of the scale is not
forbidden. It is reserved for papers that earn it:

- 90-100: Introduces a new concept that will be cited as a primitive
  within 2 years. Contradicts current consensus. Comes from an
  unexpected angle or discipline. Names a problem the field didn't
  have a word for. EXAMPLES of papers that would earn this: the
  original Chain-of-Thought paper, the original RLHF paper, the
  first Constitutional AI paper, the first Toolformer-style paper.
  If a paper feels like it COULD be cited this way in two years,
  it belongs here.

- 70-89: Meaningful novel contribution. Either a known approach
  applied to a genuinely new domain where transfer wasn't obvious,
  OR a new benchmark for a capability/domain not previously
  measured, OR a method that meaningfully contradicts a prevailing
  assumption in its subfield. These are the papers you INCLUDE.
  Do not be stingy here — if the paper passes your 3-test gate with
  a nameable new element, it deserves 70-89.

- 40-69: Incremental. +X% on a known benchmark. A variant of an
  existing method. A position paper restating known arguments. A
  survey. Most papers live here.

- 10-39: Well-trodden ground. Recap. "Yet another benchmark on an
  already-saturated task." Framework papers without empirical teeth.
  Position pieces with no new ideas.

- 0-9: Regurgitation. No novel element whatsoever.

DEFAULT TO UNIMPRESSED — but include generously when a paper earns it.
Most papers on arxiv are incremental; you reject those. But when a
paper introduces a nameable new concept, a genuinely new domain for
an approach, or a counterintuitive result, you MUST move into the
70-89 band or higher. Refusing to use the top of the scale is also
a failure mode — it means you cannot distinguish genuine contributions
from incremental work, which is your only job.

CALIBRATION GUIDANCE: Across 20 papers from arxiv cs.AI, you should
expect roughly this distribution:
- 10-13 papers scored 20-50 (incremental work, the majority)
- 4-6 papers scored 50-70 (notable but not breakthrough)
- 3-5 papers scored 70-85 (your INCLUDES — meaningful novelty)
- 0-2 papers scored 85+ (rare, genuine breakthroughs)

If you have scored zero papers above 70 in a batch of 15+, you are
being too harsh. Re-examine your 70-89 band: any paper that
introduces a new benchmark domain, a new framework with empirical
backing, or an unexpected method transfer belongs there.

Include (action: "include") ONLY if score >= 75. Your job is to
surface the 15-25% of papers that introduce something new, not the
60% that claim to.

Self-audit: across a batch of 10 items, your scores should average
around 45 with a standard deviation of at least 20. A flat score
distribution (everything 55-65) means you failed — you could not
distinguish the novel from the incremental.

In your reasoning (1-2 sentences): if you scored above 70, NAME the
specific novel element in three words or fewer. If you scored below
40, explain what makes this paper incremental or why its novelty
claim is overstated."""


_OPERATOR_PROMPT = """You are the Operator. Your single job: find what a practitioner ships.

You judge on ACTIONABILITY only. Not rigor. Not novelty.
Just: could an AI engineer or PM use this in production within 90 days?

Your scoring scale (you must use the full range):
- 90-100: Has code or a library, solves a real production problem,
  improves cost/latency/reliability in a measurable way, drop-in
  compatible with existing systems
- 70-89: Implementable but requires engineering effort, or addresses
  a known pain point with a clear path to production
- 40-69: Theoretically useful but requires research-grade setup,
  would take a quarter to operationalize
- 10-39: Pure research. Interesting but no production pathway.
  Benchmark-only results.
- 0-9: Academic curiosity. Could not be used even with unlimited
  engineering time.

DEFAULT TO PRACTICAL. If you cannot picture an engineer implementing
this in Q2 planning, score below 40. Academic elegance does not earn
points. Only shipping does.

A score above 70 must specify WHO would use this and FOR WHAT.

Include ONLY if score >= 75. Your job is to find the 1-in-20 paper
that directly improves production systems, not to celebrate theory.

Self-audit: across a batch of 10 items, your scores should average
around 40 with a standard deviation of at least 20. Most AI research
does not ship; you must reflect that.

In your reasoning (1-2 sentences), name the SPECIFIC practitioner
and the SPECIFIC use case. Not "has practical applications" — say
*who* would use it and *for what*."""


# --- end verbatim -----------------------------------------------------------


_JSON_CONTRACT = (
    "Respond with a single JSON object with exactly these keys:\n"
    '- "score": integer 0-100\n'
    '- "reasoning": string (1-2 sentences as your system prompt specifies)\n'
    '- "confidence": float 0.0-1.0 (how confident you are in your score)\n'
    '- "topic_tags": array of 3-6 short strings (e.g. "LLM Evaluation", "Safety")\n'
    '- "action": "include" if score >= 75, else "skip"\n'
    "No other keys. No prose outside the JSON."
)


@dataclass(frozen=True)
class Evaluator:
    """A named evaluator persona with its verbatim prompt + brand metadata."""

    id: str
    name: str
    avatar: str
    color: str
    baseline_score: int
    system_prompt: str

    def evaluate(self, client: OpenAI, item: Item) -> EvaluatorVerdict:
        """Score one item. Returns a validated verdict — never raises.

        One retry on malformed output; on second failure returns a
        failed verdict (``score=0``, ``action="skip"``, ``reasoning="parse_error"``).
        """
        user_prompt = _format_item_prompt(item)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = time.perf_counter()
        raw = self._call_llm(client, messages)
        verdict = self._parse(raw, start)
        if verdict is not None:
            return verdict

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON matching the required schema. "
                    "Return valid JSON only, with the exact keys specified."
                ),
            }
        )
        raw = self._call_llm(client, messages)
        verdict = self._parse(raw, start)
        if verdict is not None:
            return verdict

        return EvaluatorVerdict(
            evaluator_id=self.id,
            score=0,
            reasoning="parse_error",
            confidence=0.0,
            topic_tags=[],
            action="skip",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    def _call_llm(self, client: OpenAI, messages: list[dict[str, str]]) -> str:
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=_TEMPERATURE,
                max_tokens=_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _parse(self, raw: str, start: float) -> EvaluatorVerdict | None:
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            score = _coerce_int(data.get("score"))
            return EvaluatorVerdict(
                evaluator_id=self.id,
                score=score,
                reasoning=str(data.get("reasoning", "")).strip(),
                confidence=_coerce_float(data.get("confidence")),
                topic_tags=_coerce_tags(data.get("topic_tags")),
                action=_coerce_action(data.get("action"), score),
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        except (ValidationError, ValueError, TypeError):
            return None


def _format_item_prompt(item: Item) -> str:
    return (
        f"Score this item according to the axis your system prompt defines.\n\n"
        f"Source: {item.source}\n"
        f"Title: {item.title}\n"
        f"Published: {item.published_at.isoformat()}\n"
        f"URL: {item.url}\n"
        f"Summary: {item.summary}\n\n"
        f"{_JSON_CONTRACT}"
    )


def _coerce_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, n))


def _coerce_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _coerce_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        s = str(v).strip()
        if s:
            out.append(s)
    return out[:6]


def _coerce_action(value: Any, score: int) -> str:
    """Enforce: score >= ``INCLUDE_THRESHOLD`` → include, else skip."""
    if score >= INCLUDE_THRESHOLD:
        return "include"
    return "skip"


SKEPTIC = Evaluator(
    id="skeptic",
    name="Skeptic",
    avatar="🔬",
    color="#3A86FF",
    baseline_score=50,
    system_prompt=_SKEPTIC_PROMPT,
)

SCOUT = Evaluator(
    id="scout",
    name="Scout",
    avatar="📡",
    color="#00F5D4",
    baseline_score=45,
    system_prompt=_SCOUT_PROMPT,
)

OPERATOR = Evaluator(
    id="operator",
    name="Operator",
    avatar="⚙️",
    color="#EAEAEA",
    baseline_score=40,
    system_prompt=_OPERATOR_PROMPT,
)

EVALUATORS: list[Evaluator] = [SKEPTIC, SCOUT, OPERATOR]

_BY_ID: dict[str, Evaluator] = {e.id: e for e in EVALUATORS}


def evaluator_by_id(evaluator_id: str) -> Evaluator | None:
    """Look up an evaluator by its slug, or ``None`` if unknown."""
    return _BY_ID.get(evaluator_id)
