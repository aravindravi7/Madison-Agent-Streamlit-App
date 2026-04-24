"""Phase 2 calibration gate.

Pulls ~15 recent arXiv cs.AI items, runs all three evaluators in parallel,
and reports whether the metrics fall inside the Phase 2 addendum's targets.

Run:
    python scripts/calibrate.py
or with a custom count:
    python scripts/calibrate.py 20

Secrets:
- Reads ``OPENAI_API_KEY`` from the environment first.
- Falls back to ``.streamlit/secrets.toml`` in the repo root.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from openai import OpenAI  # noqa: E402

from signalscout.evaluators import EVALUATORS  # noqa: E402
from signalscout.pipeline import run_arena_on_item  # noqa: E402
from signalscout.sources import BUILTIN_SOURCES, clean_and_validate  # noqa: E402


TARGETS = {
    "skeptic":  {"baseline": 50, "tolerance": 10, "min_stdev": 15},
    "scout":    {"baseline": 45, "tolerance": 10, "min_stdev": 20},
    "operator": {"baseline": 40, "tolerance": 10, "min_stdev": 20},
}
INCLUDE_RATE_MIN = 0.15
INCLUDE_RATE_MAX = 0.35
MIXED_VERDICT_MIN = 0.40  # ≥40% of items must have at least one evaluator disagreeing.


def _load_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    secrets_path = REPO / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    # Minimal TOML read without extra deps.
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        return None
    with open(secrets_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("OPENAI_API_KEY")


def _fetch_items(n: int) -> list:
    items = []
    for src in BUILTIN_SOURCES:
        if src.id == "arxiv_cs_ai":
            items.extend(src.fetch(n * 2))  # oversample before cleaning
            break
    items = clean_and_validate(items)[:n]
    return items


def _metric_status(value: float, low: float, high: float) -> str:
    return "PASS" if low <= value <= high else "FAIL"


def main(n_items: int = 15) -> int:
    api_key = _load_api_key()
    if not api_key:
        print(
            "ERROR: no OpenAI API key. Put it in .streamlit/secrets.toml "
            "(key: OPENAI_API_KEY) or export OPENAI_API_KEY=... in your shell."
        )
        return 2
    client = OpenAI(api_key=api_key)

    print(f"\n=== Phase 2 calibration · fetching up to {n_items} arXiv items ===")
    items = _fetch_items(n_items)
    print(f"Got {len(items)} items after cleaning.")
    if len(items) < 5:
        print("Not enough items to judge calibration. Try re-running later.")
        return 2

    print("\nRunning all three evaluators on each item (parallel per item)…")
    results = []
    for i, item in enumerate(items, 1):
        result = run_arena_on_item(client, item)
        results.append((item, result))
        print(f"  [{i:>2}/{len(items)}] disagreement={result.disagreement:5.1f}  {item.title[:60]}")

    # ---- Per-evaluator metrics
    print("\n=== Per-evaluator metrics ===")
    print(f"{'Evaluator':10} {'n':>3}  {'mean':>6}  {'stdev':>6}  {'include%':>9}  {'notes'}")
    ev_stats = {}
    for ev in EVALUATORS:
        scores = [next(v for v in r.verdicts if v.evaluator_id == ev.id).score for _, r in results]
        actions = [next(v for v in r.verdicts if v.evaluator_id == ev.id).action for _, r in results]
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        include_rate = sum(1 for a in actions if a == "include") / len(actions)
        ev_stats[ev.id] = {"mean": mean, "stdev": stdev, "include_rate": include_rate}

        t = TARGETS[ev.id]
        baseline_lo = t["baseline"] - t["tolerance"]
        baseline_hi = t["baseline"] + t["tolerance"]
        mean_ok = _metric_status(mean, baseline_lo, baseline_hi)
        stdev_ok = "PASS" if stdev >= t["min_stdev"] else "FAIL"
        incl_ok = _metric_status(include_rate, INCLUDE_RATE_MIN, INCLUDE_RATE_MAX)
        print(
            f"{ev.id:10} {len(scores):>3}  {mean:6.1f}  {stdev:6.1f}  {include_rate*100:8.1f}%  "
            f"mean[{baseline_lo}-{baseline_hi}]:{mean_ok}  σ>={t['min_stdev']}:{stdev_ok}  "
            f"incl[{int(INCLUDE_RATE_MIN*100)}-{int(INCLUDE_RATE_MAX*100)}%]:{incl_ok}"
        )

    # ---- Cross-item metrics
    print("\n=== Cross-item metrics ===")
    mixed = 0
    for _, r in results:
        actions = {v.action for v in r.verdicts}
        if len(actions) > 1:
            mixed += 1
    mixed_rate = mixed / len(results)
    mixed_ok = "PASS" if mixed_rate >= MIXED_VERDICT_MIN else "FAIL"
    print(
        f"Mixed-verdict rate: {mixed}/{len(results)} = {mixed_rate*100:.1f}% "
        f"(target ≥ {int(MIXED_VERDICT_MIN*100)}%) → {mixed_ok}"
    )

    # ---- Illustrative disagreement examples
    print("\n=== Top-3 most-contested items ===")
    for _, r in sorted(results, key=lambda x: -x[1].disagreement)[:3]:
        item = next(it for it, rr in results if rr.item_id == r.item_id)
        print(f"\n--- {item.title[:90]}")
        print(f"    disagreement σ={r.disagreement:.1f}   winner={r.winner_id}")
        for v in sorted(r.verdicts, key=lambda v: v.evaluator_id):
            reason = (v.reasoning[:120] + "…") if len(v.reasoning) > 120 else v.reasoning
            print(f"    {v.evaluator_id:9} score={v.score:>3} action={v.action:<7} — {reason}")

    # ---- Overall gate
    gates = []
    for ev in EVALUATORS:
        t = TARGETS[ev.id]
        s = ev_stats[ev.id]
        gates.append(abs(s["mean"] - t["baseline"]) <= t["tolerance"])
        gates.append(s["stdev"] >= t["min_stdev"])
        gates.append(INCLUDE_RATE_MIN <= s["include_rate"] <= INCLUDE_RATE_MAX)
    gates.append(mixed_rate >= MIXED_VERDICT_MIN)

    print("\n=== GATE ===")
    if all(gates):
        print("PASS — Phase 2 calibration targets met.")
        return 0
    print("FAIL — one or more targets not met. See per-metric PASS/FAIL above.")
    return 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    raise SystemExit(main(n))
