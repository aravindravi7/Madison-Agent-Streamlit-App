"""Context featurization for the TasteBandit.

Produces a fixed-length, interpretable context vector for each item so
the bandit can learn which evaluator to trust *for which kind of item*.
Deliberately small (15 dims) — enough signal to personalize, few enough
to fit online linear bandit math without overfitting.

Feature order is LOCKED. Reordering breaks persisted bandit weights.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import numpy as np

from .models import Item
from .storage import Storage

__all__ = ["FEATURE_DIM", "extract_context", "FEATURE_NAMES"]

FEATURE_DIM = 15

FEATURE_NAMES: tuple[str, ...] = (
    "is_arxiv",            # 0
    "is_smol",             # 1
    "is_user_source",      # 2
    "title_length_norm",   # 3
    "summary_length_norm", # 4
    "days_since_pub_norm", # 5
    "has_code_link",       # 6
    "has_benchmark_kw",    # 7
    "has_safety_kw",       # 8
    "has_agent_kw",        # 9
    "has_interp_kw",       # 10
    "has_novel_kw",        # 11
    "user_has_history",    # 12
    "user_thumbs_up_rate", # 13
    "bias",                # 14
)

_BUILTIN_SOURCES = ("arxiv_cs_ai", "smol")

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "benchmark": ("benchmark", "evaluation suite", "leaderboard"),
    "safety":    ("safety", "alignment", "red-team", "red team", "jailbreak", "harmful"),
    "agent":     ("agent", "tool use", "tool-use", "toolformer", "multi-agent"),
    "interp":    ("interpretability", "mechanistic", "probe", "attribution", "circuit"),
    "novel":     ("novel", "new", "first", "introduce"),
}

_CODE_LINK_PATTERNS = ("github.com/", "gitlab.com/", "huggingface.co/", "code:", "code release")

_TITLE_NORM_CHARS = 120.0
_SUMMARY_NORM_CHARS = 2000.0
_DAYS_NORM = 30.0
_HISTORY_THRESHOLD = 5
_RECENT_FEEDBACK_WINDOW = 20
_DEFAULT_THUMBS_UP_RATE = 0.5


def _contains_any(haystack: str, needles: tuple[str, ...]) -> float:
    return 1.0 if any(n in haystack for n in needles) else 0.0


def _days_since(published_at: datetime) -> float:
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - published_at
    return max(0.0, delta.total_seconds() / 86400.0)


def _source_flags(source: str) -> tuple[float, float, float]:
    is_arxiv = 1.0 if source == "arxiv_cs_ai" else 0.0
    is_smol = 1.0 if source == "smol" else 0.0
    is_user = 0.0 if source in _BUILTIN_SOURCES else 1.0
    return is_arxiv, is_smol, is_user


def _user_features(user_id: str, storage: Storage) -> tuple[float, float]:
    """Return (has_history, thumbs_up_rate) summarized from recent feedback.

    ``has_history`` = 1.0 iff user has ≥ _HISTORY_THRESHOLD feedback events total.
    ``thumbs_up_rate`` uses the last ``_RECENT_FEEDBACK_WINDOW`` events; 0.5 default
    when empty. ``thumbs_up`` and ``saved`` count positive; ``thumbs_down`` and
    ``dismissed`` count negative. Ambiguous signals (``clicked``) are excluded.
    """
    if not user_id:
        return 0.0, _DEFAULT_THUMBS_UP_RATE
    events = storage.get_feedback_for_user(user_id)
    has_history = 1.0 if len(events) >= _HISTORY_THRESHOLD else 0.0
    recent = events[:_RECENT_FEEDBACK_WINDOW]  # newest first from DAL
    up = sum(1 for e in recent if e.signal in ("thumbs_up", "saved"))
    down = sum(1 for e in recent if e.signal in ("thumbs_down", "dismissed"))
    total = up + down
    rate = (up / total) if total > 0 else _DEFAULT_THUMBS_UP_RATE
    return has_history, rate


def extract_context(item: Item, user_id: str, storage: Storage) -> np.ndarray:
    """Return a 15-dim float64 context vector for ``item``.

    Stable ordering — see ``FEATURE_NAMES``. All values clipped to [0, 1]
    except the bias term which is always 1.0. Float64 so ``A + xxᵀ`` stays
    numerically stable when accumulated over many updates.
    """
    title = (item.title or "").lower()
    summary = (item.summary or "").lower()
    combined = f"{title} {summary}"

    is_arxiv, is_smol, is_user_src = _source_flags(item.source)
    title_len = min(1.0, len(item.title or "") / _TITLE_NORM_CHARS)
    summary_len = min(1.0, len(item.summary or "") / _SUMMARY_NORM_CHARS)
    days_norm = min(1.0, _days_since(item.published_at) / _DAYS_NORM)

    has_code = _contains_any(summary + " " + (item.url or "").lower(), _CODE_LINK_PATTERNS)
    has_bench = _contains_any(combined, _KEYWORDS["benchmark"])
    has_safety = _contains_any(combined, _KEYWORDS["safety"])
    has_agent = _contains_any(combined, _KEYWORDS["agent"])
    has_interp = _contains_any(combined, _KEYWORDS["interp"])
    has_novel = _contains_any(combined, _KEYWORDS["novel"])

    has_history, thumbs_up_rate = _user_features(user_id, storage)

    vec = np.array(
        [
            is_arxiv,
            is_smol,
            is_user_src,
            title_len,
            summary_len,
            days_norm,
            has_code,
            has_bench,
            has_safety,
            has_agent,
            has_interp,
            has_novel,
            has_history,
            thumbs_up_rate,
            1.0,  # bias
        ],
        dtype=np.float64,
    )
    assert vec.shape == (FEATURE_DIM,), f"vec shape {vec.shape} != ({FEATURE_DIM},)"
    return vec
