"""Pydantic v2 data contracts between SignalScout modules.

Every model here is the single source of truth for its concept — no dict
plumbing, no schema drift. All fields typed; all models exported at module
level for clean imports elsewhere.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Item",
    "EvaluatorVerdict",
    "ArenaResult",
    "Feedback",
    "Brief",
    "BanditArmState",
]


class Item(BaseModel):
    """A single fetched piece of content (paper, post, news item)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    title: str
    url: str
    published_at: datetime
    summary: str
    fetched_at: datetime


class EvaluatorVerdict(BaseModel):
    """One evaluator persona's judgment of one item."""

    model_config = ConfigDict(extra="forbid")

    evaluator_id: str
    score: int = Field(ge=0, le=100)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    topic_tags: list[str] = Field(default_factory=list)
    action: Literal["include", "skip"]
    latency_ms: int = Field(ge=0)


class ArenaResult(BaseModel):
    """All three evaluators' verdicts for one item plus the bandit's pick."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    verdicts: list[EvaluatorVerdict]
    winner_id: str
    winner_reason: str
    bandit_sampled_values: dict[str, float] = Field(default_factory=dict)
    final_score: int = Field(ge=0, le=100)
    final_action: Literal["include", "skip"]
    disagreement: float = Field(ge=0.0)


class Feedback(BaseModel):
    """One user interaction signal attached to an item and the evaluator that decided it."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    user_id: str
    signal: Literal["thumbs_up", "thumbs_down", "saved", "clicked", "dismissed"]
    evaluator_id: str
    timestamp: datetime


class Brief(BaseModel):
    """One generated brief: the set of included items plus the run's synthesis and config."""

    model_config = ConfigDict(extra="forbid")

    id: str
    generated_at: datetime
    user_id: str
    included_items: list[str] = Field(default_factory=list)
    theme_synthesis: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)


class BanditArmState(BaseModel):
    """Per-arm Bayesian linear regression parameters for Thompson Sampling."""

    model_config = ConfigDict(extra="forbid")

    evaluator_id: str
    A: list[list[float]]
    b: list[float]
    n_pulls: int = Field(default=0, ge=0)
    n_rewards_positive: int = Field(default=0, ge=0)
