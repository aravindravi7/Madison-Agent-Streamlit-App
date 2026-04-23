"""Tests for signalscout.features.extract_context.

Locks:
- FEATURE_DIM == 15 and FEATURE_NAMES has 15 stable names
- Source indicator channels fire correctly
- Each keyword channel fires on expected inputs and stays off otherwise
- All features except bias are clipped to [0, 1]
- User-history features fall back safely for anonymous users
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from signalscout.features import FEATURE_DIM, FEATURE_NAMES, extract_context
from signalscout.models import Feedback, Item
from signalscout.storage import Storage


def _storage(tmp_path: Path) -> Storage:
    s = Storage(str(tmp_path / "features.db"))
    s.init_db()
    return s


def _item(
    *,
    source: str = "arxiv_cs_ai",
    title: str = "On evaluation",
    summary: str = "This paper studies LLM behavior " * 3,
    url: str = "https://example.com/paper",
    published_at: datetime | None = None,
) -> Item:
    return Item(
        id="i-1",
        source=source,
        title=title,
        url=url,
        published_at=published_at or datetime.now(UTC),
        summary=summary,
        fetched_at=datetime.now(UTC),
    )


def test_feature_dim_is_fifteen_and_names_are_stable() -> None:
    assert FEATURE_DIM == 15
    assert len(FEATURE_NAMES) == 15
    assert FEATURE_NAMES[0] == "is_arxiv"
    assert FEATURE_NAMES[1] == "is_smol"
    assert FEATURE_NAMES[2] == "is_user_source"
    assert FEATURE_NAMES[-1] == "bias"


def test_shape_and_bias(tmp_path: Path) -> None:
    vec = extract_context(_item(), user_id="", storage=_storage(tmp_path))
    assert vec.shape == (15,)
    assert vec.dtype == np.float64
    assert vec[14] == 1.0  # bias always on


def test_source_channels_are_one_hot(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    arxiv = extract_context(_item(source="arxiv_cs_ai"), user_id="", storage=s)
    smol = extract_context(_item(source="smol"), user_id="", storage=s)
    user = extract_context(_item(source="some_user_rss"), user_id="", storage=s)
    assert (arxiv[0], arxiv[1], arxiv[2]) == (1.0, 0.0, 0.0)
    assert (smol[0], smol[1], smol[2]) == (0.0, 1.0, 0.0)
    assert (user[0], user[1], user[2]) == (0.0, 0.0, 1.0)


def test_all_features_are_clipped_to_unit_interval_except_bias(tmp_path: Path) -> None:
    # Hostile input: enormous title + summary, ancient publish date.
    huge_summary = ("benchmark safety agent interpretability novel " * 400)
    item = _item(
        title="x" * 10_000,
        summary=huge_summary,
        published_at=datetime(2020, 1, 1, tzinfo=UTC),
    )
    vec = extract_context(item, user_id="", storage=_storage(tmp_path))
    for i, name in enumerate(FEATURE_NAMES):
        if name == "bias":
            assert vec[i] == 1.0
        else:
            assert 0.0 <= vec[i] <= 1.0, f"{name} out of bounds: {vec[i]}"


def test_keyword_channels_fire_when_present(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    item = _item(
        title="A novel benchmark for safety",
        summary=(
            "We introduce an agent evaluation framework using interpretability probes. "
            "Code available at https://github.com/example/repo"
        ),
    )
    vec = extract_context(item, user_id="", storage=s)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert vec[idx["has_code_link"]] == 1.0
    assert vec[idx["has_benchmark_kw"]] == 1.0
    assert vec[idx["has_safety_kw"]] == 1.0
    assert vec[idx["has_agent_kw"]] == 1.0
    assert vec[idx["has_interp_kw"]] == 1.0
    assert vec[idx["has_novel_kw"]] == 1.0


def test_keyword_channels_stay_off_when_absent(tmp_path: Path) -> None:
    item = _item(
        title="A paper about logistics",
        summary=(
            "We describe a shipment routing system for warehouse operations. "
            "There are no public links to this work."
        ),
        url="https://example.com/no-code",
    )
    vec = extract_context(item, user_id="", storage=_storage(tmp_path))
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    for k in ("has_code_link", "has_benchmark_kw", "has_safety_kw",
              "has_agent_kw", "has_interp_kw"):
        assert vec[idx[k]] == 0.0, f"{k} spuriously fired"


def test_user_history_defaults_for_anonymous(tmp_path: Path) -> None:
    vec = extract_context(_item(), user_id="", storage=_storage(tmp_path))
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert vec[idx["user_has_history"]] == 0.0
    assert vec[idx["user_thumbs_up_rate"]] == 0.5  # neutral prior


def test_user_history_reflects_recent_feedback(tmp_path: Path) -> None:
    s = _storage(tmp_path)
    # Seed 5 events: 4 positive, 1 negative → rate 0.8, has_history=1
    base_ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    for i, sig in enumerate(["thumbs_up", "saved", "thumbs_up", "saved", "thumbs_down"]):
        s.save_feedback(
            Feedback(
                item_id=f"it-{i}",
                user_id="u1",
                signal=sig,  # type: ignore[arg-type]
                evaluator_id="scout",
                timestamp=base_ts.replace(hour=12 + i),
            )
        )
    vec = extract_context(_item(), user_id="u1", storage=s)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    assert vec[idx["user_has_history"]] == 1.0
    assert vec[idx["user_thumbs_up_rate"]] == 0.8
