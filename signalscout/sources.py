"""RSS source protocol + built-in sources (arXiv cs.AI, Smol AI News).

``fetch`` returns ``list[Item]`` (Pydantic models) — never raw dicts.
Cleaning/validation (``clean_and_validate``) preserves the existing
rules: summary ≥ 50 chars, published ≥ 2023-01-01, dedupe by URL.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import feedparser

from .models import Item

__all__ = [
    "Source",
    "RSSSource",
    "BUILTIN_SOURCES",
    "clean_and_validate",
    "ARXIV_CS_AI_URL",
    "SMOL_URL",
    "CUTOFF_DATE",
]

ARXIV_CS_AI_URL = "https://export.arxiv.org/rss/cs.AI"
SMOL_URL = "https://news.smol.ai/rss.xml"
CUTOFF_DATE = datetime(2023, 1, 1, tzinfo=UTC)


class Source(Protocol):
    """Any source that produces ``Item`` records given a ``limit``."""

    id: str
    name: str

    def fetch(self, limit: int) -> list[Item]: ...


def _hash_url(url: str) -> str:
    """Deterministic short id from a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _extract_summary(entry: Any) -> str:
    """Pull the best available summary text from a feedparser entry."""
    summary = entry.get("summary", "") or ""
    if not summary and entry.get("content"):
        try:
            summary = entry["content"][0].get("value", "") or ""
        except (AttributeError, IndexError, TypeError):
            summary = ""
    if not summary:
        summary = entry.get("content_encoded", "") or entry.get("description", "") or ""
    if isinstance(summary, dict):
        summary = summary.get("value", "") or ""
    return str(summary).strip()


def _parse_published(raw: str | None) -> datetime | None:
    """Parse an RSS pubDate / ISO date into a tz-aware datetime, or return None."""
    if not raw:
        return None
    s = str(raw).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class RSSSource:
    """A source backed by a single RSS/Atom feed URL."""

    def __init__(self, id: str, name: str, url: str) -> None:
        self.id = id
        self.name = name
        self.url = url

    def fetch(self, limit: int) -> list[Item]:
        """Fetch up to ``limit`` entries and map them to Item records.

        Entries with no URL or no parseable published date are dropped at this
        stage. Further validation (summary length, cutoff date, dedupe) lives
        in ``clean_and_validate``.
        """
        feed = feedparser.parse(self.url)
        fetched_at = datetime.now(UTC)
        out: list[Item] = []
        for entry in feed.entries[: max(0, int(limit))]:
            url = entry.get("link") or ""
            if not url:
                continue
            published = _parse_published(entry.get("published") or entry.get("updated"))
            if published is None:
                continue
            out.append(
                Item(
                    id=_hash_url(url),
                    source=self.id,
                    title=str(entry.get("title", "")).strip(),
                    url=url,
                    published_at=published,
                    summary=_extract_summary(entry),
                    fetched_at=fetched_at,
                )
            )
        return out


BUILTIN_SOURCES: list[RSSSource] = [
    RSSSource("arxiv_cs_ai", "arXiv cs.AI", ARXIV_CS_AI_URL),
    RSSSource("smol", "Smol AI News", SMOL_URL),
]


def clean_and_validate(items: list[Item]) -> list[Item]:
    """Drop short-summary items, pre-cutoff items, and duplicate URLs.

    Rules (preserved from the original pipeline):
    - ``len(summary) > 50``
    - ``published_at >= 2023-01-01 UTC``
    - dedupe by URL (first occurrence wins)
    """
    seen: set[str] = set()
    cleaned: list[Item] = []
    for item in items:
        if not item.url or item.url in seen:
            continue
        if len(item.summary) <= 50:
            continue
        pub = item.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=UTC)
        if pub < CUTOFF_DATE:
            continue
        seen.add(item.url)
        cleaned.append(item)
    return cleaned
