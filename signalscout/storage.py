"""SQLite data-access layer.

Every SQL statement in the codebase lives in this file. Nothing outside
this module should import ``sqlite3``. Every public method is typed and
returns Pydantic models (or dicts/lists of them) — raw rows never escape.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .models import (
    ArenaResult,
    BanditArmState,
    Brief,
    EvaluatorVerdict,
    Feedback,
    Item,
)

__all__ = ["Storage", "DEFAULT_DB_PATH"]

DEFAULT_DB_PATH = "signalscout.db"

_POSITIVE_SIGNALS = ("thumbs_up", "saved", "clicked")
_NEGATIVE_SIGNALS = ("thumbs_down", "dismissed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    published_at TEXT NOT NULL,
    summary TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS arena_results (
    item_id TEXT PRIMARY KEY,
    verdicts_json TEXT NOT NULL,
    winner_id TEXT NOT NULL,
    winner_reason TEXT DEFAULT '',
    bandit_sampled_values_json TEXT,
    final_score INTEGER,
    final_action TEXT,
    disagreement REAL,
    created_at TEXT NOT NULL,
    user_id TEXT DEFAULT '',
    context_vector_json TEXT DEFAULT '',
    FOREIGN KEY (item_id) REFERENCES items(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    signal TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS briefs (
    id TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    user_id TEXT NOT NULL,
    included_item_ids_json TEXT NOT NULL,
    theme_synthesis_json TEXT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS bandit_arm_state (
    evaluator_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    A_json TEXT NOT NULL,
    b_json TEXT NOT NULL,
    n_pulls INTEGER DEFAULT 0,
    n_rewards_positive INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (evaluator_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_item ON feedback(item_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_arena_created ON arena_results(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique
    ON feedback(item_id, user_id, signal);
"""

# Indexes that depend on columns added by migrations — applied AFTER the
# ``ALTER TABLE`` statements in ``init_db`` so a pre-migration DB doesn't
# blow up trying to index a column that doesn't exist yet.
_POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_arena_user ON arena_results(user_id);
"""


def _iso(dt: datetime) -> str:
    """Serialize a datetime as an ISO 8601 string."""
    return dt.isoformat()


def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 datetime string back into a datetime."""
    return datetime.fromisoformat(s)


def _verdicts_to_json(verdicts: list[EvaluatorVerdict]) -> str:
    return json.dumps([v.model_dump(mode="json") for v in verdicts])


def _verdicts_from_json(s: str) -> list[EvaluatorVerdict]:
    return [EvaluatorVerdict.model_validate(v) for v in json.loads(s)]


class Storage:
    """Thin DAL over SQLite. One instance per app; path overridable."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(Path(db_path))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection with foreign keys and Row row_factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create all tables and indexes. Idempotent — safe to call repeatedly.

        Also applies in-place migrations for schema additions made in later
        phases (e.g. ``arena_results.winner_reason`` was introduced in Phase 2;
        a Phase-1 DB is migrated on first open).
        """
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(arena_results)")}
            if "winner_reason" not in cols:
                conn.execute("ALTER TABLE arena_results ADD COLUMN winner_reason TEXT DEFAULT ''")
            if "user_id" not in cols:
                conn.execute("ALTER TABLE arena_results ADD COLUMN user_id TEXT DEFAULT ''")
            if "context_vector_json" not in cols:
                conn.execute(
                    "ALTER TABLE arena_results ADD COLUMN context_vector_json TEXT DEFAULT ''"
                )
            conn.executescript(_POST_MIGRATION_INDEXES)

    # ------------------------------------------------------------------ items

    def upsert_item(self, item: Item) -> None:
        """Insert an item or update it if the id already exists."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO items (id, source, title, url, published_at, summary, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source=excluded.source,
                    title=excluded.title,
                    url=excluded.url,
                    published_at=excluded.published_at,
                    summary=excluded.summary,
                    fetched_at=excluded.fetched_at
                """,
                (
                    item.id,
                    item.source,
                    item.title,
                    item.url,
                    _iso(item.published_at),
                    item.summary,
                    _iso(item.fetched_at),
                ),
            )

    # ------------------------------------------------------------ arena_results

    def save_arena_result(
        self,
        result: ArenaResult,
        user_id: str = "",
        context_vector: list[float] | None = None,
    ) -> None:
        """Insert or replace the arena result for an item.

        ``user_id`` scopes the result to a user for mood / per-user analytics.
        ``context_vector`` is the bandit's 15-dim feature vector captured at
        decision time; stored so feedback that arrives late can still update
        the bandit against the frozen context.
        """
        ctx_json = json.dumps(list(context_vector)) if context_vector is not None else ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO arena_results (
                    item_id, verdicts_json, winner_id, winner_reason,
                    bandit_sampled_values_json, final_score, final_action,
                    disagreement, created_at, user_id, context_vector_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    verdicts_json=excluded.verdicts_json,
                    winner_id=excluded.winner_id,
                    winner_reason=excluded.winner_reason,
                    bandit_sampled_values_json=excluded.bandit_sampled_values_json,
                    final_score=excluded.final_score,
                    final_action=excluded.final_action,
                    disagreement=excluded.disagreement,
                    created_at=excluded.created_at,
                    user_id=excluded.user_id,
                    context_vector_json=excluded.context_vector_json
                """,
                (
                    result.item_id,
                    _verdicts_to_json(result.verdicts),
                    result.winner_id,
                    result.winner_reason,
                    json.dumps(result.bandit_sampled_values),
                    result.final_score,
                    result.final_action,
                    result.disagreement,
                    _iso(datetime.now(UTC)),
                    user_id,
                    ctx_json,
                ),
            )

    def get_arena_result(self, item_id: str) -> ArenaResult | None:
        """Fetch the arena result for one item, or None if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM arena_results WHERE item_id = ?", (item_id,)
            ).fetchone()
        if row is None:
            return None
        return ArenaResult(
            item_id=row["item_id"],
            verdicts=_verdicts_from_json(row["verdicts_json"]),
            winner_id=row["winner_id"],
            winner_reason=row["winner_reason"] or "",
            bandit_sampled_values=json.loads(row["bandit_sampled_values_json"] or "{}"),
            final_score=int(row["final_score"]),
            final_action=row["final_action"],
            disagreement=float(row["disagreement"]),
        )

    def get_context_vector_for_item(self, item_id: str) -> list[float] | None:
        """Return the context vector captured at decision time, or None if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT context_vector_json FROM arena_results WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        raw = row["context_vector_json"] or ""
        if not raw:
            return None
        try:
            return [float(x) for x in json.loads(raw)]
        except (ValueError, json.JSONDecodeError):
            return None

    # ---------------------------------------------------------------- feedback

    def save_feedback(self, feedback: Feedback) -> bool:
        """Append a feedback row. Returns ``True`` if inserted, ``False`` on duplicate.

        Idempotency is enforced by a unique index on ``(item_id, user_id, signal)``;
        re-clicking 👍 on the same item is a no-op.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO feedback (item_id, user_id, signal, evaluator_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback.item_id,
                    feedback.user_id,
                    feedback.signal,
                    feedback.evaluator_id,
                    _iso(feedback.timestamp),
                ),
            )
            return cur.rowcount > 0

    def get_feedback_for_user(self, user_id: str) -> list[Feedback]:
        """All feedback rows for a given user, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,),
            ).fetchall()
        return [
            Feedback(
                item_id=r["item_id"],
                user_id=r["user_id"],
                signal=r["signal"],
                evaluator_id=r["evaluator_id"],
                timestamp=_parse_iso(r["timestamp"]),
            )
            for r in rows
        ]

    # --------------------------------------------------------------- bandit

    def get_bandit_state(self, user_id: str) -> dict[str, BanditArmState]:
        """Load all per-evaluator bandit arm states for a user."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM bandit_arm_state WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {
            r["evaluator_id"]: BanditArmState(
                evaluator_id=r["evaluator_id"],
                A=json.loads(r["A_json"]),
                b=json.loads(r["b_json"]),
                n_pulls=int(r["n_pulls"]),
                n_rewards_positive=int(r["n_rewards_positive"]),
            )
            for r in rows
        }

    def get_total_pulls(self, user_id: str) -> int:
        """Sum of ``n_pulls`` across all bandit arms for this user.

        Used by the bandit's cold-start logic (first ``COLD_START_PULLS``
        decisions across the whole user history are round-robin).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(n_pulls), 0) AS total FROM bandit_arm_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["total"]) if row is not None else 0

    def save_bandit_state(
        self, user_id: str, state: dict[str, BanditArmState]
    ) -> None:
        """Upsert every arm's state for a user."""
        now = _iso(datetime.now(UTC))
        with self._connect() as conn:
            for arm in state.values():
                conn.execute(
                    """
                    INSERT INTO bandit_arm_state (
                        evaluator_id, user_id, A_json, b_json,
                        n_pulls, n_rewards_positive, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(evaluator_id, user_id) DO UPDATE SET
                        A_json=excluded.A_json,
                        b_json=excluded.b_json,
                        n_pulls=excluded.n_pulls,
                        n_rewards_positive=excluded.n_rewards_positive,
                        updated_at=excluded.updated_at
                    """,
                    (
                        arm.evaluator_id,
                        user_id,
                        json.dumps(arm.A),
                        json.dumps(arm.b),
                        arm.n_pulls,
                        arm.n_rewards_positive,
                        now,
                    ),
                )

    # ----------------------------------------------------------------- briefs

    def save_brief(self, brief: Brief) -> None:
        """Insert or replace a brief by id."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO briefs (
                    id, generated_at, user_id, included_item_ids_json,
                    theme_synthesis_json, config_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    generated_at=excluded.generated_at,
                    user_id=excluded.user_id,
                    included_item_ids_json=excluded.included_item_ids_json,
                    theme_synthesis_json=excluded.theme_synthesis_json,
                    config_json=excluded.config_json
                """,
                (
                    brief.id,
                    _iso(brief.generated_at),
                    brief.user_id,
                    json.dumps(brief.included_items),
                    json.dumps(brief.theme_synthesis),
                    json.dumps(brief.config),
                ),
            )

    def list_briefs(self, user_id: str, limit: int = 20) -> list[Brief]:
        """Most recent briefs for a user (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM briefs
                WHERE user_id = ?
                ORDER BY generated_at DESC
                LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
        return [
            Brief(
                id=r["id"],
                generated_at=_parse_iso(r["generated_at"]),
                user_id=r["user_id"],
                included_items=json.loads(r["included_item_ids_json"] or "[]"),
                theme_synthesis=json.loads(r["theme_synthesis_json"] or "{}"),
                config=json.loads(r["config_json"] or "{}"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------ verdicts

    def get_recent_verdicts_for_evaluator(
        self,
        evaluator_id: str,
        user_id: str,
        limit: int = 20,
    ) -> list[EvaluatorVerdict]:
        """Most recent ``limit`` verdicts from this evaluator for this user.

        Spans all items the user has processed. Used by ``moods.compute_mood``.
        Empty list if there are none.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT verdicts_json
                FROM arena_results
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, int(max(1, limit * 3))),
            ).fetchall()
        out: list[EvaluatorVerdict] = []
        for r in rows:
            for verdict in _verdicts_from_json(r["verdicts_json"]):
                if verdict.evaluator_id == evaluator_id:
                    out.append(verdict)
                    if len(out) >= int(limit):
                        return out
                    break
        return out

    # ---------------------------------------------------------- leaderboard

    def get_evaluator_win_rates(self, user_id: str) -> dict[str, float]:
        """Positive-signal rate per evaluator across all of this user's feedback.

        win_rate = positive / (positive + negative).
        Evaluators with zero feedback are omitted.
        """
        positive_placeholders = ",".join("?" * len(_POSITIVE_SIGNALS))
        negative_placeholders = ",".join("?" * len(_NEGATIVE_SIGNALS))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    evaluator_id,
                    SUM(CASE WHEN signal IN ({positive_placeholders}) THEN 1 ELSE 0 END) AS pos,
                    SUM(CASE WHEN signal IN ({negative_placeholders}) THEN 1 ELSE 0 END) AS neg
                FROM feedback
                WHERE user_id = ?
                GROUP BY evaluator_id
                """,
                (*_POSITIVE_SIGNALS, *_NEGATIVE_SIGNALS, user_id),
            ).fetchall()
        rates: dict[str, float] = {}
        for r in rows:
            pos = int(r["pos"] or 0)
            neg = int(r["neg"] or 0)
            total = pos + neg
            if total == 0:
                continue
            rates[r["evaluator_id"]] = pos / total
        return rates
