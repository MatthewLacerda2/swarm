"""SQLite schema and access. One database per run, written only by the orchestrator."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    parent TEXT,
    born_tick INTEGER NOT NULL,
    status TEXT NOT NULL,              -- active | sleeping | finished
    spent REAL NOT NULL DEFAULT 0,
    sleep_until INTEGER,               -- tick to wake at; NULL = until woken
    finished_reason TEXT,
    spawn_prompt TEXT,
    turns INTEGER NOT NULL DEFAULT 0,
    context_est INTEGER NOT NULL DEFAULT 0,   -- estimated prompt tokens of its next call
    api_failures INTEGER NOT NULL DEFAULT 0,
    over_quota INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'normal',  -- normal | compaction
    content TEXT NOT NULL,             -- JSON, exactly as sent to / returned by the API
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_agent ON messages(agent, id);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    input_tokens INTEGER, cache_write_tokens INTEGER, cache_read_tokens INTEGER,
    output_tokens INTEGER, prompt_tokens INTEGER,
    compaction INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL,
    latency REAL,
    stop_reason TEXT,
    raw TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    cmd TEXT NOT NULL,
    exit_code INTEGER,
    seconds REAL,
    output_bytes INTEGER,
    output_path TEXT,
    timed_out INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
    name TEXT PRIMARY KEY,
    description TEXT,
    creator TEXT,
    tick INTEGER
);
CREATE TABLE IF NOT EXISTS members (
    channel TEXT NOT NULL,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    PRIMARY KEY (channel, agent)
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    channel TEXT,                      -- NULL for DMs
    dm_to TEXT,
    author TEXT NOT NULL,
    tick INTEGER NOT NULL,
    parent_id INTEGER,
    body TEXT NOT NULL,
    attachment TEXT
);
CREATE TABLE IF NOT EXISTS mentions (
    post_id INTEGER NOT NULL,
    agent TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    agent TEXT NOT NULL,
    channel TEXT NOT NULL,
    last_post INTEGER NOT NULL,
    PRIMARY KEY (agent, channel)
);
CREATE TABLE IF NOT EXISTS reads (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    kind TEXT NOT NULL,
    target TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pushes (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    branch TEXT NOT NULL,
    result TEXT NOT NULL,              -- new | ff | merge | rejected | unchanged
    detail TEXT,
    old TEXT, new TEXT
);
CREATE TABLE IF NOT EXISTS commits (
    hash TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY,
    tick INTEGER NOT NULL,
    agent TEXT,                        -- NULL = everyone
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS exams (
    tick INTEGER PRIMARY KEY,
    commit_hash TEXT,
    backtest_skill REAL,
    report TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embargo (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    tick INTEGER NOT NULL,
    kind TEXT NOT NULL,                -- command | file
    evidence TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS disk (
    tick INTEGER NOT NULL,
    agent TEXT NOT NULL,
    home_bytes INTEGER NOT NULL,
    bulk_bytes INTEGER NOT NULL,
    PRIMARY KEY (tick, agent)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    tick INTEGER,
    agent TEXT,
    kind TEXT NOT NULL,
    detail TEXT,
    created REAL NOT NULL
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # --- generic -------------------------------------------------------------------

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def transaction(self):
        return _Tx(self.conn)

    # --- meta ----------------------------------------------------------------------

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM meta WHERE key = ?", (key,))
        return default if row is None else json.loads(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    # --- agents --------------------------------------------------------------------

    def add_agent(self, agent_id: str, tick: int, parent: str | None = None, prompt: str | None = None) -> None:
        self.execute(
            "INSERT INTO agents(id, parent, born_tick, status, spawn_prompt) VALUES(?, ?, ?, 'active', ?)",
            (agent_id, parent, tick, prompt),
        )
        self.event(tick, agent_id, "born", {"parent": parent})

    def agents(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.all("SELECT * FROM agents WHERE status = ? ORDER BY id", (status,))
        return self.all("SELECT * FROM agents ORDER BY id")

    def agent(self, agent_id: str) -> sqlite3.Row:
        row = self.one("SELECT * FROM agents WHERE id = ?", (agent_id,))
        if row is None:
            raise KeyError(agent_id)
        return row

    def set_status(self, agent_id: str, status: str, tick: int, reason: str | None = None,
                   sleep_until: int | None = None) -> None:
        self.execute(
            "UPDATE agents SET status = ?, finished_reason = COALESCE(?, finished_reason), sleep_until = ? WHERE id = ?",
            (status, reason, sleep_until, agent_id),
        )
        self.event(tick, agent_id, status, {"reason": reason, "sleep_until": sleep_until})

    def next_agent_id(self) -> str:
        row = self.one("SELECT COUNT(*) AS n FROM agents")
        return f"agent-{row['n'] + 1:03d}"

    # --- messages ------------------------------------------------------------------

    def add_message(self, agent: str, tick: int, role: str, content: Any, kind: str = "normal") -> None:
        self.execute(
            "INSERT INTO messages(agent, tick, role, kind, content, created) VALUES(?, ?, ?, ?, ?, ?)",
            (agent, tick, role, kind, json.dumps(content, ensure_ascii=False), time.time()),
        )

    def history(self, agent: str) -> list[dict]:
        """What the API gets: everything from the latest compaction block on."""
        last = self.one(
            "SELECT MAX(id) AS id FROM messages WHERE agent = ? AND kind = 'compaction'", (agent,)
        )
        start = last["id"] if last and last["id"] is not None else 0
        rows = self.all(
            "SELECT role, content FROM messages WHERE agent = ? AND id >= ? ORDER BY id", (agent, start)
        )
        return [{"role": r["role"], "content": json.loads(r["content"])} for r in rows]

    # --- events --------------------------------------------------------------------

    def event(self, tick: int | None, agent: str | None, kind: str, detail: Any = None) -> None:
        self.execute(
            "INSERT INTO events(tick, agent, kind, detail, created) VALUES(?, ?, ?, ?, ?)",
            (tick, agent, kind, json.dumps(detail) if detail is not None else None, time.time()),
        )

    def notice(self, tick: int, kind: str, text: str, agent: str | None = None) -> None:
        self.execute(
            "INSERT INTO notices(tick, agent, kind, text) VALUES(?, ?, ?, ?)", (tick, agent, kind, text)
        )

    def total_spent(self) -> float:
        row = self.one("SELECT COALESCE(SUM(cost), 0) AS s FROM usage")
        return float(row["s"])


class _Tx:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False
