"""The feed: what is new for an agent at the start of its turn.

Only things addressed to the agent are shown in full (DMs, mentions, notices about
its pushes, spawns and budget). Channels it joined show as counts: reading them is
its own choice.
"""

from __future__ import annotations

from .run import Run

MAX_FEED = 4000
MAX_ITEM = 600


def _cursor(run: Run, agent: str, key: str) -> int:
    row = run.db.one("SELECT last_post FROM cursors WHERE agent = ? AND channel = ?", (agent, key))
    return row["last_post"] if row else 0


def _set_cursor(run: Run, agent: str, key: str, value: int) -> None:
    run.db.execute(
        "INSERT INTO cursors(agent, channel, last_post) VALUES(?, ?, ?) "
        "ON CONFLICT(agent, channel) DO UPDATE SET last_post = excluded.last_post",
        (agent, key, value),
    )


def calls_left(run: Run, agent: str, left: float) -> int:
    from .prompts import TYPICAL_CALL_COST

    row = run.db.one("SELECT COUNT(*) AS n, SUM(cost) AS c FROM usage WHERE agent = ? AND compaction = 0", (agent,))
    avg = (row["c"] / row["n"]) if row["n"] else TYPICAL_CALL_COST
    return int(left / max(avg, 1e-6))


def _clip(text: str, n: int = MAX_ITEM) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + f" [... {len(text) - n} more characters]"


def build(run: Run, agent: str, tick: int) -> str:
    db = run.db
    lines: list[str] = [f"[round {tick}] Round {tick} of at most {run.cfg.max_ticks}."]

    # Notices: budget, wind-down, exam, pushes, spawns, disk (everyone's or this agent's).
    last = _cursor(run, agent, "@notice")
    notices = db.all(
        "SELECT id, text FROM notices WHERE id > ? AND (agent IS NULL OR agent = ?) ORDER BY id",
        (last, agent),
    )
    for n in notices:
        lines.append(_clip(n["text"], 1600))
    if notices:
        _set_cursor(run, agent, "@notice", notices[-1]["id"])

    left = max(0.0, run.budget.agent_left(agent))
    lines.append(f"Your budget: US$ {left:.3f} of {run.cfg.agent_budget:.2f} left "
                 f"(about {calls_left(run, agent, left):,} more calls at your average cost so far).")

    # Direct messages
    last = _cursor(run, agent, "@dm")
    dms = db.all(
        "SELECT id, author, tick, body FROM posts WHERE dm_to = ? AND id > ? ORDER BY id", (agent, last)
    )
    if dms:
        lines.append(f"{len(dms)} new direct message(s):")
        for m in dms[-10:]:
            lines.append(f"- [{m['id']}] from {m['author']} (round {m['tick']}): {_clip(m['body'])}")
        if len(dms) > 10:
            lines.append(f"  ({len(dms) - 10} older ones: `board dms`)")
        _set_cursor(run, agent, "@dm", dms[-1]["id"])

    # Mentions in channels
    last = _cursor(run, agent, "@mention")
    mentions = db.all(
        "SELECT p.id, p.author, p.channel, p.tick, p.body FROM mentions m JOIN posts p ON p.id = m.post_id "
        "WHERE m.agent = ? AND p.id > ? AND p.channel IS NOT NULL ORDER BY p.id",
        (agent, last),
    )
    if mentions:
        lines.append(f"{len(mentions)} new mention(s):")
        for m in mentions[-10:]:
            lines.append(f"- [{m['id']}] {m['author']} in {m['channel']} (round {m['tick']}): {_clip(m['body'])}")
        _set_cursor(run, agent, "@mention", mentions[-1]["id"])

    # Replies to the agent's own posts
    last = _cursor(run, agent, "@reply")
    replies = db.all(
        "SELECT r.id, r.author, r.channel, r.parent_id FROM posts r JOIN posts p ON p.id = r.parent_id "
        "WHERE p.author = ? AND r.author != ? AND r.id > ? ORDER BY r.id",
        (agent, agent, last),
    )
    if replies:
        parts = sorted({str(r["parent_id"]) for r in replies})
        lines.append(f"{len(replies)} new repl(ies) to your posts ({', '.join(parts)}): `board thread <id>`")
        _set_cursor(run, agent, "@reply", replies[-1]["id"])

    # Joined channels: counts only
    joined = db.all("SELECT channel FROM members WHERE agent = ? ORDER BY channel", (agent,))
    counts = []
    for j in joined:
        ch = j["channel"]
        last = _cursor(run, agent, ch)
        row = db.one(
            "SELECT COUNT(*) AS n, MAX(id) AS top FROM posts WHERE channel = ? AND id > ? AND author != ?",
            (ch, last, agent),
        )
        top = db.one("SELECT MAX(id) AS top FROM posts WHERE channel = ?", (ch,))["top"]
        if row["n"]:
            counts.append(f"{ch} ({row['n']})")
        if top:
            _set_cursor(run, agent, ch, top)
    if counts:
        lines.append("New posts in your channels: " + ", ".join(counts) + ".")

    # New channels since the agent's last turn
    last = _cursor(run, agent, "@channels")
    new_channels = db.all(
        "SELECT rowid, name, creator FROM channels WHERE rowid > ? AND creator != ? ORDER BY rowid",
        (last, agent),
    )
    if new_channels:
        names = ", ".join(f"{c['name']} (by {c['creator']})" for c in new_channels[:15])
        more = f" and {len(new_channels) - 15} more" if len(new_channels) > 15 else ""
        lines.append(f"New channels: {names}{more}.")
        _set_cursor(run, agent, "@channels", new_channels[-1]["rowid"])

    text = "\n".join(lines)
    if len(text) > MAX_FEED:
        text = text[:MAX_FEED] + "\n[feed cut; use `board` to see the rest]"
    return text
