"""`swarm status`: a one-screen summary of a run (reads the database, never writes)."""

from __future__ import annotations

import json
import sqlite3

from . import config


def render(run_id: str) -> str:
    path = config.RUNS / run_id / "swarm.db"
    if not path.exists():
        return f"no run {run_id}"
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    def meta(key, default=None):
        row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    cfg = config.RunConfig(**meta("config"))
    hard = cfg.hard_stop
    spent = db.execute("SELECT COALESCE(SUM(cost), 0) FROM usage").fetchone()[0]
    out = []
    out.append(f"Run {run_id} ({cfg.task}) - status {meta('status')} - round {meta('tick', 0)}")
    phase = "normal" if spent < cfg.run_budget else ("wind-down" if spent < hard else "hard stop")
    out.append(f"Spent US$ {spent:.3f} of {cfg.run_budget:.2f} ({100 * spent / cfg.run_budget:.1f}%), "
               f"hard stop {hard:.2f} - phase {phase}")

    u = db.execute("SELECT COUNT(*) n, SUM(input_tokens) i, SUM(cache_write_tokens) w, SUM(cache_read_tokens) r,"
                   " SUM(output_tokens) o, SUM(compaction) c, AVG(latency) l FROM usage WHERE agent != 'orchestrator'").fetchone()
    if u["n"]:
        total_in = (u["i"] or 0) + (u["w"] or 0) + (u["r"] or 0)
        hit = (u["r"] or 0) / total_in if total_in else 0
        out.append(f"Calls {u['n']} - cache hit {hit:.0%} - output {u['o'] or 0:,} tokens - "
                   f"compactions {u['c'] or 0} - avg latency {u['l'] or 0:.1f}s - "
                   f"US$ {spent / u['n']:.4f}/call")

    agents = db.execute("SELECT * FROM agents ORDER BY id").fetchall()
    by = {}
    for a in agents:
        by.setdefault(a["status"], []).append(a)
    out.append("Agents: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by.items())))
    for a in agents:
        extra = f" ({a['finished_reason']})" if a["status"] == "finished" and a["finished_reason"] else ""
        parent = f" <- {a['parent']}" if a["parent"] else ""
        out.append(f"  {a['id']} {a['status']:<9} US$ {a['spent']:.3f} turns {a['turns']:>3} "
                   f"ctx ~{a['context_est'] // 1000}k{parent}{extra}")

    ex = db.execute("SELECT tick, backtest_skill, report FROM exams ORDER BY tick DESC LIMIT 1").fetchone()
    if ex:
        rep = json.loads(ex["report"])
        skill = "n/a" if ex["backtest_skill"] is None else f"{ex['backtest_skill']:.3f}"
        out.append(f"Exam (round {ex['tick']}): backtest skill {skill}; validity {rep['validity']}")
    emb = db.execute("SELECT COUNT(*) n, COUNT(DISTINCT agent) a FROM embargo").fetchone()
    out.append(f"Embargo flags: {emb['n']} from {emb['a']} agents")
    pushes = db.execute("SELECT result, COUNT(*) n FROM pushes GROUP BY result").fetchall()
    if pushes:
        out.append("Pushes: " + ", ".join(f"{p['result']} {p['n']}" for p in pushes))
    disk = db.execute("SELECT SUM(home_bytes) h, SUM(bulk_bytes) b FROM disk WHERE tick = (SELECT MAX(tick) FROM disk)").fetchone()
    if disk and disk["h"] is not None:
        out.append(f"Disk: $HOME total {disk['h'] / 1e9:.2f} GB, $BULK total {disk['b'] / 1e9:.2f} GB")

    chans = db.execute("SELECT c.name, (SELECT COUNT(*) FROM posts p WHERE p.channel = c.name) n,"
                       " (SELECT COUNT(*) FROM members m WHERE m.channel = c.name) m FROM channels c ORDER BY n DESC LIMIT 8").fetchall()
    if chans:
        out.append("Channels: " + ", ".join(f"{c['name']} ({c['n']} posts, {c['m']} members)" for c in chans))
    dms = db.execute("SELECT COUNT(*) FROM posts WHERE channel IS NULL").fetchone()[0]
    out.append(f"Direct messages: {dms}")
    recent = db.execute("SELECT author, channel, dm_to, tick, body FROM posts ORDER BY id DESC LIMIT 5").fetchall()
    if recent:
        out.append("Recent posts:")
        for p in reversed(recent):
            where = p["channel"] or f"DM to {p['dm_to']}"
            body = p["body"].replace("\n", " ")
            out.append(f"  r{p['tick']} {p['author']} in {where}: {body[:110]}")

    anomalies = db.execute(
        "SELECT kind, COUNT(*) n FROM events WHERE kind IN ('command_timeout', 'sandbox_error', 'api_error',"
        " 'refusal', 'turn_crashed', 'compaction_error', 'shutdown', 'fetch_failed', 'land_failed', 'outbox_error',"
        " 'low_disk', 'exam_error', 'hard_stop') GROUP BY kind").fetchall()
    if anomalies:
        out.append("Anomalies: " + ", ".join(f"{a['kind']} {a['n']}" for a in anomalies))
    return "\n".join(out)
