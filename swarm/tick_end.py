"""End of tick: apply outboxes, land pushes, spawn agents, rebuild the snapshot.

Nothing here runs while a sandbox is up, so agents can't race the orchestrator.
Files in agent folders are only read through safeio (no symlinks followed), and
outbox files are moved out of the agent's folder before being read.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

from . import gitops, safeio
from .run import Run, create_agent, write_status

CHANNEL_RE = re.compile(r"^#[a-z0-9][a-z0-9_-]{0,39}$")
AGENT_RE = re.compile(r"@(agent-\d{3})\b")
MAX_OPS_BYTES = 5 * 1024 * 1024
MAX_BODY = 8000
MAX_PUBLISH = 20 * 1024 * 1024


async def end_tick(run: Run, tick: int, agents_with_turns: list[str]) -> None:
    db = run.db
    alive = [r["id"] for r in db.agents() if r["status"] != "finished"] + [
        a for a in agents_with_turns if db.agent(a)["status"] == "finished"
    ]
    order = sorted(set(alive))
    run.rng.shuffle(order)

    # 1. outboxes: creates first (so joins/posts in the same round can use them), then the rest
    ops: list[tuple[str, dict]] = []
    for agent in order:
        ops += [(agent, op) for op in _take_ops(run, agent, tick)]
    for agent, op in ops:
        if op.get("op") == "create":
            _apply(run, tick, agent, op)
    spawns = []
    for agent, op in ops:
        if op.get("op") == "spawn":
            spawns.append((agent, op))
        elif op.get("op") != "create":
            _apply(run, tick, agent, op)

    # 2. pushes
    for agent in order:
        _land_pushes(run, tick, agent)

    # 3. spawns
    for agent, op in spawns:
        _spawn(run, tick, agent, op)

    # 4. snapshot and exports
    rebuild_snapshot(run, tick)
    for agent in order:
        gitops.commit_home(run.paths.home_history(agent), run.paths.home(agent), agent, tick)
    _disk(run, tick)
    _embargo_scan(run, tick)
    for r in db.agents():
        if r["status"] != "finished":
            write_status(run, r["id"], tick + 1)


# --- outbox ops -------------------------------------------------------------------------


def _take_ops(run: Run, agent: str, tick: int) -> list[dict]:
    """Move $HOME/.outbox/ops.jsonl out of the agent's folder, then read it."""
    home = run.paths.home(agent)
    archive = run.paths.archive(agent)
    archive.mkdir(parents=True, exist_ok=True)
    name = f"t{tick:04d}.jsonl"
    try:
        home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
        try:
            out_fd = os.open(".outbox", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd)
        finally:
            os.close(home_fd)
        arch_fd = os.open(archive, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.rename("ops.jsonl", name, src_dir_fd=out_fd, dst_dir_fd=arch_fd)
        finally:
            os.close(out_fd)
            os.close(arch_fd)
    except FileNotFoundError:
        return []
    except OSError as e:
        run.db.event(tick, agent, "outbox_error", str(e))
        return []
    try:
        data = safeio.read_beneath(archive, name, MAX_OPS_BYTES)
    except safeio.UnsafePath as e:
        run.db.event(tick, agent, "outbox_error", str(e))
        return []
    ops = []
    for line in data.decode(errors="replace").splitlines():
        try:
            op = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(op, dict):
            ops.append(op)
    return ops


def _text(value, limit: int = MAX_BODY) -> str:
    return str(value or "")[:limit].strip()


def _apply(run: Run, tick: int, agent: str, op: dict) -> None:
    db = run.db
    kind = op.get("op")
    try:
        if kind == "create":
            ch = _text(op.get("channel"), 41).lower()
            if not CHANNEL_RE.match(ch):
                return
            if db.one("SELECT 1 FROM channels WHERE name = ?", (ch,)):
                db.notice(tick, "board", f"Channel {ch} already existed; you joined it.", agent)
            else:
                db.execute("INSERT INTO channels(name, description, creator, tick) VALUES(?, ?, ?, ?)",
                           (ch, _text(op.get("description"), 300), agent, tick))
                db.event(tick, agent, "channel_created", ch)
            db.execute("INSERT OR IGNORE INTO members(channel, agent, tick) VALUES(?, ?, ?)", (ch, agent, tick))
        elif kind in ("join", "leave"):
            ch = _text(op.get("channel"), 41).lower()
            if not db.one("SELECT 1 FROM channels WHERE name = ?", (ch,)):
                db.notice(tick, "board", f"Could not {kind} {ch}: no such channel.", agent)
                return
            if kind == "join":
                db.execute("INSERT OR IGNORE INTO members(channel, agent, tick) VALUES(?, ?, ?)", (ch, agent, tick))
            else:
                db.execute("DELETE FROM members WHERE channel = ? AND agent = ?", (ch, agent))
            db.event(tick, agent, kind, ch)
        elif kind == "post":
            ch = _text(op.get("channel"), 41).lower()
            if not db.one("SELECT 1 FROM channels WHERE name = ?", (ch,)):
                db.notice(tick, "board", f"Your post to {ch} was not delivered: no such channel "
                          "(create it with `board create`).", agent)
                return
            _post(run, tick, agent, ch, None, _text(op.get("body")), None)
        elif kind == "reply":
            parent = db.one("SELECT id, channel, parent_id FROM posts WHERE id = ?", (int(op.get("parent", 0)),))
            if not parent or parent["channel"] is None:
                db.notice(tick, "board", f"Your reply to post {op.get('parent')} was not delivered: no such post.", agent)
                return
            root = parent["parent_id"] or parent["id"]
            _post(run, tick, agent, parent["channel"], root, _text(op.get("body")), None)
        elif kind == "dm":
            to = _text(op.get("to"), 20)
            if to == agent or not db.one("SELECT 1 FROM agents WHERE id = ?", (to,)):
                db.notice(tick, "board", f"Your message to {to} was not delivered: no such agent.", agent)
                return
            body = _text(op.get("body"))
            if body:
                db.execute("INSERT INTO posts(channel, dm_to, author, tick, body) VALUES(NULL, ?, ?, ?, ?)",
                           (to, agent, tick, body))
        elif kind == "publish":
            _publish(run, tick, agent, op)
        elif kind == "sleep":
            n = op.get("rounds")
            until = tick + int(n) if isinstance(n, int) and n > 0 else None
            if db.agent(agent)["status"] == "active":
                db.set_status(agent, "sleeping", tick, sleep_until=until)
        elif kind == "read":
            db.execute("INSERT INTO reads(agent, tick, kind, target) VALUES(?, ?, ?, ?)",
                       (agent, tick, _text(op.get("kind"), 20), _text(op.get("target"), 100)))
    except (ValueError, TypeError, sqlite3.Error) as e:
        db.event(tick, agent, "bad_op", {"op": str(op)[:500], "error": str(e)})


def _post(run: Run, tick: int, agent: str, channel: str, parent: int | None, body: str,
          attachment: str | None) -> None:
    if not body:
        return
    cur = run.db.execute(
        "INSERT INTO posts(channel, dm_to, author, tick, parent_id, body, attachment) VALUES(?, NULL, ?, ?, ?, ?, ?)",
        (channel, agent, tick, parent, body, attachment),
    )
    post_id = cur.lastrowid
    # Like Slack: posting in a channel makes you a member, so replies show in your feed.
    run.db.execute("INSERT OR IGNORE INTO members(channel, agent, tick) VALUES(?, ?, ?)", (channel, agent, tick))
    for mentioned in set(AGENT_RE.findall(body)):
        if mentioned != agent and run.db.one("SELECT 1 FROM agents WHERE id = ?", (mentioned,)):
            run.db.execute("INSERT INTO mentions(post_id, agent) VALUES(?, ?)", (post_id, mentioned))


def _publish(run: Run, tick: int, agent: str, op: dict) -> None:
    path = _text(op.get("path"), 1000)
    roots = {
        f"/swarm/agents/{agent}/": run.paths.home(agent),
        f"/swarm/bulk/{agent}/": run.paths.bulk(agent),
    }
    for prefix, host_root in roots.items():
        if path.startswith(prefix):
            rel = path[len(prefix):]
            break
    else:
        run.db.notice(tick, "board", f"Not published: {path} is not in your $HOME or $BULK.", agent)
        return
    try:
        data = safeio.read_beneath(host_root, rel, MAX_PUBLISH)
    except safeio.UnsafePath as e:
        run.db.notice(tick, "board", f"Not published: {path} ({e}).", agent)
        return
    dest_dir = run.paths.snapshot / "shared" / agent
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(rel).name
    tmp = dest_dir / f".{name}.tmp"
    tmp.write_bytes(data)
    tmp.replace(dest_dir / name)
    shared = f"/swarm/shared/{agent}/{name}"
    run.db.event(tick, agent, "publish", shared)
    ch = _text(op.get("channel"), 41).lower()
    if ch and run.db.one("SELECT 1 FROM channels WHERE name = ?", (ch,)):
        _post(run, tick, agent, ch, None, f"Published {shared}", shared)


# --- pushes and spawns ---------------------------------------------------------------------


def _land_pushes(run: Run, tick: int, agent: str) -> None:
    home = run.paths.home(agent)
    ok, err = gitops.sandboxed_fetch(home / ".outbox" / "push.git", run.paths.staging(agent), home)
    if not ok:
        run.db.event(tick, agent, "fetch_failed", err)
        run.db.notice(tick, "git", "Your outbox could not be read, so your pushes did not land. "
                      f"Make sure $HOME/.outbox/push.git is a valid bare repository. ({err[:300]})", agent)
        return
    try:
        seen = run.db.get_meta(f"outbox_seen:{agent}", {})
        results = gitops.land(run.paths.central, run.paths.staging(agent), agent, tick,
                              run.cfg.push_limit_mb * 1024 * 1024, seen)
    except gitops.GitError as e:
        run.db.event(tick, agent, "land_failed", str(e)[:2000])
        run.db.notice(tick, "git", "Your pushes could not be processed this round.", agent)
        return
    for r in results:
        if r.pushed:
            seen[r.branch] = r.pushed
    run.db.set_meta(f"outbox_seen:{agent}", seen)
    for r in results:
        if r.result == "unchanged":
            continue
        run.db.execute(
            "INSERT INTO pushes(agent, tick, branch, result, detail, old, new) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (agent, tick, r.branch, r.result, r.detail, r.old, r.new),
        )
        for c in r.commits or []:
            run.db.execute("INSERT OR IGNORE INTO commits(hash, agent, tick) VALUES(?, ?, ?)", (c, agent, tick))
        text = {
            "new": f"Your push created branch {r.branch}.",
            "ff": f"Your push to {r.branch} landed.",
            "merge": f"Your push to {r.branch} landed, merged with others' changes. Pull before your next push.",
            "rejected": f"Your push to {r.branch} was rejected: {r.detail}. Pull, fix and push again.",
        }[r.result]
        run.db.notice(tick, "git", text, agent)


def _spawn(run: Run, tick: int, parent: str, op: dict) -> None:
    db = run.db
    prompt = _text(op.get("prompt"))
    alive = db.one("SELECT COUNT(*) AS n FROM agents WHERE status != 'finished'")["n"]
    total = db.one("SELECT COUNT(*) AS n FROM agents")["n"]
    reason = None
    if run.budget.winding_down:
        reason = "the run's budget is used up"
    elif alive >= run.cfg.max_alive:
        reason = f"{run.cfg.max_alive} agents are already alive"
    elif total >= run.cfg.max_total:
        reason = f"{run.cfg.max_total} agents have already been created"
    elif not prompt:
        reason = "the message was empty"
    if reason:
        db.notice(tick, "spawn", f"Your new agent was not created: {reason}.", parent)
        db.event(tick, parent, "spawn_refused", reason)
        return
    child = create_agent(run, tick + 1, parent, prompt)
    db.notice(tick, "spawn", f"Your new agent {child} was created and starts next round.", parent)
    db.event(tick, parent, "spawn", child)


# --- snapshot -----------------------------------------------------------------------------


def rebuild_snapshot(run: Run, tick: int) -> None:
    snap = run.paths.snapshot
    (snap / "shared").mkdir(parents=True, exist_ok=True)
    (snap / "transcripts").mkdir(parents=True, exist_ok=True)

    nxt = snap / "repo.git.next"
    shutil.rmtree(nxt, ignore_errors=True)
    gitops.mirror(run.paths.central, nxt)
    old = snap / "repo.git.old"
    shutil.rmtree(old, ignore_errors=True)
    if (snap / "repo.git").exists():
        (snap / "repo.git").rename(old)
    nxt.rename(snap / "repo.git")
    shutil.rmtree(old, ignore_errors=True)

    _export_board(run, snap / "board.db")
    agents = [
        {"id": r["id"], "status": r["status"], "parent": r["parent"], "born": r["born_tick"]}
        for r in run.db.agents()
    ]
    alive = sum(a["status"] != "finished" for a in agents)
    data = {"round": tick, "alive": alive, "total": len(agents), "max_alive": run.cfg.max_alive,
            "max_total": run.cfg.max_total, "agents": agents}
    _atomic_write(snap / "agents.json", json.dumps(data, indent=1))
    for a in agents:
        _export_dms(run, a["id"])
        _export_transcript(run, a["id"])


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _export_board(run: Run, dest: Path) -> None:
    tmp = dest.with_name("board.db.tmp")
    tmp.unlink(missing_ok=True)
    out = sqlite3.connect(tmp)
    out.executescript(
        "CREATE TABLE channels(name TEXT PRIMARY KEY, description TEXT, creator TEXT, tick INTEGER);"
        "CREATE TABLE members(channel TEXT, agent TEXT, tick INTEGER);"
        "CREATE TABLE posts(id INTEGER PRIMARY KEY, channel TEXT, author TEXT, tick INTEGER,"
        " parent_id INTEGER, body TEXT, attachment TEXT);"
        "CREATE INDEX posts_channel ON posts(channel, id);"
    )
    db = run.db
    out.executemany("INSERT INTO channels VALUES(?, ?, ?, ?)",
                    [tuple(r) for r in db.all("SELECT name, description, creator, tick FROM channels")])
    out.executemany("INSERT INTO members VALUES(?, ?, ?)",
                    [tuple(r) for r in db.all("SELECT channel, agent, tick FROM members")])
    out.executemany("INSERT INTO posts VALUES(?, ?, ?, ?, ?, ?, ?)", [
        tuple(r) for r in db.all(
            "SELECT id, channel, author, tick, parent_id, body, attachment FROM posts WHERE channel IS NOT NULL")
    ])
    out.commit()
    out.close()
    tmp.replace(dest)


def _export_dms(run: Run, agent: str) -> None:
    rows = run.db.all(
        "SELECT id, author, dm_to, tick, body FROM posts WHERE channel IS NULL AND (author = ? OR dm_to = ?) ORDER BY id",
        (agent, agent),
    )
    lines = [json.dumps({"id": r["id"], "from": r["author"], "to": r["dm_to"], "tick": r["tick"],
                         "body": r["body"]}, ensure_ascii=False) for r in rows]
    private = run.paths.private(agent)
    private.mkdir(parents=True, exist_ok=True)
    _atomic_write(private / "dms.jsonl", "\n".join(lines) + ("\n" if lines else ""))


def _export_transcript(run: Run, agent: str) -> None:
    """Append messages since the last export to /swarm/transcripts/<agent>.md."""
    key = f"transcript_last:{agent}"
    last = run.db.get_meta(key, 0)
    rows = run.db.all("SELECT id, tick, role, kind, content FROM messages WHERE agent = ? AND id > ? ORDER BY id",
                      (agent, last))
    if not rows:
        return
    parts = []
    current_tick = run.db.get_meta(f"transcript_tick:{agent}", None)
    for r in rows:
        if r["tick"] != current_tick:
            current_tick = r["tick"]
            parts.append(f"\n## Round {current_tick}\n")
        parts.append(_render(r["role"], r["kind"], json.loads(r["content"])))
    with (run.paths.snapshot / "transcripts" / f"{agent}.md").open("a") as f:
        if last == 0:
            f.write(f"# Transcript of {agent}\n")
        f.write("\n".join(parts) + "\n")
    run.db.set_meta(key, rows[-1]["id"])
    run.db.set_meta(f"transcript_tick:{agent}", current_tick)


def _render(role: str, kind: str, content) -> str:
    if kind == "compaction":
        summary = content[0].get("content", "") if content else ""
        return f"**[memory compacted]**\n\n{summary}\n"
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    out = []
    for b in content:
        t = b.get("type")
        if t == "text":
            label = "**feed**" if role == "user" else "**says**"
            out.append(f"{label}\n\n{b['text']}\n")
        elif t == "thinking":
            if b.get("thinking"):
                out.append("**thinking (summary)**\n\n> " + b["thinking"].replace("\n", "\n> ") + "\n")
        elif t == "tool_use":
            inp = b.get("input") or {}
            cmd = inp.get("command", "(restart)" if inp.get("restart") else json.dumps(inp))
            out.append(f"**runs**\n\n```bash\n{cmd}\n```\n")
        elif t == "tool_result":
            text = b.get("content")
            if isinstance(text, list):
                text = "\n".join(x.get("text", "") for x in text if isinstance(x, dict))
            text = str(text or "")
            if len(text) > 3000:
                text = text[:1500] + "\n[...]\n" + text[-1500:]
            out.append(f"**output**\n\n```\n{text}\n```\n")
    return "\n".join(out)


# --- disk and embargo -----------------------------------------------------------------------


def _du(path: Path) -> int:
    if not path.exists():
        return 0
    res = subprocess.run(["du", "-s", "-B1", "-x", str(path)], capture_output=True, text=True)
    try:
        return int(res.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def _disk(run: Run, tick: int) -> None:
    cfg = run.cfg
    totals = {"home": 0, "bulk": 0}
    for r in run.db.agents():
        agent = r["id"]
        home, bulk = _du(run.paths.home(agent)), _du(run.paths.bulk(agent))
        totals["home"] += home
        totals["bulk"] += bulk
        run.db.execute("INSERT OR REPLACE INTO disk(tick, agent, home_bytes, bulk_bytes) VALUES(?, ?, ?, ?)",
                       (tick, agent, home, bulk))
        if r["status"] == "finished":
            continue
        over = home > cfg.home_quota_gb * 1e9 or bulk > cfg.bulk_quota_gb * 1e9
        run.db.execute("UPDATE agents SET over_quota = ? WHERE id = ?", (int(over), agent))
        for label, used, quota in (("$HOME", home, cfg.home_quota_gb), ("$BULK", bulk, cfg.bulk_quota_gb)):
            _threshold_notice(run, tick, agent, f"disk:{agent}:{label}", label, used / (quota * 1e9), agent)
    run.extra["disk_totals"] = totals
    _threshold_notice(run, tick, None, "disk:all:home", "all agents' $HOME", totals["home"] / (cfg.home_total_gb * 1e9))
    _threshold_notice(run, tick, None, "disk:all:bulk", "all agents' $BULK", totals["bulk"] / (cfg.bulk_total_gb * 1e9))

    low = []
    for path in (run.paths.root, cfg.bulk_dir):
        st = os.statvfs(path)
        if st.f_bavail * st.f_frsize < cfg.min_free_gb * 1e9:
            low.append(str(path))
    if totals["home"] > cfg.home_total_gb * 1e9 or totals["bulk"] > cfg.bulk_total_gb * 1e9:
        low.append("run total quota")
    run.extra["low_disk"] = low
    if low:
        run.db.event(tick, None, "low_disk", low)


def _threshold_notice(run: Run, tick: int, agent: str | None, key: str, label: str, frac: float,
                      who: str | None = None) -> None:
    level = 95 if frac >= 0.95 else 80 if frac >= 0.80 else 0
    prev = run.db.get_meta(key, 0)
    if level > prev:
        whose = "Your" if who else "Disk:"
        text = (f"{whose} {label} is at {frac * 100:.0f}% of its quota." if who
                else f"Disk: {label} is at {frac * 100:.0f}% of the run's quota.")
        run.db.notice(tick, "disk", text, agent)
    if level != prev:
        run.db.set_meta(key, level)


EMBARGO_URL = re.compile(
    r"(?i)(bls\.gov|census\.gov|bea\.gov|data\.gov|stlouisfed\.org|ipums|kaggle|github)[^\s'\"]*"
)
OLD_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|200\d|201\d|2020|2021)(?!\d)")
DATA_WORDS = re.compile(r"(?i)(qcew|cew|qtrly|annual|cbp|zbp|laus|ces|gdp|cagdp|cainc|emp|wage|unemploy|naics)")
# Structural tables of any year are allowed (input-output, commuting, freight, geography).
STRUCTURAL = re.compile(
    r"(?i)(input-output|io-annual|io-benchmark|/io/|makeuse|make_?use|_use_|use_|_use\b|supply|"
    r"lodes|lehd|/cfs|commodity|tiger|gazetteer|shapefile|crosswalk)"
)


def _embargo_scan(run: Run, tick: int) -> None:
    """Flag commands and files that look like pre-2022 outcome data. Heuristic; reviewed by us."""
    seen = {r["evidence"] for r in run.db.all("SELECT evidence FROM embargo")}

    def flag(agent: str, kind: str, evidence: str) -> None:
        if evidence not in seen:
            seen.add(evidence)
            run.db.execute("INSERT INTO embargo(agent, tick, kind, evidence) VALUES(?, ?, ?, ?)",
                           (agent, tick, kind, evidence[:1000]))

    for c in run.db.all("SELECT agent, cmd FROM commands WHERE tick = ?", (tick,)):
        for m in EMBARGO_URL.finditer(c["cmd"]):
            url = m.group(0)
            if OLD_YEAR.search(url) and DATA_WORDS.search(url) and not STRUCTURAL.search(url):
                flag(c["agent"], "command", url)

    for r in run.db.agents():
        agent = r["id"]
        for root in (run.paths.home(agent), run.paths.bulk(agent)):
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in (".git", ".outbox", "node_modules", ".venv")]
                for name in filenames:
                    if OLD_YEAR.search(name) and DATA_WORDS.search(name) and not STRUCTURAL.search(name):
                        flag(agent, "file", os.path.join(dirpath, name))
