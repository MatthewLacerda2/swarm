"""A whole run with a scripted fake model: turns, board, git, spawn, sleep, compaction, budget.

Real sandboxes and git, no API calls.
"""

from __future__ import annotations

import itertools
import re
import shutil
import uuid

import pytest

from swarm import config, gitops, llm, scheduler
from swarm import run as runmod

_ids = itertools.count()


def tool(cmd: str) -> dict:
    return {"type": "tool_use", "id": f"toolu_{next(_ids):06d}", "name": "bash", "input": {"command": cmd}}


SCRIPTS = {
    ("agent-001", 1): [
        'board create "#engine" "the simulator"',
        "git clone -q /swarm/repo.git repo && cd repo && echo hi > a.txt && git add a.txt "
        "&& git commit -qm 'add a' && git push -q 2>&1; echo pushed",
        'agent spawn "please fetch QCEW 2023 data"',
    ],
    ("agent-002", 1): [
        'board post "#engine" "hello @agent-001, I will do the data"',
        'board dm agent-001 "psst, private note"',
        "board sleep",
    ],
    ("agent-003", 1): ["quota", "ls /swarm; cat /swarm/agents.json | head -5"],
    ("agent-001", 2): ["board read '#engine'", "cd repo && git pull -q && cat a.txt"],
}


class FakeLLM:
    def __init__(self, big_context_agent: str | None = None, scripts: dict | None = None):
        self.scripts = SCRIPTS if scripts is None else scripts
        self.calls: list[tuple[str, int]] = []
        self.big = big_context_agent
        self.compactions = 0
        self.seen_feeds: dict[tuple[str, int], str] = {}

    @staticmethod
    def _who(messages: list[dict]) -> tuple[str, int]:
        agent, rnd = "?", 0
        for m in messages:
            if m["role"] != "user" or not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b.get("type") != "text":
                    continue
                if mm := re.search(r"You are (agent-\d{3})", b["text"]):
                    agent = mm.group(1)
                if mm := re.search(r"\[round (\d+)\]", b["text"]):
                    rnd = int(mm.group(1))
        return agent, rnd

    async def sample(self, system, messages, max_tokens):
        agent, rnd = self._who(messages)
        if agent == "?":
            # after a compaction the identity line is gone; the summary carries it
            for b in messages[0]["content"]:
                if b.get("type") == "compaction":
                    agent = b["content"].split()[0]
        last_user = messages[-1]["content"]
        feed_text = "".join(b.get("text", "") for b in last_user if b.get("type") == "text")
        if feed_text:
            self.seen_feeds[(agent, rnd)] = feed_text
        self.calls.append((agent, rnd))
        script = self.scripts.get((agent, rnd), [])
        done = sum(
            1 for m in messages if m["role"] == "user" and any(b.get("type") == "tool_result" for b in m["content"])
            and self._round_of(messages, m) == rnd
        )
        prompt = 89_000 if agent == self.big else 3000
        usage = {"input_tokens": 500, "cache_creation_input_tokens": 0, "cache_read_input_tokens": prompt - 500,
                 "output_tokens": 2000 if agent == self.big else 200}
        thinking = {"type": "thinking", "thinking": f"{agent} thinks", "signature": "sig"}
        if done < len(script):
            content = [thinking, tool(script[done])]
            stop = "tool_use"
        else:
            content = [thinking, {"type": "text", "text": "done for this round"}]
            stop = "end_turn"
        return llm.LLMResponse(content, stop, llm.usage_from_dict(usage), usage)

    def _round_of(self, messages, target):
        rnd = 0
        for m in messages:
            for b in m["content"] if isinstance(m["content"], list) else []:
                if b.get("type") == "text" and (mm := re.search(r"\[round (\d+)\]", b["text"])):
                    rnd = int(mm.group(1))
            if m is target:
                return rnd
        return rnd

    async def compact(self, system, messages, max_tokens):
        self.compactions += 1
        agent, _ = self._who(messages)
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "iterations": [{"type": "compaction", "input_tokens": 91_000, "output_tokens": 800,
                                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]}
        block = {"type": "compaction", "content": f"{agent} summary of everything", "signature": "s"}
        return llm.LLMResponse([block], "compaction", llm.usage_from_dict(usage), usage)


@pytest.fixture
def fake_run():
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    cfg = config.RunConfig(run_id=run_id, task="pilot", initial_agents=3, max_alive=5, max_total=5,
                           run_budget=1.0, agent_budget=0.3, exam_every=100, commands_per_turn=5)
    fake = FakeLLM(big_context_agent="agent-003")
    run = runmod.open_run(cfg, fake)
    yield run, fake
    runmod.delete_run(cfg)


async def test_three_rounds(fake_run):
    run, fake = fake_run
    db = run.db
    await scheduler.run_ticks(run, max_new_ticks=3)
    assert run.tick == 3

    # board
    assert db.one("SELECT creator FROM channels WHERE name = '#engine'")["creator"] == "agent-001"
    post = db.one("SELECT * FROM posts WHERE channel = '#engine'")
    assert post["author"] == "agent-002" and post["tick"] == 1
    assert db.one("SELECT 1 FROM mentions WHERE agent = 'agent-001'")
    feed2 = fake.seen_feeds[("agent-001", 2)]
    assert "psst, private note" in feed2 and "hello @agent-001" in feed2
    assert db.one("SELECT 1 FROM reads WHERE agent = 'agent-001' AND target = '#engine'")

    # git: the push landed on main and agent-001 saw it after pulling
    main = gitops.rev(run.paths.central, "refs/heads/main")
    files = gitops.git("ls-tree", "--name-only", main, git_dir=run.paths.central).stdout.split()
    assert "a.txt" in files
    assert db.one("SELECT result FROM pushes WHERE agent = 'agent-001'")["result"] == "ff"
    assert db.one("SELECT agent FROM commits")["agent"] == "agent-001"
    out = db.one("SELECT output_path FROM commands WHERE agent = 'agent-001' AND tick = 2 AND seq = 2")
    assert open(out["output_path"]).read().strip() == "hi"

    # spawn: agent-004 created in round 1, ran from round 2 with its parent's message
    child = db.agent("agent-004")
    assert child["parent"] == "agent-001" and child["born_tick"] == 2 and child["turns"] >= 1
    assert "please fetch QCEW 2023 data" in fake.seen_feeds[("agent-004", 2)]

    # sleep: agent-002 slept after round 1 and did not run in round 2
    assert ("agent-002", 2) not in fake.calls
    assert db.agent("agent-002")["status"] == "sleeping"

    # compaction: agent-003's context passed 90k and was compacted on demand
    assert fake.compactions >= 1
    assert db.one("SELECT 1 FROM messages WHERE agent = 'agent-003' AND kind = 'compaction'")
    hist = db.history("agent-003")
    assert hist[0]["content"][0]["type"] == "compaction"

    # accounting
    spent = db.total_spent()
    assert spent > 0 and abs(spent - run.budget.spent) < 1e-9
    assert abs(sum(r["spent"] for r in db.agents()) - spent) < 1e-9

    # snapshot exports
    snap = run.paths.snapshot
    assert (snap / "transcripts" / "agent-001.md").read_text().count("## Round") >= 2
    assert (snap / "board.db").exists() and (snap / "agents.json").exists()
    assert "psst" not in (snap / "board.db").read_bytes().decode(errors="ignore")
    dms = (run.paths.private("agent-001") / "dms.jsonl").read_text()
    assert "psst" in dms


async def test_hard_stop_and_wind_down(fake_run):
    run, fake = fake_run
    b = run.budget
    # pretend most of the money is gone
    b.spent = run.cfg.run_budget * 0.995
    b.db.set_meta("thresholds_noticed", [20, 40, 60, 80, 85, 90, 95, 96, 97, 98, 99])
    await scheduler.run_ticks(run, max_new_ticks=1)
    assert b.spent >= run.cfg.run_budget
    assert run.db.one("SELECT 1 FROM notices WHERE text LIKE 'The run''s budget is used up%'")

    b.spent = b.hard_stop - 1e-6
    reason = await scheduler.run_ticks(run, max_new_ticks=3)
    assert reason == "hard_stop"
    assert run.db.get_meta("status") == "paused"
    assert b.spent <= b.hard_stop + 1e-9


async def test_budget_reservation():
    class DummyDB:
        def __init__(self):
            self.meta = {}

        def total_spent(self):
            return 0.0

        def agents(self):
            return []

        def event(self, *a):
            pass

        def execute(self, *a):
            pass

        def get_meta(self, k, d=None):
            return self.meta.get(k, d)

        def set_meta(self, k, v):
            self.meta[k] = v

    cfg = config.RunConfig(run_id="x", run_budget=18, agent_budget=100)
    b = llm.Budget(DummyDB(), cfg, hard_stop=22)
    r1 = await b.reserve("a", 21.0)
    assert isinstance(r1, llm.Reservation)
    assert await b.reserve("b", 1.5) == "hard_stop"
    assert b.hard_stopped
    await b.settle(r1, 18.5)
    assert b.winding_down and b.crossed_thresholds() == list(config.BUDGET_THRESHOLDS)
    assert b.crossed_thresholds() == []


def test_usage_cost():
    u = llm.usage_from_dict({"input_tokens": 1000, "cache_creation_input_tokens": 2000,
                             "cache_creation": {"ephemeral_1h_input_tokens": 2000, "ephemeral_5m_input_tokens": 0},
                             "cache_read_input_tokens": 10000, "output_tokens": 1000})
    assert u.cost == pytest.approx((1000 * 0.10 + 2000 * 0.20 + 10000 * 0.01 + 1000 * 0.50) / 1e6)
    assert u.prompt_tokens == 13000
    big = llm.usage_from_dict({"input_tokens": 101_000, "output_tokens": 0})
    assert big.cost == pytest.approx(101_000 * 0.10 * 5 / 1e6)
    comp = llm.usage_from_dict({"input_tokens": 0, "output_tokens": 0, "iterations": [
        {"type": "compaction", "input_tokens": 50_000, "output_tokens": 1000,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]})
    assert comp.compaction and comp.cost == pytest.approx((50_000 * 0.10 + 1000 * 0.5) / 1e6)


async def test_tick_end_attacks():
    """Symlinks planted in an agent's folder must not make the orchestrator read host files."""
    secret = config.PROJECT / "ARCHITECTURE.md"
    marker = secret.read_text().splitlines()[0]
    scripts = {
        ("agent-001", 1): [
            # outbox file is a symlink to a hidden host file
            f"ln -sf {secret} ~/.outbox/ops.jsonl && echo planted",
        ],
        ("agent-002", 1): [
            # a real outbox op that publishes a symlink to the hidden file
            f"ln -sf {secret} ~/link.md && echo '{{\"op\": \"publish\", \"path\": \"/swarm/agents/agent-002/link.md\"}}' >> ~/.outbox/ops.jsonl",
            'board create "#ok" "still works"',
        ],
        ("agent-003", 1): [
            # replace the push outbox with a symlink to another agent's
            "rm -rf ~/.outbox/push.git && ln -s /swarm/agents/agent-001/.outbox/push.git ~/.outbox/push.git && echo swapped",
        ],
    }
    run_id = f"test-{uuid.uuid4().hex[:8]}"
    cfg = config.RunConfig(run_id=run_id, task="pilot", initial_agents=3, exam_every=100)
    fake = FakeLLM(scripts=scripts)
    run = runmod.open_run(cfg, fake)
    try:
        await scheduler.run_ticks(run, max_new_ticks=1)
        db = run.db
        assert db.one("SELECT 1 FROM events WHERE agent = 'agent-001' AND kind = 'outbox_error'")
        assert not (run.paths.snapshot / "shared" / "agent-002" / "link.md").exists()
        assert db.one("SELECT 1 FROM notices WHERE agent = 'agent-002' AND text LIKE 'Not published%'")
        assert db.one("SELECT 1 FROM channels WHERE name = '#ok'")
        assert db.one("SELECT 1 FROM notices WHERE agent = 'agent-003' AND text LIKE 'Your outbox could not be read%'")
        for path in run.paths.snapshot.rglob("*"):
            if path.is_file() and path.stat().st_size < 5_000_000:
                assert marker not in path.read_text(errors="ignore"), path
        for row in db.all("SELECT text FROM notices"):
            assert marker not in row["text"]
    finally:
        runmod.delete_run(cfg)
